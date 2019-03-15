# coding: utf-8
import json
import numpy as np
import os
from PIL import Image
import glob
from pycocotools.coco import COCO
import math
import pandas as pd
import torch
from torch import nn
import itertools
from tqdm import tqdm
from skimage import io
import cv2
import matplotlib.pyplot as plt
import time
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector
from mmdet.ops.nms.nms_wrapper import nms, soft_nms


# from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


# copy from detectron
def box_voting(top_dets, all_dets, thresh, scoring_method='ID', beta=1.0):
    """Apply bounding-box voting to refine `top_dets` by voting with `all_dets`.
    See: https://arxiv.org/abs/1505.01749. Optional score averaging (not in the
    referenced  paper) can be applied by setting `scoring_method` appropriately.
    """
    # top_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    # all_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    top_dets_out = top_dets.copy()
    top_boxes = top_dets[:, :4]
    all_boxes = all_dets[:, :4]
    all_scores = all_dets[:, 4]
    top_to_all_overlaps = bbox_overlaps(top_boxes, all_boxes)
    for k in range(top_dets_out.shape[0]):
        inds_to_vote = np.where(top_to_all_overlaps[k] >= thresh)[0]
        boxes_to_vote = all_boxes[inds_to_vote, :]
        ws = all_scores[inds_to_vote]
        top_dets_out[k, :4] = np.average(boxes_to_vote, axis=0, weights=ws)
        if scoring_method == 'ID':
            # Identity, nothing to do
            pass
        elif scoring_method == 'TEMP_AVG':
            # Average probabilities (considered as P(detected class) vs.
            # P(not the detected class)) after smoothing with a temperature
            # hyperparameter.
            P = np.vstack((ws, 1.0 - ws))
            P_max = np.max(P, axis=0)
            X = np.log(P / P_max)
            X_exp = np.exp(X / beta)
            P_temp = X_exp / np.sum(X_exp, axis=0)
            P_avg = P_temp[0].mean()
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'AVG':
            # Combine new probs from overlapping boxes
            top_dets_out[k, 4] = ws.mean()
        elif scoring_method == 'IOU_AVG':
            P = ws
            ws = top_to_all_overlaps[k, inds_to_vote]
            P_avg = np.average(P, weights=ws)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'GENERALIZED_AVG':
            P_avg = np.mean(ws**beta)**(1.0 / beta)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'QUASI_SUM':
            top_dets_out[k, 4] = ws.sum() / float(len(ws))**beta
        else:
            raise NotImplementedError(
                'Unknown scoring method {}'.format(scoring_method)
            )

    return top_dets_out


class ShowDetResult(object):
    @staticmethod
    def store_bbox(img, result, score_thr=0.3, out_file=None):
        class_names = ['steel', ]
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(result)
        ]
        labels = np.concatenate(labels)
        bboxes = np.vstack(result)
        img = mmcv.imread(img)
        mmcv.imshow_det_bboxes(
            img.copy(),
            bboxes,
            labels,
            show=False,
            class_names=class_names,
            score_thr=score_thr,
            out_file=out_file,
        )

    @staticmethod
    def infer_and_store_bbox(gpu_id=0, show=True):
        work_dir = 'cascade_rcnn_r101_fpn_120x'
        config_file_name = 'cascade_rcnn_x101_64x4d_fpn_1x.py'

        mode = 'train'
        # dump_path = f'weights/{work_dir}/{mode}_b120.pkl'
        # csv_path = f'weights/{work_dir}/cascade120_{mode}.csv'
        checkpoint_path = 'weights/{}/epoch_120.pth'.format(work_dir)
        save_path = f'/workspace/nas/gangjin/show120_{mode}/'
        img_path = f'/workspace/mmdetection/gangjin/{mode}_dataset/'
        os.makedirs(save_path, exist_ok=True)
        # ############
        cfg = mmcv.Config.fromfile(f'myconfigs/{config_file_name}')
        cfg.model.pretrained = None
        # construct the model and load checkpoint
        model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
        _ = load_checkpoint(model, checkpoint_path, map_location=f'cuda:{str(gpu_id)}')

        # test a single image
        # img = mmcv.imread(img_path+'0.jpg')
        # result = inference_detector(model, img, cfg, device='cuda:0')
        # store_result(img, result, out_file=save_path+'det_0.jpg')

        # test a list of images
        imgs = sorted(glob.glob(img_path+'*jpg'))
        print(len(imgs))
        # all_rlt = []
        for i, result in enumerate(inference_detector(model, imgs, cfg, f'cuda:{str(gpu_id)}')):
            print(i, imgs[i])
            # all_rlt.append(result)
            if show:
                ShowDetResult.store_bbox(imgs[i], result, score_thr=0.5, out_file=save_path+'det_{}'.format(os.path.basename(imgs[i])))
        # mmcv.dump(all_rlt, dump_path)
        # TestPkl2SubmitFormat.test_pkl2csv_format(all_rlt, csv_path, train=(mode == 'train'))

    @staticmethod
    def show_pkl(pkl_file, mode, save_path):
        from mmdet.datasets.coco import CocoDataset
        # mode = 'train'
        # save_path = f'/workspace/nas/gangjin/show120_{mode}/'
        data_path = f'gangjin/{mode}_dataset/'
        os.makedirs(save_path, exist_ok=True)

        pkl = mmcv.load(pkl_file)
        print('pkl len: ', len(pkl))
        if mode == 'train':
            ann_file = 'gangjin/round1_tra.json'
        else:
            ann_file = 'gangjin/round1_test_a.json'

        dataset = CocoDataset(ann_file, None, (0, 0), {}, test_mode=True)
        img_names = [dataset.img_infos[idx]['filename'] for idx in range(len(dataset))]
        print('img num: ', len(img_names))
        for idx, (img_nm, result) in enumerate(zip(img_names, pkl)):
            print(idx, img_nm)
            ShowDetResult.store_bbox(data_path+img_nm, result, 0.5, save_path + f'det_{img_nm}')

    @staticmethod
    def infer_and_store_mask(gpu_id=0):
        work_dir = 'mask_rcnn_r50_fpn_1x'
        config_file_name = 'dp_mask_rcnn_r50_fpn_1x.py'

        checkpoint_path = 'dp_weights/{}/epoch_120.pth'.format(work_dir)
        save_path = f'/workspace/mmdetection-0p6rc/dp_data_shuangyangqu/show_all_1/'
        img_path = f'/workspace/mmdetection-0p6rc/dp_data_shuangyangqu/all_data_jpg/'  # images  all_data_jpg
        img_path1 = f'/workspace/mmdetection-0p6rc/dp_data_shuangyangqu/images/'  # images  all_data_jpg
        imgs1 = sorted(glob.glob1(img_path1, '*jpg'))
        os.makedirs(save_path, exist_ok=True)
        # ############
        cfg = mmcv.Config.fromfile(f'myconfigs/{config_file_name}')
        cfg.model.pretrained = None
        # construct the model and load checkpoint
        model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
        _ = load_checkpoint(model, checkpoint_path, map_location=f'cuda:{str(gpu_id)}')

        # test a single image
        # img = mmcv.imread(img_path+'0.jpg')
        # result = inference_detector(model, img, cfg, device='cuda:0')
        # store_result(img, result, out_file=save_path+'det_0.jpg')

        # test a list of images
        imgs = sorted(glob.glob(img_path + '*jpg'))
        print('img num: ', len(imgs))
        # all_rlt = []
        from pycocotools import mask as mask_utils
        t_start = time.time()
        for i, result in enumerate(inference_detector(model, imgs, cfg, f'cuda:{str(gpu_id)}')):
            print(i, imgs[i])
            img = io.imread(imgs[i])
            img_nm = os.path.basename(imgs[i])

            det_bboxes = result[0]  # [cls_ndarray]
            det_masks = result[1]

            if len(det_masks[0]) == 0:
                # det_masks = np.zeros_like(img)
                io.imsave(save_path+img_nm, img)
                continue
            else:
                # try:
                if img_nm in imgs1:
                    prefix = 'det_train_'
                else:
                    prefix = 'det_'
                det_masks = mask_utils.decode(det_masks[0])  # for class
                # except:
                #     print(det_masks)
            ShowDetResult.show_mask(det_masks, img, save_path, prefix+img_nm)
            # img_nm = os.path.basename(imgs[i])
            # img_nm = os.path.splitext(img_nm)[0] + '.npy'
            # np.save(save_path + img_nm, det_masks)
            # print(img_nm, det_masks.dtype)
            # continue
        print(f'time elapsed: {(time.time()-t_start)/60:.3f} min')

    @staticmethod
    def show_mask(masks, im, output_dir, output_name):
        """
        :param masks: list of mask, each mask is a ndarray(h, w, c) which represents the mask of the same class
        :param output_dir:
        :param output_name:
        :param im:
        :return:
        """
        if isinstance(masks, list):
            masks = masks[0]
        # only one class
        dpi = 72
        cmap = plt.get_cmap('jet')  # rainbow
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]
        im = np.concatenate((im, im), axis=1)
        fig = plt.figure(frameon=False)
        fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)
        ax.imshow(im)

        for i in range(masks.shape[-1]):
            e = masks[:, :, i]
            contour, hier = cv2.findContours(
                e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            for c in contour:
                polygon = plt.Polygon(
                    c.reshape((-1, 2)),
                    fill=True, facecolor=cmap(np.random.rand()),  # colors[i],
                    edgecolor='w', linewidth=1.2,
                    alpha=0.5)
                ax.add_patch(polygon)
        fig.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=dpi)
        plt.close('all')


class SubmitFormat(object):
    @staticmethod  # multi class
    def test_pkl2json_format(pkl_file='test_a.pkl', suffix=None):
        """
        convert out.pkl generated by test.py to required json format
        .pkl file format generated by test.py is a list of len (pic_num, ) of list
        like [[cls1_det, cls2_det, ...], ...],
        cls1_det is a numpy.array of shape (box_num, 5) [xmin, ymin, xmax, ymax, confidence].

        :return: json format
        """

        pkl = mmcv.load(pkl_file)

        img_path = '/workspace/nas/lvdefect/guangdong_round2_test_b_20181106/'

        # order of img_names must match img order generated by json_annotations_for_test
        # use the generated json file is ok
        img_names = sorted([nm for nm in os.listdir(img_path) if 'jpg' in nm])
        assert len(img_names) == 1000

        rlt = dict()
        rlt['results'] = []
        for i in range(len(img_names)):
            tp_dict = dict()
            tp_dict['filename'] = img_names[i]
            tp_dict['rects'] = []

            for idx, boxes in enumerate(pkl[i]):
                if len(boxes) == 0:
                    continue
                for j in range(boxes.shape[0]):
                    b_dict = dict()
                    b_dict['xmin'] = int(boxes[j, 0])
                    b_dict['ymin'] = int(boxes[j, 1])
                    b_dict['xmax'] = int(boxes[j, 2])
                    b_dict['ymax'] = int(boxes[j, 3])
                    b_dict['confidence'] = float(boxes[j, 4])
                    b_dict['label'] = 'defect{}'.format(idx)

                    tp_dict['rects'].append(b_dict)

            rlt['results'].append(tp_dict)

        with open('/workspace/nas/lvdefect/bbox_rlt_{}.json'.format(suffix), 'w') as fp:
            json.dump(rlt, fp=fp, ensure_ascii=False, indent=4, separators=(',', ': '))

        print('done!')

    @staticmethod  # single class
    def test_pkl2csv_format(pkl_file=None, save_path=None, train=False):
        """
        convert out.pkl generated by test.py to required json format
        .pkl file format generated by test.py is a list of len (pic_num, ) of list
        like [[cls1_det, cls2_det, ...], ...],
        cls1_det is a numpy.array of shape (box_num, 5) [xmin, ymin, xmax, ymax, confidence].

        :return: csv format
        """
        if isinstance(pkl_file, str):
            pkl = mmcv.load(pkl_file)
        else:
            pkl = pkl_file

        img_names = SubmitFormat.get_img_names(train)
        rlt = []
        for idx, img_nm in enumerate(img_names):
            det = pkl[idx][0]  # ndarray n*5  (['stage0', 'stage1', 'stage2', 'ensemble']
            # det = pkl[idx][0]  # ndarray n*5
            det = np.round(det,).astype(np.int32)

            for i in range(det.shape[0]):
                out_str = str(det[i, 0])+' '+str(det[i, 1])+' '+str(det[i, 2])+' '+str(det[i, 3])
                rlt.append([img_nm, out_str])
        df = pd.DataFrame(rlt)
        df.to_csv(save_path, index=None, header=None)

    @staticmethod
    def get_img_names(train: bool):
        from mmdet.datasets.coco import CocoDataset
        data_root = 'gangjin/'
        if train:
            # img_path = '/workspace/mmdetection/gangjin/train_dataset/'
            ann_file = data_root + 'round1_tra.json'
            dataset = CocoDataset(ann_file, None, (0, 0), {}, test_mode=True)
        else:
            # img_path = '/workspace/mmdetection/gangjin/test_dataset/'
            ann_file = data_root + 'round1_test_a.json'
            dataset = CocoDataset(ann_file, None, (0, 0), {}, test_mode=True)
        # order of img_names must match img order generated by json_annotations_for_test
        # use the generated json file is ok
        # img_names = sorted([nm for nm in os.listdir(img_path) if 'jpg' in nm])
        img_names = [dataset.img_infos[idx]['filename'] for idx in range(len(dataset))]

        return img_names

    @staticmethod
    def infer_and_gen_coco_json(gpu_id=0):
        work_dir = 'mask_rcnn_r50_fpn_1x'
        config_file_name = 'dp_mask_rcnn_r50_fpn_1x.py'

        checkpoint_path = 'dp_weights/{}/epoch_120.pth'.format(work_dir)
        save_path = f'/workspace/mmdetection-0p6rc/dp_data_shuangyangqu/show_all/'
        img_path = f'/workspace/mmdetection-0p6rc/dp_data_shuangyangqu/images/'  # images  all_data_jpg
        os.makedirs(save_path, exist_ok=True)
        # ############
        cfg = mmcv.Config.fromfile(f'myconfigs/{config_file_name}')
        cfg.model.pretrained = None
        # construct the model and load checkpoint
        model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
        _ = load_checkpoint(model, checkpoint_path, map_location=f'cuda:{str(gpu_id)}')

        # test a single image
        # img = mmcv.imread(img_path+'0.jpg')
        # result = inference_detector(model, img, cfg, device='cuda:0')
        # store_result(img, result, out_file=save_path+'det_0.jpg')

        # test a list of images
        imgs = sorted(glob.glob(img_path + '*jpg'))
        print('img num: ', len(imgs))
        # all_rlt = []
        from pycocotools import mask as mask_utils
        final_json = dict()
        final_json['categories'] = []
        final_json['images'] = []
        final_json['annotations'] = []
        cat_list = ['dp']
        for i, cat in enumerate(cat_list):
            category = dict()
            category['id'] = i+1
            category['name'] = cat
            category['supercategory'] = cat
            final_json['categories'].append(category)

        t_start = time.time()
        annotation_id = 0
        for i, result in enumerate(inference_detector(model, imgs, cfg, f'cuda:{str(gpu_id)}')):
            print(i, imgs[i])
            img = io.imread(imgs[i])
            img_nm = os.path.basename(imgs[i])
            #
            image = dict()
            image['id'] = i
            image['file_name'] = img_nm
            image['height'] = img.shape[0]
            image['width'] = img.shape[1]
            final_json['images'].append(image)

            det_bboxes = result[0]  # [cls_ndarray]
            det_masks = result[1]
            assert len(det_bboxes[0]) == len(det_masks[0])

            for j, cat_masks in enumerate(det_masks):
                cls_id = j + 1
                if len(cat_masks) == 0:
                    continue
                # print(cat_masks)
                # npy_masks = mask_utils.decode(cls_masks)  # for class

                for cat_mask in cat_masks:
                    cat_mask['counts'] = cat_mask['counts'].decode()
                    annotation = dict()
                    annotation['id'] = annotation_id
                    annotation['image_id'] = i
                    annotation['category_id'] = cls_id
                    annotation['iscrowd'] = 0
                    annotation['area'] = float(mask_utils.area(cat_mask))
                    annotation['bbox'] = mask_utils.toBbox(cat_mask).tolist()
                    annotation['segmentation'] = cat_mask
                    annotation_id += 1
                    final_json['annotations'].append(annotation)
        with open('dp_data_shuangyangqu/det.json', 'w') as fp:
            json.dump(final_json, fp, ensure_ascii=False, indent=4)

        print(f'time elapsed: {time.time()-t_start/60:.3f} sec')


class FilterPKL(object):
    @staticmethod
    def filter_pkl_by_score(pkl, score=0.6):
        new_all_pic = []
        box_num = 0

        for i in range(len(pkl)):
            pic = []
            for cs in pkl[i]:
                if len(cs) == 0:
                    pic.append(cs)
                    continue
                k = cs[:, -1] > score
                box_num += np.sum(k)
                pic.append(cs[k, :])
            new_all_pic.append(pic)
        return new_all_pic, box_num

    @staticmethod
    def filter_pkl_by_scale(pkl, scale=750, gt=True):
        new_all_pic = []
        box_num = 0
        for i in range(len(pkl)):
            pic = []
            for cs in pkl[i]:
                if len(cs) == 0:
                    pic.append(cs)
                    continue
                if gt:
                    k = (cs[:, 2]-cs[:, 0])*(cs[:, 3]-cs[:, 1]) > (scale*scale)
                else:
                    k = (cs[:, 2] - cs[:, 0]) * (cs[:, 3] - cs[:, 1]) < (scale * scale)
                box_num += np.sum(k)
                pic.append(cs[k, :])
            new_all_pic.append(pic)
        return new_all_pic, box_num

    @staticmethod
    def filter_pkl_by_nms(pkl, nms_thr=0.5):
        new_all_pic = []
        box_num = 0
        for i in range(len(pkl)):
            pic = []
            for cb in pkl[i]:
                if len(cb) == 0:
                    pic.append(cb)
                    continue
                new_dets, k = nms(cb, nms_thr)
                box_num += k.shape[0]
                pic.append(new_dets)
            new_all_pic.append(pic)
        return new_all_pic, box_num


class MergePKLs(object):

    @staticmethod
    def merge_pkls_by_nms(nms_thr=0.7, is_save=False):
        work_dir1 = 'cascade_rcnn_r50_fpn_1x_blur_800_rf_rf'
        work_dir2 = 'cascade_rcnn_r50_fpn_1x_blur_800_rf_rf'
        work_dir3 = 'cascade_rcnn_dconv_r50_fpn_1x_3s_all_fz-1_sm0p5'
        if is_save:
            # dump_name = f'weights/{work_dir}/merge.pkl'
            dump_name = f'weights/merge.pkl'

        def pkls_prepare():
            pkls = [
                f'weights/{work_dir1}/test_b50_800_rf_rf.pkl',
                f'weights/{work_dir2}/test_b50_800_rf_rf_2000.pkl',
                # f'weights/{work_dir3}/test_b150.pkl',
            ]
            ps = [mmcv.load(p) for p in pkls]

            assert len(ps[0]) == len(ps[1])
            print('pic num: ', len(ps[0]))

            return ps

        fs = pkls_prepare()
        new_all_pic = MergePKLs._merge_pkls_by_nms(fs, nms_thr, soft=False)
        # mean_ap(pkl_file=new_all_pic)
        if is_save:
            mmcv.dump(new_all_pic, dump_name)
        return new_all_pic

    @staticmethod
    def _merge_pkls_by_nms(pkls, nms_ratio, soft):
        new_all_pic = []

        for i in range(len(pkls[0])):
            # print('pic: ', i+1)
            pic = []
            pk = [pkl[i] for pkl in pkls]
            for cs in zip(*pk):
                c = np.vstack(cs)
                if nms_ratio < 1.0:
                    if soft:
                        new_dets, keep = soft_nms(c, nms_ratio, 'linear')
                    else:
                        new_dets, keep = nms(c, nms_ratio)
                else:
                    new_dets = c
                pic.append(new_dets)
            new_all_pic.append(pic)
        return new_all_pic


def aug_test(gpu_id=0):
    work_dir = 'cascade_rcnn_r50_fpn_1x_3s_all_aug_fz-1_ohem12_sm0p5'
    config_file_name = 'cascade_rcnn_r50_fpn_1x_0p991586.py'

    mode = 'test'
    dump_path = f'weights/{work_dir}/aug_{mode}_150_nms0p7.pkl'
    csv_path = f'weights/{work_dir}/aug_{mode}_150_nms0p7.csv'
    checkpoint_path = 'weights/{}/epoch_150.pth'.format(work_dir)
    img_path = f'/workspace/mmdetection/gangjin/{mode}_dataset/'
    # ############
    from mmdet.datasets.transforms import bbox_flip, bbox_flip_ud, bbox_transpose
    cfg = mmcv.Config.fromfile(f'myconfigs/{config_file_name}')
    cfg.model.pretrained = None

    # construct the model and load checkpoint
    model = build_detector(cfg.model, test_cfg=cfg.test_cfg)

    _ = load_checkpoint(model, checkpoint_path, map_location=f'cuda:{str(gpu_id)}')

    img_nms = SubmitFormat.get_img_names(mode == "train")
    print(len(img_nms))
    # test a single image
    final_rlt = []
    for nm in tqdm(img_nms):
        orig_img = mmcv.imread(img_path+nm)
        aug_rlt = []
        img_shape = orig_img.shape
        for flip, flip_ud, transpose in itertools.product(*itertools.repeat([True, False], 3)):
            img = orig_img.copy()
            img = img if not flip else mmcv.imflip(img, direction='horizontal')
            img = img if not flip_ud else mmcv.imflip(img, direction='vertical')
            img = img if not transpose else img.transpose(1, 0, 2)
            result1 = inference_detector(model, img, cfg, device=f'cuda:{str(gpu_id)}')
            result = [rlt[:, 0:4] for rlt in result1]
            result = result if not transpose else [bbox_transpose(bboxes, None) for bboxes in result]
            result = result if not flip else [bbox_flip(bboxes, img_shape) for bboxes in result]
            result = result if not flip_ud else [bbox_flip_ud(bboxes, img_shape) for bboxes in result]
            out_rlt = []
            for rlt1, rlt in zip(result1, result):
                rlt1[:, 0:4] = rlt
                out_rlt.append(rlt1)
            aug_rlt.append(out_rlt)
        # print(nm)

        merge_rlt = [np.concatenate(ls, axis=0) for ls in zip(*aug_rlt)]
        # print('aug_rlt: ', len(aug_rlt))
        # print('aug_rlt: ', len(aug_rlt[0]))
        # print('aug_rlt: ', aug_rlt[0][0].shape)
        # print('merge_rlt: ', len(merge_rlt))
        # print('merge_rlt: ', merge_rlt[0].shape, np.sum(merge_rlt[0][:, -1] > 0.9))
        final_rlt.append(merge_rlt)

    mmcv.dump(final_rlt, dump_path)
    final_rlt, box_num = FilterPKL.filter_pkl_by_score(final_rlt, 0.9)
    print('box num gt 0.9: ', box_num)
    # aa = np.round(final_rlt[0][0]).astype(np.int32)
    # print(aa.shape)
    # with open('file.txt', 'w') as f:
    #     for i in range(aa.shape[0]):
    #         f.write(' '.join(map(str, aa[i].tolist())) + '\r\n')
    #         print(aa[i])
    # ShowDetResult.store_result(orig_img, final_rlt[0], 0.2, 'a.jpg')
    final_rlt, box_num = FilterPKL.filter_pkl_by_nms(final_rlt, 0.7)
    print('box num nms 0.5: ', box_num)

    SubmitFormat.test_pkl2csv_format(final_rlt, csv_path, train=(mode == 'train'))


def box_vote(pkl, nms_thresh=0.7):

    rlt = pkl
    final_rlt, box_num = FilterPKL.filter_pkl_by_score(rlt, 0.9)
    print('box num score 0.9: ', box_num)
    final_rlt, box_num = FilterPKL.filter_pkl_by_nms(final_rlt, nms_thresh)
    print('box num nms {:.2f}: '.format(nms_thresh), box_num)

    save_rlt = []
    for fr, r in zip(final_rlt, rlt):
        class_r = []
        for fr_np, r_np in zip(fr, r):
            top_dets_out = box_voting(fr_np, r_np, nms_thresh, scoring_method='AVG', )
            class_r.append(top_dets_out)
        save_rlt.append(class_r)
    return save_rlt


def mean_ap(wt_path=None, filter_sc=True, filter_nms=True, eval_iou_thr=0.5, pkl_file=None, ):
    """
    calculate mean Average Precision
    :param wt_path:
    :param filter_sc:
    :param filter_nms:
    :param eval_iou_thr:
    :param pkl_file:
    :return:
    """
    print(wt_path)
    from mmdet.core.evaluation.mean_ap import eval_map
    from mmdet.core import coco_eval

    pred_box_pkl = 'data/{}/eval_train.pkl'.format(wt_path)
    if pkl_file is None:
        det_results = mmcv.load(pred_box_pkl)
    else:
        det_results = pkl_file
    gt_json_file = 'data/lvdefect/train_annotations.json'

    coco = COCO(gt_json_file)
    print('=' * 20 + 'coco_mean_ap' + '=' * 20)
    pred_box_json = 'data/{}/eval_train.pkl.json'.format(wt_path)
    coco_eval(pred_box_json, ['bbox'], coco)
    exit()

    # mmdet eval
    # ground truth
    gt_bboxes = []
    gt_labels = []
    gt_box_num = 0
    for i in range(len(coco.dataset["images"])):
        haha = coco.getAnnIds(imgIds=[i], catIds=[])
        anns = coco.loadAnns(ids=haha)
        gt_box_num += len(anns)
        gt_bx = np.zeros((len(anns), 4))
        gt_lb = np.zeros(len(anns))
        for idx, a in enumerate(anns):
            gt_lb[idx] = a['category_id']
            bx = a['bbox']
            xmin = bx[0]
            ymin = bx[1]
            xmax = bx[2] + bx[0] + 1
            ymax = bx[3] + bx[1] + 1
            gt_bx[idx, :] = [xmin, ymin, xmax, ymax]
        gt_bboxes.append(gt_bx)
        gt_labels.append(gt_lb)
        # print(gt_bboxes[-1])
        # return
    print('gt box num {:d}'.format(gt_box_num))
    print('=' * 20 + 'base cls_mean_ap' + '=' * 20)
    eval_map(det_results=det_results, gt_bboxes=gt_bboxes, gt_labels=gt_labels, iou_thr=eval_iou_thr, )

    if filter_sc:
        # filter score
        print('=' * 20 + 'diff score thr' + '=' * 20)
        for sc in np.arange(0.05, 0.55, 0.05, dtype=np.float32):
            f_rlt, box_num = FilterPKL.filter_pkl_by_score(det_results, sc)
            print('=' * 20 + 'score thr: {:.3f}'.format(sc) + '=' * 20 + 'box_num: {:d}'.format(box_num))
            eval_map(det_results=f_rlt, gt_bboxes=gt_bboxes, gt_labels=gt_labels, iou_thr=eval_iou_thr, )

    if filter_nms:
        # filter nms
        print('=' * 20 + 'diff nms thr' + '=' * 20)
        for thr in np.arange(0.05, 0.5, 0.05, dtype=np.float32):
            f_rlt, box_num = FilterPKL.filter_pkl_by_nms(det_results, float(thr))
            print('=' * 20 + 'num thr: {:.3f}'.format(thr) + '=' * 20 + 'box_num: {:d}'.format(box_num))
            eval_map(det_results=f_rlt, gt_bboxes=gt_bboxes, gt_labels=gt_labels, iou_thr=eval_iou_thr, )


def split_annotations():
    path = 'data/lvdefect/train_annotations.json'
    coco = COCO(path)
    gt_750 = []
    lt_1000 = []
    for i in range(1, 11):
        anns_id = coco.getAnnIds(catIds=[i])
        anns = coco.loadAnns(anns_id)
        for ann in anns:
            w, h = ann['bbox'][2:]
            scale = math.sqrt(w*h)
            if scale > 750:
                gt_750.append(ann)
            if scale < 1000:
                lt_1000.append(ann)

    js_file = mmcv.load(path)
    print('gt_750: {:.3f}'.format(len(gt_750) / len(js_file['annotations'])))
    print('lt_1000: {:.3f}'.format(len(lt_1000) / len(js_file['annotations'])))

    # js_file['annotations'] = gt_750
    # mmcv.dump(js_file, 'data/lvdefect/train_annotations_gt750.json', ensure_ascii=False, indent=4, separators=(',', ': '))
    #
    # js_file['annotations'] = lt_1000
    # mmcv.dump(js_file, 'data/lvdefect/train_annotations_lt1000.json', ensure_ascii=False, indent=4, separators=(',', ': '))


def data_analysis():
    path = '../gangjin/round1_tra.json'
    coco = COCO(path)
    all_scales = []
    all_ratios = []
    # coco.cats
    for i in range(1, 2):
        scales = []
        ratios = []
        # anns_id = coco.getAnnIds(catIds=[i])
        anns_id = coco.getAnnIds(catIds=['gangjin'])
        anns = coco.loadAnns(anns_id)
        for ann in anns:
            w, h = ann['bbox'][2:]
            if w == 0:
                print(ann)
                continue
            s = math.sqrt(w * h)
            # if s < 30:
            #     continue
            scales.append(s)
            ratios.append(h/w)

        all_scales.append(scales)
        all_ratios.append(ratios)

    for idx, (ratio, scale) in enumerate(zip(all_ratios, all_scales)):
        r_stas = []
        s_stas = []
        r_stas.append(min(ratio))
        r_stas.append(max(ratio))
        r_stas.append(np.mean(ratio))
        r_stas.append(np.std(ratio))
        r_stas.append(np.median(ratio))

        s_stas.append(min(scale))
        s_stas.append(max(scale))
        s_stas.append(np.mean(scale))
        s_stas.append(np.std(scale))
        s_stas.append(np.median(scale))
        # print(f"cls {idx}: {defect_map_e2c[f'defect{idx}']}")
        # print("cls {}: {}".format(idx, defect_map_e2c['defect{}'.format(idx)]))

        # print(f"scale min: {s_stas[0]:.2f}, max: {s_stas[1]:.2f}, mean: {s_stas[2]:.2f},"
        #       f" std: {s_stas[3]:.2f}, median: {s_stas[4]:.2f}")
        # print(f"ratio min: {r_stas[0]:.2f}, max: {r_stas[1]:.2f}, mean: {r_stas[2]:.2f},"
        #       f" std: {r_stas[3]:.2f}, median: {r_stas[4]:.2f}")
        print("scale min: {:.2f}, max: {:.2f}, mean: {:.2f},".format(s_stas[0], s_stas[1], s_stas[2]),
              " std: {:.2f}, median: {:.2f}".format(s_stas[3], s_stas[4]))
        print("ratio min: {:.2f}, max: {:.2f}, mean: {:.2f},".format(s_stas[0], s_stas[1], s_stas[2]),
              " std: {:.2f}, median: {:.2f}".format(s_stas[3], s_stas[4]))
    import matplotlib.pyplot as plt
    plt.figure()
    for idx, ratio in enumerate(all_ratios):
        x, y = np.histogram(ratio, bins=10)
        plt.plot(y[0:-1], np.cumsum(x / len(ratio)))
    plt.title('ratio')
    # plt.xlim([0, 1])
    plt.ylim([0, 1.1])
    # plt.legend(['defect{}'.format(i) for i in range(10)])
    plt.show()
    plt.figure()
    for idx, scale in enumerate(all_scales):
        x, y = np.histogram(scale, bins=10)
        plt.plot(y[0:-1], np.cumsum(x / len(scale)))
    plt.title('scale')
    plt.ylim([0, 1.1])
    # plt.legend(['defect{}'.format(i) for i in range(10)])
    plt.show()
    plt.close()

    plt.figure()
    plt.scatter(np.array(all_ratios[0]), np.array(all_scales[0]))
    plt.title('scale ratio')
    # plt.legend(['defect{}'.format(i) for i in range(10)])
    plt.show()
    plt.close()


def dataset_test():
    from mmdet.datasets.coco import CocoDataset
    from PIL import Image, ImageDraw
    img_norm_cfg = dict(
        mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)

    data_root = 'gangjin/'
    ds = CocoDataset(ann_file=data_root + 'round1_tra.json',
                     img_prefix=data_root + 'train_dataset/',
                     img_scale=[(9999, 1200), (9999, 1000), (9999, 800)],
                     img_norm_cfg=img_norm_cfg,
                     size_divisor=32,
                     flip_ratio=1,
                     flip_ud_ratio=1,
                     transpose_ratio=1,
                     extra_aug={'photo_metric_distortion': {'brightness_delta': 16,
                                                           'contrast_range': (0.8, 1.2),
                                                           'saturation_range': (0.8, 1.2),
                                                           'hue_delta': 18}, },

                     with_mask=False,
                     with_crowd=True,
                     with_label=True)
    dt_dict = ds[0]
    img = dt_dict['img'].data.permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    boxes = dt_dict['gt_bboxes'].data.cpu().detach().numpy()
    print(img.shape)
    print(boxes.shape)
    print(type(img))
    print(type(boxes))
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    for i in range(boxes.shape[0]):
        draw.rectangle(tuple(boxes[i, :].tolist()), fill=None, outline='red', width=2)

    img.save('ds_test.jpg')


def change_shape_of_coco_wt():
    load_from = '/root/.torch/models/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-dfa53166.pth'
    save_to = '/root/.torch/models/cascade_rcnn_dconv_c3-c5_r50_fpn_2cls.pth'
    wt = torch.load(load_from)
    for i in range(3):
        wt['state_dict'][f'bbox_head.{i}.fc_cls.weight'] = nn.init.kaiming_normal_(torch.empty(2, 1024),
                                                                                   mode='fan_in', nonlinearity='relu')
        wt['state_dict'][f'bbox_head.{i}.fc_cls.bias'] = torch.rand(2)

    torch.save(wt, save_to)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # dataset_test()
    # COCOStyleDataset.json_annotations_for_train()
    # COCOStyleDataset.merge_annotations()
    # ShowDetResult.infer_one_and_store(0)
    # ShowDetResult.infer_and_store_mask(0)
    pkl = MergePKLs.merge_pkls_by_nms(0.5, False)
    SubmitFormat.test_pkl2csv_format(pkl, 'sub_merge.csv')


