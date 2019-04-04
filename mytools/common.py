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
from mytools.prepare_and_submit import SubmitFormat


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
    class_names = ['iron_lighter', 'lighter', 'knife', 'battery', 'scissors']

    @staticmethod
    def store_bbox(img, result, score_thr=0.3, out_file=None):
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
            class_names=ShowDetResult.class_names,
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
    def show_pkl(pkl_file, save_path):
        save_path = save_path+'/' if save_path[-1] != '/' else save_path
        from mmdet.datasets.coco import CocoDataset
        # save_path = f'x-ray/show_test/'
        data_path = f'x-ray/jinnan2_round1_test_a_20190306/'
        ann_file = 'x-ray/round1_test_a.json'

        os.makedirs(save_path, exist_ok=True)
        pkl = mmcv.load(pkl_file)
        print('pkl len: ', len(pkl))

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
    work_dir = 'cascade_dc3_5_r101_ep30_bs2_ohem1_cw_3s'
    config_file_name = 'cascade_rcnn_dconv_c3-c5_fpn.py'

    mode = 'test'
    img_path = f'x-ray/jinnan2_round1_test_a_20190306/'
    checkpoint_path = f'myweights/{work_dir}/epoch_30.pth'
    dump_path = f'myweights/{work_dir}/aug_{mode}.pkl'
    submit_path = f'myweights/{work_dir}/aug_{mode}.json'

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
    # final_rlt, box_num = FilterPKL.filter_pkl_by_score(final_rlt, 0.9)
    # print('box num gt 0.9: ', box_num)
    final_rlt, box_num = FilterPKL.filter_pkl_by_nms(final_rlt, 0.5)
    print('box num nms 0.5: ', box_num)

    SubmitFormat.test_pkl2json_format(final_rlt, submit_path)


def box_vote(pkl, nms_thresh=0.7, filter_score=0.05):

    rlt = pkl
    final_rlt, box_num = FilterPKL.filter_pkl_by_score(rlt, filter_score)
    print(f'box num score {filter_score}: ', box_num)
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


def mean_ap(pred_pkl=None, pred_coco_json=None, filter_sc=True, filter_nms=True, eval_iou_thr=0.5):
    """
    calculate mean Average Precision
    :param pred_pkl:
    :param filter_sc:
    :param filter_nms:
    :param pred_coco_json:
    :param eval_iou_thr:
    :return:
    """
    print(pred_pkl)
    from mmdet.core.evaluation.mean_ap import eval_map
    from mmdet.core import coco_eval

    if isinstance(pred_pkl, str):
        det_results = mmcv.load(pred_pkl)
    else:
        det_results = pred_pkl

    gt_json_file = 'x-ray/jinnan2_round1_train_20190305/train_no_poly_coco.json'
    coco = COCO(gt_json_file)
    print('=' * 20 + 'coco_mean_ap' + '=' * 20)
    coco_eval(pred_coco_json, ['bbox'], coco)
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
    exit()
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
    cls_num = 6
    load_from = '/root/.torch/models/cascade_rcnn_dconv_c3-c5_r101_fpn_1x_20190125-aaa877cc.pth'
    save_to = f'/root/.torch/models/cascade_rcnn_dconv_c3-c5_r101_fpn_{cls_num}cls.pth'
    wt = torch.load(load_from)
    for i in range(3):
        wt['state_dict'][f'bbox_head.{i}.fc_cls.weight'] = nn.init.kaiming_normal_(torch.empty(cls_num, 1024),
                                                                                   mode='fan_in', nonlinearity='relu')
        wt['state_dict'][f'bbox_head.{i}.fc_cls.bias'] = torch.rand(cls_num)

    torch.save(wt, save_to)


def statistics():
    path = 'x-ray/'
    img_nms = glob.glob(path+'**/*jpg')

    stas = np.zeros((3, len(img_nms)), dtype=np.float64)
    for i, nm in tqdm(enumerate(img_nms)):
        img = io.imread(nm)
        stas[:, i] = np.mean(img, axis=(0, 1))
        # print(stas[:, i])
    mean = np.mean(stas, axis=1)
    print(mean)
    std = np.std(stas, axis=1)
    print(std)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # aug_test()
    # ShowDetResult.show_pkl('myweights/cascade_dc3_5_r101_ep30_bs2_ohem1_cw_3s/aug_test.pkl', 'aug_test/')
    mean_ap(pred_coco_json='myweights/cascade_dc3_5_r101_ep30_bs2_ohem1_cw_3s/train_ep30_aug.pkl.json')


