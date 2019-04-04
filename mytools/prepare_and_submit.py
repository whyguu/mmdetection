import glob
import json
import numpy as np
import pandas as pd
from PIL import Image
import os
import mmcv
from mmdet.models import build_detector
from mmdet.apis import inference_detector
import time
from skimage import io
from mmcv.runner import load_checkpoint


CLASS_NAMES = ['steel']


class COCOStyleDataset(object):

    @staticmethod
    def json_info(json_path=None):
        # json_path = '/workspace/mmdetection-0p6rc/gangjin/train_annotations.json'
        from pycocotools import coco
        cd = coco.COCO(json_path)
        print('anno num: ', len(cd.anns))
        print('img num: ', len(cd.imgs))
        return len(cd.imgs), len(cd.anns)

    @staticmethod
    def json_annotations_for_train():
        class_name2id = {}
        for idx, cls_nm in enumerate(CLASS_NAMES):
            class_name2id[cls_nm] = idx+1
        final_json = dict()
        final_json['info'] = None
        final_json['licenses'] = None
        final_json['categories'] = []
        final_json['images'] = []
        final_json['annotations'] = []

        # categories
        for i in range(len(CLASS_NAMES)):
            cat = dict()
            cat['id'] = i+1
            cat['name'] = CLASS_NAMES[i]
            cat['supercategory'] = CLASS_NAMES[i]
            final_json['categories'].append(cat)

        sv_nm = 'test_annotations_991586.json'
        data_root = '/workspace/mmdetection-0p6rc/gangjin/'
        path_prefix = 'test_dataset/'
        img_path = data_root+path_prefix
        img_names = glob.glob1(img_path, '*jpg')

        def pre_data(df):
            df.iloc[:, 1] = df.apply(lambda x: [float(a) for a in x[1].split(' ')], axis=1)

        def collect(df):
            rlt = dict()
            for ci in range(df.shape[0]):
                img_nm = df.iloc[ci, 0]
                if img_nm in rlt.keys():
                    rlt[img_nm].append(df.iloc[ci, 1])
                else:
                    rlt[img_nm] = [df.iloc[ci, 1]]
            for key, val in rlt.items():
                rlt[key] = np.array(val)
            return rlt
        df = pd.read_csv(data_root+'test_991586.csv', header=None)
        pre_data(df)
        lb_dict = collect(df)

        assert len(lb_dict) == len(img_names)
        img_names = list(lb_dict.keys())

        # gene json
        image_id_counter = 0
        annotation_id_counter = 0
        for img_name in img_names:
            print(image_id_counter, annotation_id_counter, img_name)
            # image
            image = dict()
            img_data = Image.open(img_path+img_name)
            image['id'] = image_id_counter
            image['width'] = img_data.width
            image['height'] = img_data.height
            image['file_name'] = path_prefix+img_name
            final_json['images'].append(image)

            # annotation
            for i in range(lb_dict[img_name].shape[0]):
                x, y, x2, y2 = lb_dict[img_name][i]
                width, height = x2-x, y2-y

                annotation = dict()
                annotation['id'] = annotation_id_counter
                annotation['image_id'] = image_id_counter
                annotation['category_id'] = 1
                annotation['bbox'] = [x, y, width, height]
                annotation['area'] = float(width*height)
                annotation['iscrowd'] = 0
                final_json['annotations'].append(annotation)
                # update annotation id
                annotation_id_counter += 1
            # update image id
            image_id_counter += 1
        # write json
        with open(data_root+sv_nm, 'w') as fp:
            json.dump(final_json, fp=fp, ensure_ascii=False, indent=4, separators=(',', ': '))

    @staticmethod
    def json_annotations_for_test():
        final_json = dict()
        final_json['info'] = None
        final_json['licenses'] = None
        final_json['categories'] = []
        final_json['images'] = []
        final_json['annotations'] = []

        img_path = 'x-ray/jinnan2_round1_test_a_20190306/'
        sv_path = 'x-ray/round1_test_a.json'
        img_names = sorted([nm for nm in os.listdir(img_path) if 'jpg' in nm])
        print('img num: ', len(img_names))
        image_id_counter = 0
        for img_name in img_names:
            print(image_id_counter, img_name)
            img_data = Image.open(img_path + img_name)
            # image
            image = dict()
            image['id'] = image_id_counter
            image['width'] = img_data.width
            image['height'] = img_data.height
            image['file_name'] = img_name
            final_json['images'].append(image)
            image_id_counter += 1

        with open(sv_path, 'w') as fp:
            json.dump(final_json, fp=fp, ensure_ascii=False, indent=4, separators=(',', ': '))

    @staticmethod
    def merge_annotations():
        root = '/workspace/mmdetection/gangjin/'
        train_anno = root + 'train_annotations.json'
        test_anno = root + 'test_annotations_991586.json'
        with open(train_anno, 'r') as fp:
            train = json.load(fp)
        with open(test_anno, 'r') as fp:
            test = json.load(fp)
        train['images'] += test['images']
        train['annotations'] += test['annotations']
        with open(root+'merge_991586.json', 'w') as fp:
            json.dump(train, fp=fp, ensure_ascii=False, indent=4, separators=(',', ': '))

    @staticmethod
    def format_json():
        path = 'x-ray/jinnan2_round1_train_20190305/train_no_poly.json'
        sv_path = 'x-ray/jinnan2_round1_train_20190305/train_no_poly_format.json'

        with open(path, 'r') as fp:
            js = json.load(fp)
        with open(sv_path, 'w') as fp:
            json.dump(js, fp, ensure_ascii=False, indent=4, separators=(', ', ': '))

    @staticmethod
    def jinnan2_format2coco_format():
        path = 'x-ray/jinnan2_round1_train_20190305/train_no_poly.json'
        sv_path = 'x-ray/jinnan2_round1_train_20190305/train_no_poly_coco.json'

        with open(path, 'r') as fp:
            js = json.load(fp)
        annos = js['annotations']
        for idx in range(len(annos)):
            annos[idx]['area'] = annos[idx]['bbox'][2] * annos[idx]['bbox'][3]

        with open(sv_path, 'w') as fp:
            json.dump(js, fp, ensure_ascii=False, indent=4, separators=(', ', ': '))


class SubmitFormat(object):
    @staticmethod
    def get_img_names(train: bool):
        from mmdet.datasets.coco import CocoDataset
        data_root = 'x-ray/'
        if train:
            ann_file = data_root + 'round1_tra.json'
            dataset = CocoDataset(ann_file, None, (0, 0), {}, test_mode=True)
        else:
            ann_file = data_root + 'round1_test_a.json'
            dataset = CocoDataset(ann_file, None, (0, 0), {}, test_mode=True)
        # order of img_names must match img order generated by json_annotations_for_test
        # use the generated json file is ok
        # img_names = sorted([nm for nm in os.listdir(img_path) if 'jpg' in nm])
        img_names = [dataset.img_infos[idx]['filename'] for idx in range(len(dataset))]

        return img_names

    @staticmethod  # multi class
    def test_pkl2json_format(pkl_file, sv_path):
        """
        convert out.pkl generated by test.py to required json format
        .pkl file format generated by test.py is a list of len (pic_num, ) of list
        like [[cls1_det, cls2_det, ...], ...],
        cls1_det is a numpy.array of shape (box_num, 5) [xmin, ymin, xmax, ymax, confidence].

        :return: json format
        """
        # # output format example
        """
        {
            "results": [
                {
                    "filename": "1.jpg",
                    "rects": []
                },
                {
                    "filename": "10.jpg",
                    "rects": [
                        {
                            "xmin": 432,
                            "ymin": 330,
                            "xmax": 497,
                            "ymax": 366,
                            "confidence": 0.980955,
                            "label": 1,
                        },
                        {
                            "xmin": 549,
                            "ymin": 199,
                            "xmax": 598,
                            "ymax": 337,
                            "confidence": 0.000179,
                            "label": 5,
                        }
                    ]
                }
            ]
        }
        """

        if isinstance(pkl_file, str):
            pkl = mmcv.load(pkl_file)
        else:
            pkl = pkl_file
        # order of img_names must match img order generated by json_annotations_for_test
        # img_names = sorted([nm for nm in os.listdir(img_path) if 'jpg' in nm])
        img_names = SubmitFormat.get_img_names(False)
        print(f'img num: {len(img_names)}')
        # assert len(img_names) == 1000

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
                    b_dict['label'] = idx+1

                    tp_dict['rects'].append(b_dict)

            rlt['results'].append(tp_dict)

        with open(sv_path, 'w') as fp:
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
    def infer_and_gen_coco_json(gpu_id=0, with_mask=False):
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

            # if with_mask:
            #     det_bboxes = result[0]  # [cls_ndarray]
            #     det_masks = result[1]
            #     assert len(det_bboxes[0]) == len(det_masks[0])
            # else:
            #     det_bboxes = result

            for j, rlt in enumerate(zip(result)):
                cls_id = j + 1

                cat_bboxes = rlt[0]
                if with_mask:
                    cat_masks = rlt[1]
                if len(cat_bboxes) == 0:
                    continue

                for k in range(cat_bboxes.shape[0]):
                    cat_bbox = cat_bboxes[k]

                    annotation = dict()
                    annotation['id'] = annotation_id
                    annotation['image_id'] = i
                    annotation['category_id'] = cls_id
                    annotation['iscrowd'] = 0

                    x1, y1, x2, y2 = list(map(round, cat_bbox[0:4].tolist()))
                    annotation['bbox'] = [x1, y1, x2-x1, y2-y1]  # mask_utils.toBbox(cat_mask).tolist()
                    annotation['area'] = (x2-x1) * (y2-y1)  # float(mask_utils.area(cat_mask))
                    if with_mask:
                        # cat_mask = cat_masks[k]
                        # cat_mask['counts'] = cat_mask['counts'].decode()
                        annotation['segmentation'] = cat_masks[k]
                        annotation['area'] = mask_utils.area(cat_masks[k])
                    annotation_id += 1
                    final_json['annotations'].append(annotation)
        with open('dp_data_shuangyangqu/det.json', 'w') as fp:
            json.dump(final_json, fp, ensure_ascii=False, indent=4)

        print(f'time elapsed: {(time.time()-t_start)/60:.3f} min')


if __name__ == '__main__':
    # COCOStyleDataset.json_annotations_for_test()

    SubmitFormat.test_pkl2json_format([], '')



