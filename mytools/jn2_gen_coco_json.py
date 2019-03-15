import glob
import json
import numpy as np
import pandas as pd
from PIL import Image
import os


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

        img_path = '/workspace/mmdetection/gangjin/test_dataset/'
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

        with open('/workspace/mmdetection/gangjin/round1_test_a.json', 'w') as fp:
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


if __name__ == '__main__':
    pass


