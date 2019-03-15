# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 08:31:49 2018

@author: wangq
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from PIL import Image
import glob

# labelme format -> csv format
# labelme format -> coco json format
# csv format -> coco json format


def labelme2csv(labeled_json_path=None, csv_path=None):
    csv_path = './pig_tra.csv' if csv_path is None else csv_path
    json_path = './pigs/images/' if labeled_json_path is None else labeled_json_path

    output_file = open(csv_path, 'w')
    output_file.write('img_name,img_w,img_h,label,points_xy\n')
    namelist = glob.glob1(json_path, '*json')
    for i in range(len(namelist)):
        json_name = namelist[i]
        # if json_name.split('.')[-1] != 'json':
        #     continue
        print(i, json_name)
        with open(json_path + json_name, 'r') as load_f:
            json_data = json.load(load_f)
        # print(json_data)

        try:
            w, h, name, objs = json_data['imageWidth'], json_data['imageHeight'], json_data['imagePath'], json_data['shapes']
        except KeyError:
            # label me version 3.4 compatible
            name, objs, name = json_data['imagePath'], json_data['shapes'], os.path.basename(json_data['imagePath'])
            img = Image.open(json_path+name)
            h = img.height
            w = img.width

        for ii in range(len(objs)):
            obj = objs[ii]
            label, points = obj['label'], obj['points']
            points = np.array(points, dtype=str).reshape([-1, ]).tolist()
            points = ' '.join(points)
            # pts = np.array(obj['points'], dtype=int)
            # plt.plot(pts[:, 0], pts[:, 1], color='yellow')

            out_line = ','.join([name, str(w), str(h), label, points])
            out_line += '\n'

            output_file.write(out_line)
        # plt.show()
    output_file.close()


def csv2json(csv_path=None, json_path=None):
    json_path = './pig_tra.json' if json_path is None else json_path
    csv_path = './pig_tra.csv' if csv_path is None else csv_path
    vis = False
    img_path = './data/train_dataset/'
    json_obj = {}
    images = []
    annotations = []
    categories = []
    categories_list = []
    annotation_id = 1

    print("-----------------Start------------------")
    # data = get_npy()
    data = np.array(pd.read_csv(csv_path))[:, :5]

    for i in range(len(data)):

        name, w, h, label = data[i, :4]
        segs = np.round(np.array(data[i, 4].split(' '), dtype=float)).astype(int).reshape([-1, 2])
        segs = np.concatenate([segs, segs[0].reshape([1, 2])], axis=0)
        print(i, name)

        w, h = int(w), int(h)

        image = dict()
        image["file_name"] = name
        image["width"] = w
        image["height"] = h
        image["id"] = name[:-4]

        if image not in images:
            images.append(image)

        # print(box)
        xmin_nodes, ymin_nodes = segs[:, 0].min(), segs[:, 1].min()
        xmax_nodes, ymax_nodes = segs[:, 0].max(), segs[:, 1].max()

        if vis:
            img = Image.open(img_path + name)
            plt.imshow(img)
            plt.plot([xmin_nodes, xmax_nodes, xmax_nodes, xmin_nodes, xmin_nodes],
                     [ymin_nodes, ymin_nodes, ymax_nodes, ymax_nodes, ymin_nodes])
            plt.show()

        annotation = {}
        segmentation = []
        bbox = []

        segmentation.append(segs.reshape([-1, ]).tolist())
        width = float(xmax_nodes) - float(xmin_nodes) + 1
        height = float(ymax_nodes) - float(ymin_nodes) + 1
        area = width * height
        if area < 9:
            continue
        bbox.append(float(xmin_nodes))
        bbox.append(float(ymin_nodes))
        bbox.append(width)
        bbox.append(height)

        annotation["segmentation"] = segmentation
        annotation["area"] = area
        annotation["iscrowd"] = 0
        annotation["image_id"] = name[:-4]
        annotation["bbox"] = bbox
        annotation["category_id"] = label
        annotation["id"] = annotation_id
        annotation_id += 1
        annotation["ignore"] = 0
        # annotation["num_keypoints"] = 16

        annotations.append(annotation)
        if label in categories_list:
            pass
        else:
            categories_list.append(label)
            categorie = dict()
            categorie["supercategory"] = label
            categorie["id"] = label
            categorie["name"] = label
            categories.append(categorie)

    json_obj["images"] = images
    # json_obj["type"] = "instances"
    json_obj["annotations"] = annotations
    json_obj["categories"] = categories

    f = open(json_path, "w")
    json_str = json.dumps(json_obj, indent=4)
    f.write(json_str)
    f.close()
    print("------------------End-------------------")


def labelme2json(labeled_json_path, out_json_pathname):
    labelme2csv(labeled_json_path, './.temp.csv')
    csv2json('./.temp.csv', out_json_pathname)
    os.remove('./.temp.csv')


if __name__ == '__main__':
    labelme2json('./dp_data_shuangyangqu/images/', './dp_data_shuangyangqu/dp_train.json')





