#!/usr/bin/python
# encoding: utf-8

import os
import time
import json
import yaml
from PIL import Image
from shapely.geometry import Polygon
from glob import glob

from text_detection.detecting import TextDetector


en_alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

detector = TextDetector(
    detecting_model='pretrained_models/final_detection.pb',
    recognition_model='pretrained_models/crnn.pth',
    gpu_mode=False,
    gpu_id=0,
    alphabet=en_alphabet
)


def get_all_yaml_supervising(name):
    all_annotation_files = glob(f"{name}*yaml")
    annotations = []
    for file in all_annotation_files:
        annotation = yaml.load(open(file))
        annotations.append(annotation)
    return annotations


def get_bbox_from_ann(ann):
    poly = list(map(int, ann['plate_corners_gt'].split()))
    bbox = []
    bbox.append(min(poly[0],poly[2],poly[4], poly[6]))
    bbox.append(min(poly[1],poly[3],poly[5], poly[7]))
    bbox.append(max(poly[0],poly[2],poly[4], poly[6]))
    bbox.append(max(poly[1],poly[3],poly[5], poly[7]))
    return bbox


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
        Mx = max(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
        my = min(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
        My = max(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea


def get_pairs(pred, name, iou_threshold=0.5):
    gt = get_all_yaml_supervising(name)
    pred_bboxes = [[bbox['topleft']['x'], bbox['topleft']['y'], bbox['bottomright']['x'], bbox['bottomright']['y']]
                       for bbox in pred['words']]
    pred_words = [ bbox['label']
                       for bbox in pred['words']]
    gt_bboxes = [get_bbox_from_ann(x) for x in gt]
    gt_words = [x['plate_number_gt'] for x in gt]
    pairs_to_check = []
    for i, bbox in enumerate(pred_bboxes):
        for j, bbox_gt in enumerate(gt_bboxes):
            if bbox_iou(bbox, bbox_gt) >= iou_threshold:
                pairs_to_check.append([pred_words[i], gt_words[j]])
    return pairs_to_check, len(gt_bboxes)


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def char_precision(word_pred, word_true):
    word_pred = str(word_pred).lower()
    word_true = str(word_true).lower()
    return (len(word_true) - levenshteinDistance(word_pred, word_true)) / len(word_true)


def word_precision(word_pred, word_true):
    word_pred = str(word_pred).lower()
    word_true = str(word_true).lower()
    return word_pred == word_true



t0 = time.time()
rez = []
total_char_precision = 0
total_word_precision = 0
total_gt_count = 0
for image_file in glob('data/validation_ds/*.jpg'):
    image = Image.open(image_file)
    try:
      result = detector.predict(image)
    except:
      continue
    matched_pairs, gt_count = get_pairs(result, os.path.splitext(image_file)[0])

    total_gt_count += gt_count
    for pair in matched_pairs:
      total_char_precision += char_precision(*pair)
      total_word_precision += word_precision(*pair)

    print(matched_pairs)

t1 = time.time()
print('Average char precision', "%.4f" % (total_char_precision / total_gt_count))
print('Average word precision', "%.4f" % (total_word_precision / total_gt_count))
print('time seq', t1 - t0)