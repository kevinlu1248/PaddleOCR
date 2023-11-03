# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import sys
import shapely
from shapely.geometry import Polygon
import numpy as np
from collections import defaultdict
import operator
import editdistance


def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def polygon_from_str(polygon_points):
    """
    Create a shapely polygon object from gt or dt line.
    """
    polygon_points = np.array(polygon_points).reshape(4, 2)
    polygon = Polygon(polygon_points).convex_hull
    return polygon


def polygon_iou(poly1, poly2):
    """
    Intersection over union between two shapely polygons.
    """
    if not poly1.intersects(
            poly2):  # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            union_area = poly1.area + poly2.area - inter_area
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            # except Exception as e:
            #     print(e)
            print('shapely.geos.TopologicalError occurred, iou set to 0')
            iou = 0
    return iou


def ed(str1, str2):
    return editdistance.eval(str1, str2)


def e2e_eval(gt_dir, res_dir, ignore_blank=False):
    print('start testing...')
    iou_thresh = 0.5
    val_names = os.listdir(gt_dir)
    num_gt_chars = 0
    gt_count = 0
    dt_count = 0
    hit = 0
    ed_sum = 0

    for i, val_name in enumerate(val_names):
        gts, ignore_masks = read_gt_file(os.path.join(gt_dir, val_name))
        dts = read_dt_file(os.path.join(res_dir, val_name))
            dt_lines = []
        else:
            with open(val_path, encoding='utf-8') as f:
                dt_lines = [o.strip() for o in f.readlines()]
        dts = []
        for line in dt_lines:
            # print(line)
            parts = line.strip().split("\t")
            assert (len(parts) < 10), "line error: {}".format(line)
            if len(parts) == 8:
                dts.append(parts + [''])
            else:
                dts.append(parts)

        dt_match = [False] * len(dts)
        gt_match = [False] * len(gts)
        all_ious = defaultdict(tuple)
        dt_match = [False] * len(dts)
        gt_match = [False] * len(gts)
        all_ious = calculate_ious(gts, dts, iou_thresh)
        sorted_gt_dt_pairs = sort_ious(all_ious)

        # matched gt and dt
        hit, ed_sum, num_gt_chars, gt_count, dt_count = calculate_metrics_for_matched_pairs(sorted_gt_dt_pairs, gt_match, dt_match, gts, dts, ignore_masks, ignore_blank, hit, ed_sum, num_gt_chars, gt_count, dt_count)

        # unmatched dt
        for tindex, dt_match_flag in enumerate(dt_match):
            if dt_match_flag == False:
                dt_str = dts[tindex][8]
                gt_str = ''
                ed_sum += ed(dt_str, gt_str)
        ed_sum, dt_count = calculate_metrics_for_unmatched_dts(dt_match, dts, ed_sum, dt_count)
        ed_sum, num_gt_chars, gt_count = calculate_metrics_for_unmatched_gts(gt_match, gts, ignore_masks, ed_sum, num_gt_chars, gt_count)
        for tindex, gt_match_flag in enumerate(gt_match):
            if gt_match_flag == False and ignore_masks[tindex] == '0':
                dt_str = ''
                gt_str = gts[tindex][8]
                ed_sum += ed(gt_str, dt_str)
                num_gt_chars += len(gt_str)
    eps = 1e-9
    precision, recall, fmeasure, avg_edit_dist_img, avg_edit_dist_field, character_acc = calculate_final_metrics(hit, dt_count, gt_count, ed_sum, len(val_names), num_gt_chars, eps)
    print_results(hit, dt_count, gt_count, character_acc, avg_edit_dist_field, avg_edit_dist_img, precision, recall, fmeasure)


if __name__ == '__main__':
    gt_folder = sys.argv[1]
    pred_folder = sys.argv[2]
    e2e_eval(gt_folder, pred_folder)


if __name__ == '__main__':
# Code moved to read_files function
    gt_folder = sys.argv[1]
    pred_folder = sys.argv[2]
    e2e_eval(gt_folder, pred_folder)
