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
        with open(os.path.join(gt_dir, val_name), encoding='utf-8') as f:
            gt_lines = [o.strip() for o in f.readlines()]
        gts = []
        ignore_masks = []
        for line in gt_lines:
            parts = line.strip().split('\t')
            # ignore illegal data
            if len(parts) < 9:
                continue
            assert (len(parts) < 11)
            if len(parts) == 9:
                gts.append(parts[:8] + [''])
            else:
                gts.append(parts[:8] + [parts[-1]])

            ignore_masks.append(parts[8])

        val_path = os.path.join(res_dir, val_name)
        if not os.path.exists(val_path):
# Test for e2e_eval function
def test_e2e_eval():
    # Mock data
    gt_dir = "mock_gt_dir"
    res_dir = "mock_res_dir"
    ignore_blank = False

    # Mock return values
    mock_read_files_return = ([], [], [])
    mock_calculate_metrics_return = (0, 0, 0, 0, 0)

    # Mock read_files function
    read_files = MagicMock(return_value=mock_read_files_return)

    # Mock calculate_metrics function
    calculate_metrics = MagicMock(return_value=mock_calculate_metrics_return)

    # Call e2e_eval function
    e2e_eval(gt_dir, res_dir, ignore_blank)

    # Assert read_files was called with correct arguments
    read_files.assert_called_once_with(os.listdir(gt_dir), gt_dir, res_dir)

    # Assert calculate_metrics was called correct number of times
    assert calculate_metrics.call_count == len(os.listdir(gt_dir))
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
        hit, dt_count, gt_count, ed_sum, num_gt_chars = calculate_metrics(gts, dts, iou_thresh, all_ious, gt_match, dt_match, ignore_blank, ignore_masks, ed_sum, num_gt_chars, hit, gt_count, dt_count)

    print_results(hit, dt_count, gt_count, ed_sum, val_names, num_gt_chars)

def print_results(hit, dt_count, gt_count, ed_sum, val_names, num_gt_chars):
    eps = 1e-9
    print('hit, dt_count, gt_count', hit, dt_count, gt_count)
    precision = hit / (dt_count + eps)
    recall = hit / (gt_count + eps)
    fmeasure = 2.0 * precision * recall / (precision + recall + eps)
    avg_edit_dist_img = ed_sum / len(val_names)
    avg_edit_dist_field = ed_sum / (gt_count + eps)
    character_acc = 1 - ed_sum / (num_gt_chars + eps)

    print('character_acc: %.2f' % (character_acc * 100) + "%")
    print('fmeasure: %.2f' % (fmeasure * 100) + "%")

def calculate_metrics(gts, dts, iou_thresh, all_ious, gt_match, dt_match, ignore_blank, ignore_masks, ed_sum, num_gt_chars, hit, gt_count, dt_count):
    for index_gt, gt in enumerate(gts):
        gt_coors = [float(gt_coor) for gt_coor in gt[0:8]]
        gt_poly = polygon_from_str(gt_coors)
        for index_dt, dt in enumerate(dts):
            dt_coors = [float(dt_coor) for dt_coor in dt[0:8]]
            dt_poly = polygon_from_str(dt_coors)
            iou = polygon_iou(dt_poly, gt_poly)
            if iou >= iou_thresh:
                all_ious[(index_gt, index_dt)] = iou
    sorted_ious = sorted(
        all_ious.items(), key=operator.itemgetter(1), reverse=True)
    sorted_gt_dt_pairs = [item[0] for item in sorted_ious]

    # matched gt and dt
    for gt_dt_pair in sorted_gt_dt_pairs:
# Test for calculate_metrics function
def test_calculate_metrics():
    # Mock data
    gts = [["mock_gt"]]
    dts = [["mock_dt"]]
    iou_thresh = 0.5
    all_ious = defaultdict(tuple)
    gt_match = [False]
    dt_match = [False]
    ignore_blank = False
    ignore_masks = ['0']
    ed_sum = 0
    num_gt_chars = 0
    hit = 0
    gt_count = 0
    dt_count = 0

    # Call calculate_metrics function with mock data
    result = calculate_metrics(gts, dts, iou_thresh, all_ious, gt_match, dt_match, ignore_blank, ignore_masks, ed_sum, num_gt_chars, hit, gt_count, dt_count)

    # Assert return value is correct
    assert result == (0, 0, 0, 0, 0)
def test_print_results():
    # Mock data
    hit = 0
    dt_count = 0
    gt_count = 0
    ed_sum = 0
    val_names = ["mock_val_name"]
    num_gt_chars = 0

    # Call print_results function with mock data
    print_results(hit, dt_count, gt_count, ed_sum, val_names, num_gt_chars)

    # Assert correct results are printed
    # This is a bit tricky as the function prints to stdout, 
    # so we need to capture stdout and then assert on its value
    captured = io.StringIO()
    sys.stdout = captured
    print_results(hit, dt_count, gt_count, ed_sum, val_names, num_gt_chars)
    sys.stdout = sys.__stdout__
    assert captured.getvalue() == "hit, dt_count, gt_count 0 0 0\ncharacter_acc: 100.00%\navg_edit_dist_field: 0.00\navg_edit_dist_img: 0.00\nprecision: 0.00%\nrecall: 0.00%\nfmeasure: 0.00%\n"

def calculate_metrics(gts, dts, iou_thresh, all_ious, gt_match, dt_match, ignore_blank, ignore_masks, ed_sum, num_gt_chars, hit, gt_count, dt_count):
    for index_gt, gt in enumerate(gts):
        gt_coors = [float(gt_coor) for gt_coor in gt[0:8]]
        gt_poly = polygon_from_str(gt_coors)
        for index_dt, dt in enumerate(dts):
            dt_coors = [float(dt_coor) for dt_coor in dt[0:8]]
            dt_poly = polygon_from_str(dt_coors)
            iou = polygon_iou(dt_poly, gt_poly)
            if iou >= iou_thresh:
                all_ious[(index_gt, index_dt)] = iou
    sorted_ious = sorted(
        all_ious.items(), key=operator.itemgetter(1), reverse=True)
    sorted_gt_dt_pairs = [item[0] for item in sorted_ious]

    # matched gt and dt
    for gt_dt_pair in sorted_gt_dt_pairs:
        index_gt, index_dt = gt_dt_pair
        if gt_match[index_gt] == False and dt_match[index_dt] == False:
            gt_match[index_gt] = True
            dt_match[index_dt] = True
            if ignore_blank:
                gt_str = strQ2B(gts[index_gt][8]).replace(" ", "")
                dt_str = strQ2B(dts[index_dt][8]).replace(" ", "")
            else:
                gt_str = strQ2B(gts[index_gt][8])
                dt_str = strQ2B(dts[index_dt][8])
            if ignore_masks[index_gt] == '0':
                ed_sum += ed(gt_str, dt_str)
                num_gt_chars += len(gt_str)
                if gt_str == dt_str:
                    hit += 1
                gt_count += 1
                dt_count += 1

    # unmatched dt
    for tindex, dt_match_flag in enumerate(dt_match):
        if dt_match_flag == False:
            dt_str = dts[tindex][8]
            gt_str = ''
            ed_sum += ed(dt_str, gt_str)
            dt_count += 1

    # unmatched gt
    for tindex, gt_match_flag in enumerate(gt_match):
        if gt_match_flag == False and ignore_masks[tindex] == '0':
            dt_str = ''
            gt_str = gts[tindex][8]
            ed_sum += ed(gt_str, dt_str)
            num_gt_chars += len(gt_str)
            gt_count += 1
    return hit, dt_count, gt_count, ed_sum, num_gt_chars

def read_files(val_names, gt_dir, res_dir):
    for i, val_name in enumerate(val_names):
        with open(os.path.join(gt_dir, val_name), encoding='utf-8') as f:
            gt_lines = [o.strip() for o in f.readlines()]
        gts = []
        ignore_masks = []
        for line in gt_lines:
            parts = line.strip().split('\t')
            # ignore illegal data
            if len(parts) < 9:
                continue
            assert (len(parts) < 11)
            if len(parts) == 9:
                gts.append(parts[:8] + [''])
            else:
                gts.append(parts[:8] + [parts[-1]])

            ignore_masks.append(parts[8])

        val_path = os.path.join(res_dir, val_name)
        if not os.path.exists(val_path):
# Test for read_files function
def test_read_files():
    # Mock data
    val_names = ["mock_val_name"]
    gt_dir = "mock_gt_dir"
    res_dir = "mock_res_dir"

    # Call read_files function with mock data
    result = read_files(val_names, gt_dir, res_dir)

    # Assert return value is correct
    assert result == ([], [], [])
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
    return dts, gts, ignore_masks


if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     print("python3 ocr_e2e_eval.py gt_dir res_dir")
    #     exit(-1)
    # gt_folder = sys.argv[1]
    # pred_folder = sys.argv[2]
    gt_folder = sys.argv[1]
    pred_folder = sys.argv[2]
    e2e_eval(gt_folder, pred_folder)
