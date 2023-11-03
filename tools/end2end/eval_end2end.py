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



def convert_unicode_to_ascii():
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
    return strQ2B

strQ2B = convert_unicode_to_ascii()



def create_polygon_from_str():
def read_files(gt_dir, res_dir):
    val_names = os.listdir(gt_dir)
    results = []
    for i, val_name in enumerate(val_names):
        with open(os.path.join(gt_dir, val_name), encoding='utf-8') as f:
            gt_lines = [o.strip() for o in f.readlines()]
        gts = []
        ignore_masks = []
        for line in gt_lines:
            parts = line.strip().split('\t')
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
            dt_lines = []
        else:
            with open(val_path, encoding='utf-8') as f:
                dt_lines = [o.strip() for o in f.readlines()]
        dts = []
        for line in dt_lines:
            parts = line.strip().split("\t")
            assert (len(parts) < 10), "line error: {}".format(line)
            if len(parts) == 8:
                dts.append(parts + [''])
            else:
                dts.append(parts)
        results.append((gts, dts, ignore_masks))
    return results
    def polygon_from_str(polygon_points):
        """
        Create a shapely polygon object from gt or dt line.
        """
        polygon_points = np.array(polygon_points).reshape(4, 2)
        polygon = Polygon(polygon_points).convex_hull
        return polygon
    return polygon_from_str

polygon_from_str = create_polygon_from_str()



def calculate_polygon_iou():
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
            if not os.path.exists(val_path):
def test_read_files():
    # Add test code here

def test_calculate_metrics():
    # Create dummy ground truth and prediction data
    gts = [[['0', '0', '10', '0', '10', '10', '0', '10', '0'], ['0', '0', '10', '0', '10', '10', '0', '10', '0']]]
    dts = [[['0', '0', '10', '0', '10', '10', '0', '10', '0'], ['0', '0', '10', '0', '10', '10', '0', '10', '0']]]
    ignore_masks = ['0', '0']

    # Call the calculate_metrics function with this data
    precision, recall, fmeasure, avg_edit_dist_img, avg_edit_dist_field, character_acc = calculate_metrics(gts, dts, ignore_masks, polygon_iou, strQ2B, ed)

    # Assert that the returned metrics match the expected metrics
    assert precision == 1.0, f'Expected 1.0, but got {precision}'
    assert recall == 1.0, f'Expected 1.0, but got {recall}'
    assert fmeasure == 1.0, f'Expected 1.0, but got {fmeasure}'
    assert avg_edit_dist_img == 0.0, f'Expected 0.0, but got {avg_edit_dist_img}'
    assert avg_edit_dist_field == 0.0, f'Expected 0.0, but got {avg_edit_dist_field}'
    assert character_acc == 1.0, f'Expected 1.0, but got {character_acc}'

def test_print_results():
    # Redirect stdout to a stringIO object
    stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    # Call the print_results function with dummy metrics
    print_results(1.0, 1.0, 1.0, 0.0, 0.0, 1.0)
    
    # Get the output and restore stdout
    output = sys.stdout.getvalue()
    sys.stdout = stdout
    
    # Assert that the output matches the expected output
    expected_output = 'character_acc: 100.00%\navg_edit_dist_field: 0.00\navg_edit_dist_img: 0.00\nprecision: 100.00%\nrecall: 100.00%\nfmeasure: 100.00%\n'
    assert output == expected_output, f'Expected "{expected_output}", but got "{output}"'
    
    # Test with different decimal places
    stdout = sys.stdout
    sys.stdout = io.StringIO()
    print_results(0.123456789, 0.987654321, 0.555555555, 0.111111111, 0.999999999, 0.666666666)
    output = sys.stdout.getvalue()
    sys.stdout = stdout
    expected_output = 'character_acc: 66.67%\navg_edit_dist_field: 0.11\navg_edit_dist_img: 1.00\nprecision: 12.35%\nrecall: 98.77%\nfmeasure: 55.56%\n'
    assert output == expected_output, f'Expected "{expected_output}", but got "{output}"'
    
    # Test with zero values
    stdout = sys.stdout
    sys.stdout = io.StringIO()
    print_results(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    output = sys.stdout.getvalue()
    sys.stdout = stdout
    expected_output = 'character_acc: 0.00%\navg_edit_dist_field: 0.00\navg_edit_dist_img: 0.00\nprecision: 0.00%\nrecall: 0.00%\nfmeasure: 0.00%\n'
    assert output == expected_output, f'Expected "{expected_output}", but got "{output}"'

def test_e2e_eval():
    # Create dummy ground truth and prediction files
    gt_dir = 'test_gt_dir'
    res_dir = 'test_res_dir'
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(gt_dir, 'test.txt'), 'w', encoding='utf-8') as f:
        f.write('0\t0\t10\t0\t10\t10\t0\t10\t0\n')
    with open(os.path.join(res_dir, 'test.txt'), 'w', encoding='utf-8') as f:
        f.write('0\t0\t10\t0\t10\t10\t0\t10\t0\n')

    # Redirect stdout to a stringIO object
    stdout = sys.stdout
    sys.stdout = io.StringIO()

    # Call the e2e_eval function with these files
    e2e_eval(gt_dir, res_dir)

    # Get the output and restore stdout
    output = sys.stdout.getvalue()
    sys.stdout = stdout

    # Assert that the output matches the expected output
    expected_output = 'start testing...\nhit, dt_count, gt_count 1 1 1\ncharacter_acc: 100.00%\navg_edit_dist_field: 0.00\navg_edit_dist_img: 0.00\nprecision: 100.00%\nrecall: 100.00%\nfmeasure: 100.00%\n'
    assert output == expected_output, f'Expected "{expected_output}", but got "{output}"'

    # Clean up
    shutil.rmtree(gt_dir)
    shutil.rmtree(res_dir)
                dt_lines = []
            else:
                with open(val_path, encoding='utf-8') as f:
                    dt_lines = [o.strip() for o in f.readlines()]
            dts = []
            for line in dt_lines:
        return iou
    return polygon_iou
def print_results(precision, recall, fmeasure, avg_edit_dist_img, avg_edit_dist_field, character_acc):
    print('character_acc: %.2f' % (character_acc * 100) + "%")
    print('avg_edit_dist_field: %.2f' % (avg_edit_dist_field))
    print('avg_edit_dist_img: %.2f' % (avg_edit_dist_img))
    print('precision: %.2f' % (precision * 100) + "%")
    print('recall: %.2f' % (recall * 100) + "%")
    print('fmeasure: %.2f' % (fmeasure * 100) + "%")

polygon_iou = calculate_polygon_iou()



def calculate_edit_distance():
    def ed(str1, str2):
        return editdistance.eval(str1, str2)
    return ed

ed = calculate_edit_distance()



def process_results_and_calculate_metrics(polygon_iou, strQ2B, ed, polygon_from_str):
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
def test_read_files():
    # Create dummy ground truth and prediction files
    gt_dir = 'test_gt_dir'
    res_dir = 'test_res_dir'
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(gt_dir, 'test.txt'), 'w', encoding='utf-8') as f:
        f.write('0\t0\t10\t0\t10\t10\t0\t10\t0\n')
    with open(os.path.join(res_dir, 'test.txt'), 'w', encoding='utf-8') as f:
        f.write('0\t0\t10\t0\t10\t10\t0\t10\t0\n')

    # Call the read_files function with these files
    results = read_files(gt_dir, res_dir)

    # Assert that the returned results match the expected results
    expected_results = [([['0', '0', '10', '0', '10', '10', '0', '10', '']], [['0', '0', '10', '0', '10', '10', '0', '10', '']], ['0'])]
    assert results == expected_results, f'Expected {expected_results}, but got {results}'

    # Clean up
    shutil.rmtree(gt_dir)
    shutil.rmtree(res_dir)

def test_calculate_metrics():
    # Add test code here

def test_print_results():
    def test_e2e_eval():
        # Create dummy ground truth and prediction files
        gt_dir = 'test_gt_dir'
        res_dir = 'test_res_dir'
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(res_dir, exist_ok=True)
        with open(os.path.join(gt_dir, 'test.txt'), 'w', encoding='utf-8') as f:
            f.write('0\t0\t10\t0\t10\t10\t0\t10\t0\n')
        with open(os.path.join(res_dir, 'test.txt'), 'w', encoding='utf-8') as f:
            f.write('0\t0\t10\t0\t10\t10\t0\t10\t0\n')
    
        # Redirect stdout to a stringIO object
        stdout = sys.stdout
        sys.stdout = io.StringIO()
    
        # Call the e2e_eval function with these files
        e2e_eval(gt_dir, res_dir)
    
        # Get the output and restore stdout
        output = sys.stdout.getvalue()
        sys.stdout = stdout
    
        # Assert that the output matches the expected output
        expected_output = 'start testing...\nhit, dt_count, gt_count 1 1 1\ncharacter_acc: 100.00%\navg_edit_dist_field: 0.00\navg_edit_dist_img: 0.00\nprecision: 100.00%\nrecall: 100.00%\nfmeasure: 100.00%\n'
        assert output == expected_output, f'Expected "{expected_output}", but got "{output}"'
    
        # Clean up
        shutil.rmtree(gt_dir)
        shutil.rmtree(res_dir)
    res_dir = 'test_res_dir'
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(gt_dir, 'test.txt'), 'w', encoding='utf-8') as f:
        f.write('0\t0\t10\t0\t10\t10\t0\t10\t0\n')
    with open(os.path.join(res_dir, 'test.txt'), 'w', encoding='utf-8') as f:
        f.write('0\t0\t10\t0\t10\t10\t0\t10\t0\n')

    # Redirect stdout to a stringIO object
    stdout = sys.stdout
    sys.stdout = io.StringIO()

    # Call the e2e_eval function with these files
    e2e_eval(gt_dir, res_dir)

    # Get the output and restore stdout
    output = sys.stdout.getvalue()
    sys.stdout = stdout

    # Assert that the output matches the expected output
    stdout = sys.stdout
    sys.stdout = io.StringIO()

    # Call the print_results function with dummy metrics
    print_results(1.0, 1.0, 1.0, 0.0, 0.0, 1.0)

    # Get the output and restore stdout
    output = sys.stdout.getvalue()
    sys.stdout = stdout

    # Assert that the output matches the expected output
    expected_output = 'character_acc: 100.00%\navg_edit_dist_field: 0.00\navg_edit_dist_img: 0.00\nprecision: 100.00%\nrecall: 100.00%\nfmeasure: 100.00%\n'
    assert output == expected_output, f'Expected "{expected_output}", but got "{output}"'

def test_e2e_eval():
    # Create dummy ground truth and prediction files
    gt_dir = 'test_gt_dir'
    res_dir = 'test_res_dir'
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(gt_dir, 'test.txt'), 'w', encoding='utf-8') as f:
        f.write('0\t0\t10\t0\t10\t10\t0\t10\t0\n')
    with open(os.path.join(res_dir, 'test.txt'), 'w', encoding='utf-8') as f:
        f.write('0\t0\t10\t0\t10\t10\t0\t10\t0\n')

    # Redirect stdout to a stringIO object
    stdout = sys.stdout
    sys.stdout = io.StringIO()

    # Call the e2e_eval function with these files
    e2e_eval(gt_dir, res_dir)

    # Get the output and restore stdout
    output = sys.stdout.getvalue()
    sys.stdout = stdout

    # Assert that the output matches the expected output

def test_e2e_eval():
    # Add test code here
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

        eps = 1e-9
        print('hit, dt_count, gt_count', hit, dt_count, gt_count)
        precision = hit / (dt_count + eps)
        recall = hit / (gt_count + eps)
        fmeasure = 2.0 * precision * recall / (precision + recall + eps)
        avg_edit_dist_img = ed_sum / len(val_names)
        avg_edit_dist_field = ed_sum / (gt_count + eps)
        character_acc = 1 - ed_sum / (num_gt_chars + eps)

        print('character_acc: %.2f' % (character_acc * 100) + "%")
        print('avg_edit_dist_field: %.2f' % (avg_edit_dist_field))
        print('avg_edit_dist_img: %.2f' % (avg_edit_dist_img))
        print('precision: %.2f' % (precision * 100) + "%")
        print('recall: %.2f' % (recall * 100) + "%")
        print('fmeasure: %.2f' % (fmeasure * 100) + "%")
    return e2e_eval

e2e_eval = process_results_and_calculate_metrics(polygon_iou, strQ2B, ed, polygon_from_str)


if __name__ == '__main__':
    def create_polygon_from_str():
        def polygon_from_str(polygon_points):
            """
            Create a shapely polygon object from gt or dt line.
            """
            polygon_points = np.array(polygon_points).reshape(4, 2)
            polygon = Polygon(polygon_points).convex_hull
            return polygon
        return polygon_from_str
    
    polygon_from_str = create_polygon_from_str()
    
    
    
    def calculate_polygon_iou():
    def calculate_metrics(gts, dts, ignore_masks, polygon_iou, strQ2B, ed):
        iou_thresh = 0.5
        num_gt_chars = 0
        gt_count = 0
        dt_count = 0
        hit = 0
        ed_sum = 0
    
        dt_match = [False] * len(dts)
        gt_match = [False] * len(gts)
        all_ious = defaultdict(tuple)
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
    
        for tindex, dt_match_flag in enumerate(dt_match):
            if dt_match_flag == False:
                dt_str = dts[tindex][8]
                gt_str = ''
                ed_sum += ed(dt_str, gt_str)
                dt_count += 1
    
        for tindex, gt_match_flag in enumerate(gt_match):
            if gt_match_flag == False and ignore_masks[tindex] == '0':
                dt_str = ''
                gt_str = gts[tindex][8]
                ed_sum += ed(gt_str, dt_str)
                num_gt_chars += len(gt_str)
                gt_count += 1
    
        eps = 1e-9
        precision = hit / (dt_count + eps)
        recall = hit / (gt_count + eps)
        fmeasure = 2.0 * precision * recall / (precision + recall + eps)
        avg_edit_dist_img = ed_sum / len(val_names)
        avg_edit_dist_field = ed_sum / (gt_count + eps)
        character_acc = 1 - ed_sum / (num_gt_chars + eps)
    
        return precision, recall, fmeasure, avg_edit_dist_img, avg_edit_dist_field, character_acc
    gt_folder = sys.argv[1]
    pred_folder = sys.argv[2]
    e2e_eval(gt_folder, pred_folder)
