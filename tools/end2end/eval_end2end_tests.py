import unittest
from tools.end2end.eval_end2end import calculate_iou, calculate_edit_distance, calculate_metrics, match_gt_and_dt, handle_unmatched_dt, handle_unmatched_gt
from shapely.geometry import Polygon

class TestEvalEnd2End(unittest.TestCase):

    def test_calculate_iou(self):
        poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly2 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly3 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])

        self.assertEqual(calculate_iou(poly1, poly2), 1.0)
        self.assertEqual(calculate_iou(poly1, poly3), 0.0)
        self.assertEqual(calculate_iou(poly1, Polygon([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)])), 0.25)

    def test_calculate_edit_distance(self):
        self.assertEqual(calculate_edit_distance('test', 'test'), 0)
        self.assertEqual(calculate_edit_distance('test', 'tset'), 2)
        self.assertEqual(calculate_edit_distance('test', 'abcd'), 4)

    def test_calculate_metrics(self):
        self.assertEqual(calculate_metrics(10, 10, 10), (1.0, 1.0, 1.0))
        self.assertEqual(calculate_metrics(5, 10, 10), (0.5, 0.5, 0.5))
        self.assertEqual(calculate_metrics(0, 10, 10), (0.0, 0.0, 0.0))

    def test_match_gt_and_dt(self):
        sorted_gt_dt_pairs = [(0, 0), (1, 1), (2, 2)]
        gt_match = [False, False, False]
        dt_match = [False, False, False]
        ignore_blank = False
        gts = [['', '', '', '', '', '', '', '', 'test'], ['', '', '', '', '', '', '', '', 'test'], ['', '', '', '', '', '', '', '', 'test']]
        dts = [['', '', '', '', '', '', '', '', 'test'], ['', '', '', '', '', '', '', '', 'test'], ['', '', '', '', '', '', '', '', 'test']]
        ignore_masks = ['0', '0', '0']
        ed_sum = 0
        num_gt_chars = 0
        hit = 0
        gt_count = 0
        dt_count = 0

        ed_sum, num_gt_chars, hit, gt_count, dt_count = match_gt_and_dt(sorted_gt_dt_pairs, gt_match, dt_match, ignore_blank, gts, dts, ignore_masks, ed_sum, num_gt_chars, hit, gt_count, dt_count)

        self.assertEqual(ed_sum, 0)
        self.assertEqual(num_gt_chars, 12)
        self.assertEqual(hit, 3)
        self.assertEqual(gt_count, 3)
        self.assertEqual(dt_count, 3)

    def test_handle_unmatched_dt(self):
        dt_match = [False, False, False]
        dts = [['', '', '', '', '', '', '', '', 'test'], ['', '', '', '', '', '', '', '', 'test'], ['', '', '', '', '', '', '', '', 'test']]
        ed_sum = 0
        dt_count = 0

        ed_sum, dt_count = handle_unmatched_dt(dt_match, dts, ed_sum, dt_count)

        self.assertEqual(ed_sum, 12)
        self.assertEqual(dt_count, 3)

    def test_handle_unmatched_gt(self):
        gt_match = [False, False, False]
        ignore_masks = ['0', '0', '0']
        gts = [['', '', '', '', '', '', '', '', 'test'], ['', '', '', '', '', '', '', '', 'test'], ['', '', '', '', '', '', '', '', 'test']]
        num_gt_chars = 0
        ed_sum = 0
        gt_count = 0

        ed_sum, gt_count, num_gt_chars = handle_unmatched_gt(gt_match, ignore_masks, gts, num_gt_chars, ed_sum, gt_count)

        self.assertEqual(ed_sum, 12)
        self.assertEqual(gt_count, 3)
        self.assertEqual(num_gt_chars, 12)

if __name__ == '__main__':
    unittest.main()
