import unittest
from tools.end2end.eval_end2end import calculate_final_metrics, calculate_edit_distance

class TestEvalEnd2End(unittest.TestCase):

    def test_calculate_final_metrics(self):
        gt_dir = "/path/to/mock/gt_dir"
        val_name = "mock_val_name"
        res_dir = "/path/to/mock/res_dir"

        dts, gts, ignore_masks = calculate_final_metrics(gt_dir, val_name, res_dir)

        self.assertIsInstance(dts, list)
        self.assertIsInstance(gts, list)
        self.assertIsInstance(ignore_masks, list)

    def test_calculate_edit_distance(self):
        dts = [("mock_dt1", "mock_dt2")]
        gts = [("mock_gt1", "mock_gt2")]
        iou_thresh = 0.5

        sorted_gt_dt_pairs, gt_match, dt_match = calculate_edit_distance(dts, gts, iou_thresh)

        self.assertIsInstance(sorted_gt_dt_pairs, list)
        self.assertIsInstance(gt_match, list)
        self.assertIsInstance(dt_match, list)

if __name__ == '__main__':
    unittest.main()
