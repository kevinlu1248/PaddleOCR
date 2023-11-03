import unittest
from tools.end2end.eval_end2end import calculate_iou, calculate_edit_distance

class TestEvalEnd2End(unittest.TestCase):

    def test_calculate_iou(self):
        poly1 = [(0, 0), (1, 0), (1, 1), (0, 1)]
        poly2 = [(0, 0), (1, 0), (1, 1), (0, 1)]
        expected_iou = 1.0
        actual_iou = calculate_iou()(poly1, poly2)
        self.assertEqual(expected_iou, actual_iou)

        poly1 = [(0, 0), (1, 0), (1, 1), (0, 1)]
        poly2 = [(1, 1), (2, 1), (2, 2), (1, 2)]
        expected_iou = 0.0
        actual_iou = calculate_iou()(poly1, poly2)
        self.assertEqual(expected_iou, actual_iou)

    def test_calculate_edit_distance(self):
        str1 = "hello"
        str2 = "hello"
        expected_distance = 0
        actual_distance = calculate_edit_distance()(str1, str2)
        self.assertEqual(expected_distance, actual_distance)

        str1 = "hello"
        str2 = "helli"
        expected_distance = 1
        actual_distance = calculate_edit_distance()(str1, str2)
        self.assertEqual(expected_distance, actual_distance)

if __name__ == '__main__':
    unittest.main()
