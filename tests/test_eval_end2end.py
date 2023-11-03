import unittest
from tools.end2end.eval_end2end import calculate_iou, calculate_edit_distance, calculate_metrics, calculate_avg_edit_distance, calculate_character_acc

class TestEvalEnd2End(unittest.TestCase):

    def test_calculate_iou(self):
        # Test with two overlapping polygons
        poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly2 = Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)])
        self.assertAlmostEqual(calculate_iou(poly1, poly2), 0.25)

        # Test with two non-overlapping polygons
        poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
        self.assertEqual(calculate_iou(poly1, poly2), 0)

    def test_calculate_edit_distance(self):
        # Test with two identical strings
        self.assertEqual(calculate_edit_distance("abc", "abc"), 0)

        # Test with two completely different strings
        self.assertEqual(calculate_edit_distance("abc", "def"), 3)

        # Test with two strings with some common characters
        self.assertEqual(calculate_edit_distance("abc", "abd"), 1)

    def test_calculate_metrics(self):
        # Test with perfect prediction
        self.assertEqual(calculate_metrics(10, 10, 1e-9, 10), (1.0, 1.0, 1.0))

        # Test with no correct predictions
        self.assertEqual(calculate_metrics(0, 10, 1e-9, 10), (0.0, 0.0, 0.0))

        # Test with some correct predictions
        self.assertAlmostEqual(calculate_metrics(5, 10, 1e-9, 10), (0.5, 0.5, 0.5))

    def test_calculate_avg_edit_distance(self):
        # Test with no errors
        self.assertEqual(calculate_avg_edit_distance(0, ["img1", "img2"], 2, 1e-9), (0.0, 0.0))

        # Test with some errors
        self.assertEqual(calculate_avg_edit_distance(4, ["img1", "img2"], 2, 1e-9), (2.0, 2.0))

    def test_calculate_character_acc(self):
        # Test with no errors
        self.assertEqual(calculate_character_acc(0, 10, 1e-9), 1.0)

        # Test with some errors
        self.assertEqual(calculate_character_acc(2, 10, 1e-9), 0.8)

if __name__ == '__main__':
    unittest.main()
