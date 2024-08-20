import unittest
import pandas as pd
import numpy as np

from SD.general_utility import label_epochs

class TestSum(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")



class TestLabelEpochs(unittest.TestCase):
    
    def setUp(self):
        # This method will run before each test case.
        self.annotations = pd.DataFrame({
            'start_time': [10],
            'stop_time': [30],
            'label': ['seiz']
        })
    
    def test_incorrect_num_epochs(self):
        with self.assertRaises(ValueError):
            label_epochs(-5, 20, self.annotations)
    
    def test_incorrect_epoch_length(self):
        with self.assertRaises(ValueError):
            label_epochs(5, -20, self.annotations)

    def test_correct_labels_seizure(self):
        annotations_df_bi = pd.DataFrame({
            'start_time': [10],
            'stop_time': [60],
            'label': ['seiz']

        })
        labels = label_epochs(5, 30, annotations_df_bi)
        # the first two epochs contain seizure
        expected_labels = [1, 1, 0, 0, 0]
        self.assertTrue(np.array_equal(labels, expected_labels))

    def test_correct_labels_no_seizure(self):
        annotations_df_bi = pd.DataFrame({
            'start_time': [0],
            'stop_time': [150],
            'label': ['bckg']

        })
        labels = label_epochs(5, 30, annotations_df_bi)
        expected_labels = [0, 0, 0, 0, 0]
        self.assertTrue(np.array_equal(labels, expected_labels))

    
    def test_correct_labels_multiple_seizure(self):
        annotations_df_bi = pd.DataFrame({
            'start_time': [10, 50, 90],
            'stop_time': [30, 70, 110],
            'label': ['seiz', 'bckg', 'seiz']

        })
        labels = label_epochs(5, 30, annotations_df_bi)
        expected_labels = [1, 0, 0, 1, 0]
        print(labels)
        self.assertTrue(np.array_equal(labels, expected_labels))
    
        
    def test_annotation_missing_columns(self):
        annotations_df_bi = pd.DataFrame({
            'start_time': [10],
            'label': ['seiz']

        })
        with self.assertRaises(ValueError):
            label_epochs(5, 30, annotations_df_bi)

if __name__ == '__main__':
    unittest.main()