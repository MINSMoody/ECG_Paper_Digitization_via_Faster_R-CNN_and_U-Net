
import unittest
import pytest
from ..src import helper_code
import os
import numpy as np

class TestHelperCode(unittest.TestCase):

    def setUp(self):
        self.data_folder = './TeamCode/tests/resources/example_data'


    def skip_test_trim_signal(self):
        fake_signal = np.random.rand(1000, 12)
        num_samples = 100
        with pytest.raises(NameError) as e:
            out = helper_code.trim_signal(fake_signal, num_samples)
            self.assertTrue("NameError: name 'num_channels' is not defined" in str(e))
            print(str(e))

    def test_load_signal(self):
        signal, header = helper_code.load_signals(os.path.join(self.data_folder, "00001_hr"))
        self.assertEqual(signal.shape, (5000, 12))
        self.assertTrue(signal.dtype == np.float64)


        

