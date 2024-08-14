
import unittest
import pytest

from ..src import interface, helper_code, implementation, evaluate_model
import numpy as np

import os


class TestTools(unittest.TestCase):

    def setUp(self) -> None:
        self.data_folder = './TeamCode/tests/resources/example_data'
        self.model_folder = './TeamCode/tests/resources/example_model'
        self.output_folder = './TeamCode/tests/resources/example_output'
        self.verbose = True
        self.allowFailures = False

    def test_paths_exist(self):
        self.assertTrue(os.path.exists(self.data_folder))
        self.assertTrue(os.path.exists(self.model_folder))



    def _run_models(self, digitization_class, classification_class):
        # copy pasted from run_model.py by the challenge organizers on Mar 28
        self.assertTrue(issubclass(digitization_class, interface.AbstractDigitizationModel))
        self.assertTrue(issubclass(classification_class, interface.AbstractClassificationModel))

        digitization_model = digitization_class()
        classification_model = classification_class()

        ## train model
        digitization_model.train_model(self.data_folder, self.model_folder, self.verbose) 
        classification_model.train_model(self.data_folder, self.model_folder, self.verbose) 

        digitization_model = None
        classification_model = None

        ## run model
        trained_digitization_model = digitization_class.from_folder(self.model_folder, self.verbose) 
        trained_classification_model = classification_class.from_folder(self.model_folder, self.verbose) 

        records = helper_code.find_records(self.data_folder)
        num_records = len(records)

        for i in range(num_records):
            if self.verbose:
                width = len(str(num_records))
                print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

            data_record = os.path.join(self.data_folder, records[i])
            output_record = os.path.join(self.output_folder, records[i])


            try:
                signal = trained_digitization_model.run_digitization_model(data_record, self.verbose) 
            except:
                if self.allowFailures:
                    if self.verbose:
                        print('... digitization failed.')
                    signal = None
                else:
                    raise


            try:
                dx = trained_classification_model.run_classification_model(data_record, signal, self.verbose) 
            except:
                if self.allowFailures:
                    if self.verbose >= 2:
                        print('... dx classification failed.')
                    dx = None
                else:
                    raise

                    # Save Challenge outputs.
                
            output_path = os.path.split(output_record)[0]
            os.makedirs(output_path, exist_ok=True)

            data_header = helper_code.load_header(data_record)
            helper_code.save_header(output_record, data_header)

            signals = signal # they renamed
            labels = dx

            if signals is not None:
                comments = [l for l in data_header.split('\n') if l.startswith('#')]
                helper_code.save_signals(output_record, signals, comments)
            if labels is not None:
                helper_code.save_labels(output_record, labels)

        print('finished')
        
    def _run_evaluation(self):
        label_folder = self.data_folder
        output_folder = self.output_folder
        extra_scores = False
        no_shift = False

        scores = evaluate_model.evaluate_model(folder_ref=label_folder, folder_est=output_folder, no_shift=no_shift, extra_scores=extra_scores)

                # Unpack the scores.
        snr, snr_median, ks_metric, asci_metric, mean_weighted_absolute_difference_metric, f_measure = scores

        # Construct a string with scores.
        if not extra_scores:
            output_string = \
                f'SNR: {snr:.3f}\n' + \
                f'F-measure: {f_measure:.3f}\n'
        else:
            output_string = \
                f'SNR: {snr:.3f}\n' + \
                f'SNR median: {snr_median:.3f}\n' \
                f'KS metric: {ks_metric:.3f}\n' + \
                f'ASCI metric: {asci_metric:.3f}\n' \
                f'Weighted absolute difference metric: {mean_weighted_absolute_difference_metric:.3f}\n' \
                f'F-measure: {f_measure:.3f}\n'

        print(output_string)

    def _test_both_models(self, digitization_class, classification_class):
        self._run_models(digitization_class, classification_class)
        self._run_evaluation()

    def test_our_implementation(self):
        self._test_both_models(implementation.OurDigitizationModel, implementation.VoidClassificationModel)
        

# Haoliang added this for testing
if __name__ == '__main__':
    # Create a test suite containing the specific test
    suite = unittest.TestSuite()
    suite.addTest(TestTools('test_our_implementation'))
    
    # Run the test suite
    runner = unittest.TextTestRunner()
    runner.run(suite)