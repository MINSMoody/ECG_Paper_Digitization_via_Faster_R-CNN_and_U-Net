


import unittest
import pytest
from ..src import sample_implementation, interface, helper_code
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

            if signal is not None:
                helper_code.save_signal(output_record, signal)

                comment_lines = [l for l in data_header.split('\n') if l.startswith('#')]
                signal_header = helper_code.load_header(output_record)
                signal_header += ''.join(comment_lines) + '\n'
                helper_code.save_header(output_record, signal_header)

            if dx is not None:
                helper_code.save_dx(output_record, dx)

        print('finished')
        

    def _run_evaluation(self):
        # copy pasted from evaluate_model.py by the challenge organizers on Mar 28
        label_folder = self.data_folder
        output_folder = self.output_folder
        extra_scores = False
        # Find the records.
        records = helper_code.find_records(label_folder)
        num_records = len(records)

        # Compute the signal reconstruction metrics.
        records_completed_signal_reconstruction = list()
        snr = dict()
        snr_median = dict()
        ks_metric = dict()
        asci_metric = dict()
        weighted_absolute_difference_metric = dict()

        # Iterate over the records.
        for record in records:
            # Load the signals, if available.
            label_record = os.path.join(label_folder, record)
            label_signal, label_fields = helper_code.load_signal(label_record)

            if label_signal is not None:
                label_channels = label_fields['sig_name']
                label_num_channels = label_fields['n_sig']
                label_num_samples = label_fields['sig_len']
                label_sampling_frequency = label_fields['fs']
                label_units = label_fields['units']

                output_record = os.path.join(output_folder, record)
                output_signal, output_fields = helper_code.load_signal(output_record)

                if output_signal is not None:
                    output_channels = output_fields['sig_name']
                    output_num_channels = output_fields['n_sig']
                    output_num_samples = output_fields['sig_len']
                    output_sampling_frequency = output_fields['fs']
                    output_units = output_fields['units']

                    records_completed_signal_reconstruction.append(record)

                    # Check that the label and output signals match as expected.
                    assert(label_sampling_frequency == output_sampling_frequency)
                    assert(label_units == output_units)

                    # Reorder the channels in the output signal to match the channels in the label signal.
                    output_signal = helper_code.reorder_signal(output_signal, output_channels, label_channels)

                    # Trim or pad the channels in the output signal to match the channels in the label signal.
                    output_signal = helper_code.trim_signal(output_signal, label_num_samples)

                    # Replace the samples with NaN values in the output signal with zeros.
                    output_signal[np.isnan(output_signal)] = 0

                else:
                    output_signal = np.zeros(np.shape(label_signal), dtype=label_signal.dtype)

                # Compute the signal reconstruction metrics.
                channels = label_channels
                num_channels = label_num_channels
                sampling_frequency = label_sampling_frequency

                for j, channel in enumerate(channels):
                    value = helper_code.compute_snr(label_signal[:, j], output_signal[:, j])
                    snr[(record, channel)] = value

                    if extra_scores:
                        value = helper_code.compute_snr_median(label_signal[:, j], output_signal[:, j])
                        snr_median[(record, channel)] = value

                        value = helper_code.compute_ks_metric(label_signal[:, j], output_signal[:, j])
                        ks_metric[(record, channel)] = value

                        value = helper_code.compute_asci_metric(label_signal[:, j], output_signal[:, j])
                        asci_metric[(record, channel)] = value

                        value = helper_code.compute_weighted_absolute_difference(label_signal[:, j], output_signal[:, j], sampling_frequency)
                        weighted_absolute_difference_metric[(record, channel)] = value

        # Compute the metrics.
        if len(records_completed_signal_reconstruction) > 0:
            snr = np.array(list(snr.values()))
            if not np.all(np.isnan(snr)):
                mean_snr = np.nanmean(snr)
            else:
                mean_snr = float('nan')

            if extra_scores:
                snr_median = np.array(list(snr_median.values()))
                if not np.all(np.isnan(snr_median)):
                    mean_snr_median = np.nanmean(snr_median)
                else:
                    mean_snr_median = float('nan')

                ks_metric = np.array(list(ks_metric.values()))
                if not np.all(np.isnan(ks_metric)):
                    mean_ks_metric = np.nanmean(ks_metric)
                else:
                    mean_ks_metric = float('nan')

                asci_metric = np.array(list(asci_metric.values()))
                if not np.all(np.isnan(asci_metric)):
                    mean_asci_metric = np.nanmean(asci_metric)
                else:
                    mean_asci_metric = float('nan')

                weighted_absolute_difference_metric = np.array(list(weighted_absolute_difference_metric.values()))
                if not np.all(np.isnan(weighted_absolute_difference_metric)):
                    mean_weighted_absolute_difference_metric = np.nanmean(weighted_absolute_difference_metric)
                else:
                    mean_weighted_absolute_difference_metric = float('nan')
            else:
                mean_snr_median = float('nan')
                mean_ks_metric = float('nan')
                mean_asci_metric = float('nan')
                mean_weighted_absolute_difference_metric = float('nan')

        else:
            mean_snr = float('nan')
            mean_snr_median = float('nan')
            mean_ks_metric = float('nan')
            mean_asci_metric = float('nan')
            mean_weighted_absolute_difference_metric = float('nan')

        # Compute the classification metrics.
        records_completed_classification = list()
        label_dxs = list()
        output_dxs = list()

        # Iterate over the records.
        for record in records:
            # Load the classes, if available.
            label_record = os.path.join(label_folder, record)
            label_dx = helper_code.load_dx(label_record)

            if label_dx:
                output_record = os.path.join(output_folder, record)
                output_dx = helper_code.load_dx(output_record)

                if output_dx:
                    records_completed_classification.append(record)

                label_dxs.append(label_dx)
                output_dxs.append(output_dx)

        # Compute the metrics.
        if len(records_completed_classification) > 0:
            f_measure, _, _ = helper_code.compute_f_measure(label_dxs, output_dxs)
        else:
            f_measure = float('nan')


    def _test_both_models(self, digitization_class, classification_class):
        self._run_models(digitization_class, classification_class)
        self._run_evaluation()

    def test_sample_implementation(self):
        self._test_both_models(sample_implementation.ExampleDigitizationModel, sample_implementation.ExampleClassificationModel)
        
    @pytest.mark.skip(reason = "not used any more")
    def test_Kmeans_implementation(self):
        self._test_both_models(sample_implementation.KMeansDigitizationModel, sample_implementation.ExampleClassificationModel)

# Haoliang added this for testing
if __name__ == '__main__':
    # Create a test suite containing the specific test
    suite = unittest.TestSuite()
    suite.addTest(TestTools('test_sample_implementation'))
    
    # Run the test suite
    runner = unittest.TextTestRunner()
    runner.run(suite)