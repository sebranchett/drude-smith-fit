import unittest
import numpy as np
from ..drude_smith_fit import read_csv, perform_fit
from ..drude_smith_fit import set_input_parameters


class ExcitonFitTestCase(unittest.TestCase):

    def test_exciton_fit(self):
        set_input_parameters(0., 1., 0., 0., 0., 0., 1., False, False, False)
        frequencies, complex_numbers = read_csv(
            'test/test_exciton.csv', 0., 2.E13
        )
        num_variable_params = 3
        fitted_complex_numbers, params_fit, std_dev_fit = perform_fit(
            frequencies, complex_numbers, num_variable_params
        )
        self.assertIsInstance(fitted_complex_numbers, np.ndarray)
        self.assertIsInstance(params_fit, list)
        self.assertIsInstance(std_dev_fit, list)
        self.assertEqual(len(fitted_complex_numbers), len(frequencies))
        self.assertEqual(len(params_fit), 10)
        self.assertEqual(len(std_dev_fit), 10)
        expected_result = np.array([
            0., 1., 0., 0., 0., 0., 1., 300., 6.28319E13, 2.E13
        ])
        assert np.isclose(params_fit, expected_result).all()

        expected_result = np.array([
            0., 0., 0., 0., 0., 0., 0., 4.133830e-11, 1.194149, 3.648919
        ])
        assert np.isclose(std_dev_fit, expected_result).all()

    def test_mixed_fit(self):
        # Test fitting a mixed model with Drude and Lorentz
        # Not a good fit, but it should work
        set_input_parameters(False, 1., False, False, False, False,
                             1., False, False, False)
        frequencies, complex_numbers = read_csv(
            'test/test_exciton.csv', 0., 2.E13
        )
        num_variable_params = 8
        fitted_complex_numbers, params_fit, std_dev_fit = perform_fit(
            frequencies, complex_numbers, num_variable_params
        )
        self.assertEqual(len(params_fit), 10)
        self.assertEqual(len(std_dev_fit), 10)
        expected_result = np.array([
            0.10186, 1., 1.913E-15, -1., .9999, .9999,
            1., 4.396E-8, 8.706E7, 7.083E7
        ])
        assert np.isclose(params_fit, expected_result, rtol=.001).all()

        expected_result = np.array([
            2.581E2, 0., 1.276E-12, 8.937E3, 1.548E4, 1.293E4,
            0., 2.587E1, 0., 0.
        ])
        assert np.isclose(std_dev_fit, expected_result, rtol=.001).all()


if __name__ == '__main__':
    unittest.main()
