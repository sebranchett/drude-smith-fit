import unittest
import numpy as np
from ..drude_smith_fit import read_csv, perform_fit
from ..drude_smith_fit import set_input_parameters


class ExcitonFitTestCase(unittest.TestCase):

    def test_exciton_fit(self):
        set_input_parameters(0., 1., 0., 0., 0., 0., 1., False, 0., False)
        frequencies, complex_numbers = read_csv(
            'test/test_exciton.csv', 0., 2.5E12
        )
        num_variable_params = 2
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
            0., 1., 0., 0., 0., 0., 1., 40., 0., 5.E12
        ])
        assert np.isclose(params_fit, expected_result).all()

        expected_result = np.array([
            0., 0., 0., 0., 0., 0., 0., 1.432E-15, 0., 2.259E-04
        ])
        assert np.isclose(std_dev_fit, expected_result, rtol=.001).all()

        # Test the same data, but with Drude only
        set_input_parameters(1., False, False, 0., 0., 0., 0., 0., 0., 0.)
        num_variable_params = 2
        fitted_complex_numbers, params_fit, std_dev_fit = perform_fit(
            frequencies, complex_numbers, num_variable_params
        )

        expected_result = np.array([
            1., 0.025, 200.E-15, 0., 0., 0., 0., 0., 0., 0.
        ])
        assert np.isclose(params_fit, expected_result).all()

        expected_result = np.array([
            0., 1.366E-18, 1.379E-29, 0., 0., 0., 0., 0., 0., 0.
        ])
        assert np.isclose(std_dev_fit, expected_result, rtol=.001).all()

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
            5.408e-02, 1.000e+00, 1.165e-14,
            -5.061e-28, -4.918e-01, -1.000e+00,
            1.000e+00, 3.495e-03, 2.137e+12, 6.882e+12
        ])
        assert np.isclose(params_fit, expected_result, rtol=.001).all()

        expected_result = np.array([
            1.094e-01, 0.000e+00, 1.469e-14, 7.190e+00, 1.716e+01, 9.861e+00,
            0.000e+00, 7.358e-03, 1.732e+13, 1.526e+13
        ])
        assert np.isclose(std_dev_fit, expected_result, rtol=.001).all()


if __name__ == '__main__':
    unittest.main()
