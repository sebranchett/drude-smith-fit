import unittest
import numpy as np
from drude_smith_fit import read_csv, perform_fit
from drude_smith_fit import drude_smith_c3, arrange_parameters
from drude_smith_fit import set_input_parameters, fit_function


class DrudeSmithFitTestCase(unittest.TestCase):
    set_input_parameters(1., False, False, False, 0., 0.)

    def test_read_csv(self):
        frequencies, complex_numbers = read_csv(
            'test_data.csv', 0.3E12, 2.2E12
        )
        self.assertIsInstance(frequencies, np.ndarray)
        self.assertIsInstance(complex_numbers, np.ndarray)
        self.assertEqual(len(frequencies), 15)
        self.assertEqual(len(complex_numbers), 15)

    def test_drude_smith_c3(self):
        frequencies = np.array([1E12, 2E12, 3E12, 4E12, 5E12])
        phi = 1.0
        m = 2.0
        tau = 3.0
        c1 = 4.0
        c2 = 5.0
        c3 = 6.0
        expected_result = np.array([
            42.112363 + 2.383551j, 41.824256 + 4.747365j,
            41.348111 + 7.072056j, 40.689876 + 9.338918j,
            39.857688 + 11.530251j
        ])
        result = drude_smith_c3(frequencies, phi, m, tau, c1, c2, c3)
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_arrange_parameters_with_std_dev_false(self):
        fit_values = [11.0, 22.0, 33.0]
        expected_result = [
            1., 11., 22., 33., 0., 0.,
        ]
        result = arrange_parameters(fit_values, std_dev=False)
        self.assertEqual(result, expected_result)

    def test_arrange_parameters_with_std_dev_true(self):
        fit_values = [11.0, 22.0, 33.0]
        expected_result = [
            0., 11., 22., 33., 0., 0.,
        ]
        result = arrange_parameters(fit_values, std_dev=True)
        self.assertEqual(result, expected_result)

    def test_fit_function(self):
        frequencies = np.array([1E12, 2E12, 3E12, 4E12, 5E12])
        fit_values = [11.0, 22.0, 33.0]
        expected_result = np.array([
            113.078342,  95.787243,  72.977641,  49.962049,  30.145426,
            31.374944,  56.292759,  71.334921,  76.777795,  75.125706
        ])
        result = fit_function(frequencies, fit_values)
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_perform_fit(self):
        frequencies = np.array([1E12, 2E12, 3E12, 4E12, 5E12])
        complex_numbers = np.array([1+2j, 2+3j, 3+4j, 4+5j, 5+6j])
        num_variable_params = 3
        fitted_complex_numbers, params_fit, std_dev_fit = perform_fit(
            frequencies, complex_numbers, num_variable_params
        )
        self.assertIsInstance(fitted_complex_numbers, np.ndarray)
        self.assertIsInstance(params_fit, list)
        self.assertIsInstance(std_dev_fit, list)
        self.assertEqual(len(fitted_complex_numbers), len(frequencies))
        self.assertEqual(len(params_fit), 6)
        self.assertEqual(len(std_dev_fit), 6)
        expected_result = np.array([
            4.804727+0.869923j, 4.404369+1.760269j, 3.696767+2.368778j,
            2.952127+2.621344j, 2.323706+2.638324j
        ])
        np.testing.assert_array_almost_equal(
            fitted_complex_numbers, expected_result
        )
        expected_result = np.array([
            1., 12.365495, 50.029188, -0.31461, 0., 0.0
        ])
        np.testing.assert_array_almost_equal(
            params_fit, expected_result
        )
        expected_result = np.array([
            0., 8.801072, 92.359791,  1.231787, 0., 0.0
        ])
        np.testing.assert_array_almost_equal(
            std_dev_fit, expected_result
        )


if __name__ == '__main__':
    unittest.main()
