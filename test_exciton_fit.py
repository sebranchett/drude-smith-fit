import unittest
import numpy as np
from drude_smith_fit import read_csv, perform_fit
from drude_smith_fit import set_input_parameters


class ExcitonFitTestCase(unittest.TestCase):

    def test_exciton_fit(self):
        set_input_parameters(1., 1., 0., 0., 0., 0., False, False, False)
        frequencies, complex_numbers = read_csv(
            'test_exciton.csv', 0., 2.E13
        )
        num_variable_params = 3
        fitted_complex_numbers, params_fit, std_dev_fit = perform_fit(
            frequencies, complex_numbers, num_variable_params
        )
        self.assertIsInstance(fitted_complex_numbers, np.ndarray)
        self.assertIsInstance(params_fit, list)
        self.assertIsInstance(std_dev_fit, list)
        self.assertEqual(len(fitted_complex_numbers), len(frequencies))
        self.assertEqual(len(params_fit), 9)
        self.assertEqual(len(std_dev_fit), 9)
        expected_result = np.array([
            1., 1., 0., 0., 0., 0., 300., 6.28319E13, 2.E13
        ])
        assert np.isclose(params_fit, expected_result).all()

        expected_result = np.array([
            0., 0., 0., 0., 0., 0., 4.133830e-11, 1.194149, 3.648919
        ])
        assert np.isclose(std_dev_fit, expected_result).all()


if __name__ == '__main__':
    unittest.main()
