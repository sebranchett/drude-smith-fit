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
        complex_numbers *= 1.e-4
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

        print(std_dev_fit)
        expected_result = np.array([
            0., 0., 0., 0., 0., 0., 0., 2.669E-15, 0., 4.211E-04
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
            0., 1.128E-18, 1.139E-29, 0., 0., 0., 0., 0., 0., 0.
        ])
        assert np.isclose(std_dev_fit, expected_result, rtol=.001).all()

    def test_mixed_fit(self):
        # Test fitting a mixed model with Drude and Lorentz
        set_input_parameters(phi=False, m=0.2, tau=False,
                             c1=False, c2=0., c3=0.,
                             phi_ex=0.5, fbn=False,
                             wbn=(1.91 * 2. * 3.1416), gamma=False)
        frequencies, complex_numbers = read_csv(
            'test/test_combine.csv', 0., 2.5E13
        )
        complex_numbers *= 1.e-4
        num_variable_params = 5
        fitted_complex_numbers, params_fit, std_dev_fit = perform_fit(
            frequencies, complex_numbers, num_variable_params
        )
        self.assertEqual(len(params_fit), 10)
        self.assertEqual(len(std_dev_fit), 10)
        expected_result = np.array([
            0.5, 0.2, 700.E-15, -0.7, 0., 0.,
            0.5, 0.4, 12.E12, 1.E12
        ])
        assert np.isclose(params_fit, expected_result, rtol=.001).all()

        expected_result = np.array([
            2.644e-05, 0.000e+00, 5.786e-17, 6.108e-05, 0.000e+00, 0.000e+00,
            0.000e+00, 3.565e-05, 0.000e+00, 1.211e+08
        ])
        assert np.isclose(std_dev_fit, expected_result, rtol=.001).all()


if __name__ == '__main__':
    unittest.main()
