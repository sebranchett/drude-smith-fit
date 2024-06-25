import unittest
from ..drude_smith_fit import read_csv
from ..drude_smith_fit import guess_min_Lorentz_f


class GuessMinLorentzFTestCase(unittest.TestCase):

    def test_guess_min_Lorentz_f(self):
        frequencies, complex_numbers = read_csv(
            'test/test_combine.csv', 0., 2.5E12
        )
        expected_result = 1.89
        result = guess_min_Lorentz_f(frequencies, complex_numbers.real)
        self.assertEqual(result, expected_result)

    def test_guess_zero(self):
        frequencies, complex_numbers = read_csv(
            'test/test_exciton.csv', 0., 2.5E12
        )
        expected_result = 0
        result = guess_min_Lorentz_f(frequencies, complex_numbers.real)
        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
