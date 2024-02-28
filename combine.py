import csv
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def read_csv(filename, min_frequency, max_frequency):
    frequencies = []
    complex_numbers = []

    with open(filename, 'r') as file:
        reader = csv.reader(file)

        for row in reader:
            if min_frequency <= float(row[0]) <= max_frequency:
                frequency = float(row[0])
                real_part = float(row[2])
                imaginary_part = float(row[1])
                complex_number = complex(real_part, imaginary_part)

                frequencies.append(frequency)
                complex_numbers.append(complex_number)

    return np.array(frequencies), np.array(complex_numbers)


def drude_smith_c1(frequency, m, tau, c1):
    # Calculate the Drude-Smith mobility with only 1 c coefficient
    e = 1.602 * 10E-19
    m0 = 9.109 * 10E-31

    mstar = m * m0
    factor = e * tau / mstar
    omega = 2 * np.pi * frequency
    onePlusw2t2 = 1. + (omega * omega * tau * tau)
    oneMinusw2t2 = 1. - (omega * omega * tau * tau)

    result = onePlusw2t2 + c1 * oneMinusw2t2 + \
        1j * omega * tau * (onePlusw2t2 + 2. * c1)

    return factor * result / (onePlusw2t2*onePlusw2t2)


def drude_smith_c3(frequencies, m, tau, c1, c2=0., c3=0.):
    # Calculate the Drude-Smith mobility with 3 c coefficients
    e = 1.602 * 10E-19
    m0 = 9.109 * 10E-31

    mstar = m * m0
    f1 = e * tau / mstar
    f2 = 1 / (1 - 1j * 2 * np.pi * frequencies * tau)
    f3 = 1 + (c1 / (1 - 1j * 2 * np.pi * frequencies * tau)) + \
             (c2 / (1 - 1j * 2 * np.pi * frequencies * tau) ** 2) + \
             (c3 / (1 - 1j * 2 * np.pi * frequencies * tau) ** 3)
    complex_argument = f1 * f2 * f3
    return complex_argument


def fit_function(frequencies, m, tau, c1):
    # Define the function to fit
    results = drude_smith_c1(frequencies, m, tau, c1)
    stretched_results = np.concatenate((np.real(results), np.imag(results)))
    return stretched_results


def plot_experimental_and_fitted_data(
    frequencies, complex_numbers, fitted_complex_numbers
):
    plt.scatter(
        frequencies,
        [complex_number.real for complex_number in complex_numbers],
        marker='.',
        label='Experimental',
        color='red'
    )
    plt.scatter(
        frequencies,
        [complex_number.imag for complex_number in complex_numbers],
        marker='.',
        color='red'
    )
    plt.plot(
        frequencies,
        [complex_number.real for complex_number in fitted_complex_numbers],
        label='Fitted',
        color='blue'
    )
    plt.plot(
        frequencies,
        [complex_number.imag for complex_number in fitted_complex_numbers],
        color='blue'
    )
    plt.legend()
    plt.xlabel('Frequency')
    plt.ylabel('Real and Imaginary Parts')
    plt.title('Experimental and Fitted Data')
    plt.savefig('experimental_and_fitted_data.png')
    plt.show()


if __name__ == "__main__":
    filename = "mobility.csv"

    min_frequency = 0.3E12
    max_frequency = 2.2E12

    frequencies, complex_numbers = read_csv(
        filename, min_frequency, max_frequency
    )

    stretched_complex_numbers = np.concatenate(
        (np.real(complex_numbers), np.imag(complex_numbers))
    )

    # encourage imaginary part to be negative
    max_tau = 0.5 / (2. * np.pi * (max_frequency + min_frequency))

    # initial_guess = [m, tau, c1]
    minima = [-np.inf, 0., -1.]
    maxima = [np.inf, max_tau, 0.]

    # Perform the fit
    params, _ = curve_fit(
        fit_function, frequencies, stretched_complex_numbers,
        bounds=(minima, maxima)
    )

    # Extract the fitted parameters
    m_fit, tau_fit, c1_fit = params
    print("Fitted value of m:", m_fit)
    print("Fitted value of tau:", tau_fit)
    print("Fitted value of c1:", c1_fit)

    # Use the fitted parameters to calculate the fitted complex numbers
    fitted_stretched_complex_numbers = fit_function(
        frequencies, m_fit, tau_fit, c1_fit
    )

    fitted_complex_numbers = \
        fitted_stretched_complex_numbers[:len(frequencies)] + \
        1j * fitted_stretched_complex_numbers[len(frequencies):]

    plot_experimental_and_fitted_data(
        frequencies, complex_numbers, fitted_complex_numbers
    )
