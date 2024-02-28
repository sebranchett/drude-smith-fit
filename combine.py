import csv
import numpy as np
from scipy.optimize import curve_fit
from reproduce import drude_smith_c1
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

    # Initial guess for the parameters
    m = 0.18  # mstar = m * m0
    tau = .80 * 10E-15  # 80 fs = 80 * 10E-15 s
    c1 = -0.82
    initial_guess = [m, tau, c1]

    # Perform the fit
    params, _ = curve_fit(
        fit_function, frequencies, stretched_complex_numbers,
        p0=initial_guess
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
