import sys
import csv
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def print_help():
    print("Usage: python drude_smith_fit.py [input_filename [min_frequency" +
          " [max_frequency [output_filename]]]]")


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


# def drude_smith_c3(frequencies, m, tau, c1, c2=0., c3=0.):
#     # Calculate the Drude-Smith mobility with 3 c coefficients
#     e = 1.602 * 10E-19
#     m0 = 9.109 * 10E-31

#     mstar = m * m0
#     f1 = e * tau / mstar
#     f2 = 1 / (1 - 1j * 2 * np.pi * frequencies * tau)
#     f3 = 1 + (c1 / (1 - 1j * 2 * np.pi * frequencies * tau)) + \
#              (c2 / (1 - 1j * 2 * np.pi * frequencies * tau) ** 2) + \
#              (c3 / (1 - 1j * 2 * np.pi * frequencies * tau) ** 3)
#     complex_argument = f1 * f2 * f3
#     return complex_argument


def fit_function(frequencies, m, tau, c1):
    # To get the fit to work, curve_fit needs to work in seconds,
    # but the Drude-Smith model uses femtoseconds
    results = drude_smith_c1(frequencies, m, tau * 1E-15, c1)
    stretched_results = np.concatenate((np.real(results), np.imag(results)))
    return stretched_results


def plot_experimental_and_fitted_data(
    frequencies, complex_numbers, fitted_complex_numbers, title,
    output_filename
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
    plt.title(title)
    plt.savefig(output_filename)
    plt.show()


if __name__ == "__main__":
    filename = "mobility.csv"
    min_frequency = 0.3E12
    max_frequency = 2.2E12
    output_filename = "experimental_and_fitted_data.png"

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if filename == "-h":
            print_help()
            sys.exit(0)
        if len(sys.argv) > 2:
            min_frequency = float(sys.argv[2])
            if len(sys.argv) > 3:
                max_frequency = float(sys.argv[3])
                if len(sys.argv) > 4:
                    output_filename = sys.argv[4]

    frequencies, complex_numbers = read_csv(
        filename, min_frequency, max_frequency
    )

    # To fit both the real and imaginary parts of the complex numbers
    # create a 'stretched' array
    stretched_complex_numbers = np.concatenate(
        (np.real(complex_numbers), np.imag(complex_numbers))
    )

    # Set some physics boundaries
    min_c1 = -1.
    max_c1 = 0.
    min_tau = 0.
    max_tau = np.inf
    print(min_tau, max_tau)

    minima = [-np.inf, min_tau, min_c1]
    maxima = [np.inf, max_tau, max_c1]

    # Perform the fit
    params, pcov = curve_fit(
        fit_function, frequencies, stretched_complex_numbers,
        bounds=(minima, maxima)
    )

    # Extract the fitted parameters
    m_fit, tau_fit, c1_fit = params
    print("Fitted value of m:", m_fit)
    print("Fitted value of tau:", tau_fit * 1E-15)  # Convert to femtoseconds
    print("Fitted value of c1:", c1_fit)
    # print("One standard deviation:", np.sqrt(np.diag(pcov)))

    # Use the fitted parameters to calculate the fitted complex numbers
    fitted_stretched_complex_numbers = fit_function(
        frequencies, m_fit, tau_fit, c1_fit
    )

    fitted_complex_numbers = \
        fitted_stretched_complex_numbers[:len(frequencies)] + \
        1j * fitted_stretched_complex_numbers[len(frequencies):]

    plot_experimental_and_fitted_data(
        frequencies, complex_numbers, fitted_complex_numbers,
        "m_fit = %.3e, tau_fit = %.3e, c_fit = %.3e"
        % (m_fit, tau_fit * 1E-15, c1_fit),
        output_filename
    )  # Convert to femtoseconds
