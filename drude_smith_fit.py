import sys
import csv
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def print_help():
    print("Usage: python drude_smith_fit.py [input_filename [min_frequency" +
          " [max_frequency [output_filename]]]]")
    print("")
    print("Fits experimental data to the Drude-Smith model for mobility.")
    print("")
    print("")
    print("input_filename: The name of the file containing the experimental" +
          " data. The file should be a CSV file with the first column" +
          " containing the frequency in Hz, the second column containing" +
          " the imaginary part of the complex number, and the third column" +
          " containing the real part of the complex number.")
    print("*** UNITS must be cm2 V-1 s-1. ***")
    print("")
    print("min_frequency: The minimum frequency to include in the fit. " +
          "Default is 0.3 THz.")
    print("")
    print("max_frequency: The maximum frequency to include in the fit. " +
          "Default is 2.2 THz.")
    print("")
    print("The input file basename is used to create the output file names.")
    print("This program creates a .png file with a graph of the fit and")
    print("a .txt file with the fitted parameters.")


def read_csv(filename, min_frequency, max_frequency):
    frequencies = []
    complex_numbers = []

    with open(filename, 'r') as file:
        content = file.read()
        find_comma = content.find(',')
    with open(filename, 'r') as file:
        if find_comma != -1:
            reader = csv.reader(file, delimiter=',')
        else:
            # try a semi-colon
            reader = csv.reader(file, delimiter=';')

        for row in reader:
            if min_frequency <= float(row[0]) <= max_frequency:
                frequency = float(row[0])
                real_part = float(row[2])
                imaginary_part = float(row[1])
                complex_number = complex(real_part, imaginary_part)

                frequencies.append(frequency)
                complex_numbers.append(complex_number)

    return np.array(frequencies), np.array(complex_numbers)


def drude_smith_c3(frequencies, m, tau, c1, c2=0., c3=0., phi=1.):
    # Calculate the Drude-Smith mobility with 3 c coefficients
    e = 1.602E-19
    m0 = 9.109E-31
    conversion = 10000.  # input is in cm^2

    mstar = m * m0
    f1 = conversion * phi * e * tau / mstar
    f2 = 1 / (1 - 1j * 2 * np.pi * frequencies * tau)
    f3 = 1 + (c1 / (1 - 1j * 2 * np.pi * frequencies * tau)) + \
             (c2 / (1 - 1j * 2 * np.pi * frequencies * tau) ** 2) + \
             (c3 / (1 - 1j * 2 * np.pi * frequencies * tau) ** 3)
    complex_argument = f1 * f2 * f3
    return complex_argument


def fit_function(frequencies, m, tau, c1):
    # To get the fit to work, curve_fit needs to work in seconds,
    # but the Drude-Smith model uses femtoseconds
    results = drude_smith_c3(frequencies, m, tau * 1E-15, c1)
    stretched_results = np.concatenate((np.real(results), np.imag(results)))
    return stretched_results


def perform_fit(frequencies, complex_numbers):
    # Set some physics boundaries
    min_m = 0.  # this helps the fit to converge
    max_m = 10.  # this helps the fit to converge
    min_c1 = -1.
    max_c1 = 0.
    min_tau = 0.
    max_tau = np.inf

    minima = [min_m, min_tau, min_c1]
    maxima = [max_m, max_tau, max_c1]

    # To fit both the real and imaginary parts of the complex numbers
    # create a 'stretched' array
    stretched_complex_numbers = np.concatenate(
        (np.real(complex_numbers), np.imag(complex_numbers))
    )

    # Perform the fit
    params, pcov = curve_fit(
        fit_function, frequencies, stretched_complex_numbers,
        bounds=(minima, maxima)
    )
    std_dev = np.sqrt(np.diag(pcov))

    # Extract the fitted parameters
    m_fit, tau_fit, c1_fit = params

    # Use the fitted parameters to calculate the fitted complex numbers
    fitted_stretched_complex_numbers = fit_function(
        frequencies, m_fit, tau_fit, c1_fit
    )

    fitted_complex_numbers = \
        fitted_stretched_complex_numbers[:len(frequencies)] + \
        1j * fitted_stretched_complex_numbers[len(frequencies):]

    return [fitted_complex_numbers, m_fit, tau_fit, c1_fit, std_dev]


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
    image_filename = filename.split('.')[0] + '.png'
    txt_filename = filename.split('.')[0] + '.txt'

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if filename == "-h":
            print_help()
            sys.exit(0)
        if len(sys.argv) > 2:
            min_frequency = float(sys.argv[2])
            if len(sys.argv) > 3:
                max_frequency = float(sys.argv[3])

    frequencies, complex_numbers = read_csv(
        filename, min_frequency, max_frequency
    )

    fitted_complex_numbers, m_fit, tau_fit, c1_fit, std_dev = perform_fit(
        frequencies, complex_numbers
    )

    print("Fitted value of m/phi:", '{:.3e}'.format(m_fit),
          "+/-", '{:.3e}'.format(std_dev[0]))
    print("Fitted value of tau:", '{:.3e}'.format(tau_fit),
          "femtoseconds +/-", '{:.3e}'.format(std_dev[1]), 'femtoseconds')
    print("Fitted value of c1:", '{:.3e}'.format(c1_fit),
          "+/-", '{:.3e}'.format(std_dev[2]))

    plot_experimental_and_fitted_data(
        frequencies, complex_numbers, fitted_complex_numbers,
        "m_fit/phi = %.3e, tau_fit = %.3e, c_fit = %.3e"
        % (m_fit, tau_fit * 1E-15, c1_fit),
        image_filename
    )  # Convert to femtoseconds

    with open(txt_filename, 'w') as file:
        file.writelines("# m/phi, std, tau(fs), std, c1, std\n")
        file.writelines(
            "{:.3e}".format(m_fit) + ", " +
            "{:.3e}".format(std_dev[0]) + ", " +
            "{:.3e}".format(tau_fit) + ", " +
            "{:.3e}".format(std_dev[1]) + ", " +
            "{:.3e}".format(c1_fit) + ", " +
            "{:.3e}".format(std_dev[2]) +
            "\n"
        )
