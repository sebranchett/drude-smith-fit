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


def drude_smith_c3(frequencies, phi, m, tau, c1, c2=0., c3=0.):
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


def arrange_parameters(fit_values, std_dev=False):
    global input_parameters
    index = 0
    if isinstance(input_parameters[0], float):
        if std_dev:
            phi = 0.
        else:
            phi = input_parameters[0]
    else:
        phi = fit_values[index]
        index += 1
    if isinstance(input_parameters[1], float):
        if std_dev:
            m = 0.
        else:
            m = input_parameters[1]
    else:
        m = fit_values[index]
        index += 1
    if isinstance(input_parameters[2], float):
        if std_dev:
            tau = 0.
        else:
            tau = input_parameters[2]
    else:
        tau = fit_values[index]
        index += 1
    if isinstance(input_parameters[3], float):
        if std_dev:
            c1 = 0.
        else:
            c1 = input_parameters[3]
    else:
        c1 = fit_values[index]
        index += 1
    if isinstance(input_parameters[4], float):
        if std_dev:
            c2 = 0.
        else:
            c2 = input_parameters[4]
    else:
        c2 = fit_values[index]
        index += 1
    if isinstance(input_parameters[5], float):
        if std_dev:
            c3 = 0.
        else:
            c3 = input_parameters[5]
    else:
        c3 = fit_values[index]
    return [phi, m, tau, c1, c2, c3]


def fit_function(frequencies, fit_values):
    # To get the fit to work, curve_fit needs to work in seconds,
    # but the Drude-Smith model uses femtoseconds
    phi, m, tau, c1, c2, c3 = arrange_parameters(fit_values)
    results = drude_smith_c3(frequencies, phi, m, tau * 1E-15, c1, c2, c3)
    stretched_results = np.concatenate((np.real(results), np.imag(results)))
    return stretched_results


# The following 5 defs are needed because curve_fit requires a function with
# the right number of parameters
def fit_function_5(frequencies, fit01, fit02, fit03, fit04, fit05):
    fit_values = [fit01, fit02, fit03, fit04, fit05]
    return fit_function(frequencies, fit_values)


def fit_function_4(frequencies, fit01, fit02, fit03, fit04):
    fit_values = [fit01, fit02, fit03, fit04]
    return fit_function(frequencies, fit_values)


def fit_function_3(frequencies, fit01, fit02, fit03):
    fit_values = [fit01, fit02, fit03]
    return fit_function(frequencies, fit_values)


def fit_function_2(frequencies, fit01, fit02):
    fit_values = [fit01, fit02]
    return fit_function(frequencies, fit_values)


def fit_function_1(frequencies, fit01):
    fit_values = [fit01]
    return fit_function(frequencies, fit_values)


def perform_fit(frequencies, complex_numbers, num_variable_params):
    global input_parameters
    # Set some physics boundaries
    min_phi = 0.
    max_phi = 1.
    min_m = 0.  # this helps the fit to converge
    max_m = 10.  # this helps the fit to converge
    min_tau = 0.
    max_tau = np.inf
    min_c1 = -1.
    max_c1 = 0.
    min_c2 = -1.
    max_c2 = 1.
    min_c3 = -1.
    max_c3 = 1.

    minima = []
    maxima = []

    if not isinstance(input_parameters[0], float):
        minima.append(min_phi)
        maxima.append(max_phi)
    if not isinstance(input_parameters[1], float):
        minima.append(min_m)
        maxima.append(max_m)
    if not isinstance(input_parameters[2], float):
        minima.append(min_tau)
        maxima.append(max_tau)
    if not isinstance(input_parameters[3], float):
        minima.append(min_c1)
        maxima.append(max_c1)
    if not isinstance(input_parameters[4], float):
        minima.append(min_c2)
        maxima.append(max_c2)
    if not isinstance(input_parameters[5], float):
        minima.append(min_c3)
        maxima.append(max_c3)

    # To fit both the real and imaginary parts of the complex numbers
    # create a 'stretched' array
    stretched_complex_numbers = np.concatenate(
        (np.real(complex_numbers), np.imag(complex_numbers))
    )

    # Perform the fit (curve_fit requires separate functions for
    # different numbers of parameters)
    if num_variable_params == 5:
        params, pcov = curve_fit(
            fit_function_5, frequencies, stretched_complex_numbers,
            bounds=(minima, maxima)
        )
        fitted_stretched_complex_numbers = fit_function_5(
            frequencies, params[0], params[1], params[2], params[3], params[4]
        )
    elif num_variable_params == 4:
        params, pcov = curve_fit(
            fit_function_4, frequencies, stretched_complex_numbers,
            bounds=(minima, maxima)
        )
        fitted_stretched_complex_numbers = fit_function_4(
            frequencies, params[0], params[1], params[2], params[3]
        )
    elif num_variable_params == 3:
        params, pcov = curve_fit(
            fit_function_3, frequencies, stretched_complex_numbers,
            bounds=(minima, maxima)
        )
        fitted_stretched_complex_numbers = fit_function_3(
            frequencies, params[0], params[1], params[2]
        )
    elif num_variable_params == 2:
        params, pcov = curve_fit(
            fit_function_2, frequencies, stretched_complex_numbers,
            bounds=(minima, maxima)
        )
        fitted_stretched_complex_numbers = fit_function_2(
            frequencies, params[0], params[1]
        )
    elif num_variable_params == 1:
        params, pcov = curve_fit(
            fit_function_1, frequencies, stretched_complex_numbers,
            bounds=(minima, maxima)
        )
        fitted_stretched_complex_numbers = fit_function_1(
            frequencies, params[0]
        )

    fitted_complex_numbers = \
        fitted_stretched_complex_numbers[:len(frequencies)] + \
        1j * fitted_stretched_complex_numbers[len(frequencies):]

    std_dev = np.sqrt(np.diag(pcov))
    params_fit = arrange_parameters(params)
    std_dev_fit = arrange_parameters(std_dev, True)

    return [fitted_complex_numbers, params_fit, std_dev_fit]


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


def set_input_parameters(phi, m, tau, c1, c2, c3):
    global input_parameters
    input_parameters = [phi, m, tau, c1, c2, c3]
    num_variable_params = sum(
        not isinstance(param, float) for param in input_parameters
    )
    return num_variable_params


if __name__ == "__main__":
    global input_parameters

    filename = "mobility.csv"
    min_frequency = 0.3E12
    max_frequency = 2.2E12

    fix_phi = 1.
    fix_m = False
    fix_tau = False
    fix_c1 = False
    fix_c2 = 0.
    fix_c3 = 0.

    if not isinstance(fix_phi, float) and not isinstance(fix_m, float):
        print("Error: phi and m cannot both be variable")
        sys.exit(1)
    num_variable_params = set_input_parameters(
        fix_phi, fix_m, fix_tau, fix_c1, fix_c2, fix_c3
    )

    if num_variable_params < 1 or num_variable_params > 5:
        print("Error: only 1, 2, 3, 4 or 5 variable parameters allowed,")
        print("found", num_variable_params, "variable parameters")
        sys.exit(1)
    image_filename = filename.split('.')[0] + '.png'
    txt_filename = filename.split('.')[0] + '.txt'

    if len(sys.argv) > 1:
        if filename == "-h" or filename == "--help":
            print_help()
            sys.exit(0)
        if len(sys.argv) > 2:
            min_frequency = float(sys.argv[2])
            if len(sys.argv) > 3:
                max_frequency = float(sys.argv[3])

    frequencies, complex_numbers = read_csv(
        filename, min_frequency, max_frequency
    )

    fitted_complex_numbers, \
        [phi_fit, m_fit, tau_fit, c1_fit, c2_fit, c3_fit], \
        std_dev = perform_fit(
            frequencies, complex_numbers, num_variable_params
        )

    print("Value of phi:", '{:.3e}'.format(phi_fit),
          "+/-", '{:.3e}'.format(std_dev[0]))
    print("Value of m:", '{:.3e}'.format(m_fit),
          "+/-", '{:.3e}'.format(std_dev[1]))
    print("Value of tau:", '{:.3e}'.format(tau_fit),
          "femtoseconds +/-", '{:.3e}'.format(std_dev[2]), 'femtoseconds')
    print("Value of c1:", '{:.3e}'.format(c1_fit),
          "+/-", '{:.3e}'.format(std_dev[3]))
    print("Value of c2:", '{:.3e}'.format(c2_fit),
          "+/-", '{:.3e}'.format(std_dev[4]))
    print("Value of c3:", '{:.3e}'.format(c3_fit),
          "+/-", '{:.3e}'.format(std_dev[5]))

    plot_experimental_and_fitted_data(
        frequencies, complex_numbers, fitted_complex_numbers,
        "phi = %.3e, m = %.3e, tau = %.3e,\nc1 = %.3e, c2 = %.3e, c3 = %.3e"
        % (phi_fit, m_fit, tau_fit * 1E-15, c1_fit, c2_fit, c3_fit),
        image_filename
    )  # Convert to femtoseconds

    with open(txt_filename, 'w') as file:
        file.writelines(
            "# phi, std, m, std, tau(fs), std, c1, std, c2, std, c3, std\n"
        )
        file.writelines(
            "{:.3e}".format(phi_fit) + ", " +
            "{:.3e}".format(std_dev[0]) + ", " +
            "{:.3e}".format(m_fit) + ", " +
            "{:.3e}".format(std_dev[1]) + ", " +
            "{:.3e}".format(tau_fit) + ", " +
            "{:.3e}".format(std_dev[2]) + ", " +
            "{:.3e}".format(c1_fit) + ", " +
            "{:.3e}".format(std_dev[3]) + ", " +
            "{:.3e}".format(c2_fit) + ", " +
            "{:.3e}".format(std_dev[4]) + ", " +
            "{:.3e}".format(c3_fit) + ", " +
            "{:.3e}".format(std_dev[5]) +
            "\n"
        )
