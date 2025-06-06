import sys
import csv
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# set the default global variable 'factor'
factor = 1.


def print_help():
    print("Usage: python drude_smith_fit.py [input_filename [min_frequency" +
          " [max_frequency [fix_phi [fix_m [fix_tau" +
          " [fix_c1 [fix_c2 [fix_c3 [fix_phi_ex [fix_fbn [fix_wbn" +
          " [fix_gamma [min_Lorentz_f]]]]]]]]]]]]]]")
    print("")
    print("Fits experimental data to the Drude-Smith model for mobility " +
          "and Lorentz oscillator for exciton response.")
    print("")
    print("")
    print("input_filename: The name of the file containing the experimental" +
          " data. The file should be:\n",
          " either\n",
          " a CSV file with UNITS m2 V-1 s-1\n",
          " first column contains the frequency in Hz,\n",
          " second column contains the imaginary part of the mobility,\n",
          " and third column contains the real part of the mobility;\n",
          " or\n",
          " a TXT file with UNITS m2 V-1 s-1\n",
          " first column contains the frequency in Hz,\n",
          " second column contains delta |E(omega)|, not used,\n",
          " third column contains |E(omega)|, not used,\n",
          " fourth column contains the imaginary part of the mobility,\n",
          " and fifth column contains the real part of the mobility.")
    print("Default is 'mobility.csv'.")
    print("")
    print("min_frequency, max_frequency: The minimum and maximum " +
          "frequencies to include in the fit.")
    print("Defaults are 0.3E12 and 2.2E12.")
    print("")
    print("fix_phi, fix_m, fix_tau: Drude fit parameters.")
    print("Defaults are 1., False, False. 'False' means parameter will be " +
          "fit.")
    print("")
    print("fix_c1, fix_c2, fix_c3: Smith fit parameters.")
    print("Defaults are False, 0., 0. 'False' means parameter will be fit.")
    print("")
    print("fix_phi_ex, fix_fbn, fix_wbn, fix_gamma: Lorentz fit parameters.")
    print("Defaults are 0., 0., 0., 0. 'False' means parameter will be fit.")
    print("")
    print("min_Lorentz_f: Minimum frequency [THz] to start search for " +
          "Lorentz peak in Real part of signal.")
    print("Default is -1., which means the program will try to guess.")
    print("")
    print("The input file basename is used to create the output file names.")
    print("This program creates a .png file with a graph of the fit,")
    print("a _param.csv file with the fitted parameters and a _fitted.csv")
    print("file with both the experimental and fitted data.")


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
            # skip the row if the columns cannot be converted to float
            # these are comments, annotations, etc.
            try:
                float(row[0])
                float(row[1])
                float(row[2])
            except ValueError:
                continue
            if min_frequency <= float(row[0]) <= max_frequency:
                frequency = float(row[0])
                real_part = float(row[2])
                imaginary_part = float(row[1])
                complex_number = complex(real_part, imaginary_part)

                frequencies.append(frequency)
                complex_numbers.append(complex_number)

    return np.array(frequencies), np.array(complex_numbers)


def read_txt(filename, min_frequency, max_frequency):
    frequencies = []
    complex_numbers = []

    # read the 5 columns separated by spaces
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("#"):
                continue
            columns = line.split()
            frequency = float(columns[0])

            if min_frequency <= frequency <= max_frequency:
                real_part = float(columns[4])
                imaginary_part = float(columns[3])
                complex_number = complex(real_part, imaginary_part)
                frequencies.append(frequency)
                complex_numbers.append(complex_number)

    return np.array(frequencies), np.array(complex_numbers)


def plot_checks(filename, min_frequency, max_frequency):
    # Throw an error if the file is not a txt file
    if filename[-4:] != ".txt":
        raise ValueError("File is not a txt file")
    # Read the first column as frequencies, the second as check_01 and
    # the third as check_02
    frequencies, check_01, check_02 = np.loadtxt(
        filename, usecols=(0, 1, 2), unpack=True
    )
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(frequencies, check_02)
    # Draw a vertical line at min_frequency and max_frequency
    plt.axvline(x=min_frequency, color='r', linestyle='--')
    plt.axvline(x=max_frequency, color='r', linestyle='--')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"$|E(\omega)|$")
    plt.subplot(1, 2, 2)
    plt.plot(frequencies, check_01)
    # Draw a vertical line at min_frequency and max_frequency
    plt.axvline(x=min_frequency, color='r', linestyle='--')
    plt.axvline(x=max_frequency, color='r', linestyle='--')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"$\Delta |E(\omega)|$")
    plt.tight_layout()


def drude_smith_c3(
    frequencies, phi, m, tau, c1, c2=0., c3=0.,
    phi_ex=0., fbn=0., wbn=0., gamma=0.
):
    global factor
    # Calculate the Drude-Smith mobility with 3 c coefficients
    # interpret tau as if it is in fs, this helps the fit to converge
    e = 1.602E-19
    m0 = 9.109E-31
    tau = tau * 1E-15  # convert tau from fs to s
    w = 2. * np.pi * frequencies
    w_tau = w * tau
    wbn *= 1E12  # convert wbn from THz to Hz
    gamma *= 1E12  # convert gamma from THz to Hz

    mstar = m * m0
    f1 = phi * e * tau / mstar
    f2 = 1 / (1 - 1j * w_tau)
    f3 = 1 + (c1 / (1 - 1j * w_tau)) + \
             (c2 / ((1 - 1j * w_tau) ** 2)) + \
             (c3 / ((1 - 1j * w_tau) ** 3))
    complex_argument = f1 * f2 * f3

    # Include the exciton response which originates from transitions from
    # lowest exciton state to higher states.
    # This is described in equation (4) of Nanoscale, 2019, 11, p. 21571
    # with n = 1
    ex = (e/(1j * mstar)) * (fbn * w) / \
        (wbn**2 - w**2 - (1j * w * gamma))
    ex = phi_ex * ex
    complex_argument += ex
    return factor * complex_argument


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
        index += 1
    if isinstance(input_parameters[6], float):
        if std_dev:
            phi_ex = 0.
        else:
            phi_ex = input_parameters[6]
    else:
        phi_ex = fit_values[index]
        index += 1
    if isinstance(input_parameters[7], float):
        if std_dev:
            fbn = 0.
        else:
            fbn = input_parameters[7]
    else:
        fbn = fit_values[index]
        index += 1
    if isinstance(input_parameters[8], float):
        if std_dev:
            wbn = 0.
        else:
            wbn = input_parameters[8]
    else:
        wbn = fit_values[index]
        index += 1
    if isinstance(input_parameters[9], float):
        if std_dev:
            gamma = 0.
        else:
            gamma = input_parameters[9]
    else:
        gamma = fit_values[index]
        index += 1
    return [phi, m, tau, c1, c2, c3, phi_ex, fbn, wbn, gamma]


def fit_function(frequencies, fit_values):
    phi, m, tau, c1, c2, c3, phi_ex, fbn, wbn, gamma = \
        arrange_parameters(fit_values)
    results = drude_smith_c3(
        frequencies, phi, m, tau, c1, c2, c3, phi_ex, fbn, wbn, gamma
    )
    stretched_results = np.concatenate((np.real(results), np.imag(results)))
    return stretched_results


# The following 9 defs are needed because curve_fit requires a function with
# the right number of parameters
def fit_function_9(
    frequencies, fit01, fit02, fit03, fit04, fit05, fit06, fit07, fit08,
    fit09
):
    fit_values = [fit01, fit02, fit03, fit04, fit05, fit06, fit07, fit08,
                  fit09]
    return fit_function(frequencies, fit_values)


def fit_function_8(
    frequencies, fit01, fit02, fit03, fit04, fit05, fit06, fit07, fit08
):
    fit_values = [fit01, fit02, fit03, fit04, fit05, fit06, fit07, fit08]
    return fit_function(frequencies, fit_values)


def fit_function_7(
    frequencies, fit01, fit02, fit03, fit04, fit05, fit06, fit07
):
    fit_values = [fit01, fit02, fit03, fit04, fit05, fit06, fit07]
    return fit_function(frequencies, fit_values)


def fit_function_6(frequencies, fit01, fit02, fit03, fit04, fit05, fit06):
    fit_values = [fit01, fit02, fit03, fit04, fit05, fit06]
    return fit_function(frequencies, fit_values)


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


def perform_fit(frequencies, orig_complex_numbers, num_variable_params):
    global input_parameters, factor
    # fitting works better if the values are scaled to be between ~ 0 and 1000.
    factor = 1000. / orig_complex_numbers.max().real
    complex_numbers = factor * orig_complex_numbers
    # Set some physics boundaries
    min_Lorentz_f = input_parameters[10]
    min_phi = 0.
    max_phi = 1.
    min_m = 0.
    max_m = np.inf
    min_tau = 0.
    max_tau = np.inf
    min_c1 = -1.
    max_c1 = 0.
    min_c2 = -1.
    max_c2 = 1.
    min_c3 = -1.
    max_c3 = 1.
    min_phi_ex = 0.
    max_phi_ex = 1.
    min_fbn = 0.
    max_fbn = np.inf
    min_wbn = 2. * np.pi * min_Lorentz_f
    max_wbn = np.inf
    min_gamma = 0.
    max_gamma = np.inf

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
    if not isinstance(input_parameters[6], float):
        minima.append(min_phi_ex)
        maxima.append(max_phi_ex)
    if not isinstance(input_parameters[7], float):
        minima.append(min_fbn)
        maxima.append(max_fbn)
    if not isinstance(input_parameters[8], float):
        minima.append(min_wbn)
        maxima.append(max_wbn)
    if not isinstance(input_parameters[9], float):
        minima.append(min_gamma)
        maxima.append(max_gamma)

    # To fit both the real and imaginary parts of the complex numbers
    # create a 'stretched' array
    stretched_complex_numbers = np.concatenate(
        (np.real(complex_numbers), np.imag(complex_numbers))
    )

    # Perform the fit (curve_fit requires separate functions for
    # different numbers of parameters)
    # There are 10 parameters in total, but only one of phi_ex, fbn and m
    # can be variable, so the number of variable parameters is between 1 and 8
    if num_variable_params == 9:
        params, pcov = curve_fit(
            fit_function_9, frequencies, stretched_complex_numbers,
            bounds=(minima, maxima)
        )
        fitted_stretched_complex_numbers = fit_function_9(
            frequencies, params[0], params[1], params[2], params[3],
            params[4], params[5], params[6], params[7], params[8]
        )
    elif num_variable_params == 8:
        params, pcov = curve_fit(
            fit_function_8, frequencies, stretched_complex_numbers,
            bounds=(minima, maxima)
        )
        fitted_stretched_complex_numbers = fit_function_8(
            frequencies, params[0], params[1], params[2], params[3],
            params[4], params[5], params[6], params[7]
        )
    elif num_variable_params == 7:
        params, pcov = curve_fit(
            fit_function_7, frequencies, stretched_complex_numbers,
            bounds=(minima, maxima)
        )
        fitted_stretched_complex_numbers = fit_function_7(
            frequencies, params[0], params[1], params[2], params[3],
            params[4], params[5], params[6]
        )
    elif num_variable_params == 6:
        params, pcov = curve_fit(
            fit_function_6, frequencies, stretched_complex_numbers,
            bounds=(minima, maxima)
        )
        fitted_stretched_complex_numbers = fit_function_6(
            frequencies, params[0], params[1], params[2], params[3],
            params[4], params[5]
        )
    elif num_variable_params == 5:
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

    # fitting was scaled, so scale back the fitted values
    fitted_complex_numbers /= factor

    std_dev = np.sqrt(np.diag(pcov))
    params_fit = arrange_parameters(params)
    std_dev_fit = arrange_parameters(std_dev, True)

    # convert tau from s to fs
    params_fit[2] *= 1E-15
    std_dev_fit[2] *= 1E-15
    # convert wbn from THz to Hz
    params_fit[8] *= 1E12
    std_dev_fit[8] *= 1E12
    # convert gamma from THz to Hz
    params_fit[9] *= 1E12
    std_dev_fit[9] *= 1E12

    return [fitted_complex_numbers, params_fit, std_dev_fit]


def write_csv(filename, frequencies, complex_numbers, fitted_complex_numbers):
    data = np.column_stack((
        frequencies,
        np.imag(complex_numbers), np.real(complex_numbers),
        np.imag(fitted_complex_numbers), np.real(fitted_complex_numbers)
    ))
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            '# Frequencies', 'Experiment (Imag)', 'Experiment (Real)',
            'Fitted (Imag)', 'Fitted (Real)'
        ])
        writer.writerows(data)


def plot_experimental_and_fitted_data(
    frequencies, complex_numbers, fitted_complex_numbers, title,
    filename
):
    plt.scatter(
        frequencies,
        [complex_number.real for complex_number in complex_numbers],
        marker='.',
        label='Experimental Real',
        color='red'
    )
    plt.scatter(
        frequencies,
        [complex_number.imag for complex_number in complex_numbers],
        marker='x',
        label='Experimental Imaginary',
        color='red'
    )
    plt.plot(
        frequencies,
        [complex_number.real for complex_number in fitted_complex_numbers],
        label='Fitted Real',
        color='blue'
    )
    plt.plot(
        frequencies,
        [complex_number.imag for complex_number in fitted_complex_numbers],
        label='Fitted Imaginary',
        color='blue',
        linestyle='--'
    )
    plt.legend()
    plt.xlabel('Frequency ($Hz$)')
    plt.ylabel('Real and Imaginary Parts ($m^2$ $V^{-1}$ $s^{-1}$)')
    plt.title(title)
    plt.savefig(filename)
    plt.show()


def set_input_parameters(phi, m, tau, c1, c2, c3,
                         phi_ex, fbn, wbn, gamma, min_Lorentz_f=0.):
    global input_parameters
    input_parameters = [phi, m, tau, c1, c2, c3,
                        phi_ex, fbn, wbn, gamma, min_Lorentz_f]
    num_variable_params = sum(
        not isinstance(param, float) for param in input_parameters
    )
    return num_variable_params


def print_fit_results(
        phi_fit, m_fit, tau_fit, c1_fit, c2_fit, c3_fit,
        phi_ex_fit, fbn_fit, wbn_fit, gamma_fit, std_dev
):
    print("Value of phi:", '{:.3e}'.format(phi_fit), end=" ")
    if not isinstance(input_parameters[0], float):
        print("+/-", '{:.3e}'.format(std_dev[0]))
    else:
        print('fixed')
    print("Value of m:", '{:.3e}'.format(m_fit), end=" ")
    if not isinstance(input_parameters[1], float):
        print("+/-", '{:.3e}'.format(std_dev[1]))
    else:
        print('fixed')
    print("Value of tau:", '{:.3e}'.format(tau_fit), end=" ")
    if not isinstance(input_parameters[2], float):
        print("+/-", '{:.3e}'.format(std_dev[2]))
    else:
        print('fixed')
    print("Value of c1:", '{:.3e}'.format(c1_fit), end=" ")
    if not isinstance(input_parameters[3], float):
        print("+/-", '{:.3e}'.format(std_dev[3]))
    else:
        print('fixed')
    print("Value of c2:", '{:.3e}'.format(c2_fit), end=" ")
    if not isinstance(input_parameters[4], float):
        print("+/-", '{:.3e}'.format(std_dev[4]))
    else:
        print('fixed')
    print("Value of c3:", '{:.3e}'.format(c3_fit), end=" ")
    if not isinstance(input_parameters[5], float):
        print("+/-", '{:.3e}'.format(std_dev[5]))
    else:
        print('fixed')
    print("Value of phi_ex:", '{:.3e}'.format(phi_ex_fit), end=" ")
    if not isinstance(input_parameters[6], float):
        print("+/-", '{:.3e}'.format(std_dev[6]))
    else:
        print('fixed')
    print("Value of fbn:", '{:.3e}'.format(fbn_fit), end=" ")
    if not isinstance(input_parameters[7], float):
        print("+/-", '{:.3e}'.format(std_dev[7]))
    else:
        print('fixed')
    print("Value of wbn:", '{:.3e}'.format(wbn_fit), end=" ")
    if not isinstance(input_parameters[8], float):
        print("+/-", '{:.3e}'.format(std_dev[8]))
    else:
        print('fixed')
    print("Value of gamma:", '{:.3e}'.format(gamma_fit), end=" ")
    if not isinstance(input_parameters[9], float):
        print("+/-", '{:.3e}'.format(std_dev[9]))
    else:
        print('fixed')


def write_parameters(
        filename, phi_fit, m_fit, tau_fit,
        c1_fit, c2_fit, c3_fit,
        phi_ex_fit, fbn_fit, wbn_fit, gamma_fit, std_dev
):
    header = (
        "# phi, std, m, std, tau, std, c1, std, c2, std, c3, std, "
        "phi_ex, std, fbn, std, wbn, std, gamma, std\n"
    )
    with open(filename, 'w') as file:
        file.writelines(header)
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
            "{:.3e}".format(std_dev[5]) + ", " +
            "{:.3e}".format(phi_ex_fit) + ", " +
            "{:.3e}".format(std_dev[6]) + ", " +
            "{:.3e}".format(fbn_fit) + ", " +
            "{:.3e}".format(std_dev[7]) + ", " +
            "{:.3e}".format(wbn_fit) + ", " +
            "{:.3e}".format(std_dev[8]) + ", " +
            "{:.3e}".format(gamma_fit) + ", " +
            "{:.3e}".format(std_dev[9]) +
            "\n"
        )


def check_input_parameters(
    fix_phi, fix_m, fix_tau, fix_c1, fix_c2, fix_c3,
    fix_phi_ex, fix_fbn, fix_wbn, fix_gamma, min_Lorentz_f=0.
):
    # convert integer values to floats
    fix_phi = float(fix_phi) if type(fix_phi) is int else fix_phi
    fix_m = float(fix_m) if type(fix_m) is int else fix_m
    fix_tau = float(fix_tau) if type(fix_tau) is int else fix_tau
    fix_c1 = float(fix_c1) if type(fix_c1) is int else fix_c1
    fix_c2 = float(fix_c2) if type(fix_c2) is int else fix_c2
    fix_c3 = float(fix_c3) if type(fix_c3) is int else fix_c3
    fix_phi_ex = float(fix_phi_ex) if type(fix_phi_ex) is int else fix_phi_ex
    fix_fbn = float(fix_fbn) if type(fix_fbn) is int else fix_fbn
    fix_wbn = float(fix_wbn) if type(fix_wbn) is int else fix_wbn
    fix_gamma = float(fix_gamma) if type(fix_gamma) is int else fix_gamma
    min_Lorentz_f = float(min_Lorentz_f)

    # phi_ex and fbn cannot be fit simultaneously
    if not isinstance(fix_phi_ex, float) and not isinstance(fix_fbn, float):
        print("Error: phi_ex and fbn cannot both be fit simultaneously")
        sys.exit(1)

    # If no Lorentz part,
    if (isinstance(fix_phi_ex, float) and fix_phi_ex == 0.) or  \
       (isinstance(fix_fbn, float) and fix_fbn == 0.):
        # then phi and m cannot both be fit
        if not isinstance(fix_phi, float) and not isinstance(fix_m, float):
            print("Error: phi and m cannot both be fit simultaneously")
            sys.exit(1)

    # If no Drude-Smith part,
    if (isinstance(fix_phi, float) and fix_phi == 0.) or \
       (isinstance(fix_tau, float) and (fix_tau == 0.)):
        # then only one of phi_ex, fbn and m can be fit simultaneously
        if sum(
            not isinstance(var, float) for var in [fix_phi_ex, fix_fbn, fix_m]
        ) > 1:
            print("Error: Only one of phi_ex, fbn and m can be fit")
            sys.exit(1)

    num_variable_params = set_input_parameters(
        fix_phi, fix_m, fix_tau, fix_c1, fix_c2, fix_c3,
        fix_phi_ex, fix_fbn, fix_wbn, fix_gamma, min_Lorentz_f
    )

    if num_variable_params < 1 or num_variable_params > 9:
        print("Error: only 1, 2, 3, 4, 5, 6, 7, 8 or 9 variable parameters,")
        print("allowed, found", num_variable_params, "variable parameters")
        sys.exit(1)

    if fix_m is not False and fix_m == 0.:
        print("Error: m cannot be 0, "
              "to avoid division by zero in Drude-Smith formula")
        sys.exit(1)

    return num_variable_params


def guess_min_Lorentz_f(frequencies, real_values):
    # Try to guess where the Lorentz peak starts.
    # Find the peak in the real experimental values at the highest frequency
    # and guess frequency just before, converting it to THz
    peaks, _ = find_peaks(real_values)

    min_Lorentz_f = 0.
    if len(peaks) > 0 and peaks[-1] > 1:
        min_Lorentz_f = frequencies[peaks[-1]-2] / 1.E12
    return min_Lorentz_f


if __name__ == "__main__":
    global input_parameters

    input_filename = "mobility.csv"

    min_frequency = 0.3E12
    max_frequency = 2.2E12

    fix_phi = 1.
    fix_m = False
    fix_tau = False  # fix tau in fs
    fix_c1 = False
    fix_c2 = 0.
    fix_c3 = 0.
    fix_phi_ex = 0.
    fix_fbn = 0.
    fix_wbn = 0.  # fix wbn in THz
    fix_gamma = 0.  # fix gamma in THz
    min_Lorentz_f = -1.  # THz

    if len(sys.argv) > 1:
        input_filename = sys.argv[1]
        if input_filename == "-h" or input_filename == "--help":
            print_help()
            sys.exit(0)
        if len(sys.argv) > 2:
            min_frequency = float(sys.argv[2])
        if len(sys.argv) > 3:
            max_frequency = float(sys.argv[3])
        if len(sys.argv) > 4:
            if sys.argv[4] == "False":
                fix_phi = False
            else:
                fix_phi = float(sys.argv[4])
        if len(sys.argv) > 5:
            if sys.argv[5] == "False":
                fix_m = False
            else:
                fix_m = float(sys.argv[5])
        if len(sys.argv) > 6:
            if sys.argv[6] == "False":
                fix_tau = False
            else:
                fix_tau = float(sys.argv[6])
        if len(sys.argv) > 7:
            if sys.argv[7] == "False":
                fix_c1 = False
            else:
                fix_c1 = float(sys.argv[7])
        if len(sys.argv) > 8:
            if sys.argv[8] == "False":
                fix_c2 = False
            else:
                fix_c2 = float(sys.argv[8])
        if len(sys.argv) > 9:
            if sys.argv[9] == "False":
                fix_c3 = False
            else:
                fix_c3 = float(sys.argv[9])
        if len(sys.argv) > 10:
            if sys.argv[10] == "False":
                fix_phi_ex = False
            else:
                fix_phi_ex = float(sys.argv[10])
        if len(sys.argv) > 11:
            if sys.argv[11] == "False":
                fix_fbn = False
            else:
                fix_fbn = float(sys.argv[11])
        if len(sys.argv) > 12:
            if sys.argv[12] == "False":
                fix_wbn = False
            else:
                fix_wbn = float(sys.argv[12])
        if len(sys.argv) > 13:
            if sys.argv[13] == "False":
                fix_gamma = False
            else:
                fix_gamma = float(sys.argv[13])
        if len(sys.argv) > 14:
            min_Lorentz_f = float(sys.argv[14])

    if input_filename[-4] != ".":
        raise ValueError("Could not find file extension")
    basename = input_filename[:-4]

    image_filename = basename + '.png'
    param_filename = basename + '_param.csv'
    data_filename = basename + '_fitted.csv'

    frequencies, complex_numbers = read_csv(
        input_filename, min_frequency, max_frequency
    )
    if min_Lorentz_f == -1.:
        min_Lorentz_f = guess_min_Lorentz_f(frequencies, complex_numbers.real)

    num_variable_params = check_input_parameters(
        fix_phi, fix_m, fix_tau, fix_c1, fix_c2, fix_c3,
        fix_phi_ex, fix_fbn, fix_wbn, fix_gamma, min_Lorentz_f
    )

    fitted_complex_numbers, \
        [phi_fit, m_fit, tau_fit, c1_fit, c2_fit, c3_fit,
         phi_ex_fit, fbn_fit, wbn_fit, gamma_fit], \
        std_dev = perform_fit(
            frequencies, complex_numbers, num_variable_params
        )

    print_fit_results(
        phi_fit, m_fit, tau_fit, c1_fit, c2_fit, c3_fit,
        phi_ex_fit, fbn_fit, wbn_fit, gamma_fit, std_dev
    )

    title = ("phi = %.3e, m = %.3e, tau = %.3e,\n"
             "c1 = %.3e, c2 = %.3e, c3 = %.3e,\n"
             "phi_ex = %.3e, fbn = %.3e, wbn = %.3e, gamma = %.3e")
    plot_experimental_and_fitted_data(
        frequencies, complex_numbers, fitted_complex_numbers,
        title
        % (phi_fit, m_fit, tau_fit, c1_fit, c2_fit, c3_fit,
           phi_ex_fit, fbn_fit, wbn_fit, gamma_fit),
        image_filename
    )

    write_parameters(
        param_filename, phi_fit, m_fit, tau_fit,
        c1_fit, c2_fit, c3_fit,
        phi_ex_fit, fbn_fit, wbn_fit, gamma_fit, std_dev
    )

    write_csv(
        data_filename, frequencies, complex_numbers,
        fitted_complex_numbers
    )
