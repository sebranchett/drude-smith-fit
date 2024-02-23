import csv
import numpy as np
from data_plots import plot_frequency_vs_real_and_imaginary_parts
from scipy.optimize import curve_fit


def read_csv(filename, min_frequency, max_frequency):
    frequencies = []
    complex_numbers = []

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row

        for row in reader:
            if min_frequency <= float(row[0]) <= max_frequency:
                frequency = float(row[0])
                real_part = float(row[2])
                imaginary_part = float(row[1])
                complex_number = complex(real_part, imaginary_part)

                frequencies.append(frequency)
                complex_numbers.append(complex_number)

    return np.array(frequencies), np.array(complex_numbers)


def fit_function(frequencies, mstar, tau, c1, c2, c3):
    # Define your function here, using the complex argument
    e = 1.602176634 * 10 ** -19
    f1 = e * tau / mstar
    f2 = 1 / (1 - 1j * 2 * np.pi * frequencies * tau)
    f3 = 1 + (c1 / (1 - 1j * 2 * np.pi * frequencies * c2)) + \
             (c2 / (1 - 1j * 2 * np.pi * frequencies * c3) ** 2) + \
             (c3 / (1 - 1j * 2 * np.pi * frequencies * c3) ** 3)
    complex_argument = f1 * f2 * f3
    return complex_argument


def real_fit_function(frequencies, mstar, tau, c1, c2, c3):
    # Define the function to fit
    return np.real(fit_function(frequencies, mstar, tau, c1, c2, c3))


if __name__ == "__main__":
    filename = "mobility.csv"

    min_frequency = 0.3E12
    max_frequency = 2.2E12

    frequencies, complex_numbers = read_csv(
        filename, min_frequency, max_frequency
    )

    # plot_frequency_vs_real_and_imaginary_parts(
    #     frequencies, complex_numbers
    # )

    # Initial guess for the parameters
    mstar = 0.18 * 9.10938356 * 10E-31
    tau = 80 * 10E-15
    c1 = -0.82
    c2 = 0.0
    c3 = 0.0
    initial_guess = [mstar, tau, c1, c2, c3]

    # Perform the least squares fit
    params, _ = curve_fit(
        real_fit_function, frequencies, complex_numbers,
        p0=initial_guess
    )

    # Extract the fitted parameters
    mstar_fit, tau_fit, c1_fit, c2_fit, c3_fit = params

    # Use the fitted parameters to calculate the fitted complex numbers
    fitted_complex_numbers = fit_function(
        frequencies, mstar_fit, tau_fit, c1_fit, c2_fit, c3_fit
    )

    # Plot the fitted complex numbers
    plot_frequency_vs_real_and_imaginary_parts(
        frequencies, fitted_complex_numbers
    )

    print("Fitted parameters:", params)
