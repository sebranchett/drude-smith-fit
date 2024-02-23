import csv
import numpy as np
from data_plots import plot_frequency_vs_real_and_imaginary_parts


def read_csv(filename):
    frequencies = []
    complex_numbers = []

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row

        for row in reader:
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


if __name__ == "__main__":
    filename = "mobility.csv"
    frequencies, complex_numbers = read_csv(filename)

    # Perform the least squares fit SEB TODO

    # Use the fitted parameters to calculate the fitted complex numbers
    mstar = 0.18 * 9.109 * 10E31
    tau = 80 * 10E-15
    c1 = -0.82
    c2 = 0.0
    c3 = 0.0
    fitted_complex_numbers = fit_function(frequencies, mstar, tau, c1, c2, c3)

    # Plot the fitted complex numbers
    plot_frequency_vs_real_and_imaginary_parts(
        frequencies, fitted_complex_numbers
    )
