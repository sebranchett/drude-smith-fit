import csv
import numpy as np
from data_plots import plot_frequency_vs_real_and_imaginary_parts
# from data_plots import plot_experimental_data
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


def plot_experimental_data(frequencies, complex_numbers):
    plt.scatter(
        frequencies,
        [complex_number.real for complex_number in complex_numbers],
        marker='.'
    )
    plt.scatter(
        frequencies,
        [complex_number.imag for complex_number in complex_numbers],
        marker='.'
    )
    plt.xlabel('Frequency')
    plt.ylabel('Real and Imaginary Parts')
    plt.title('Experimental Data')
    plt.savefig('experimental_data.png')
    plt.show()


if __name__ == "__main__":
    filename = "mobility.csv"

    min_frequency = 0.3E12
    max_frequency = 2.2E12

    frequencies, complex_numbers = read_csv(
        filename, min_frequency, max_frequency
    )

    plot_experimental_data(
        frequencies, complex_numbers
    )

    stretched_complex_numbers = np.concatenate(
        (np.real(complex_numbers), np.imag(complex_numbers))
    )

    # Initial guess for the parameters
    m = 0.18  # mstar = m * m0
    tau = .80 * 10E-15  # 80 fs = 80 * 10E-15 s
    c1 = -0.82
    initial_guess = [m, tau, c1]

    # Perform the least squares fit
    params, _ = curve_fit(
        fit_function, frequencies, stretched_complex_numbers,
        p0=initial_guess
    )

    # Extract the fitted parameters
    mstar_fit, tau_fit, c1_fit = params

    # Use the fitted parameters to calculate the fitted complex numbers
    fitted_complex_numbers = fit_function(
        frequencies, mstar_fit, tau_fit, c1_fit
    )

    # Plot the fitted complex numbers
    plot_frequency_vs_real_and_imaginary_parts(
        frequencies, fitted_complex_numbers
    )

    print("Fitted parameters:", params)
