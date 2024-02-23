import csv
import numpy as np
from data_plots import plot_complex_numbers
from data_plots import plot_frequency_vs_real_part
from data_plots import plot_frequency_vs_imaginary_part
from data_plots import plot_frequency_vs_magnitude


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


if __name__ == "__main__":
    filename = "mobility.csv"
    frequencies, complex_numbers = read_csv(filename)
    plot_complex_numbers(complex_numbers)
    plot_frequency_vs_real_part(frequencies, complex_numbers)
    plot_frequency_vs_imaginary_part(frequencies, complex_numbers)
    plot_frequency_vs_magnitude(frequencies, complex_numbers)
