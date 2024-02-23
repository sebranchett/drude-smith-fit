import csv
import numpy as np
import matplotlib.pyplot as plt


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
    return frequencies, complex_numbers


def plot_complex_numbers(complex_numbers):
    real_parts = [complex_number.real for complex_number in complex_numbers]
    imaginary_parts = [
        complex_number.imag for complex_number in complex_numbers
    ]

    plt.scatter(real_parts, imaginary_parts)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Complex Numbers Plot')
    plt.savefig('complex_numbers_plot.png')
    plt.show()


def plot_frequency_vs_real_part(frequencies, complex_numbers):
    plt.plot(
        frequencies,
        [complex_number.real for complex_number in complex_numbers]
    )
    plt.xlabel('Frequency')
    plt.ylabel('Real Part')
    plt.title('Frequency vs Real Part')
    plt.savefig('frequency_vs_real_part.png')
    plt.show()


def plot_frequency_vs_imaginary_part(frequencies, complex_numbers):
    plt.plot(
        frequencies,
        [complex_number.imag for complex_number in complex_numbers]
    )
    plt.xlabel('Frequency')
    plt.ylabel('Imaginary Part')
    plt.title('Frequency vs Imaginary Part')
    plt.savefig('frequency_vs_imaginary_part.png')
    plt.show()


def plot_frequency_vs_magnitude(frequencies, complex_numbers):
    plt.plot(
        frequencies,
        [abs(complex_number) for complex_number in complex_numbers]
    )
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('Frequency vs Magnitude')
    plt.savefig('frequency_vs_magnitude.png')
    plt.show()


if __name__ == "__main__":
    filename = "mobility.csv"
    frequencies, complex_numbers = read_csv(filename)
    plot_complex_numbers(complex_numbers)
    plot_frequency_vs_real_part(frequencies, complex_numbers)
    plot_frequency_vs_imaginary_part(frequencies, complex_numbers)
    plot_frequency_vs_magnitude(frequencies, complex_numbers)
