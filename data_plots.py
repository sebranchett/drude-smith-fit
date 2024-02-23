import matplotlib.pyplot as plt


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


def plot_frequency_vs_real_and_imaginary_parts(frequencies, complex_numbers):
    plt.plot(
        frequencies,
        [complex_number.real for complex_number in complex_numbers],
        label='Real Part'
    )
    plt.plot(
        frequencies,
        [complex_number.imag for complex_number in complex_numbers],
        label='Imaginary Part'
    )
    plt.xlabel('Frequency')
    plt.ylabel('Real and Imaginary Parts')
    plt.title('Frequency vs Real and Imaginary Parts')
    plt.legend()
    plt.savefig('frequency_vs_real_and_imaginary_parts.png')
    plt.show()
