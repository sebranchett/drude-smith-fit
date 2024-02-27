import numpy as np
# from data_plots import plot_frequency_vs_real_and_imaginary_parts
from matplotlib import pyplot as plt


def fit_function2(frequency, e, mstar, tau, c1):
    # Calculate the Drude-Smith mobility
    factor = e * tau / mstar
    omega = 2 * np.pi * frequency
    onePlusw2t2 = 1. + (omega * omega * tau * tau)
    oneMinusw2t2 = 1. - (omega * omega * tau * tau)

    result = onePlusw2t2 + c1 * oneMinusw2t2 + \
        1j * omega * tau * (onePlusw2t2 + 2. * c1)

    return factor * result / (onePlusw2t2*onePlusw2t2)


def fit_function3(frequencies, e, mstar, tau, c1, c2, c3):
    # Define your function here, using the complex argument
    f1 = e * tau / mstar
    f2 = 1 / (1 - 1j * 2 * np.pi * frequencies * tau)
    f3 = 1 + (c1 / (1 - 1j * 2 * np.pi * frequencies * tau)) + \
             (c2 / (1 - 1j * 2 * np.pi * frequencies * tau) ** 2) + \
             (c3 / (1 - 1j * 2 * np.pi * frequencies * tau) ** 3)
    complex_argument = f1 * f2 * f3
    return complex_argument


e = 1.602 * 10E-19
m0 = 9.109 * 10E-31
mstar = 0.18 * m0
tau = 80 * 10E-15
points = np.linspace(.1*10E10, .7*10E10, 21)
c1 = -0.82
# fit_result2 = fit_function2(points, e, mstar, tau, c1)
# plt.plot(points, fit_result2.real, label='real2')
# plt.plot(points, fit_result2.imag, label='imag2')
fit_result3 = fit_function3(points, e, mstar, tau, c1, 0., 0.)
plt.plot(points, fit_result3.real, label='real3')
plt.plot(points, fit_result3.imag, label='imag3')

plt.xlabel('Points')
plt.ylabel('mu_DS')
plt.legend()
plt.savefig('what_know.png')
plt.show()
