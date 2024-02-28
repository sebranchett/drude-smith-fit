import numpy as np
# from data_plots import plot_frequency_vs_real_and_imaginary_parts
from matplotlib import pyplot as plt


def drude_smith_c1(frequency, m, tau, c1):
    # Calculate the Drude-Smith mobility with only 1 c coefficient
    e = 1.602 * 10E-19
    m0 = 9.109 * 10E-31

    mstar = m * m0
    factor = e * tau / mstar
    omega = 2 * np.pi * frequency
    onePlusw2t2 = 1. + (omega * omega * tau * tau)
    oneMinusw2t2 = 1. - (omega * omega * tau * tau)

    result = onePlusw2t2 + c1 * oneMinusw2t2 + \
        1j * omega * tau * (onePlusw2t2 + 2. * c1)

    return factor * result / (onePlusw2t2*onePlusw2t2)


def drude_smith_c3(frequencies, m, tau, c1, c2=0., c3=0.):
    # Calculate the Drude-Smith mobility with 3 c coefficients
    e = 1.602 * 10E-19
    m0 = 9.109 * 10E-31

    mstar = m * m0
    f1 = e * tau / mstar
    f2 = 1 / (1 - 1j * 2 * np.pi * frequencies * tau)
    f3 = 1 + (c1 / (1 - 1j * 2 * np.pi * frequencies * tau)) + \
             (c2 / (1 - 1j * 2 * np.pi * frequencies * tau) ** 2) + \
             (c3 / (1 - 1j * 2 * np.pi * frequencies * tau) ** 3)
    complex_argument = f1 * f2 * f3
    return complex_argument


m = 0.18  # mstar = m * m0
tau = .80 * 10E-15  # 80 fs = 80 * 10E-15 s
points = np.linspace(.1*10E12, .7*10E12, 21)
c1 = -0.82
fit_result3 = drude_smith_c3(points, m, tau, c1)
plt.plot(points, fit_result3.real, label='real3')
plt.plot(points, fit_result3.imag, label='imag3')

plt.xlabel('Points')
plt.ylabel('mu_DS')
plt.legend()
plt.savefig('what_know.png')
plt.show()
