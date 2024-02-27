import numpy as np
# from data_plots import plot_frequency_vs_real_and_imaginary_parts
from matplotlib import pyplot as plt


def fit_function(frequency, e, m0, mstar, tau, c1):
    # Calculate the Drude-Smith mobility
    factor = e * tau / mstar
    print("factor", factor)
    omega = 2 * np.pi * frequency

    onebyonePlusw2t2 = 1. / (1. + (omega * omega * tau * tau))
    print("onebyonePlusw2t2", onebyonePlusw2t2[0], onebyonePlusw2t2[-1])

    result = (1 - c1) + 2. * c1 * onebyonePlusw2t2 + \
        1j * omega * tau * (1. + 2. * c1 * onebyonePlusw2t2)
    print("result", result[0], result[-1])

    return factor * onebyonePlusw2t2 * result


e = 1.  # 1.602 * 10E-19
m0 = 1.  # 9.109 * 10E-31
mstar = 1.  # 0.18 * m0
tau = 1.  # 80 * 10E-15
points = np.linspace(0.0, 1, 10)
c1 = -1.  # -0.82
fit_result = fit_function(points, e, m0, mstar, tau, c1)
plt.plot(points, fit_result.real, label='real')
plt.plot(points, fit_result.imag, label='imag')

plt.xlabel('Points')
plt.ylabel('mu_DS')
plt.legend()
plt.savefig('what_know.png')
plt.show()
