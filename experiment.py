import numpy as np
# from data_plots import plot_frequency_vs_real_and_imaginary_parts
import matplotlib.pyplot as plt


def fit_function(frequency):
    # Calculate hte Drude-Smith mobility
    # e = 1.602 * 10E-19
    m0 = 9.109 * 10E-31

    omega = 2 * np.pi * frequency
    mstar = 0.18 * m0
    tau = 1
    c1 = -0.82

    oneminiomegatau = 1. - (1j * omega * tau)
    # p1 = e * tau / mstar
    p2 = 1. / (1. - oneminiomegatau)
    p3 = 1. + c1/(1. - oneminiomegatau)

    return p2 * p3


points = np.linspace(0, 1*10E12, 20)
fit_result = fit_function(points)

# plot_frequency_vs_real_and_imaginary_parts(points, fit_result)

plt.plot(points, np.real(fit_result))
plt.xlabel('Frequency')
plt.ylabel('Real Part')
plt.title('Real Part of Fit Result')
plt.show()
