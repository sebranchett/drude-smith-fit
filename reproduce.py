import numpy as np
from data_plots import plot_frequency_vs_real_and_imaginary_parts


def fit_function(frequency):
    # Calculate hte Drude-Smith mobility
    e = 1.602 * 10E-19
    m0 = 9.109 * 10E-31

    omega = 2 * np.pi * frequency
    mstar = 0.18 * m0
    tau = 80 * 10E-15
    # c1 = -0.82
    c1 = -0.2

    oneminiomegatau = 1. - (1j * omega * tau)
    p1 = e * tau / mstar
    p2 = 1. / (1. - oneminiomegatau)
    p3 = 1. + c1/(1. - oneminiomegatau)

    print("Real part is zero for w^2 = ", (2. + 3*c1)/tau*tau*(c1 + 2))
    print("c1 must be greater than ", 2*(tau*tau-1)/(3-tau*tau))
    if c1 > 2*(tau*tau-1)/(5-tau*tau):
        print("omega has a root in real part at", np.sqrt((2. + 3*c1)/tau*tau*(c1 + 2)))
    print("Imaginary part is zero for w^2 = ",
          (6*tau*tau*(1+c1) + tau*tau*np.sqrt(36*(1+c1)*(1+c1) - 4*(1+2*c1))) /
          2*(1+2*c1))
    print("or w^2 = ",
          (6*tau*tau*(1+c1) - tau*tau*np.sqrt(36*(1+c1)*(1+c1) - 4*(1+2*c1))) /
          2*(1+2*c1))
    return p1 * p2 * p3


points = np.linspace(1, 2, 20)
fit_result = fit_function(points)

plot_frequency_vs_real_and_imaginary_parts(points, fit_result)
