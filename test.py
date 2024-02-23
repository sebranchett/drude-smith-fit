import numpy as np
from fit import fit_function
from data_plots import plot_frequency_vs_real_and_imaginary_parts

points = np.linspace(0.2E12, 0.6E12, 20)
mstar = 0.18 * 9.10938356 * 10E-31
tau = 80 * 10E-15
c1 = -0.82
c2 = 0.0
c3 = 0.0
fit_result = fit_function(points, mstar, tau, c1, c2, c3)

plot_frequency_vs_real_and_imaginary_parts(points, fit_result)
