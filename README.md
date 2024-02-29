# Fit experimental data using Drude-Smith mobility model
Created with the help of GitHub Copilot.

## Things to note
* e = 1.602E-19 and m0 = 9.109E-31, so SI units (not cm)
* m_fit is $m/\phi$ where $m^* = m * m0$

## More things to note
* Fit doesn't work well for parameters of very different orders of magnitude, so 'omptimize' works with Femtoseconds and Drude-Smith works with seconds (there is a conversion)
* 'Omptimize' can only work with real numbers, so it works with an array containing first the real parts of the experimental points and then the imaginary parts (there is a conversion)
