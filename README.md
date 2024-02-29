# Fit experimental data using Drude-Smith mobility model
Created with the help of GitHub Copilot.


See Figure 3 and Equation 3 in this paper: https://www.nature.com/articles/ncomms9195 also [available here](https://repository.tudelft.nl/islandora/object/uuid:f809d8dd-b180-4564-af78-17170851451a?collection=research).


## Things to note
* e = 1.602E-19 and m0 = 9.109E-31, so SI units (not cm)
* m_fit is $m/\phi$ where $m^* = m * m0$

## More things to note
* Fit doesn't work well for parameters of very different orders of magnitude, so 'omptimize' works with Femtoseconds and Drude-Smith works with seconds (there is a conversion)
* 'Omptimize' can only work with real numbers, so it works with an array containing first the real parts of the experimental points and then the imaginary parts (there is a conversion)
