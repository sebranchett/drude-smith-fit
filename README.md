# Fit experimental data using Drude-Smith mobility model and Lorentz oscillator exciton response
Created with the help of GitHub Copilot.


For Drude-Smith mobility model, see Figure 3 and Equation 3 in this paper: https://www.nature.com/articles/ncomms9195 also [available here](https://repository.tudelft.nl/islandora/object/uuid:f809d8dd-b180-4564-af78-17170851451a?collection=research).

For exciton response, See Equations 2 and 4 in this paper: [Nanoscale, 2019,11, 21569-21576](https://pubs.rsc.org/en/content/articlelanding/2019/nr/c9nr07927k) also [available here](https://repository.tudelft.nl/islandora/object/uuid:68d763e3-f3f2-40fd-ac0c-36b9749d321a?collection=research).


## Fit Equation
$Signal [cm^2 V^{-1} s^{-1}] =$

$\phi \cdot \frac{{e \tau}}{{m^*}} \cdot 
\frac{{1}}{{1 - i \omega \tau}} \cdot
(1 + \frac{{c1}}{{1 - i \omega \tau}} + \frac{{c2}}{{(1 - i \omega \tau)^2}} + \frac{{c3}}{{(1 - i \omega \tau)^3}}) +$

$\phi_{EX} \cdot \frac{{e}}{{i m^*}} \cdot
\frac{{f_{bn} \omega}}{{\omega_{bn}^2 - \omega^2 - i \omega \Gamma}}$

where $\omega = 2 \pi \cdot frequency$.

## Things to note
* Experimental input data must be in $cm^2$ $V^{-1}$ $s^{-1}$ (not SI units)
* m_fit is $m$ where $m^* = m * m0$


## More things to note
* Fit doesn't work well for parameters of very different orders of magnitude, so 'optimize' works with Femtoseconds and Terahertz. Functions `drude_smith_c3` and `perform_fit` handle the conversions
* 'Optimize' can only work with real numbers, so it is fed an array containing first the real parts of the experimental points and then the imaginary parts (there is a conversion)
* There are 10 possible model parameters which can be fit: `phi`, `m`, `tau`, `c1`, `c2` and `c3` for Drude-Smith, and `phi_ex`, `fbn`, `wbn` and `gamma` for Lorentz oscillator exciton response. If you want to fix a parameter, set `fix_<parameter>` to a real value. For example, if you do not wish to model exciton response, set `fix_fbn = 0.`. If you wish to fit a parameter, then do not fix it. You can do this by setting `fix_<parameter> = False`.
* Some combinations of parameters cannot be fit. For example `phi_ex` and `fbn` cannot both be fit. However, if you fix one of them, you can fit the other. There are warnings in place.
