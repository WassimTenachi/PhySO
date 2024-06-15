## Highlights

$\Phi$-SO's symbolic regression module uses deep reinforcement learning to infer analytical physical laws that fit data points, searching in the space of functional forms.  

`physo` is able to leverage:

* Physical units constraints, reducing the search space with dimensional analysis ([[Tenachi et al 2023]](https://arxiv.org/abs/2303.03192))

* Class constraints, searching for a single analytical functional form that accurately fits multiple datasets - each governed by its own (possibly) unique set of fitting parameters ([[Tenachi et al 2024]](https://arxiv.org/abs/2312.01816))

$\Phi$-SO recovering the equation for a damped harmonic oscillator:

https://github.com/WassimTenachi/PhySO/assets/63928316/655b0eea-70ba-4975-8a80-00553a6e2786

Performances on the standard Feynman benchmark from [SRBench](https://github.com/cavalab/srbench/tree/master)) comprising 120 expressions from the Feynman Lectures on Physics against popular SR packages.

$\Phi$-SO achieves state-of-the-art performance in the presence of noise (exceeding 0.1%) and shows robust performances even in the presence of substantial (10%) noise:

![feynman_results](https://github.com/WassimTenachi/PhySO/assets/63928316/bbb051a2-2737-40ca-bfbf-ed185c48aa71)

