## What can PhySO do ?

$\Phi$-SO is a symbolic optimization package built for physics. Its symbolic regression (SR) module uses deep reinforcement learning to infer an analytical law from data points, searching in the space of functional forms by trial and error.
It is designed to be, as fast as technically possible, user-friendly and easy to install.

`physo` is unique in its capability to leverage :
- Physical units constraints : using the rules of dimensional analysis to constrain the search space. Ref paper: [[Tenachi 2023]](https://arxiv.org/abs/2303.03192)
- Class constraints : to infer a single analytical functional form that accurately fits multiple datasets, each governed by its own (possibly) unique set of fitting parameters Ref paper: [[Tenachi 2024]](https://arxiv.org/abs/2312.01816).

## Quick start guides

Quick start guide for __Symbolic Regression__ : [HERE](https://physo.readthedocs.io/en/latest/r_sr.html#getting-started-sr)  

Quick start guide for __Class Symbolic Regression__ : [HERE](https://physo.readthedocs.io/en/latest/r_class_sr.html#getting-started-class-sr)  
