Highlights
----------

$\Phi$-SO's symbolic regression module uses deep reinforcement learning to infer analytical physical laws that fit data points, searching in the space of functional forms.

PhySO is able to leverage:

* Physical units constraints, reducing the search space with dimensional analysis (`[Tenachi et al 2023] <https://arxiv.org/abs/2303.03192>`_)

* Class constraints, searching for a single analytical functional form that accurately fits multiple datasets - each governed by its own (possibly) unique set of fitting parameters (`[Tenachi et al 2024] <https://arxiv.org/abs/2312.01816>`_)


$\Phi$-SO recovering the equation for a damped harmonic oscillator:

.. raw:: html

    <div style="position: relative; width: 100%; overflow: hidden; padding-top: 56.25%;">
        <iframe src="https://www.youtube.com/embed/wubzZMkoTUY?autoplay=1&mute=1&loop=1&playlist=wubzZMkoTUY&controls=0" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>



Performances on the standard Feynman benchmark from `SRBench <https://github.com/cavalab/srbench/tree/master>`_ comprising 120 expressions from the Feynman Lectures on Physics against popular SR packages.

$\Phi$-SO achieves state-of-the-art performance in the presence of noise (exceeding 0.1%) and shows robust performances even in the presence of substantial (10%) noise:

.. image:: https://raw.githubusercontent.com/WassimTenachi/PhySO/main/docs/assets/feynman_results.gif
   :alt: StreamPlayer
   :align: center

