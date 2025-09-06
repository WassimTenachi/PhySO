import matplotlib.pyplot as plt
import numpy as np

def soft_length_plot(run_config, do_show=True, save_path=None):
    """
    Plots the soft length prior config, float epsilons and max length limit from a run_config.
    Parameters
    ----------
    run_config : dict
        SR run configuration dictionary.
        Eg. physo.config.config0.config0
    do_show : bool, default True
        Whether to show the figure.
    save_path : bool, default None
        If not None, path to save the figure to.
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        The created figure and axes.
    """
    # Soft Length Prior plot
    for prior in run_config["priors_config"]:
        if prior[0] == "SoftLengthPrior":
            soft_loc = prior[1]["length_loc"]
            scale_loc = prior[1]["scale"]
    max_time_step = run_config["learning_config"]['max_time_step']

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(1, max_time_step + 10, 0.1)
    color_prior = "grey"
    ax.plot(x, np.exp(-((x - soft_loc) / scale_loc) ** 2), '-', color=color_prior, lw=1, label='Length Prior')
    # Max length
    ax.axvline(x=max_time_step, color='k', linestyle='-', label=r'Max. Length')
    ax.axvline(x=soft_loc, color=color_prior, linestyle='--', label=r'${\mu}$')
    # Float limits
    ax.axhline(y=np.finfo(np.float32).eps, color='#C00028', linestyle='-', label=r'${\epsilon}_{f32}$')
    ax.axhline(y=np.finfo(np.float64).eps, color='#C00028', linestyle='-', label=r'${\epsilon}_{f64}$')
    ax.legend(loc='best')
    ax.set_yscale('log')
    # Grid log
    ax.grid(True, which='both', ls='--', alpha=0.7)
    ax.set_xlabel("Length")
    ax.set_ylabel("Prior Probability")
    ax.set_title(r"Soft Length Prior ($\mu$=%i, $\sigma^2$=%i)" % (soft_loc, scale_loc))
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if do_show:
        plt.show()
    return fig, ax