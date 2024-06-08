import time
import warnings
import torch
import numpy as np

import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Internal imports
from physo.physym import batch as Batch
from physo.physym import program
from physo.learn import rnn
from physo.learn import learn

def dummy_epoch_ClassSR (multi_X, multi_y, run_config, multi_y_weights=1.):
    """
    Dummy epoch for class SR task.
    Plots reward distribution and programs lengths distribution.

    Parameters
    ----------
    multi_X : list of len (n_realizations,) of np.array of shape (n_dim, ?,) of float
        List of X (one per realization). With X being values of the input variables of the problem with n_dim = nb
        of input variables.
    multi_y :  list of len (n_realizations,) of np.array of shape (?,) of float
        List of y (one per realization). With y being values of the target symbolic function on input variables
        contained in X.
    multi_y_weights : list of len (n_realizations,) of np.array of shape (?,) of float
                       or array_like of (n_realizations,) of float
                       or float, optional
        List of y_weights (one per realization). With y_weights being weights to apply to y data.
        Or list of weights one per entire realization.
        Or single float to apply to all (for default value = 1.).
    run_config : dict
        Run configuration.

    Returns
    -------
    rewards : np.array of shape (batch_size,)
        Rewards distribution.
    n_lengths : np.array of shape (batch_size,)
        Programs lengths distribution.
    """
    # Batch reseter
    def batch_reseter():
        return Batch.Batch (library_args          = run_config["library_config"],
                            priors_config         = run_config["priors_config"],
                            batch_size            = run_config["learning_config"]["batch_size"],
                            max_time_step         = run_config["learning_config"]["max_time_step"],
                            rewards_computer      = run_config["learning_config"]["rewards_computer"],
                            free_const_opti_args  = run_config["free_const_opti_args"],
                            multi_X         = multi_X,
                            multi_y         = multi_y,
                            multi_y_weights = multi_y_weights,
                            )

    batch = batch_reseter()

    fig, ax = plt.subplots(2, 1, figsize=(14, 7))
    ax0 = ax[0]

    for step in range (batch.max_time_step):
        # print("Programs:\n",batch.programs)

        # ---- Prior ----
        prior = torch.tensor(batch.prior().astype(np.float32))                                                # (batch_size, n_choices,)                                                                                 # (batch_size, obs_size,)

        # ---- Dummy cell output ----
        probs   = torch.tensor(np.random.rand(batch.batch_size, batch.library.n_choices).astype(np.float32)) # (batch_size, n_choices,)

        # Sampled actions
        actions = torch.multinomial(probs*prior, num_samples=1)[:, 0]                                              # (batch_size,)
        #print("Choosing actions:\n", batch.library.lib_tokens[actions])

        # # ---- Display ----
        # ax0.clear()
        # ax0.set_title("Programs lengths distribution at step = %i"%(step))
        # ax0.hist(batch.programs.n_lengths, bins=batch.max_time_step, range=(0, batch.max_time_step), color='k')
        # ax0.axvline(step, color='r', label="current step")
        # ax0.legend()
        #
        # display(fig)
        # clear_output(wait=True)
        # plt.pause(0.2)

        # ---- Appending actions ----
        batch.programs.append(actions)

    # ---- Display ----
    ax0.clear()
    ax0.set_title("Programs lengths distribution")
    ax0.hist(batch.programs.n_lengths, bins=batch.max_time_step, range=(0, batch.max_time_step), color='k')

    # ---- Embedding output (per epoch) ----
    # programs lengths
    n_lengths  = batch.programs.n_lengths
    # ---- Embedding output (per epoch) ----
    rewards = batch.get_rewards()

    # ---- Rewards distribution ----
    bins_dens = np.linspace(0., 1, int(batch.batch_size/10))
    kde = KernelDensity(kernel="gaussian", bandwidth=0.05,
                       ).fit(rewards[:, np.newaxis])
    dens = kde.score_samples(bins_dens[:, np.newaxis])
    # Plot
    ax1 = ax[1]
    ax1.set_title("Rewards distribution")
    ax1.plot(bins_dens, dens, alpha=1., linewidth=0.5, c='red')
    ax1.set_xlabel("reward")
    ax1.set_ylabel("logprobs")

    try:
        from IPython.display import display, clear_output
        display(fig)
    except:
        print("Unable to import IPython, showing plot using plt.show().")
        plt.show()

    return rewards, n_lengths

def dummy_epoch_SR (X, y, run_config, y_weights = 1.):
    """
    Dummy epoch for SR task.
    Plots reward distribution and programs lengths distribution.
    Parameters
    ----------
    X : numpy.array of shape (n_dim, ?,) of float
        Values of the input variables of the problem with n_dim = nb of input variables.
    y : numpy.array of shape (?,) of float
        Values of the target symbolic function to recover when applied on input variables contained in X.
    y_weights : np.array of shape (?,) of float
                or float, optional
        Weight values to apply to y data.
        Or single float to apply to all (for default value = 1.).

    run_config : dict
        Run configuration.

    Returns
    -------
    rewards : np.array of shape (batch_size,)
        Rewards distribution.
    n_lengths : np.array of shape (batch_size,)
        Programs lengths distribution.
    """

    res = dummy_epoch_ClassSR(multi_X         = [X, ],
                              multi_y         = [y, ],
                              multi_y_weights = [y_weights, ],
                              run_config      = run_config,
                        )
    return res


def sanity_check_ClassSR (multi_X, multi_y, run_config, multi_y_weights = 1., candidate_wrapper = None, target_program_str = None, expected_ideal_reward = 1.):
    """
    Checks if finding the target program would give the expected ideal reward.
    Parameters
    ----------
    multi_X : list of len (n_realizations,) of np.array of shape (n_dim, ?,) of float
        List of X (one per realization). With X being values of the input variables of the problem with n_dim = nb
        of input variables.
    multi_y :  list of len (n_realizations,) of np.array of shape (?,) of float
        List of y (one per realization). With y being values of the target symbolic function on input variables
        contained in X.
    multi_y_weights : list of len (n_realizations,) of np.array of shape (?,) of float
                       or array_like of (n_realizations,) of float
                       or float, optional
        List of y_weights (one per realization). With y_weights being weights to apply to y data.
        Or list of weights one per entire realization.
        Or single float to apply to all (for default value = 1.).
    run_config : dict
        Run configuration.
    candidate_wrapper : callable
        Wrapper to apply to candidate program's output, candidate_wrapper taking func, X as arguments where func is
        a candidate program callable (taking X as arg). By default = None, no wrapper is applied (identity).
    target_program_str : list of str, optional
        Polish notation of the target program.
    expected_ideal_reward : float, optional
        Expected ideal reward. By default = 1.
    Returns
    -------
    target_program : physo.physym.program.Program
        Target program.
    """

    # --------------- Batch ---------------
    def batch_reseter():
        return Batch.Batch (library_args          = run_config["library_config"],
                            priors_config         = run_config["priors_config"],
                            batch_size            = run_config["learning_config"]["batch_size"],
                            max_time_step         = run_config["learning_config"]["max_time_step"],
                            rewards_computer      = run_config["learning_config"]["rewards_computer"],
                            free_const_opti_args  = run_config["free_const_opti_args"],
                            multi_X         = multi_X,
                            multi_y         = multi_y,
                            multi_y_weights = multi_y_weights,

                            )

    batch = batch_reseter()
    n_choices = batch.n_choices
    print(batch.library.lib_choosable_name_to_idx)
    print(batch)

    # --------------- Data ---------------

    dataset = batch.dataset
    print("Data")
    n_realizations = dataset.n_realizations
    n_dim          = dataset.n_dim
    fig, ax = plt.subplots(n_realizations, n_dim, figsize=(10,5))
    if n_realizations == 1:
        ax = ax [np.newaxis, :]
    if n_dim == 1:
        ax = ax [:, np.newaxis]
    for i_real in range (n_realizations):
        for i_dim in range (n_dim):
            curr_ax = ax[i_real, i_dim]
            curr_ax.plot(dataset.multi_X[i_real][i_dim].detach().cpu().numpy(),
                         dataset.multi_y[i_real]       .detach().cpu().numpy(), 'k.',)
            curr_ax.set_xlabel("X[%i]"%(i_dim))
            curr_ax.set_ylabel("y")
    plt.show()

    # --------------- Learning config ---------------
    print("-------------------------- Learning config ------------------------")
    print(run_config["learning_config"])

    # --------------- Cell ---------------
    def cell_reseter ():
        input_size  = batch.obs_size
        output_size = batch.n_choices
        cell = rnn.Cell (input_size  = input_size,
                         output_size = output_size,
                         **run_config["cell_config"])

        return cell
    cell = cell_reseter ()
    print("-------------------------- Cell ------------------------")
    print(cell)
    print("n_params= %i"%(cell.count_parameters()))
    print("config:")
    print(run_config["cell_config"])

    # --------------- Reward config ---------------
    print("-------------------------- Reward config ------------------------")
    print(run_config["reward_config"])

    # --------------- Ideal reward ---------------
    if target_program_str is not None:
        print("-------------------------- Ideal reward ------------------------")
        # Ignoring warning that no free const table was provided and that a default one will be created with initial
        # values from library as this is what we want to do here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            target_program = program.Program(tokens  = [batch.library.lib_name_to_token[name] for name in target_program_str],
                                             library = batch.library,
                                             is_physical = True,
                                             candidate_wrapper = candidate_wrapper,
                                             n_realizations    = dataset.n_realizations,
                                                )

        print("---- Target ----")
        print("Tokens in prefix notation:")
        print(target_program)
        print("Raw expression ascii:")
        print(target_program.get_infix_pretty())

        print("Raw expression:")
        target_program.show_infix()
        print("Simplified expression:")
        target_program.show_infix(do_simplify=True, )

        # Computing ideal reward
        # optimize free const if necessary
        if batch.library.n_free_const > 0:
            t0 = time.perf_counter()
            history = target_program.optimize_constants(X         = batch.dataset.multi_X_flatten,
                                                        y_target  = batch.dataset.multi_y_flatten,
                                                        y_weights = batch.dataset.multi_y_weights_flatten,
                                                        n_samples_per_dataset = dataset.n_samples_per_dataset,
                                                        args_opti = run_config["free_const_opti_args"],
                                                         )
            t1 = time.perf_counter()
            print("free const opti time = %f ms"%((t1-t0)*1e3))
            print("class free constants found: %s"%(target_program.free_consts.class_values))
            print("spe   free constants found: %s"%(target_program.free_consts.spe_values  ))

            fig, ax = plt.subplots(1,)
            ax.plot(np.log10(history),)
            ax.axhline(np.log10(run_config["free_const_opti_args"]["method_args"]["tol"]), color='r')
            ax.axvline(run_config["free_const_opti_args"]["method_args"]["n_steps"]-1,     color='k')
            ax.set_title("Free const optimization history")
            ax.set_ylabel("log error")
            ax.set_xlabel("step")
            plt.show()

        ideal_reward = run_config["reward_config"]["reward_function"](
                                             y_pred    = target_program(batch.dataset.multi_X_flatten,
                                                                        n_samples_per_dataset = dataset.n_samples_per_dataset),
                                             y_target  = batch.dataset.multi_y_flatten,
                                             y_weights = batch.dataset.multi_y_weights_flatten,
        ).detach().cpu().numpy()

        print("Ideal reward :", ideal_reward)
        # todo: assert that it is physical and compute reward through a batch

        eps = 1e-5
        assert (expected_ideal_reward - ideal_reward) <= 2*eps, 'Ideal reward should be >= %f +/- %f '% (expected_ideal_reward, eps)

        return target_program

def sanity_check_SR (X, y, run_config, y_weights = 1., candidate_wrapper = None, target_program_str = None, expected_ideal_reward = 1.):
    """
    Checks if finding the target program would give the expected ideal reward.
    Parameters
    ----------
    X : numpy.array of shape (n_dim, ?,) of float
        Values of the input variables of the problem with n_dim = nb of input variables.
    y : numpy.array of shape (?,) of float
        Values of the target symbolic function to recover when applied on input variables contained in X.
    y_weights : np.array of shape (?,) of float
                or float, optional
        Weight values to apply to y data.
        Or single float to apply to all (for default value = 1.).
    run_config : dict
        Run configuration.
    candidate_wrapper : callable
        Wrapper to apply to candidate program's output, candidate_wrapper taking func, X as arguments where func is
        a candidate program callable (taking X as arg). By default = None, no wrapper is applied (identity).
    target_program_str : list of str, optional
        Polish notation of the target program.
    expected_ideal_reward : float, optional
        Expected ideal reward. By default = 1.
    Returns
    -------
    target_program : physo.physym.program.Program
        Target program.
    """
    target_program = sanity_check_ClassSR(multi_X         = [X, ],
                                          multi_y         = [y, ],
                                          multi_y_weights = [y_weights, ],
                                          run_config      = run_config,
                                          candidate_wrapper     = candidate_wrapper,
                                          target_program_str    = target_program_str,
                                          expected_ideal_reward = expected_ideal_reward,
                                          )
    return target_program
