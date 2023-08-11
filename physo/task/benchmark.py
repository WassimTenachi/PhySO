import time

import torch
import numpy as np

import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from IPython.display import display, clear_output

# Internal imports
from physo.physym import batch as Batch
from physo.physym import program
from physo.learn import rnn
from physo.learn import learn



def dummy_epoch (X, y, run_config):
    # Batch reseter
    def batch_reseter():
        return Batch.Batch (library_args          = run_config["library_config"],
                            priors_config         = run_config["priors_config"],
                            batch_size            = run_config["learning_config"]["batch_size"],
                            max_time_step         = run_config["learning_config"]["max_time_step"],
                            rewards_computer      = run_config["learning_config"]["rewards_computer"],
                            free_const_opti_args  = run_config["free_const_opti_args"],
                            X        = X,
                            y_target = y,
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

    display(fig)

    return None


def sanity_check (X, y, run_config, candidate_wrapper = None, target_program_str = None, expected_ideal_reward = 1.):

    # --------------- Data ---------------
    print("Data")
    n_dim = X.shape[0]
    fig, ax = plt.subplots(n_dim, 1, figsize=(10,5))
    for i in range (n_dim):
        curr_ax = ax if n_dim==1 else ax[i]
        curr_ax.plot(X[i].detach().cpu().numpy(), y.detach().cpu().numpy(), 'k.',)
        curr_ax.set_xlabel("X[%i]"%(i))
        curr_ax.set_ylabel("y")
    plt.show()

    # --------------- Batch ---------------
    def batch_reseter():
        return Batch.Batch (library_args          = run_config["library_config"],
                            priors_config         = run_config["priors_config"],
                            batch_size            = run_config["learning_config"]["batch_size"],
                            max_time_step         = run_config["learning_config"]["max_time_step"],
                            rewards_computer      = run_config["learning_config"]["rewards_computer"],
                            free_const_opti_args  = run_config["free_const_opti_args"],
                            X        = X,
                            y_target = y,
                            candidate_wrapper = candidate_wrapper
                            )

    batch = batch_reseter()
    n_choices = batch.n_choices
    print(batch.library.lib_choosable_name_to_idx)
    print(batch)

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
        target_program = program.Program(tokens  = [batch.library.lib_name_to_token[name] for name in target_program_str],
                                         library = batch.library,
                                         is_physical = True,
                                         free_const_values=torch.tensor(batch.library.free_constants_init_val).to(batch.dataset.detected_device),
                                         candidate_wrapper=candidate_wrapper,
                                         is_opti=np.array([False]),
                                         opti_steps=np.array([0]),
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
            history = target_program.optimize_constants(X         = batch.dataset.X,
                                                        y_target  = batch.dataset.y_target,
                                                        args_opti = run_config["free_const_opti_args"],
                                                         )
            t1 = time.perf_counter()
            print("free const opti time = %f ms"%((t1-t0)*1e3))
            print("free constants found: %s"%(target_program.free_const_values))

            fig, ax = plt.subplots(1,)
            ax.plot(np.log10(history),)
            ax.axhline(np.log10(run_config["free_const_opti_args"]["method_args"]["tol"]), color='r')
            ax.axvline(run_config["free_const_opti_args"]["method_args"]["n_steps"]-1,     color='k')
            ax.set_title("Free const optimization history")
            ax.set_ylabel("log error")
            ax.set_xlabel("step")
            plt.show()

        ideal_reward = run_config["reward_config"]["reward_function"](
                                             y_pred = target_program(X),
                                             y_target = batch.dataset.y_target,
                                            ).detach().cpu().numpy()

        print("Ideal reward :", ideal_reward)
        # todo: assert that it is physical and compute reward through a batch

        eps = 1e-5
        assert (expected_ideal_reward - ideal_reward) <= 2*eps, 'Ideal reward should be >= %f +/- %f '% (expected_ideal_reward, eps)

        return target_program
