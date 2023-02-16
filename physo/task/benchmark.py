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


def sanity_check (X, y, run_config, target_program_str = None, expected_ideal_reward = 1):
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
                            )

    batch = batch_reseter()
    n_choices = batch.n_choices
    print(batch.library.lib_choosable_name_to_idx)
    print(batch)

    # --------------- Cell ---------------
    def cell_reseter ():
        input_size  = batch.obs_size
        output_size = batch.n_choices
        cell = rnn.Cell (input_size  = input_size,
                         output_size = output_size,
                         hidden_size = run_config["cell_config"]["hidden_size"],
                         n_layers    = run_config["cell_config"]["n_layers"])

        return cell
    cell = cell_reseter ()
    print("-------------------------- Cell ------------------------")
    print(cell)
    print("n_params= %i"%(cell.count_parameters()))

    # --------------- Ideal reward ---------------
    if target_program_str is not None:
        print("-------------------------- Ideal reward ------------------------")
        target_program = program.Program(tokens  = [batch.library.lib_name_to_token[name] for name in target_program_str],
                                         library = batch.library,
                                         is_physical = True,
                                         free_const_values=torch.tensor(batch.library.free_constants_init_val).to(batch.dataset.detected_device),
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
            target_program.optimize_constants(X         = batch.dataset.X,
                                              y_target  = batch.dataset.y_target,
                                              args_opti = run_config["free_const_opti_args"])

        ideal_reward = run_config["reward_config"]["reward_function"](
                                             y_pred = target_program(X),
                                             y_target = batch.dataset.y_target,
                                            ).cpu().detach().numpy()

        print("Ideal reward :", ideal_reward)

        eps = 1e-5
        assert (expected_ideal_reward - ideal_reward) <= 2*eps, 'Ideal reward should be >= %f'

        return target_program
