import torch
import numpy as np
import time

# Internal imports
from . import loss

def learner ( model,
             optimizer,
             n_epochs,
             batch_reseter,
             risk_factor,
             gamma_decay,
             entropy_weight,
             verbose = True,
             stop_reward = 1.,
             stop_after_n_epochs = 50,
             max_n_evaluations   = None,
             run_logger     = None,
             run_visualiser = None,
            ):
    """
    Trains model to generate symbolic programs satisfying a reward by reinforcing on best candidates at each epoch.
    Parameters
    ----------
    model : torch.nn.Module
        Differentiable RNN cell.
    optimizer : torch.optim
        Optimizer to use.
    n_epochs : int
        Number of epochs.
    batch_reseter : callable
        Function returning a new empty physym.batch.Batch.
    risk_factor : float
        Fraction between 0 and 1 of elite programs to reinforce on.
    gamma_decay : float
        Weight of power law to use along program length: gamma_decay**t where t is the step in the sequence in the loss
        function (gamma_decay < 1 gives more important to first tokens and gamma_decay > 1 gives more weight to last
        tokens).
    entropy_weight : float
        Weight to give to entropy part of the loss.
    verbose : int, optional
        If verbose = False or 0, print nothing, if True or 1, prints learning time, if > 1 print epochs progression.
    stop_reward : float, optional
        Early stops if stop_reward is reached by a program (= 1 by default), use stop_reward = (1-1e-5) when using free
        constants.
    stop_after_n_epochs : int, optional
        Number of additional epochs to do after early stop condition is reached.
    max_n_evaluations : int or None, optional
        Maximum number of unique expression evaluations allowed (for benchmarking purposes). Immediately terminates
        the symbolic regression task if the limit is about to be reached. The parameter max_n_evaluations is distinct
        from batch_size * n_epochs because batch_size * n_epochs sets the number of expressions generated but a lot of
        these are not evaluated because they have inconsistent units.
    run_logger : object or None, optional
        Custom run logger to use having a run_logger.log method taking as args (epoch, batch, model, rewards, keep,
        notkept, loss_val).
    run_visualiser : object or None, optional
        Custom run visualiser to use having a run_visualiser.visualise method taking as args (run_logger, batch).
    Returns
    -------
    hall_of_fame_R, hall_of_fame : list of float, list of physym.program.Program
        hall_of_fame : history of overall best programs found.
        hall_of_fame_R : Corresponding reward values.
        Use hall_of_fame[-1] to access best model found.
    """
    t000 = time.perf_counter()

    # Basic logs
    overall_max_R_history = []
    hall_of_fame          = []
    # Nb. of expressions evaluated
    n_evaluated           = 0

    for epoch in range (n_epochs):

        if verbose>1: print("Epoch %i/%i"%(epoch, n_epochs))

        # -------------------------------------------------
        # --------------------- INIT  ---------------------
        # -------------------------------------------------

        # Reset new batch (embedding reset)
        batch = batch_reseter()
        batch_size    = batch.batch_size
        max_time_step = batch.max_time_step

        # Initial RNN cell input
        states = model.get_zeros_initial_state(batch_size)  # (n_layers, 2, batch_size, hidden_size)

        # Optimizer reset
        optimizer.zero_grad()

        # Candidates
        logits        = []
        actions       = []

        # Number of elite candidates to keep
        n_keep = int(risk_factor*batch_size)

        # -------------------------------------------------
        # -------------------- RNN RUN  -------------------
        # -------------------------------------------------

        # RNN run
        for i in range (max_time_step):

            # ------------ OBSERVATIONS ------------
            # (embedding output)
            observations = torch.tensor(batch.get_obs().astype(np.float32), requires_grad=False,) # (batch_size, obs_size)

            # ------------ MODEL ------------

            # Giving up-to-date observations
            output, states = model(input_tensor = observations,    # (batch_size, output_size), (n_layers, 2, batch_size, hidden_size)
                                            states = states      )

            # Getting raw prob distribution for action n°i
            outlogit = output                                         # (batch_size, output_size)

            # ------------ PRIOR ------------

            # (embedding output)
            prior_array = batch.prior().astype(np.float32)         # (batch_size, output_size)

            # 0 protection so there is always something to sample
            epsilon = 0 #1e-14 #1e0*np.finfo(np.float32).eps
            prior_array[prior_array==0] = epsilon

            # To log
            prior    = torch.tensor(prior_array, requires_grad=False) # (batch_size, output_size)
            logprior = torch.log(prior)                               # (batch_size, output_size)

            # ------------ SAMPLING ------------

            logit  = outlogit + logprior                              # (batch_size, output_size)
            action = torch.multinomial(torch.exp(logit),              # (batch_size,)
                                       num_samples=1)[:, 0]

            # ------------ ACTION ------------

            # Saving action n°i
            logits       .append(logit)
            actions      .append(action)

            # Informing embedding of new action
            # (embedding input)
            batch.programs.append(action.detach().cpu().numpy())

        # -------------------------------------------------
        # ------------------ CANDIDATES  ------------------
        # -------------------------------------------------

        # Keeping prob distribution history for backpropagation
        logits         = torch.stack(logits        , dim=0)         # (max_time_step, batch_size, n_choices, )
        actions        = torch.stack(actions       , dim=0)         # (max_time_step, batch_size,)

        # Programs as numpy array for black box reward computation
        actions_array  = actions.detach().cpu().numpy()             # (max_time_step, batch_size,)

        # -------------------------------------------------
        # -------------------- REWARD ---------------------
        # -------------------------------------------------

        # (embedding output)
        R = batch.get_rewards()

        # -------------------------------------------------
        # ---------------- BEST CANDIDATES ----------------
        # -------------------------------------------------

        # index of elite candidates
        # copy to avoid negative stride problem
        # https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/7
        keep    = R.argsort()[::-1][0:n_keep].copy()                              # (n_keep,)
        notkept = R.argsort()[::-1][n_keep: ].copy()                              # (batch_size-n_keep,)

        # ----------------- Train batch : black box part (NUMPY) -----------------

        # Elite candidates
        actions_array_train     = actions_array [:, keep]                         # (max_time_step, n_keep,)
        # Elite candidates as one-hot target probs
        ideal_probs_array_train = np.eye(batch.n_choices)[actions_array_train]    # (max_time_step, n_keep, n_choices,)

        # Elite candidates rewards
        R_train = torch.tensor(R[keep], requires_grad=False)                      # (n_keep,)
        R_lim   = R_train.min()

        # Elite candidates as one-hot in torch
        # (non-differentiable tensors)
        ideal_probs_train = torch.tensor(                                         # (max_time_step, n_keep, n_choices,)
                                ideal_probs_array_train.astype(np.float32),
                                requires_grad=False,)

        # -------------- Train batch : differentiable part (TORCH) ---------------
        # Elite candidates pred logprobs
        logits_train            = logits[:, keep]                                 # (max_time_step, n_keep, n_choices,)

        # -------------------------------------------------
        # ---------------------- LOSS ---------------------
        # -------------------------------------------------

        # Lengths of programs
        lengths = batch.programs.n_lengths[keep]                                  # (n_keep,)

        # Reward baseline
        #baseline = RISK_FACTOR - 1
        baseline = R_lim

        # Loss
        loss_val = loss.loss_func (logits_train      = logits_train,
                                  ideal_probs_train = ideal_probs_train,
                                  R_train           = R_train,
                                  baseline          = baseline,
                                  lengths           = lengths,
                                  gamma_decay       = gamma_decay,
                                  entropy_weight    = entropy_weight, )

        # -------------------------------------------------
        # ---------------- BACKPROPAGATION ----------------
        # -------------------------------------------------
        # No need to do backpropagation if model is lobotomized (ie. is just a random number generator).
        if model.is_lobotomized:
            pass
        else:
            loss_val  .backward()
            optimizer .step()

        # -------------------------------------------------
        # ----------------- LOGGING VALUES ----------------
        # -------------------------------------------------

        # Basic logging (necessary for early stopper)
        if epoch == 0:
            overall_max_R_history       = [R.max()]
            hall_of_fame                = [batch.programs.get_prog(R.argmax())]
        if epoch> 0:
            if R.max() > np.max(overall_max_R_history):
                overall_max_R_history.append(R.max())
                hall_of_fame.append(batch.programs.get_prog(R.argmax()))
            else:
                overall_max_R_history.append(overall_max_R_history[-1])

        # Custom logging
        if run_logger is not None:
            run_logger.log(epoch    = epoch,
                           batch    = batch,
                           model    = model,
                           rewards  = R,
                           keep     = keep,
                           notkept  = notkept,
                           loss_val = loss_val)

        # -------------------------------------------------
        # ----------------- VISUALISATION -----------------
        # -------------------------------------------------

        # Custom visualisation
        if run_visualiser is not None:
            run_visualiser.visualise(run_logger = run_logger, batch = batch)

        # -------------------------------------------------
        # ----------------- EARLY STOPPER -----------------
        # -------------------------------------------------
        early_stop_reward_eps = 2*np.finfo(np.float32).eps

        # If above stop_reward (+/- eps) stop after [stop_after_n_epochs] epochs.
        if (stop_reward - overall_max_R_history[-1]) <= early_stop_reward_eps:
            if stop_after_n_epochs == 0:
                try:
                    run_visualiser.save_visualisation()
                    run_visualiser.save_data()
                    run_visualiser.save_pareto_data()
                    run_visualiser.save_pareto_fig()
                except:
                    print("Unable to save last plots and data before early stopping.")
                break
            stop_after_n_epochs -= 1

        # -------------------------------------------------
        # ------------ MAX EVALUATIONS STOPPER ------------
        # -------------------------------------------------

        # Update nb. of evaluated programs
        n_evaluated += (R > 0.).sum()

        # If max_n_evaluations mode is used and we are one batch away from reaching the limit, stop.
        if (max_n_evaluations is not None) and (n_evaluated + batch_size > max_n_evaluations):
            try:
                run_visualiser.save_visualisation()
                run_visualiser.save_data()
                run_visualiser.save_pareto_data()
                run_visualiser.save_pareto_fig()
            except:
                print("Unable to save last plots and data before stopping due to max evaluation limit.")
            break

    t111 = time.perf_counter()
    if verbose:
        print("  -> Time = %f s"%(t111-t000))

    hall_of_fame_R = np.array(overall_max_R_history)
    return hall_of_fame_R, hall_of_fame