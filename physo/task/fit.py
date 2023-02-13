
# Internal imports
from physo.physym import batch
from physo.learn import rnn
from physo.learn import learn


def fit(X, y, run_config, stop_reward = 1.):
    """
    Run a symbolic regression task on (X,y) data.
    Parameters
    ----------
    X : torch.tensor of shape (n_dim, ?,) of float
        Values of the input variables of the problem with n_dim = nb of input variables.
    y_target : torch.tensor of shape (?,) of float
        Values of the target symbolic function on input variables contained in X_target.
    run_config : dict
        Run configuration, see default_run_config.sr_run_config for an example.
    stop_reward : float, optional
        Early stops if stop_reward is reached by a program (= 1 by default), use stop_reward = (1-1e-5) when using free
        constants.
    Returns
    -------
    hall_of_fame_R, hall_of_fame : list of float, list of physym.program.Program
        hall_of_fame : history of overall best programs found.
        hall_of_fame_R : Corresponding reward values.
        Use hall_of_fame[-1] to access best model found.
    """
     # todo: assert MAX_TIME_STEP>= max_length
     # todo: if run_config is None, make const
     # todo: replace stop_reward by stop_rmse
     # todo: no plot visualiser by default, text only
     # todo: check risk_factor, gamma_decay, entropy_weight

    def batch_reseter():
        return  batch.Batch (library_args          = run_config["library_config"],
                             priors_config         = run_config["priors_config"],
                             batch_size            = run_config["learning_config"]["batch_size"],
                             max_time_step         = run_config["learning_config"]["max_time_step"],
                             rewards_computer      = run_config["learning_config"]["rewards_computer"],
                             free_const_opti_args  = run_config["free_const_opti_args"],
                             X        = X,
                             y_target = y,
                             )

    batch = batch_reseter()

    def cell_reseter ():
        input_size  = batch.obs_size
        output_size = batch.n_choices
        cell = rnn.Cell (input_size  = input_size,
                         output_size = output_size,
                         hidden_size = run_config["cell_config"]["hidden_size"],
                         n_layers    = run_config["cell_config"]["n_layers"])

        return cell

    cell      = cell_reseter ()
    optimizer = run_config["learning_config"]["get_optimizer"](cell)


    hall_of_fame_R, hall_of_fame = learn.learner (
                                                    model               = cell,
                                                    optimizer           = optimizer,
                                                    n_epochs            = run_config["learning_config"]["n_epochs"],
                                                    batch_reseter       = batch_reseter,
                                                    risk_factor         = run_config["learning_config"]["risk_factor"],
                                                    gamma_decay         = run_config["learning_config"]["gamma_decay"],
                                                    entropy_weight      = run_config["learning_config"]["entropy_weight"],
                                                    verbose             = False,
                                                    stop_reward         = stop_reward,
                                                    stop_after_n_epochs = 50,
                                                    run_logger          = run_config["run_logger"],
                                                    run_visualiser      = run_config["run_visualiser"],
                                                   )

    return hall_of_fame_R, hall_of_fame