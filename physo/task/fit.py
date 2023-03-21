
# Internal imports
from physo.physym import batch as Batch
from physo.learn import rnn
from physo.learn import learn


def fit(X, y, run_config, candidate_wrapper = None, stop_reward = 1., stop_after_n_epochs = 1):
    """
    Run a symbolic regression task on (X,y) data.
    Parameters
    ----------
    X : torch.tensor of shape (n_dim, ?,) of float
        Values of the input variables of the problem with n_dim = nb of input variables.
    y_target : torch.tensor of shape (?,) of float
        Values of the target symbolic function on input variables contained in X_target.
    run_config : dict
        Run configuration, see config.default_run_config.sr_run_config for an example.
    candidate_wrapper : callable or None, optional
        Wrapper to apply to candidate program's output, candidate_wrapper taking func, X as arguments where func is
        a candidate program callable (taking X as arg). By default = None, no wrapper is applied (identity).
    stop_reward : float, optional
        Early stops if stop_reward is reached by a program (= 1 by default), use stop_reward = (1-1e-5) when using free
        constants.
    stop_after_n_epochs : int, optional
        Number of additional epochs to do after early stop condition is reached.
    Returns
    -------
    hall_of_fame_R, hall_of_fame : list of float, list of physym.program.Program
        hall_of_fame : history of overall best programs found.
        hall_of_fame_R : Corresponding reward values.
        Use hall_of_fame[-1] to access best model found.
    """
     #todo: assert MAX_TIME_STEP>= max_length
     #todo: if run_config is None, make const
     #todo: replace stop_reward by stop_rmse
     #todo: no plot visualiser by default, text only
     #todo: check risk_factor, gamma_decay, entropy_weight

    def batch_reseter():
        return  Batch.Batch (library_args          = run_config["library_config"],
                             priors_config         = run_config["priors_config"],
                             batch_size            = run_config["learning_config"]["batch_size"],
                             max_time_step         = run_config["learning_config"]["max_time_step"],
                             rewards_computer      = run_config["learning_config"]["rewards_computer"],
                             free_const_opti_args  = run_config["free_const_opti_args"],
                             X        = X,
                             y_target = y,
                             candidate_wrapper = candidate_wrapper,
                             observe_units     = run_config["learning_config"]["observe_units"],
                             )

    batch = batch_reseter()

    def cell_reseter ():
        input_size  = batch.obs_size
        output_size = batch.n_choices
        cell = rnn.Cell (input_size  = input_size,
                         output_size = output_size,
                         **run_config["cell_config"],
                        )

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
                                                    stop_after_n_epochs = stop_after_n_epochs,
                                                    run_logger          = run_config["run_logger"],
                                                    run_visualiser      = run_config["run_visualiser"],
                                                   )

    return hall_of_fame_R, hall_of_fame