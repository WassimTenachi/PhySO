
# Internal imports
from physo.physym import batch as Batch
from physo.learn import rnn
from physo.learn import learn


def fit(multi_X, multi_y, run_config, multi_y_weights = 1., candidate_wrapper = None, stop_reward = 1., stop_after_n_epochs = 1, max_n_evaluations = None):
    """
    Run a symbolic regression task on (X,y) data.
    Parameters
    ----------
    multi_X : list of len (n_realizations,) of torch.tensor of shape (n_dim, ?,) of float
            List of X (one per realization). With X being values of the input variables of the problem with n_dim = nb
            of input variables.
    multi_y : list of len (n_realizations,) of torch.tensor of shape (?,) of float
        List of y (one per realization). With y being values of the target symbolic function on input variables
        contained in X.
    multi_y_weights : list of len (n_realizations,) of torch.tensor of shape (?,) of float
                       or array_like of (n_realizations,) of float
                       or float, optional
        List of y_weights (one per realization). With y_weights being weights to apply to y data.
        Or list of weights one per entire realization.
        Or single float to apply to all (for default value = 1.).
        Weights for each data point. By default, no weights are used.
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
    max_n_evaluations : int or None, optional
        Maximum number of unique expression evaluations allowed (for benchmarking purposes). Immediately terminates
        the symbolic regression task if the limit is about to be reached. The parameter max_n_evaluations is distinct
        from batch_size * n_epochs because batch_size * n_epochs sets the number of expressions generated but a lot of
        these are not evaluated because they have inconsistent units.
    Returns
    -------
    hall_of_fame_R, hall_of_fame : list of float, list of physym.program.Program
        hall_of_fame : history of overall best programs found.
        hall_of_fame_R : Corresponding reward values.
        Use hall_of_fame[-1] to access best model found.
    """

    def batch_reseter():
        return  Batch.Batch (library_args          = run_config["library_config"],
                             priors_config         = run_config["priors_config"],
                             batch_size            = run_config["learning_config"]["batch_size"],
                             max_time_step         = run_config["learning_config"]["max_time_step"],
                             rewards_computer      = run_config["learning_config"]["rewards_computer"],
                             free_const_opti_args  = run_config["free_const_opti_args"],
                             multi_X         = multi_X,
                             multi_y         = multi_y,
                             multi_y_weights = multi_y_weights,
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
                                                    max_n_evaluations   = max_n_evaluations,
                                                    run_logger          = run_config["run_logger"],
                                                    run_visualiser      = run_config["run_visualiser"],
                                                   )

    return hall_of_fame_R, hall_of_fame