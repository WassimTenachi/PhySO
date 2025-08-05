import warnings

import torch
import numpy as np
import pandas as pd
import time
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from sklearn.neighbors import KernelDensity

# Internal imports
from physo.physym import reward as reward_funcs
from physo.physym import program as Prog

using_tex = Prog.latex_display()

# Font size
plt.rc('font', size=16)

# Faster than searching for best loc
LEGEND_LOC = 'upper left' # "best"

try:
    import psutil
    pid = psutil.Process().pid
    psutil_available = True
except:
    psutil_available = False

def save_pareto_pkl (pareto_progs, fpath):
    """
    Save pareto programs to pickle file.
    Parameters
    ----------
    pareto_progs : list of Program.Program
        List of pareto programs.
    fpath : str
        Path to pkl file.
    """
    with open(fpath, 'wb') as f:
        pickle.dump(pareto_progs, f)
    return None

def read_pareto_pkl (fpath):
    """
    Load pareto programs from pickle file.
    Parameters
    ----------
    fpath : str
        Path to pkl file.
    Returns
    -------
    pareto_progs : list of Program.Program
        List of pareto programs.
    """
    with open(fpath, 'rb') as f:
        pareto_progs = pickle.load(f)
    return pareto_progs

class RunLogger:
    """
    Custom logger function.
    """

    def __init__ (self, save_path = None, do_save = False):
        self.save_path = save_path
        self.do_save   = do_save
        self.initialize()

    def initialize (self):

        # Epoch specific
        self.epoch = None

        self.overall_max_R_history         = []
        self.overall_best_prog_str_history = []
        self.hall_of_fame                  = []

        self.epochs_history               = []
        self.loss_history                 = []

        self.mean_R_train_history         = []
        self.mean_R_history               = []
        self.max_R_history                = []

        self.R_history                    = []
        self.R_history_train              = []

        self.best_prog_epoch_str_history  = []
        self.best_prog_complexity_history = []

        self.best_prog_epoch_str_prefix_history   = []
        self.overall_best_prog_str_prefix_history = []

        self.best_prog_epoch_free_const_history   = []
        self.overall_best_prog_free_const_history = []

        self.mean_complexity_history      = []

        self.n_physical                   = []
        self.n_rewarded                   = []
        self.lengths_of_physical          = []
        self.lengths_of_unphysical        = []

        if not psutil_available:
            warnings.warn(
                "psutil not available, RAM usage can not be displayed. You can install it with `pip install psutil`")

    def log(self, epoch, batch, model, rewards, keep, notkept, loss_val):

        # Epoch specific
        self.epoch   = epoch
        self.R       = rewards
        self.batch   = batch
        self.keep    = keep
        self.notkept = notkept
        best_prog_idx_epoch  = rewards.argmax()
        self.best_prog_epoch = batch.programs.get_prog(best_prog_idx_epoch, detach=True)
        self.programs_epoch  = batch.programs.get_programs_array(detach=True)


        if epoch == 0:
            self.free_const_names            = [tok.__str__() for tok in self.batch.library.free_constants_tokens]
            self.overall_max_R_history       = [rewards.max()]
            self.hall_of_fame                = [batch.programs.get_prog(best_prog_idx_epoch, detach=True)]
        if epoch> 0:
            if rewards.max() > np.max(self.overall_max_R_history):
                self.overall_max_R_history.append(rewards.max())
                self.hall_of_fame.append(batch.programs.get_prog(best_prog_idx_epoch, detach=True))
            else:
                self.overall_max_R_history.append(self.overall_max_R_history[-1])

        self.epochs_history         .append( epoch                             )
        self.loss_history           .append( loss_val.detach().cpu().numpy()   )

        self.mean_R_train_history   .append( rewards[keep].mean()              )
        self.mean_R_history         .append( rewards.mean()                    )
        self.max_R_history          .append( rewards.max()                     )


        self.R_history              .append( rewards                           )
        self.R_history_train        .append( rewards[keep]                     )

        self.best_prog_epoch_str_history    .append( self.best_prog_epoch .get_infix_str() )
        self.overall_best_prog_str_history  .append( self.best_prog       .get_infix_str() )

        self.best_prog_epoch_str_prefix_history    .append( self.best_prog_epoch .__str__() )
        self.overall_best_prog_str_prefix_history  .append( self.best_prog       .__str__() )

        # Logging free const as df
        self.best_prog_epoch_free_const_history   .append( self.best_prog_epoch.free_consts.cpu().detach().df() )
        self.overall_best_prog_free_const_history .append( self.best_prog      .free_consts.cpu().detach().df() )

        self.best_prog_complexity_history .append(batch.programs.tokens.complexity[best_prog_idx_epoch].sum())
        self.mean_complexity_history      .append(batch.programs.tokens.complexity.sum(axis=1).mean())

        self.R_history_array        = np.array(self.R_history)
        self.R_history_train_array  = np.array(self.R_history_train)

        self.n_physical              .append( batch.programs.is_physical.sum() )
        self.n_rewarded              .append( (rewards > 0.).sum()             )
        self.lengths_of_physical     .append( self.batch.programs.n_lengths[ self.batch.programs.is_physical] )
        self.lengths_of_unphysical   .append( self.batch.programs.n_lengths[~self.batch.programs.is_physical] )

        self.pareto_logger()

        # Saving log
        if self.do_save:
            self.save_log()

    def save_log (self):

        columns = ['epoch', 'reward', 'complexity', 'length', 'is_physical', 'is_elite', 'program', "program_prefix"]
        # Columns for free const names
        columns += self.batch.programs.free_consts.df().columns.to_list()

        # Initial df
        if self.epoch == 0:
            df0 = pd.DataFrame(columns=columns)
            df0.to_csv(self.save_path, index=False)

        # Current batch log
        is_elite = np.full(self.batch.batch_size, False)
        is_elite[self.keep] = True
        programs_str = np.array([prog.get_infix_str() for prog in self.batch.programs.get_programs_array(detach=True)])

        df = pd.DataFrame()
        df["epoch"]          = np.full(self.batch.batch_size, self.epoch)
        df["reward"]         = self.R
        df["complexity"]     = self.batch.programs.n_complexity
        df["length"]         = self.batch.programs.n_lengths
        df["is_physical"]    = self.batch.programs.is_physical
        df["is_elite"]       = is_elite
        df["program"]        = programs_str
        df["program_prefix"] = self.batch.programs.get_programs_array(detach=True)

        # Exporting free constants
        df = pd.concat([df, self.batch.programs.free_consts.df()], axis=1)

        # Saving current df
        df.to_csv(self.save_path, mode='a', index=False, header=False)

        return None

    def pareto_logger(self,):
        curr_complexities = self.batch.programs.n_complexity
        curr_rewards      = self.R
        curr_batch        = self.batch

        # Init
        if self.epoch == 0:
            self.pareto_complexities  = np.arange(0,10*curr_batch.max_time_step)
            self.pareto_rewards       = np.full(shape=(self.pareto_complexities.shape), fill_value = np.nan)
            self.pareto_programs      = np.full(shape=(self.pareto_complexities.shape), fill_value = None, dtype=object)

        # Update with current epoch info
        for i,c in enumerate(self.pareto_complexities):
            # Idx in batch of programs having complexity c
            arg_have_c = np.argwhere(curr_complexities.round() == c)
            if len(arg_have_c) > 0:
                # Idx in batch of the program having complexity c and having max reward
                arg_have_c_and_max = arg_have_c[curr_rewards[arg_have_c].argmax()]
                # Max reward of this program
                max_r_at_c = curr_rewards[arg_have_c_and_max]
                # If reward > currently max reward for this complexity or empty, replace
                if self.pareto_rewards[i] <= max_r_at_c or np.isnan(self.pareto_rewards[i]):
                    self.pareto_programs [i] = curr_batch.programs.get_prog(arg_have_c_and_max[0], detach=True)
                    self.pareto_rewards  [i] = max_r_at_c[0]

    def get_pareto_front(self,):
        # Postprocessing
        # Keeping only valid pareto candidates
        mask_pareto_valid = (~np.isnan(self.pareto_rewards)) & (self.pareto_rewards>0)
        pareto_rewards_valid      = self.pareto_rewards      [mask_pareto_valid]
        pareto_programs_valid     = self.pareto_programs     [mask_pareto_valid]
        pareto_complexities_valid = self.pareto_complexities [mask_pareto_valid]
        # Computing front
        pareto_front_r            = [pareto_rewards_valid       [0]]
        pareto_front_programs     = [pareto_programs_valid      [0]]
        pareto_front_complexities = [pareto_complexities_valid  [0]]
        for i,r in enumerate(pareto_rewards_valid):
            # Only keeping candidates with higher reward than candidates having a smaller complexity
            if r > pareto_front_r[-1]:
                pareto_front_r            .append(r)
                pareto_front_programs     .append(pareto_programs_valid     [i])
                pareto_front_complexities .append(pareto_complexities_valid [i])

        pareto_front_complexities = np.array(pareto_front_complexities)
        pareto_front_programs     = np.array(pareto_front_programs)
        pareto_front_r            = np.array(pareto_front_r)

        pareto_front_rmse         = ((1/pareto_front_r)-1)*self.batch.dataset.multi_y_flatten.std().detach().cpu().numpy()

        return pareto_front_complexities, pareto_front_programs, pareto_front_r, pareto_front_rmse

    @property
    def best_prog(self):
        return self.hall_of_fame[-1]


class RunVisualiser:
    """
    Custom run visualiser.
    """

    def __init__ (self,
                  epoch_refresh_rate = 10,
                  epoch_refresh_rate_prints = 1,
                  save_path = None,
                  do_show   = True,
                  do_prints = True,
                  do_save   = False,
                  draw_all_progs_fit = True
                  ):
        self.epoch_refresh_rate        = epoch_refresh_rate
        self.epoch_refresh_rate_prints = epoch_refresh_rate_prints
        self.figsize   = (40,18)
        self.save_path = save_path
        if save_path is not None:
            self.save_path_log        = ''.join(save_path.split('.')[:-1]) + "_data.csv"    # save_path with extension replaced by '_data.csv'
            self.save_path_pareto     = ''.join(save_path.split('.')[:-1]) + "_pareto.csv"  # save_path with extension replaced by '_pareto.csv'
            self.save_path_pareto_fig = ''.join(save_path.split('.')[:-1]) + "_pareto.pdf"  # save_path with extension replaced by '_pareto.pdf'
            self.save_path_pareto_pkl = ''.join(save_path.split('.')[:-1]) + "_pareto.pkl"  # save_path with extension replaced by '_pareto.pkl'
        self.do_show   = do_show
        self.do_save   = do_save
        self.do_prints = do_prints
        self.draw_all_progs_fit = draw_all_progs_fit

    def initialize (self):
        self.fig = plt.figure(figsize=self.figsize)
        gs  = gridspec.GridSpec(3, 3)
        self.ax0 = self.fig.add_subplot(gs[0, 0])
        self.ax1 = self.fig.add_subplot(gs[0, 1])
        div = make_axes_locatable(self.ax1)
        self.cax = div.append_axes("right", size="4%", pad=0.4)
        self.ax2 = self.fig.add_subplot(gs[1, 0])
        self.ax3 = self.fig.add_subplot(gs[1, 1])
        self.ax4 = self.fig.add_subplot(gs[:2, 2])
        # 3rd line
        self.ax5 = self.fig.add_subplot(gs[2, 0])
        self.ax6 = self.fig.add_subplot(gs[2, 1])
        div = make_axes_locatable(self.ax6)
        self.cax6 = div.append_axes("right", size="4%", pad=0.4)
        self.ax7 = self.fig.add_subplot(gs[2, 2])
        div = make_axes_locatable(self.ax7)
        self.cax7 = div.append_axes("right", size="4%", pad=0.4)

        self.t0 = time.perf_counter()

    def update_plot (self,):
        epoch      = self.run_logger.epoch
        run_logger = self.run_logger
        batch      = self.batch

        # -------- Reward vs epoch --------
        curr_ax = self.ax0
        curr_ax.clear()
        curr_ax.plot(run_logger.epochs_history, run_logger.mean_R_history        , 'b'            , linestyle='solid' , alpha = 0.6, label="Mean")
        curr_ax.plot(run_logger.epochs_history, run_logger.mean_R_train_history  , 'r'            , linestyle='solid' , alpha = 0.6, label="Mean train")
        curr_ax.plot(run_logger.epochs_history, run_logger.overall_max_R_history , 'k'            , linestyle='solid' , alpha = 1.0, label="Overall Best")
        curr_ax.plot(run_logger.epochs_history, run_logger.max_R_history         , color='orange' , linestyle='solid' , alpha = 0.6, label="Best of epoch")
        curr_ax.set_ylabel("Reward")
        curr_ax.set_xlabel("Epochs")
        curr_ax.legend(loc=LEGEND_LOC)

        # -------- Reward distrbution vs epoch --------
        curr_ax = self.ax1
        cmap = plt.get_cmap("viridis")
        fading_plot_nepochs       = epoch
        fading_plot_ncurves       = 20
        fading_plot_max_alpha     = 1.
        fading_plot_bins          = 100
        fading_plot_kde_bandwidth = 0.05
        curr_ax.clear()
        self.cax.clear()
        # Plotting last "fading_plot_nepochs" epoch on "fading_plot_ncurves" curves
        plot_epochs = []
        for i in range (fading_plot_ncurves+1):
            frac = i/fading_plot_ncurves
            plot_epoch = int(epoch - frac*fading_plot_nepochs)
            plot_epochs.append(plot_epoch)
            prog = 1 - frac
            alpha = fading_plot_max_alpha*(prog)
            # Histogram
            bins_dens = np.linspace(0., 1, fading_plot_bins)
            kde = KernelDensity(kernel="gaussian", bandwidth=fading_plot_kde_bandwidth
                               ).fit(run_logger.R_history_train_array[plot_epoch][:, np.newaxis])
            dens = 10**kde.score_samples(bins_dens[:, np.newaxis])
            # Plot
            curr_ax.plot(bins_dens, dens, alpha=alpha, linewidth=0.5, c=cmap(prog))
        # Colorbar
        normcmap = plt.matplotlib.colors.Normalize(vmin=plot_epochs[0], vmax=plot_epochs[-1])
        cbar = self.fig.colorbar(plt.cm.ScalarMappable(norm=normcmap, cmap=cmap), cax=self.cax, pad=0.005)
        cbar.set_label('epochs', rotation=90,labelpad=30)
        curr_ax.set_xlim([0, 1.])
        curr_ax.set_ylabel("Density")
        curr_ax.set_xlabel("Reward")

        # -------- Complexity --------
        curr_ax = self.ax2
        curr_ax.clear()
        curr_ax.plot(run_logger.epochs_history, run_logger.best_prog_complexity_history, 'k', linestyle='solid'   ,  label="Best of epoch")
        curr_ax.plot(run_logger.epochs_history, run_logger.mean_complexity_history     , 'k', linestyle='--'      ,  label="Mean")
        curr_ax.set_ylabel("Length")
        curr_ax.set_xlabel("Epochs")
        curr_ax.legend(loc=LEGEND_LOC)

        # -------- Loss --------
        curr_ax = self.ax3
        curr_ax.clear()
        curr_ax.plot(run_logger.epochs_history, run_logger.loss_history, 'grey', label="loss")
        curr_ax.set_ylabel("Loss")
        curr_ax.set_xlabel("Epochs")
        curr_ax.legend(loc=LEGEND_LOC)

        # -------- Fit --------
        curr_ax = self.ax4
        curr_ax.clear()
        # Cut on dim
        cut_on_dim = 0
        X_toplot        = batch.dataset.multi_X[0]
        y_target_toplot = batch.dataset.multi_y[0]
        x = X_toplot[cut_on_dim]
        # Plot data
        x_expand = 0.
        n_plot = 100
        stack = []
        for x_dim in X_toplot:
            x_dim_min = x_dim.min().detach().cpu().numpy()
            x_dim_max = x_dim.max().detach().cpu().numpy()
            x_dim_plot = torch.tensor(np.linspace(x_dim_min-x_expand, x_dim_max+x_expand, n_plot))
            stack.append(x_dim_plot)
        X_plot = torch.stack(stack).to(batch.dataset.detected_device)
        x_plot = X_plot[cut_on_dim]

        # Data points
        curr_ax.plot(x.detach().cpu().numpy(), y_target_toplot.detach().cpu().numpy(), 'ko', markersize=10)
        x_plot_cpu = x_plot.detach().cpu().numpy()

        # ------- Prog drawing -------
        unable_to_draw_a_prog = False

        # Other programs
        if self.draw_all_progs_fit:
            for prog in run_logger.programs_epoch[run_logger.notkept]:
                try:
                    y_plot =  prog(X_plot).detach().cpu().numpy()
                    if y_plot.shape == (): y_plot = np.full(n_plot, y_plot)
                    curr_ax.plot(x_plot_cpu, y_plot, color='b', alpha=0.05, linestyle='solid')
                except:
                    unable_to_draw_a_prog = True

        # Train programs
        for prog in run_logger.programs_epoch[run_logger.keep]:
            try:
                y_plot =  prog(X_plot).detach().cpu().numpy()
                if y_plot.shape == (): y_plot = np.full(n_plot, y_plot)
                curr_ax.plot(x_plot_cpu, y_plot, color='r', alpha=0.05, linestyle='solid')
            except:
                unable_to_draw_a_prog = True

        # Best program of epoch
        try:
            y_plot = run_logger.best_prog_epoch(X_plot).detach().cpu().numpy()
            if y_plot.shape == (): y_plot = np.full(n_plot, y_plot)
            curr_ax.plot(x_plot_cpu, y_plot, color='orange', linestyle='solid', linewidth=2)
        except:
            unable_to_draw_a_prog = True

        # Best overall program
        try:
            y_plot = run_logger.best_prog(X_plot).detach().cpu().numpy()
            if y_plot.shape == (): y_plot = np.full(n_plot, y_plot)
            curr_ax.plot(x_plot_cpu, y_plot, color='k', linestyle='solid', linewidth=2)
        except:
            unable_to_draw_a_prog = True

        if unable_to_draw_a_prog:
            print("Unable to draw one or more prog curve on monitoring plot.")

        # ------- Plot limits -------
        y_min = y_target_toplot.min().detach().cpu().numpy()
        y_max = y_target_toplot.max().detach().cpu().numpy()
        curr_ax.set_ylim(y_min-0.1*np.abs(y_min), y_max+0.1*np.abs(y_max))
        custom_lines = [
            Line2D([0], [0], color='k',      lw=3),
            Line2D([0], [0], color='orange', lw=3),
            Line2D([0], [0], color='r',      lw=3),
            Line2D([0], [0], color='b',      lw=3),]
        curr_ax.legend(custom_lines, ['Overall Best', 'Best of epoch', 'Train', 'Others'], loc=LEGEND_LOC)

        # -------- Number of physical progs --------
        curr_ax = self.ax5
        curr_ax.clear()
        curr_ax.plot(run_logger.epochs_history, run_logger.n_physical, 'red'   , label="Physical count")
        curr_ax.plot(run_logger.epochs_history, run_logger.n_rewarded, 'black' , label="Rewarded count")
        curr_ax.set_xlabel("Epochs")
        curr_ax.set_ylabel("Physical Count")
        curr_ax.legend(loc=LEGEND_LOC)

        # -------- Lengths of physical distribution vs epoch --------
        curr_ax  = self.ax6
        curr_cax = self.cax6

        cmap = plt.get_cmap("viridis")
        fading_plot_nepochs       = epoch
        fading_plot_ncurves       = 20
        fading_plot_max_alpha     = 1.
        fading_plot_bins          = 100
        fading_plot_kde_bandwidth = 1.
        curr_ax.clear()
        curr_cax.clear()
        # Plotting last "fading_plot_nepochs" epoch on "fading_plot_ncurves" curves
        plot_epochs = []
        for i in range (fading_plot_ncurves+1):
            frac = i/fading_plot_ncurves
            plot_epoch = int(epoch - frac*fading_plot_nepochs)
            plot_epochs.append(plot_epoch)
            prog = 1 - frac
            alpha = fading_plot_max_alpha*(prog)
            # Distribution data
            distrib_data = self.run_logger.lengths_of_physical[plot_epoch]
            # If non empty selection, compute pdf and plot it
            if distrib_data.shape[0] > 0:
                # Histogram
                bins_dens = np.linspace(0., self.run_logger.batch.max_time_step, fading_plot_bins)
                kde = KernelDensity(kernel="gaussian", bandwidth=fading_plot_kde_bandwidth
                                   ).fit(distrib_data[:, np.newaxis])
                dens = 10**kde.score_samples(bins_dens[:, np.newaxis])
                # Plot
                curr_ax.plot(bins_dens, dens, alpha=alpha, linewidth=0.5, c=cmap(prog))
        # Colorbar
        normcmap = plt.matplotlib.colors.Normalize(vmin=plot_epochs[0], vmax=plot_epochs[-1])
        cbar = self.fig.colorbar(plt.cm.ScalarMappable(norm=normcmap, cmap=cmap), cax=curr_cax, pad=0.005)
        cbar.set_label('epochs', rotation=90,labelpad=30)
        curr_ax.set_xlim([0, self.run_logger.batch.max_time_step])
        curr_ax.set_ylabel("Density")
        curr_ax.set_xlabel("Lengths (physical)")

        # -------- Lengths of unphysical distribution vs epoch --------
        curr_ax  = self.ax7
        curr_cax = self.cax7
        curr_fig = self.fig

        cmap = plt.get_cmap("viridis")
        fading_plot_nepochs       = epoch
        fading_plot_ncurves       = 20
        fading_plot_max_alpha     = 1.
        fading_plot_bins          = 100
        fading_plot_kde_bandwidth = 1.
        curr_ax.clear()
        curr_cax.clear()
        # Plotting last "fading_plot_nepochs" epoch on "fading_plot_ncurves" curves
        plot_epochs = []
        for i in range (fading_plot_ncurves+1):
            frac = i/fading_plot_ncurves
            plot_epoch = int(epoch - frac*fading_plot_nepochs)
            plot_epochs.append(plot_epoch)
            prog = 1 - frac
            alpha = fading_plot_max_alpha*(prog)
            # Distribution data
            distrib_data = self.run_logger.lengths_of_unphysical[plot_epoch]
            # If non empty selection, compute pdf and plot it
            if distrib_data.shape[0] > 0:
                # Histogram
                bins_dens = np.linspace(0., self.run_logger.batch.max_time_step, fading_plot_bins)
                kde = KernelDensity(kernel="gaussian", bandwidth=fading_plot_kde_bandwidth
                                   ).fit(distrib_data[:, np.newaxis])
                dens = 10**kde.score_samples(bins_dens[:, np.newaxis])
                # Plot
                curr_ax.plot(bins_dens, dens, alpha=alpha, linewidth=0.5, c=cmap(prog))
        # Colorbar
        normcmap = plt.matplotlib.colors.Normalize(vmin=plot_epochs[0], vmax=plot_epochs[-1])
        cbar = curr_fig.colorbar(plt.cm.ScalarMappable(norm=normcmap, cmap=cmap), cax=curr_cax, pad=0.005)
        cbar.set_label('epochs', rotation=90,labelpad=30)
        curr_ax.set_xlim([0, self.batch.max_time_step])
        curr_ax.set_ylabel("Density")
        curr_ax.set_xlabel("Lengths (unphysical)")

    def make_prints(self):
        t1 = self.t0
        t2 = time.perf_counter()
        self.t0 = t2

        print("=========== Epoch %s ==========="%(str(self.run_logger.epoch).zfill(5)))
        print("-> Time %.2f s"%(t2-t1))
        if psutil_available:
            mem_info = psutil.Process(pid).memory_info()
            print(f"-> Memory Usage: {mem_info.rss / (1024 * 1024):.2f} MB")

        # Overall best
        print("\nOverall best  at R=%f"%(self.run_logger.overall_max_R_history[-1]))
        print("-> Raw expression : \n%s"%(self.run_logger.best_prog.get_infix_pretty(do_simplify=False, )))
        #print("  -> Simplified expression : \n%s"%(run_logger.best_prog.get_infix_pretty(do_simplify=True , )))

        # Best of epoch
        print("\nBest of epoch at R=%f"%(self.run_logger.R.max()))
        print("-> Raw expression : \n%s"%(self.run_logger.best_prog_epoch.get_infix_pretty(do_simplify=False, )))
        print("\n")
        #print("  -> Simplified expression : \n%s"%(run_logger.best_prog_epoch.get_infix_pretty(do_simplify=True , )))

        #print("************************************************* Best programs *************************************************")
        ## Batch status
        #print("\nBest programs")
        #for i in range(n_keep):
        #    print("  -> R = %f: \n%s"%(run_logger.R[keep][i], programs[keep][i].get_infix_pretty(do_simplify=False)))
        #    print("------------------------------------------------------------")
        #print("*****************************************************************************************************************")

    def make_visualisation (self):
        # -------- Plot update --------
        self.update_plot()
        # -------- Display --------
        try:
            from IPython.display import display, clear_output
            display(self.fig)
            clear_output(wait=True)
        except:
            print("Unable to import IPython, showing plot using plt.show().")
            plt.show()

    def get_curves_data_df (self):
        df = pd.DataFrame()
        # Reward vs epoch
        df["epoch"]         = self.run_logger.epochs_history
        df["mean_R"]        = self.run_logger.mean_R_history
        df["mean_R_train"]  = self.run_logger.mean_R_train_history
        df["overall_max_R"] = self.run_logger.overall_max_R_history
        df["max_R"]         = self.run_logger.max_R_history
        # Complexity
        df["best_prog_complexity"] = self.run_logger.best_prog_complexity_history
        df["mean_complexity"]      = self.run_logger.mean_complexity_history
        # Loss
        df["loss"]  = self.run_logger.loss_history
        # Number of physical progs
        df["n_physical"] = self.run_logger.n_physical
        df["n_rewarded"] = self.run_logger.n_rewarded
        # Programs
        df["best_prog_of_epoch"] = np.array(self.run_logger.best_prog_epoch_str_history)
        df["overall_best_prog"]  = np.array(self.run_logger.overall_best_prog_str_history)
        # Getting free consts dfs
        best_prog_epoch_free_const_history_df   = pd.concat(self.run_logger.best_prog_epoch_free_const_history   , ignore_index=True)
        overall_best_prog_free_const_history_df = pd.concat(self.run_logger.overall_best_prog_free_const_history , ignore_index=True)
        # Best of epoch
        df["best_prog_of_epoch_prefix"] = np.array(self.run_logger.best_prog_epoch_str_prefix_history)
        df = pd.concat([df, best_prog_epoch_free_const_history_df], axis=1)
        # Overall best
        df["overall_best_prog_prefix"]  = np.array(self.run_logger.overall_best_prog_str_prefix_history)
        df = pd.concat([df, overall_best_prog_free_const_history_df], axis=1)

        return df

    def save_data (self):
        # -------- Save curves data --------
        df = self.get_curves_data_df()
        df.to_csv(self.save_path_log, index=False)
        return None

    def get_pareto_data_df(self):
        df = pd.DataFrame()
        complexity, programs, reward, rmse = self.run_logger.get_pareto_front()
        df["complexity"        ] = complexity
        df["length"            ] = np.array([len(prog.tokens) for prog in programs])
        df["reward"            ] = reward
        df["rmse"              ] = rmse
        df["r2"                ] = reward_funcs.SquashedNRMSE_to_R2(reward)
        df["expression"        ] = np.array([prog.get_infix_str() for prog in programs ])
        df["expression_prefix" ] = np.array([prog.__str__()       for prog in programs ])
        # -> UPDATE START_COL_FREE_CONST_PARETO_CSV = 7 IF CHANGES ARE MADE HERE
        # Exporting free const
        free_consts_df = pd.concat([prog.free_consts.detach().cpu().df() for prog in programs], ignore_index=True)
        df = pd.concat([df, free_consts_df], axis=1)
        return df

    def save_pareto_pkl(self):
        pareto_front_complexities, pareto_front_programs, pareto_front_r, pareto_front_rmse = self.run_logger.get_pareto_front()
        progs = [prog.detach() for prog in pareto_front_programs]
        # Saving pareto front expressions as pkl
        save_pareto_pkl (pareto_progs = progs, fpath = self.save_path_pareto_pkl)
        return None

    def save_pareto_fig(self):
        def plot_pareto_front(run_logger,
                              do_simplify                   = True,
                              show_superparent_at_beginning = True,
                              eq_text_size                  = 18,
                              delta_xlim                    = [0, 5 ],
                              delta_ylim                    = [-0.1, 0.005],
                              frac_delta_equ                = [0.01, -0.01],
                              figsize                       = (20, 10),
                             ):

            pareto_front_complexities, pareto_front_programs, pareto_front_r, pareto_front_rmse = run_logger.get_pareto_front()

            pareto_front_r2 = reward_funcs.SquashedNRMSE_to_R2(pareto_front_r)

            # Prettier fig with:
            #   plt.rc('text', usetex=True)
            #   plt.rc('font', family='serif')
            #plt.rc('font', size=32)

            # Fig
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.plot(pareto_front_complexities, pareto_front_r2, 'r-')
            ax.plot(pareto_front_complexities, pareto_front_r2, 'ro')

            # Limits
            xmin = pareto_front_complexities.min() + delta_xlim[0]
            xmax = pareto_front_complexities.max() + delta_xlim[1]
            ymin = pareto_front_r2.min() + delta_ylim[0]
            ymax = pareto_front_r2.max() + delta_ylim[1]
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.invert_yaxis()

            # Axes labels
            ax.set_xlabel("Expression complexity")
            ax.set_ylabel("$R^2$")


            for i_prog in range (len(pareto_front_programs)):
                prog = pareto_front_programs[i_prog]

                text_pos  = [pareto_front_complexities[i_prog] + frac_delta_equ[0]*(xmax-xmin),
                             pareto_front_r2[i_prog]         + frac_delta_equ[1]*(ymax-ymin)]
                # Getting latex expr
                latex_str = prog.get_infix_latex(do_simplify = do_simplify)
                # Adding "superparent =" before program to make it pretty
                if show_superparent_at_beginning:
                    latex_str = prog.library.superparent.name + ' =' + latex_str


                ax.annotate(text = f'${latex_str}$',
                            xy   = (text_pos[0], text_pos[1]),
                            size = eq_text_size,
                            ha   = "left",
                            va   = "bottom",
                           )
            return fig

        fig = plot_pareto_front(self.run_logger)
        fig.savefig(self.save_path_pareto_fig)
        plt.close(fig)

    def save_pareto_data (self):
        # -------- Save pareto data --------
        df = self.get_pareto_data_df()
        df.to_csv(self.save_path_pareto, index=False)
        return None

    def save_visualisation (self):
        # -------- Plot update --------
        self.update_plot()
        # -------- Save plot --------
        self.fig.savefig(self.save_path)
        return None

    def visualise (self, run_logger, batch):
        epoch = run_logger.epoch
        self.run_logger = run_logger
        self.batch      = batch
        if epoch == 0:
            self.initialize()
        # Plot curves
        if epoch%self.epoch_refresh_rate == 0:
            try:
                if self.do_show:
                    self.make_visualisation()
                if self.do_save:
                    self.save_visualisation()
            except Exception as e:
                print("Unable to make visualisation plots, error:")
                print(e)
        # Data curves
        if epoch%self.epoch_refresh_rate == 0:
            try:
                if self.do_save:
                    self.save_data()
            except Exception as e:
                print("Unable to save train curves data, error:")
                print(e)
        # Data Pareto
        if epoch%self.epoch_refresh_rate == 0:
            try:
                if self.do_save:
                    self.save_pareto_data()
            except Exception as e:
                print("Unable to save pareto data, error:")
                print(e)

        # Pareto pkl
        if epoch%self.epoch_refresh_rate == 0:
            try:
                if self.do_save:
                    self.save_pareto_pkl()
            except Exception as e:
                print("Unable to save pareto front expressions as pkl, error:")
                print(e)

        # Figure Pareto
        if epoch%self.epoch_refresh_rate == 0:
            try:
                if self.do_save:
                    self.save_pareto_fig()
            except Exception as e:
                print("Unable to save pareto figure, error:")
                print(e)

        # Prints
        if epoch % self.epoch_refresh_rate_prints == 0:
            try:
                if self.do_prints:
                    self.make_prints()
            except Exception as e:
                print("Unable to print status, error:")
                print(e)
