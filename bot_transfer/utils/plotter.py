'''
Original Source:
https://raw.githubusercontent.com/hill-a/stable-baselines/master/stable_baselines/results_plotter.py
Modifying plotter to provide averages before the window size.
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from stable_baselines.bench.monitor import load_results
from bot_transfer.utils.loader import ModelParams, BASE

# matplotlib.use('TkAgg')  # Can change to 'Agg' for non-interactive mode
# plt.rcParams['svg.fonttype'] = 'none'

X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 200
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
          'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
          'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']

def rolling_window(array, window):
    """
    apply a rolling window to a np.ndarray

    :param array: (np.ndarray) the input Array
    :param window: (int) length of the rolling window
    :return: (np.ndarray) rolling window on the input array
    """
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

def window_func(var_1, var_2, window, func):
    """
    apply a function to the rolling window of 2 arrays

    :param var_1: (np.ndarray) variable 1
    :param var_2: (np.ndarray) variable 2
    :param window: (int) length of the rolling window
    :param func: (numpy function) function to apply on the rolling window on variable 2 (such as np.mean)
    :return: (np.ndarray, np.ndarray)  the rolling output with applied function
    """
    var_2_window = rolling_window(var_2, window)
    function_on_var2 = func(var_2_window, axis=-1)
    return var_1[window - 1:], function_on_var2

def window_func_full(x, y, window, func):
    yw = rolling_window(y, window)
    yw_func = func(yw, axis=-1)
    initial_avgs = np.cumsum(y[:window -1]) / np.arange(1, window)
    full_avgs = np.concatenate((initial_avgs[10:], yw_func))
    return x[10:], full_avgs

def ts2xy(timesteps, xaxis):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """
    if xaxis == X_TIMESTEPS:
        x_var = np.cumsum(timesteps.l.values)
        y_var = timesteps.r.values
    elif xaxis == X_EPISODES:
        x_var = np.arange(len(timesteps))
        y_var = timesteps.r.values
    elif xaxis == X_WALLTIME:
        x_var = timesteps.t.values / 3600.
        y_var = timesteps.r.values
    else:
        raise NotImplementedError
    return x_var, y_var


def plot_curves(xy_list, xaxis, title, legend_names=None):
    """
    plot the curves

    :param xy_list: ([(np.ndarray, np.ndarray)]) the x and y coordinates to plot
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param title: (str) the title of the plot
    """
    if legend_names:
        assert len(legend_names) == len(xy_list)
    else:
        legend_names = [str(i) for i in range(len(xy_list))]

    plt.figure(figsize=(8, 2))
    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0

    for (i, (x, y)) in enumerate(xy_list):
        # y = np.clip(y, -20, 50000)
        color = COLORS[i]
        # plt.scatter(x, y, s=2)
        # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        if x.shape[0] >= EPISODES_WINDOW:
            # Compute and plot rolling mean with window of size EPISODE_WINDOW
            x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean)
            plt.plot(x, y_mean, color=color, label=legend_names[i])
        else:
            plt.plot(x, y, color=color, label=legend_names[i])
    plt.xlim(minx, maxx)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel("Episode Rewards")
    plt.legend(locl='lower right')
    # plt.tight_layout()

def plot_results(dirs, num_timesteps, xaxis, task_name, legend_names=None):
    """
    plot the results

    :param dirs: ([str]) the save location of the results to plot
    :param num_timesteps: (int or None) only plot the points below this value
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param task_name: (str) the title of the task to plot
    """
    tslist = []
    for folder in dirs:
        if not folder.startswith('/'):
            folder = os.path.join(BASE, folder)
        timesteps = load_results(folder)
        if num_timesteps is not None:
            timesteps = timesteps[timesteps.l.cumsum() <= num_timesteps]
        tslist.append(timesteps)
    xy_list = [ts2xy(timesteps_item, xaxis) for timesteps_item in tslist]
    plot_curves(xy_list, xaxis, task_name, legend_names=legend_names)

def multi_seed_plot_results(dirs, num_timesteps, xaxis, task_name, legend_names=None, individual=False, zero_shot=None, zoh=True, hiro=None, clip=None):
    '''
    Directory structure is assumed as follows:
    /experiment_name/RunName/0.monitor.csv
    '''
    import pandas as pd
    data = list()
    sns.set_context(context="paper", font_scale=1.5)
    sns.set_style("darkgrid", {'font.family': 'serif'})

    graph = None
    i = 0
    for experiment in dirs:
        x_list, y_list = list(), list()
        if not experiment.startswith('/'):
            exp_dir = os.path.join(BASE, experiment)
        else:
            exp_dir = experiment
        runs = [run for run in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, run)) ]
        legend_caption = str(legend_names[i]) if legend_names else experiment

        for run in runs:
            timesteps = load_results(os.path.join(exp_dir, run))
            # print("Params", ModelParams.load(os.path.join(exp_dir, run)))
            if num_timesteps is not None:
                timesteps = timesteps[timesteps.l.cumsum() <= num_timesteps]
            x, y = ts2xy(timesteps, xaxis)
            # Apply the window function on episodes.
            if x.shape[0] >= EPISODES_WINDOW:
                # Compute and plot rolling mean with window of size EPISODE_WINDOW
                if 'Full' in legend_caption:
                    x, y = window_func_full(x, y, EPISODES_WINDOW, np.mean)
                else:
                    x, y = window_func(x, y, EPISODES_WINDOW, np.mean)
            if individual:
                graph = sns.lineplot(x=x, y=y, label=run)           
            
            x_list.append(x)
            if clip:
                y = np.clip(y, clip, np.inf)
            y_list.append(y)

        if not individual:
            combined_x_list, combined_y_list = [], []
            if zoh:
                # Zero Order Hold interpolate the data
                joint_x_list = sorted(list(set(np.concatenate(x_list))))
                for xs, ys in zip(x_list, y_list):
                    cur_ind = 0
                    new_y_list = []
                    # last_y = ys[0]
                    # last_x = xs[0]
                    for x in joint_x_list:
                        if x > xs[cur_ind] and cur_ind < len(ys) - 1:
                            cur_ind += 1
                        new_y_list.append(ys[cur_ind])
                    if not 'Full' in legend_caption:
                        combined_x_list.extend(joint_x_list[::50])
                        combined_y_list.extend(new_y_list[::50])
                    else:
                        combined_x_list.extend(joint_x_list[::5])
                        combined_y_list.extend(new_y_list[::5])
            else:
                # Regular data
                for xs, ys in zip(x_list, y_list):
                    combined_x_list.extend(xs)
                    combined_y_list.extend(ys)

            data = pd.DataFrame({xaxis : combined_x_list, "reward": combined_y_list})
            print(len(combined_x_list))
            graph = sns.lineplot(x=xaxis, y="reward", data=data, ci="sd", sort=True, label=legend_caption)
            i += 1

    if not hiro is None:
        hiro_files = list()
        for root, dirs, files in os.walk(hiro):
            for file in files:
                if file == "train.csv":
                    hiro_files.append(os.path.join(root, file))
        print(hiro_files)
        combined_x_list, combined_y_list = [], []
        for hiro_file in hiro_files:
            df = pd.read_csv(hiro_file)
            combined_x_list.extend(df["total/steps"])
            combined_y_list.extend(df["rollout/return_history"])
        
        data = pd.DataFrame({xaxis : combined_x_list, "reward": combined_y_list})
        graph = sns.lineplot(x=xaxis, y="reward", data=data, ci="sd", n_boot=500, sort=True, label="Hiro")

    hiro_tf = None
    if not hiro_tf is None:
        hiro_files = list()
        for root, dirs, files in os.walk(hiro):
            for file in files:
                if file.endswith(".csv"):
                    hiro_files.append(os.path.join(root, file))
        
        for hiro_file in hiro_files:
            df = pd.read_csv(hiro_file)
            combined_x_list.extend(df["step"])
            combined_y_list.extend(df["value"])
        
        data = pd.DataFrame({xaxis : combined_x_list, "reward": combined_y_list})
        graph = sns.lineplot(x=xaxis, y="reward", data=data, ci="sd", n_boot=500, sort=True, label="Hiro")


    if not zero_shot is None:
        graph.axhline(zero_shot, c='purple', linestyle='dashed', label="Zero Shot")
    plt.title(task_name)
    plt.xlabel('Samples')
    plt.ylabel("Episode Rewards")
    plt.legend(loc='lower right')
    # graph.get_legend().remove()
    plt.tight_layout(pad=0)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(6,6)) 
    # plt.locator_params(axis='x', nbins=6)



def main():
    """
    Example usage in jupyter-notebook

    .. code-block:: python

        from stable_baselines import results_plotter
        %matplotlib inline
        results_plotter.plot_results(["./log"], 10e6, results_plotter.X_TIMESTEPS, "Breakout")

    Here ./log is a directory containing the monitor.csv files
    """
    import argparse
    import os
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dirs', help='List of log directories', nargs='*', default=['./log'])
    parser.add_argument('--num_timesteps', type=int, default=int(10e6))
    parser.add_argument('--xaxis', help='Varible on X-axis', default=X_TIMESTEPS)
    parser.add_argument('--task_name', help='Title of plot', default='Breakout')
    args = parser.parse_args()
    args.dirs = [os.path.abspath(folder) for folder in args.dirs]
    plot_results(args.dirs, args.num_timesteps, args.xaxis, args.task_name)
    plt.show()