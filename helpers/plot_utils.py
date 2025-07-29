import matplotlib.pyplot as plt
import yaml
import numpy as np
import os

def find_best_detector_settings(salmon_tracking_root):
    '''
    Find the detector settings with the best F1 score, and plot the results.
    
    Args:
        salmon_tracking_root (str): The root of the tracking data.
    Operations:
        Creates a scatter plot over the precission-recall data of all saved detector runs.
        Saves the plot in the folder with all the detector runs.
    Return:

    '''

    fig, ax = plt.subplots(1,1)

    # Open last used config file
    with open(salmon_tracking_root + '\\config.yml', 'r') as f:
        config = yaml.safe_load(f)

    # Store information about best settings
    best_pres_rec = [0.01,0.01]
    best_params = []

    detector_results_root = salmon_tracking_root + '\\detector_optimization\\' + config['detector_name']
    
    # Iterate over all tested settings
    for analysis_folder in os.listdir(detector_results_root):
        if not analysis_folder.split('\\')[-1].startswith('analysis'):
            continue

        # Load results
        res = {}
        with open(detector_results_root + '\\' + analysis_folder + '\\' + 'motmetrics_evaluation_results' + '\\' + 'results.txt', 'r') as f:
            for line in f:
                line = line.split(',')
                res[line[0]] = [float(l) for l in line[1:]]

        # Load config file
        with open(detector_results_root + '\\' + analysis_folder + '\\' + 'config.yml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract recall and precission
        pres = np.average(np.array(res['precision']))
        rec = np.average(np.array(res['recall']))

        # Check F1 score
        if 2*(best_pres_rec[0]*best_pres_rec[1])/(best_pres_rec[0]+best_pres_rec[1]) < 2*(pres*rec)/(pres+rec):
            best_pres_rec[0] = pres
            best_pres_rec[1] = rec
            best_params = 'salmon_conf: ' + str(round(config['salmon_conf'], 2)) + ',\nbp_conf: ' + str(round(config['bp_conf'], 2)) + ',\nsalmon_iou: ' + str(round(config['salmon_iou'], 2))
        
        # Plot point on precission-recall curve
        ax.scatter(pres, rec, s = 20, c = 'k')

    ax.scatter(best_pres_rec[0], best_pres_rec[1], marker = '*', s = 20, c = 'r', label = best_params)

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel('Precission')
    ax.set_ylabel('Recall')
    ax.set_title('Precission-Recall curve for model ' + config['detector_name'])
    print('The best detector settings are: ')
    print(best_params)
    ax.legend()
    fig.savefig(detector_results_root + '\\result.png')


def plot_tracking_results(results, motmetrics_path, title = '', plot_metrics = ['hota_alpha', 'recall', 'precision', 'transfer_ratio', 'switch_ratio'], results_file_name = 'results'):
    '''
    Plot tracking results.

    Args:
        results (dict{str: [float]}): A dictionary where keys are metric strings, and the values are list of floats.
        Each float is the value of the metric at a certain iou threshold.
        Motmetrics_path (str): Path to the motmetrics folder
        title (str): Title of the plot
        plot_metrics (list[str]): A list of all the matrics to be plotted
    Operations:
        Save tracking results plot to disk
    '''
    fig, ax = plt.subplots(1,1, figsize = (10,5))
    for m in list(results.keys()):
        if m in plot_metrics:
            ax.plot(results['th_list'], results[m], '-o', label=m + ' (' + str(round(np.nanmean(results[m]), 4)) + ')')

    ax.set_xlim(0,1)
    ax.set_ylim(0,1.05)
    ax.grid()
    ax.legend()
    ax.set_title(title)
    fig.savefig(motmetrics_path + '\\' + results_file_name + '.png')
    