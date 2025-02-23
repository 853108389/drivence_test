""" run_kitti.py

Run example:
run_kitti.py --USE_PARALLEL False --METRICS Hota --TRACKERS_TO_EVAL CIWT

Command Line Arguments: Defaults, # Comments
    Eval arguments:
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 8,
        'BREAK_ON_ERROR': True,
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': True,
    Dataset arguments:
        'GT_FOLDER': os.path.join(code_path, 'data/gt/kitti/kitti_2d_box_train'),  # Location of GT data
        'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/kitti/kitti_2d_box_train/'),  # Trackers location
        'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
        'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
        'CLASSES_TO_EVAL': ['car', 'pedestrian'],  # Valid: ['car', 'pedestrian']
        'SPLIT_TO_EVAL': 'training',  # Valid: 'training', 'val', 'training_minus_val', 'test'
        'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
        'PRINT_CONFIG': True,  # Whether to print current config
        'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        'OUTPUT_SUB_FOLDER': ''  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
    Metric arguments:
        'METRICS': ['Hota','Clear', 'ID', 'Count']
"""

import argparse
import os
import sys
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402


def eval_kitti(GT_FOLDER, TRACKERS_FOLDER, SEQ_MAP_DIR, TRACKER_SUB_FOLDER, TRACKERS_TO_EVAL, OUTPUT_FOLDER):
    freeze_support()

    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.Kitti2DBox.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs

    config["CLASSES_TO_EVAL"] = ["car"]
    config["SPLIT_TO_EVAL"] = "training"
    config["METRICS"] = ['HOTA', 'CLEAR', 'Identity']
    config["GT_FOLDER"] = GT_FOLDER
    config["TRACKERS_FOLDER"] = TRACKERS_FOLDER
    config["SEQ_MAP_DIR"] = SEQ_MAP_DIR
    config["TRACKER_SUB_FOLDER"] = TRACKER_SUB_FOLDER
    config["TRACKERS_TO_EVAL"] = TRACKERS_TO_EVAL
    config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
    # print(config["OUTPUT_FOLDER"])
    # return

    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs='+')
        else:
            parser.add_argument("--" + setting)
    args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            else:
                x = args[setting]
            config[setting] = x
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.Kitti2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    return evaluator.evaluate(dataset_list, metrics_list)


if __name__ == '__main__':
    eval_kitti()
