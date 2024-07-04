import os
dir_path = os.path.dirname(os.path.realpath(__file__))
DATA_LOC = os.path.join("/".join(dir_path.split('/')[0:-3]), "data")
EXPERIMENT_DIR = os.path.join(DATA_LOC,"paper_exp_output")
TRACKER_PATH = os.path.join(EXPERIMENT_DIR,"paper_tracker.json")