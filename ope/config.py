import os
d = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
    )
EXPERIMENT_DIR = os.path.join(d,"paper_exp_output")
TRACKER_PATH = os.path.join(EXPERIMENT_DIR,"paper_tracker.json")