import argparse
import numpy as np
import json
from copy import deepcopy
import pickle
import copy
import itertools
from pymlrf.ModelTracking import SerialisedTracker, Experiment, Option
from pymlrf.FileSystem import DirectoryHandler
import os 

from ope.envs.graph import Graph
from ope.policies.basics import BasicPolicy

from ope.experiment_tools.experiment import ExperimentRunner, analysis
from ope.experiment_tools.config import Config
from ope.experiment_tools.factory import setup_params
from ope.envs.model_fail import ModelFail

TRACKER = SerialisedTracker(
    path="./paper_tracker.json",
    u_id="exp_nm"
    )
SEEDS = [100,101,102,103,104,105,106,107,108,109,110]
EXPERIMENT_DIR = "./paper_exp_output"
exp_dir_handler = DirectoryHandler(loc=EXPERIMENT_DIR)
if not exp_dir_handler.is_created:
    exp_dir_handler.create()

def toy_graph(param):
    param = setup_params(param) # Setup parameters
    runner = ExperimentRunner() # Instantiate a runner for an experiment

    # store these credentials in an object
    for i in SEEDS:
        s_config = deepcopy(param["experiment"])
        s_config["seed"] = i
        cfg = Config(s_config)

        # initialize environment with the parameters from the config file.
        # If you'd like to use a different environment, swap this line
        env = ModelFail(make_pomdp=cfg.is_pomdp,
                        number_of_pomdp_states = cfg.pomdp_horizon,
                        transitions_deterministic=not cfg.stochastic_env,
                        max_length = cfg.horizon,
                        sparse_rewards = cfg.sparse_rewards,
                        stochastic_rewards = cfg.stochastic_rewards)

        # set seed for the experiment
        np.random.seed(cfg.seed)

        # processor processes the state for storage,  {(processor(x), a, r, processor(x'), done)}
        processor = lambda x: x

        # absorbing state for padding if episode ends before horizon is reached. This is environment dependent.
        absorbing_state = processor(np.array([env.n_dim - 1]))

        # Setup policies. BasicPolicy takes the form [P(a=0), P(a=1), ..., P(a=n)]
        # For different policies, swap in here
        actions = [0, 1]
        pi_e = BasicPolicy(
            actions, [max(.001, .2*cfg.eval_policy), 1-max(.001, .2*cfg.eval_policy)])
        pi_b = BasicPolicy(
            actions, [max(.001, .2*cfg.base_policy), 1-max(.001, .2*cfg.base_policy)])
        
        # add env, policies, absorbing state and processor
        cfg.add({
            'env': env,
            'pi_e': pi_e,
            'pi_b': pi_b,
            'processor': processor,
            'absorbing_state': absorbing_state
        })
        cfg.add({'models': param['models']})

        # Add the configuration
        runner.add(cfg)
    results = runner.run()
    return results

exp_lkp = {
    "toy_graph": toy_graph
}

def experiment_wrapper(exp_func, param, exp_rt, save_loc):
    results = exp_func(param)
    results_nm = f"{exp_rt}.pkl"
    with open(os.path.join(save_loc, results_nm), "wb") as output_file:
        pickle.dump(results, output_file)
    return {"completed":True}
    

if __name__ == "__main__":
    #configuration_filename = "toy_graph_pomdp_cfg.json"
    config_filenames = [
        "toy_graph.json"
        ]
    parser = argparse.ArgumentParser()
        # parser.add_argument(
        #     '-m', '--models', help='which models to use', type=str, nargs='+',
        #     required=True)
    args = parser.parse_args()
    for config_nm in config_filenames:
        with open('cfgs_paper_rerun/{0}'.format(config_nm), 'r') as f:
            param = json.load(f)
        exp_rt = config_nm.split(".")[0]
        lst_args = [key for key in param["experiment"].keys() if isinstance(param["experiment"][key],list)]
        for key in lst_args:
            param["experiment"][key] = [{key:val} for val in param["experiment"][key]]
        for it_val in list(itertools.product(*[param["experiment"][key] for key in lst_args])):
            exp_params = copy.deepcopy(param)
            for val in it_val:
                exp_params["experiment"].update(val)
            exp_nm = "-".join([str(val) for val in exp_params["experiment"].values()])
            exp = Experiment(
                exp_name=exp_nm,
                parent_loc=EXPERIMENT_DIR,
                mt=TRACKER
                )
            exp.status_check()
            print(exp_params)
            exp.run(
                option=Option("overwrite"),
                func=experiment_wrapper,
                exp_func=exp_lkp[param["experiment"]["env"]],
                meta_data=exp_params,
                param = exp_params,
                exp_rt = exp_rt,
                save_loc = exp.loc
            )
            