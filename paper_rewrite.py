import argparse
import numpy as np
import json
from copy import deepcopy
from typing import Dict
import pickle
import itertools
from pymlrf.ModelTracking import SerialisedTracker, Experiment, Option
from pymlrf.FileSystem import DirectoryHandler
import os 
from keras.api.saving import load_model

from ope.envs.graph import Graph
from ope.policies.basics import BasicPolicy
from ope.policies.epsilon_greedy_policy import EGreedyPolicy
from ope.policies.tabular_model import TabularPolicy

from ope.experiment_tools.experiment import ExperimentRunner, analysis
from ope.experiment_tools.config import Config
from ope.experiment_tools.factory import setup_params
from ope.envs.model_fail import ModelFail
from ope.envs.discrete_toy_mc import DiscreteToyMC
from ope.envs.modified_mountain_car import ModifiedMountainCarEnv
from ope.envs.gridworld import Gridworld

from ope.config import EXPERIMENT_DIR, TRACKER_PATH

exp_dir_handler = DirectoryHandler(loc=EXPERIMENT_DIR)
if not exp_dir_handler.is_created:
    exp_dir_handler.create()

tracker = SerialisedTracker(
    path=TRACKER_PATH,
    u_id="exp_nm"
    )

SEEDS = [100,101,102,103,104,105,106,107,108,109,110]

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

def toy_mc(param):
    param = setup_params(param) # Setup parameters
    runner = ExperimentRunner() # Instantiate a runner for an experiment

    # store these credentials in an object
    for i in SEEDS:
        s_config = deepcopy(param["experiment"])
        s_config["seed"] = i
        cfg = Config(s_config)

        # initialize environment with the parameters from the config file.
        # If you'd like to use a different environment, swap this line
        env = DiscreteToyMC()
        # set seed for the experiment
        np.random.seed(cfg.seed)

        # processor processes the state for storage,  {(processor(x), a, r, processor(x'), done)}
        processor = lambda x: x

        # absorbing state for padding if episode ends before horizon is reached. This is environment dependent.
        absorbing_state = processor(np.array([env.n_dim - 1]))

        # Setup policies. BasicPolicy takes the form [P(a=0), P(a=1), ..., P(a=n)]
        # For different policies, swap in here
        actions = [0, 1]
        pi_e = BasicPolicy(actions, [1-max(.001, cfg.eval_policy/100), max(.001, cfg.eval_policy/100)])
        pi_b = BasicPolicy(actions, [1-max(.001, cfg.base_policy/100), max(.001, cfg.base_policy/100)])
        
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


def mc(param):
    param = setup_params(param) # Setup parameters
    runner = ExperimentRunner() # Instantiate a runner for an experiment

    # store these credentials in an object
    for i in SEEDS:
        s_config = deepcopy(param["experiment"])
        s_config["seed"] = i
        cfg = Config(s_config)

        # initialize environment with the parameters from the config file.
        # If you'd like to use a different environment, swap this line
        env = ModifiedMountainCarEnv(
            deterministic_start=[-.4, -.5, -.6], 
            seed=cfg.seed
            )
        # set seed for the experiment
        np.random.seed(cfg.seed)

        # processor processes the state for storage,  {(processor(x), a, r, processor(x'), done)}
        processor = lambda x: x

        # absorbing state for padding if episode ends before horizon is reached. This is environment dependent.
        absorbing_state = processor(np.array([.5, 0]))

        # Setup policies. BasicPolicy takes the form [P(a=0), P(a=1), ..., P(a=n)]
        # For different policies, swap in here
        actions = [0,1,2]
        pi_e = EGreedyPolicy(
            model=load_model(os.path.join(os.getcwd(),'ope','trained_models','mc_trained_model_Q.h5')), 
            prob_deviation=cfg.eval_policy, 
            action_space_dim=len(actions))
        pi_b = EGreedyPolicy(
            model=load_model(os.path.join(os.getcwd(),'ope','trained_models','mc_trained_model_Q.h5')), 
            prob_deviation=cfg.base_policy, 
            action_space_dim=len(actions)
            )

        
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

def gridworld(param):
    param = setup_params(param) # Setup parameters
    runner = ExperimentRunner() # Instantiate a runner for an experiment

    # store these credentials in an object
    for i in SEEDS:
        s_config = deepcopy(param["experiment"])
        s_config["seed"] = i
        cfg = Config(s_config)

        # initialize environment with the parameters from the config file.
        # If you'd like to use a different environment, swap this line
        env = Gridworld(slippage=float(.2 * cfg.stochastic_env))
        policy = env.best_policy()
        # set seed for the experiment
        np.random.seed(cfg.seed)

        # processor processes the state for storage,  {(processor(x), a, r, processor(x'), done)}
        processor = lambda x: x
        absorbing_state = processor(np.array([len(policy)]))

        # Setup policies. BasicPolicy takes the form [P(a=0), P(a=1), ..., P(a=n)]
        # For different policies, swap in here
        pi_e = EGreedyPolicy(
            model=TabularPolicy(
                policy, 
                absorbing=absorbing_state
                ), 
            prob_deviation=cfg.eval_policy, 
            action_space_dim=env.n_actions
            )
        pi_b = EGreedyPolicy(
            model=TabularPolicy(
                policy, 
                absorbing=absorbing_state
                ), 
            prob_deviation=cfg.base_policy, 
            action_space_dim=env.n_actions
            )

        
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
    "toy_graph": toy_graph,
    "toy_mc": toy_mc,
    "gridworld": gridworld,
    "mc": mc
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
        "toy_graph.json",
        "toy_graph_pomdp.json",
        "gridworld.json",
        "mc.json",
        "toy_mc.json"
        ]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--configs', help='Which configs to load. If all specify "ALL"', 
        type=str, 
        nargs='+',
        required=True)
    args = parser.parse_args()
    if "ALL" not in args.configs:
        config_filenames = args.configs
    for config_nm in config_filenames:
        with open('cfgs_paper_rerun/{0}'.format(config_nm), 'r') as f:
            param:Dict = json.load(f)
        exp_rt = config_nm.split(".")[0]
        lst_args = [key for key in param["experiment"].keys() if isinstance(param["experiment"][key],list)]
        mdl_lst_args = [
            param["models"][key]["model"] for key in param["models"].keys()
            ][0]
        if not isinstance(mdl_lst_args, list):
            mdl_lst_args = [mdl_lst_args]
        mdl_lst_args = list(itertools.product(mdl_lst_args))
        for key in lst_args:
            param["experiment"][key] = [{key:val} for val in param["experiment"][key]]    
        for it_val in list(itertools.product(*[param["experiment"][key] for key in lst_args])):
            exp_params = deepcopy(param)
            for val in it_val:
                if (not param["experiment"]["horizon"]) and (list(val.keys())[0] == "pomdp_horizon"):
                    exp_params["experiment"].update(
                        {
                            "horizon": val["pomdp_horizon"][1],
                            "pomdp_horizon": val["pomdp_horizon"][0],
                            }
                    )
                elif (not param["experiment"]["base_policy"]) and (list(val.keys())[0] == "eval_policy"):
                    exp_params["experiment"].update(
                        {
                            "base_policy": val["eval_policy"][0],
                            "eval_policy": val["eval_policy"][1],
                            }
                    )
                else:
                    exp_params["experiment"].update(val)
                _any_mdl_hori = any(
                    [
                        exp_params["models"][key]["max_traj_length"] == "horizon"
                        for key in exp_params["models"] if 
                        "max_traj_length" in exp_params["models"][key].keys()
                        ]
                    )
                if (list(val.keys())[0] == "horizon") and (_any_mdl_hori):
                    for key in exp_params["models"]:
                        if "max_traj_length" in exp_params["models"][key].keys():
                            exp_params["models"][key]["max_traj_length"] = exp_params["experiment"]["horizon"]
            if "horizon" not in lst_args:
                 for key in exp_params["models"]:
                        if "max_traj_length" in exp_params["models"][key].keys():
                            exp_params["models"][key]["max_traj_length"] = exp_params["experiment"]["horizon"]
            for mdl_type in mdl_lst_args:
                for mdl in exp_params["models"]:
                    exp_params["models"][mdl]["model"] = mdl_type[0]
                exp_nm = "-".join([str(val) for val in exp_params["experiment"].values()])
                exp_nm = exp_nm+"-"+mdl_type[0]
                exp = Experiment(
                    exp_name=exp_nm,
                    parent_loc=EXPERIMENT_DIR,
                    mt=tracker
                    )
                if tracker.is_created:
                    tracker.read()
                exp.status_check()
                exp.run(
                    option=Option("overwrite"),
                    func=experiment_wrapper,
                    exp_func=exp_lkp[param["experiment"]["env"]],
                    meta_data=exp_params,
                    param = exp_params,
                    exp_rt = exp_rt,
                    save_loc = exp.loc
                )
                tracker.write()
            