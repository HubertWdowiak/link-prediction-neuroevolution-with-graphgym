import json
import os

import pygad
from torch_geometric import seed_everything

from graphgym.config import cfg, set_run_dir
from graphgym.loader import create_loader, create_dataset
from graphgym.logger import create_logger, setup_printing
from graphgym.model_builder import create_model
from graphgym.optimizer import create_optimizer, create_scheduler
from graphgym.register import train_dict
from graphgym.train import train
from graphgym.utils.agg_runs import agg_runs
from graphgym.utils.comp_budget import params_count
from graphgym.utils.device import auto_select_device


def update_nested_dict(config, categorical_dict, continuous_dict, solution):
    config_dict = {}

    for i, (path, values) in enumerate(sorted(categorical_dict.items())):
        keys = path.split(".")
        sub_dict = config
        for key in keys[:-1]:  # iterate until the second last key
            sub_dict = sub_dict.setdefault(key, {})  # get/create the sub-dictionary
        sub_dict[keys[-1]] = values[int(solution[i] * len(values))]
        config_dict[keys[-1]] = values[int(solution[i] * len(values))]

    for i, (path, values) in enumerate(sorted(continuous_dict.items())):
        keys = path.split(".")
        sub_dict = config
        for key in keys[:-1]:  # iterate until the second last key
            sub_dict = sub_dict.setdefault(key, {})  # get/create the sub-dictionary
        sub_dict[keys[-1]] = (
            solution[i] * (values["max"] - values["min"]) + values["min"]
        )
        config_dict[keys[-1]] = (
            solution[i] * (values["max"] - values["min"]) + values["min"]
        )
    return config_dict


def get_neuroevolution_instance(
    hyperparams_categorical: dict, hyperparams_continuous: dict, repeats: int
):
    def fitness_func(ga_instance, solution, solution_idx):
        config_dict = update_nested_dict(
            cfg, hyperparams_categorical, hyperparams_continuous, solution
        )
        curr_dir = os.path.join(
            cfg.out_dir,
            str(ga_instance.generations_completed),
            str(solution_idx),
        )

        for i in range(repeats):
            set_run_dir(curr_dir)
            setup_printing()
            cfg.seed = cfg.seed + 1
            seed_everything(cfg.seed)
            auto_select_device()
            # Set machine learning pipeline
            datasets = create_dataset()
            loaders = create_loader(datasets)
            loggers = create_logger()
            model = create_model()
            optimizer = create_optimizer(model.parameters())
            scheduler = create_scheduler(optimizer)
            # Print model info
            # logging.info(model)
            # logging.info(cfg)
            cfg.params = params_count(model)
            # logging.info("Num parameters: %s", cfg.params)
            # Start training
            if cfg.train.mode == "standard":
                train(loggers, loaders, model, optimizer, scheduler)
            else:
                train_dict[cfg.train.mode](
                    loggers, loaders, model, optimizer, scheduler
                )

        # Save config dict for specific generation/solution_idx experiment set
        with open(os.path.join(curr_dir, "config_dict.json"), "w") as fp:
            json.dump(config_dict, fp)
        # Aggregate results from different seeds
        return 1 / agg_runs(curr_dir, cfg.metric_best)["loss"]

    ga_instance = pygad.GA(
        num_generations=20,
        num_parents_mating=10,
        fitness_func=fitness_func,
        gene_space={"low": 0, "high": 1},
        sol_per_pop=20,
        num_genes=len([x for x in hyperparams_categorical if x is not None])
        + len([x for x in hyperparams_categorical if x is not None]),
        parent_selection_type="rank",
        crossover_type="two_points",
        mutation_type="adaptive",
        mutation_probability=[0.25, 0.1],
    )
    return ga_instance
