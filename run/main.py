import torch

from graphgym.cmd_args import parse_args
from graphgym.config import cfg, dump_cfg, load_cfg, set_out_dir
from graphgym.contrib.train.neuroevolution import get_neuroevolution_instance

if __name__ == "__main__":
    # Load cmd line args
    args = parse_args()
    # Load config file
    load_cfg(cfg, args)
    set_out_dir(cfg.out_dir, args.cfg_file)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    dump_cfg(cfg)

    hyperparams_categorical = {
        # "gnn.layer_type": cfg.neuroevolution.layer_types,
        "optim.optimizer": cfg.neuroevolution.optimizers,
        "train.batch_size": cfg.neuroevolution.batch_sizes,
        "dataset.transform": cfg.neuroevolution.transforms,
        "dataset.resample_negative": cfg.neuroevolution.resample_negatives,
        "model.edge_decoding": cfg.neuroevolution.model_edge_decoding,
        "model.graph_pooling": cfg.neuroevolution.model_graph_pooling,
        # "gnn.att_heads": cfg.neuroevolution.gnn_att_heads,
    }

    hyperparams_continuous = {
        "optim.base_lr": cfg.neuroevolution.lrs,
        # "gnn.keep_edge": cfg.neuroevolution.gnn_keep_edge,
    }

    ga_instance = get_neuroevolution_instance(
        hyperparams_categorical, hyperparams_continuous, args.repeat
    )
    ga_instance.run()
