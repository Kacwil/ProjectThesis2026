from .train_utils import *
from .train_dataclasses import *

import sys
sys.path.append('..')

from neural_networks.Models.CNN import *


def build_cnn_model(cfg):
    return CNN()

def make_cnn_dataset(data: DataArrays, train: bool, cfg, shuffle):
    return CNNDataset(
        data.iq,
        data.angles_hat,
        data.positions,
        data.angles,
        train,
        cfg.seq_len,
        cfg.noise_mag,
        shuffle
    )

MODEL_SPECS = {
    "cnn": ModelSpec(build_cnn_model, make_cnn_dataset),
}

def get_model_spec(model_cfg):
    model_name = model_cfg.name

    try:
        return MODEL_SPECS[model_name]
    except KeyError:
        available = ", ".join(MODEL_SPECS)
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")
    
def get_model_spec_needs_seq_len(model_cfg):
    return get_model_spec(model_cfg).needs_seq_len