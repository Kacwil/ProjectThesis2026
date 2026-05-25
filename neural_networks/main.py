from pathlib import Path
import torch

from Trainer.train_walkforward import train_walk_forward
from Trainer.train_dataclasses import DatasetConfig, ModelConfig, SplitConfig
from Trainer.post_evaluate import post_training_evaluation

import torch, numpy as np, random

seed = 1111
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATASETS = {
    "20220607": DatasetConfig(
        date="20220607",
        iq=Path(r"C:\Uni\V26\Prosjekt\Simulator\data\experimental_data.pkl"),
        baseline=Path(r"C:\Uni\V26\Prosjekt\Simulator\data\music.csv"),
        synth=[Path(r"C:\Uni\V26\Prosjekt\Simulator\data\RT_100k_temp"), Path(r"C:\Uni\V26\Prosjekt\Simulator\20220607\synth_100k_mp.pkl"), Path(r"C:\Uni\V26\Prosjekt\Simulator\data\RT_50k_lowel_temp.pkl")]
    )
}

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_roots(dataset: DatasetConfig) -> dict[str, Path | list[Path]]:
    return {
        "IQ": dataset.iq,
        "BASELINE": dataset.baseline,
        "SYNTH": dataset.synth,
    }


def main() -> None:

    device = get_device()

    dataset = DATASETS["20220607"]
    roots = make_roots(dataset)

    model_config = ModelConfig(name="cnn", seq_len=8, n_experts=1)
    split_config = SplitConfig(enable_synth=True, train_on_previous_data=True, should_freeze=False)

    if True:
        train_walk_forward(
            device=device,
            roots=roots,
            split_configs=split_config,
            model_configs=model_config,
        )

    if True:
        post_training_evaluation(
            device=device,
            roots=roots,
            split_cfg=split_config,
            model_cfg=model_config
            )
        


if __name__ == "__main__":
    main()