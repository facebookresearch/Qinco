import hydra
from omegaconf import DictConfig

# Imports experiments (necessary to register experiments)
from qinco.qinco_tasks import QincoConvertTask, QincoEvalTask, QincoTrainTask
from qinco.search.search_tasks import (
    BuildIndexTask,
    EncodeDBTask,
    IVFTrainTask,
    SearchTask,
    TrainPairwiseDecoderTask,
)

EXPERIMENTS = {
    "train": QincoTrainTask,
    "eval_valset": QincoTrainTask,
    "eval": QincoEvalTask,
    "eval_time": QincoEvalTask,
    "convert": QincoConvertTask,
    "ivf_centroids": IVFTrainTask,
    "encode": EncodeDBTask,
    "build_index": BuildIndexTask,
    "train_pairwise_decoder": TrainPairwiseDecoderTask,
    "search": SearchTask,
}


@hydra.main(version_base=None, config_path="config", config_name="qinco_cfg")
def main(cfg: DictConfig):
    if cfg.task is None:
        raise ValueError(
            "Please specify a task (train, eval, etc.) using the 'train=<...>' argument"
        )
    expe = EXPERIMENTS[cfg.task](cfg)

    expe.accelerator.print(f"====================== RUNNING TASK {cfg.task}")
    expe.run()
    expe.accelerator.print("Task done")
    expe.accelerator.end_training()  # Destroy process group


if __name__ == "__main__":
    main()  # pylint: disable=all
