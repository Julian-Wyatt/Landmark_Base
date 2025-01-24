import init_paths
import argparse
import os

import wandb

from dataset_utils.dataset import LandmarkDataset

import utils.device
from trainers.resnet import resnet_unet
from utils.early_stopping import EarlyStoppingWithWarmup

os.environ["WANDB__SERVICE_WAIT"] = "240"
import torch

torch.manual_seed(42)
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, GradientAccumulationScheduler
from lightning.pytorch.loggers import WandbLogger
import lightning as L

from core import config


def parse_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description='Train a generative model to generate landmarks conditional on an image')

    parser.add_argument('--config_path', type=str, help='Path to the configuration file',
                        default="configs/default.yaml", required=False)
    parser.add_argument("--saving_root_dir", type=str, help='Path to where to save project files',
                        default="./", required=False)
    parser.add_argument("--desc", type=str, help="Description of the run", default="", required=False)
    # Parse the command-line arguments
    args = parser.parse_args()
    return args


def main(args):
    cfg = config.get_config(args.config_path, args.saving_root_dir)
    cfg.TRAIN.DESCRIPTION = args.desc
    import dotenv
    dotenv.load_dotenv()

    if not os.path.exists(f"{args.saving_root_dir}/wandb") and args.saving_root_dir != "./":
        os.makedirs(f"{args.saving_root_dir}/wandb")

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    run = wandb.init(project=f"{cfg.TRAIN.PROJECT}", name=f"{cfg.DATASET.NAME}-{args.config_path.split('/')[-1]}",
                     dir=f"{args.saving_root_dir}/wandb",
                     notes=args.desc, reinit=True)

    logger = WandbLogger(project=f"{cfg.TRAIN.PROJECT}",
                         name=f"{cfg.DATASET.NAME}-{args.config_path.split('/')[-1]}",
                         save_dir=f"{args.saving_root_dir}/wandb",
                         experiment=run,
                         id=run.id)

    import imgaug
    import numpy as np
    torch.manual_seed(42)
    imgaug.seed(42)
    np.random.seed(42)

    # handle wandb sweep setups
    if any(i in wandb.config.__dict__["_items"] for i in
           ["DATASET", "MODEL", "AUGMENTATIONS", "TRAIN", "TRAINLOSSES"]):
        sweep_config = dict(wandb.config.__dict__["_items"])

        cfg.TRAIN.LOG_IMAGE = False
        cfg.TRAIN.LOG_VIDEO = False
        for config_parent in ["DATASET", "MODEL", "AUGMENTATIONS", "TRAIN", "TRAINLOSSES"]:
            if config_parent in wandb.config:

                if config_parent == "TRAINLOSSES" and "TRAINLOSSES" in wandb.config:
                    if "NLL_WEIGHT" in wandb.config["TRAINLOSSES"] and "BCE_WEIGHT" in wandb.config["TRAINLOSSES"]:
                        if wandb.config["TRAINLOSSES"]["NLL_WEIGHT"] + wandb.config["TRAINLOSSES"]["BCE_WEIGHT"] == 0:
                            return

                if config_parent == "AUGMENTATIONS" and "AUGMENTATIONS" in wandb.config:
                    if "TRANSLATION_X" in wandb.config["AUGMENTATIONS"]:
                        sweep_config["AUGMENTATIONS"]["TRANSLATION_X"] = sorted([
                            sweep_config["AUGMENTATIONS"]["TRANSLATION_X"]["LOWERBOUND"],
                            sweep_config["AUGMENTATIONS"]["TRANSLATION_X"]["UPPERBOUND"]])
                    if "TRANSLATION_Y" in wandb.config["AUGMENTATIONS"]:
                        sweep_config["AUGMENTATIONS"]["TRANSLATION_Y"] = sorted([
                            sweep_config["AUGMENTATIONS"]["TRANSLATION_Y"]["LOWERBOUND"],
                            sweep_config["AUGMENTATIONS"]["TRANSLATION_Y"]["UPPERBOUND"]])
                getattr(cfg, config_parent).update(sweep_config[config_parent])

    # model normalisation check
    if cfg.DATASET.INT_TO_FLOAT == "none" and cfg.DATASET.NORMALISATION != "none":
        print(f"INT_TO_FLOAT is none but NORMALISATION is {cfg.DATASET.NORMALIZE}")
        return

    logger.log_hyperparams(args)
    logger.log_hyperparams(cfg.to_dict())

    if cfg.TRAIN.MODEL_TYPE.lower() == "default":
        logger.experiment.save("./trainers/resnet.py", policy="now")
        model = resnet_unet(cfg)
    else:
        raise ValueError(f"Model type {cfg.TRAIN.MODEL_TYPE} not recognised")
    logger.experiment.save("./dataset_utils/*", policy="now")
    logger.experiment.save("./trainers/default_trainer.py", policy="now")
    logger.experiment.save("./utils/*", policy="now")
    logger.experiment.save("./core/*", policy="now")

    train_dataloader = LandmarkDataset.get_loaders(
        cfg, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.NUM_WORKERS,
        augment_train=cfg.DATASET.AUGMENT_TRAIN, partition="training")
    validation_dataloader = LandmarkDataset.get_loaders(
        cfg, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.NUM_WORKERS,
        augment_train=False, partition="validation")

    checkpoint_dir = f"{args.saving_root_dir}/tmp/checkpoints/{cfg.DATASET.NAME}-{cfg.TRAIN.MODEL_TYPE}-{logger.experiment.id}"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_callback_sdr = ModelCheckpoint(
        filename="sdr_{val/sdr_l2_scaled_2.0}",
        dirpath=checkpoint_dir,
        save_top_k=1,
        save_last=True,
        monitor="val/sdr_l2_scaled_2.0",
        mode="max", )
    checkpoint_callback_l2 = ModelCheckpoint(
        filename="l2_{val/l2_scaled}",
        dirpath=checkpoint_dir,
        save_top_k=1,
        save_last=True,
        monitor="val/l2_scaled",
        mode="min", )

    # from lightning.pytorch.profilers import PyTorchProfiler
    # profiler = PyTorchProfiler(activities=[ProfilerActivity.CPU], dirpath="./",
    #                            my_schedule=schedule(
    #                                wait=1,
    #                                warmup=1,
    #                                active=5,
    #                                repeat=2),
    #                            profile_memory=False,
    #                            row_limit=-1,
    #                            sort_by_key="cpu_time_total",
    #                            with_stack=True,
    #                            with_modules=True,
    #                            )
    # import tracemalloc
    #
    # tracemalloc.start()
    early_stopping = EarlyStoppingWithWarmup(monitor="val/l2_scaled",
                                             warmup=cfg.TRAIN.EARLY_STOPPING_WARMUP,
                                             min_delta=0.00,
                                             patience=10,
                                             verbose=True, mode="min")

    if len(cfg.TRAIN.ACCUMULATOR) == 0:
        accumulator = GradientAccumulationScheduler(scheduling={0: 1})
    else:
        accumulator = GradientAccumulationScheduler(scheduling=cfg.TRAIN.ACCUMULATOR)

    if utils.device.get_device() == "mps":
        precision = "32-true"
    else:
        precision = "16-mixed"

    trainer = L.Trainer(max_epochs=cfg.TRAIN.EPOCHS, accelerator=utils.device.get_device(),
                        logger=logger, check_val_every_n_epoch=cfg.TRAIN.VAL_EVERY_N_EPOCHS, precision=precision,
                        callbacks=[checkpoint_callback_sdr, checkpoint_callback_l2, accumulator, early_stopping],
                        default_root_dir=checkpoint_dir,
                        enable_progress_bar=False,
                        gradient_clip_algorithm="norm",
                        )

    print(f"EXPERIMENT ID {logger.experiment.id}")
    checkpoint_file = f"{args.saving_root_dir}/tmp/checkpoints/{cfg.DATASET.NAME}-{cfg.TRAIN.MODEL_TYPE}-{cfg.TRAIN.CHECKPOINT_FILE}"
    if cfg.TRAIN.RUN_TRAIN:
        trainer.fit(model, train_dataloader, validation_dataloader)

        trainer.save_checkpoint(f"{checkpoint_dir}/train_end.ckpt")

        logger.log_hyperparams({"last_checkpoint": f"{checkpoint_dir}/train_end.ckpt"})
        logger.log_hyperparams({"best_checkpoint": checkpoint_callback_sdr.best_model_path})
        checkpoint_file = checkpoint_callback_sdr.best_model_path

    if cfg.TRAIN.RUN_TEST and os.path.exists(checkpoint_file):
        test_dataloader = LandmarkDataset.get_loaders(
            cfg, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.NUM_WORKERS,
            augment_train=False, partition="testing")

        if cfg.TRAIN.MODEL_TYPE.lower() == "default":
            logger.experiment.save("./trainers/resnet.py", policy="now")
            model = resnet_unet.load_from_checkpoint(checkpoint_file, cfg=cfg)

        trainer.validate(model, validation_dataloader)
        trainer.test(model, test_dataloader)
    elif not os.path.exists(checkpoint_file):
        print(f"Checkpoint file {checkpoint_file} does not exist")


if __name__ == "__main__":
    args = parse_args()
    main(args)
