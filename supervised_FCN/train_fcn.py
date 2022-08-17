from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from supervised_FCN.experiments.exp_train import ExpFCN
from supervised_FCN.preprocessing.data_pipeline import build_data_pipeline
from supervised_FCN.utils import *


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    return parser.parse_args()


if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    # data pipeline
    train_data_loader, test_data_loader = [build_data_pipeline(config, kind) for kind in ['train', 'test']]

    # fit
    train_exp = ExpFCN(config, len(train_data_loader.dataset), len(np.unique(train_data_loader.dataset.Y)))
    wandb_logger = WandbLogger(project='supervised-FCN', name=config['dataset']['subset_name'], config=config)
    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=False,
                         callbacks=[LearningRateMonitor(logging_interval='epoch')],
                         **config['trainer_params'])
    trainer.fit(train_exp,
                train_dataloaders=train_data_loader,
                val_dataloaders=test_data_loader,)

    # test
    trainer.test(train_exp, test_data_loader)

    save_model({f"{config['dataset']['subset_name']}": train_exp.fcn})
