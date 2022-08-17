from argparse import ArgumentParser
from supervised_FCN.preprocessing.data_pipeline import build_data_pipeline
from supervised_FCN.utils import get_root_dir, load_yaml_param_settings


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
