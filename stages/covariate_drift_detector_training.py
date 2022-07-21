import argparse
from pathlib import Path

import pandas as pd
from alibi_detect.cd import MMDDrift
from alibi_detect.utils.saving import save_detector

from utils.load_params import load_params


def train_data_drift_detector(data_dir: Path, detector_dir: Path,
                              detector_fname: Path, n_permutations: int):
    """Trains and saves covariate shift detector (=MMD drift detector)

    Args:
        data_dir: Folder with saved training data for the detector
        detector_dir: Folderpath where to save detector
        detector_fname: Filename of the saved detector
        n_permutations: Number of permutations used in the permutation test.
    Returns:

    """
    X_train = pd.read_pickle(data_dir / 'X_train.pkl')
    X_ref = X_train.to_numpy()
    drift_detector = MMDDrift(X_ref, p_val=0.05, n_permutations=20)

    detector_dir.mkdir(exist_ok=True)
    save_detector(drift_detector, detector_dir / detector_fname)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    params = load_params(params_path=args.config)
    data_dir = Path(params['base']['data_dir'])
    detector_dir = Path(params['detector']['dir'])
    detector_fname = Path(params['detector']['data_drift']['fname'])
    n_permutations = params['detector']['data_drift']['n_permutations']

    train_data_drift_detector(data_dir=data_dir, detector_dir=detector_dir,
                              detector_fname=detector_fname, n_permutations=n_permutations)
