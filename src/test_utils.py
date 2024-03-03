import numpy as np
import tensorflow as tf
from bidict import bidict

from src.datasets.managers import AbstractDataset


def calculate_forecasts(dataset: AbstractDataset, config, x_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, verbose: bool = False) -> dict:
    """Calculates P forecasts based on the original paper.

    Parameters
    ----------
    dataset : AbstractDataset
    config : Dict
    x_test : np.ndarray
        The true testing inputs.
    y_test : np.ndarray
        The true testing target.
    y_pred : np.ndarray
        The model predicted targe.
    verbose : bool, optional
        Verbosity for printing logs, by default False.

    Returns
    -------
    dict
        Calculate q-risk scores.
    """
    # TODO: Make Batch Size Larger for bigger datasets
    batch = 1

    # Make Global Variables to be Used
    categorical_vocab = bidict(dataset._cat_vocabs.get('categorical_id'))
    # Get the Target Scalers
    target_scalers = dataset._targ_scaler_objs

    y_actual = None
    y_expected = None

    assert y_test.shape[0] == y_pred.shape[0]
    assert y_test.shape[1] == y_pred.shape[1]
    assert y_pred.shape[2] == len(config.quantiles)

    for i in range(0, x_test.shape[0], batch):
        if verbose:
            print("Running computation for slice from {} and {}".format(i, i+batch))

        x_test_slice = x_test[i:i+batch, :, :]
        x_test_slice = x_test_slice[:, dataset._n_enc_len:, :]

        y_test_slice = y_test[i]
        y_pred_slice = y_pred[i]

        category = np.unique(x_test_slice[:, :, 4])

        assert len(category) == 1
        category = categorical_vocab.inverse[category[0]]
        if category in target_scalers.keys():
            scaler = target_scalers[category]
        elif int(category) in target_scalers.keys():
            scaler = target_scalers[int(category)]

        y_act_res = scaler.inverse_transform(y_test_slice)
        y_exp_res = scaler.inverse_transform(y_pred_slice)

        if y_actual is None:
            y_actual = y_act_res
        else:
            y_actual = np.concatenate([y_actual, y_act_res], axis=0)

        if y_expected is None:
            y_expected = y_exp_res
        else:
            y_expected = np.concatenate([y_expected, y_exp_res], axis=0)

    y_actual = y_actual.reshape(y_test.shape)
    y_expected = y_expected.reshape(y_pred.shape)

    return _calculate_q_risk(quantiles=config.quantiles, y_actual=y_actual, y_expected=y_expected, n_targets=len(dataset._targ_idx))


def _calculate_q_risk(quantiles: list, y_actual: np.ndarray, y_expected: np.ndarray, n_targets: int):
    """ 
    Q-Risk (z, z') = 2 Σ(i,t) P(zₜ (i), zₜ' (i)) / Σ(i,t) |zₜ' (i)|

    where, P(z, z') = q( z - z' ) if z > z' else (1 - q) ( z' - z )

    Refer this paper https://proceedings.neurips.cc/paper_files/paper/2018/file/5cf68969fb67aa6082363a6d4e6468e2-Paper.pdf
    for the q-risk calculation

    Parameters
    ----------
    quantiles : list
        A list of quantiles to predict. 
    y_actual : np.ndarray
        The true target.
    y_expected : np.ndarray
        The expected target.
    n_targets : int
        Number of target values.

    Returns
    -------
    dict
        A loss map with forecasts for each quantile.
    """
    output_map = {
        'p{}'.format(int(q * 100)):
        y_expected[Ellipsis, i * n_targets:(i + 1) * n_targets]
        for i, q in enumerate(quantiles)
    }
    error_map = {
        key: np.squeeze(y_actual, axis=-1) -
        np.squeeze(output_map[key], axis=-1)
        for key in output_map.keys()
    }

    loss_map = {}

    for q in quantiles:
        key = 'p{}'.format(int(q * 100))

        q_loss = q * tf.maximum(error_map[key], 0.0) + \
            (1.0 - q) * tf.maximum(-error_map[key], 0.0)
        loss_map[key] = 2 * \
            tf.math.reduce_mean(q_loss).numpy() / np.absolute(y_actual).mean()

    return loss_map
