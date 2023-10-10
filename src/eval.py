import json
import os

import numpy as np

from src.config import Config
from src.model import TFTTransformer
from src.paths import define_outputs_dir


def quick_evaluation(model: TFTTransformer,
                     x_test: np.ndarray,
                     x_train: np.ndarray,
                     x_val: np.ndarray,
                     y_test: np.ndarray,
                     y_train: np.ndarray,
                     y_val: np.ndarray,
                     config: Config):
    """
    Perform a quick evaluation, measuring the loss in each of
    the train/val/test set.
    Metrics are saved in the outputs dir for future reference.
    """
    train_loss = model.evaluate(x_train, y_train, batch_size=config.batch_sz)[0]
    val_loss = model.evaluate(x_val, y_val, batch_size=config.batch_sz)[0]
    test_loss = model.evaluate(x_test, y_test, batch_size=config.batch_sz)[0]

    print(f'Fitted model evaluation:')
    print(f'\tTraining set loss: \t\t{train_loss}')
    print(f'\tValidation set loss: \t{val_loss}')
    print(f'\tTest set loss: \t\t\t{test_loss}')
    metrics = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss
    }
    out_dir = define_outputs_dir(config.dataset, config.model_version)
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
