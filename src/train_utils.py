import os

import tensorflow as tf  # type: ignore

from src.loss import QuantileLoss, zeros_loss_fn  # type: ignore
from src.model import TFTTransformer
from src.paths import define_outputs_dir


def save_weights_and_inference_model(model, config):
    base_out_dir = define_outputs_dir(
        dataset_name=config.dataset, model_version=config.model_version
    )
    model_weights_dir = os.path.join(
        base_out_dir, 'model_weights', f'{config.model_version}/model_weights'
    )
    export_dir = os.path.join(
        base_out_dir, 'exported_model', str(config.model_version))
    model.save_weights(model_weights_dir)
    model.export(export_dir)  # use model.serve(x) for inference


def build_model(config, dset, x_train):
    model = TFTTransformer(
        ts_len=config.ts_len,
        n_dec_steps=config.n_dec_steps,
        d_model=config.d_model,
        targ_idx=dset.targ_idx,
        k_num_idx=dset.known_num_idx,
        k_cat_idx=dset.known_cat_idx,
        k_cat_vocab_sz=dset.known_cat_vocab_sz,
        unk_num_idx=dset.unknown_num_idx,
        unk_cat_idx=dset.unknown_cat_idx,
        unk_cat_vocab_sz=dset.unknown_cat_vocab_sz,
        stat_num_idx=dset.static_num_idx,
        stat_cat_idx=dset.static_cat_idx,
        stat_cat_vocab_sz=dset.static_cat_vocab_sz,
        quantiles=config.quantiles,
        dropout_rate=config.dropout_rate,
        l2_reg=config.l2_reg,
    )
    model.build(input_shape=x_train.shape)
    if config.load_model_weights is not None:
        weights_path = config.load_model_weights
        if weights_path.endswith('/'):
            weights_path = weights_path[:-1]
        weights_fp = weights_path + '/model_weights'
        print(f'Loading model weights from: {weights_fp}')
        model.load_weights(weights_fp)

    callbacks = []
    if config.log_dir is not None:
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                os.path.join(config.log_dir, config.dataset), histogram_freq=1
            )
        )
    n_targets = len(dset.targ_idx)
    quantile_loss = QuantileLoss(
        quantiles=config.quantiles, n_targets=n_targets, mask_val=config.masked_value
    ).quantile_loss

    losses = {
        'y': quantile_loss,
        'attn_w': zeros_loss_fn,
        'h_w': zeros_loss_fn,
        'f_w': zeros_loss_fn,
        's_w': zeros_loss_fn,
    }
    optimizer_factory = {
        'sgd': tf.optimizers.SGD,
        'rmsprop': tf.optimizers.RMSprop,
        'adam': tf.optimizers.Adam,
        'adamw': tf.optimizers.AdamW,
        'adagrad': tf.optimizers.Adagrad,
        'adamax': tf.optimizers.Adamax,
        'adafactor': tf.optimizers.Adafactor,
        'nadam': tf.optimizers.Nadam,
        'ftrl': tf.optimizers.Ftrl,
    }
    optimizer = optimizer_factory[config.optimizer](
        learning_rate=config.lr, clipnorm=config.clip_norm, clipvalue=config.clip_value
    )
    model.compile(optimizer=optimizer, loss=losses,
                  sample_weight_mode='temporal')

    model.summary()
    return callbacks, model
