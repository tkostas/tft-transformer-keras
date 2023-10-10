import argparse
import os

import tensorflow as tf  # type: ignore

from src.config import Config
from src.datasets.managers import datasets_factory
from src.eval import quick_evaluation
from src.loss import QuantileLoss, zeros_loss_fn  # type: ignore
from src.model import TFTTransformer
from src.paths import define_outputs_dir
from src.plots import plot_history, plot_examples

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

general_config = parser.add_argument_group(
    title='General configuration',
    description='Define general configuration like naming etc.'
)
general_config.add_argument(
    '--dataset',
    default='synthetic',
    help='Dataset name. Is used to select appropriate data and define output paths.')
general_config.add_argument(
    '--model_version',
    default=1,
    help='Version of the model. Is used to define output paths',
    type=int)

model_init_args = parser.add_argument_group(
    title='Model definition params',
    description='Define model specific parameters.'
)
model_init_args.add_argument(
    '--n_enc_steps',
    help='Number of time steps in the encoder (number historical input time steps).',
    type=int)
model_init_args.add_argument(
    '--n_dec_steps',
    help='Number of time steps in the decoder (number of time steps to forecast).',
    type=int)
model_init_args.add_argument(
    '--d_model',
    default=16,
    help='Depth of the model. Defines dimension of hidden layers.',
    type=int)
model_init_args.add_argument(
    '--load_model_weights',
    default=None,
    help='If a path is provided, model will be initialized with these weights. Use this '
         'option to resume training.',
    type=str)

training_args = parser.add_argument_group(
    title='Model training arguments',
    description='Define training configuration.'
)
training_args.add_argument(
    '--sample_sz',
    default=0,
    help='Number of examples to sample. Rather than running the full dataset, select a fraction of it. '
         'Make sure it is greater than the batch size. If set to 0 the full dataset '
         'will be used.',
    type=int)
training_args.add_argument(
    '--epochs',
    default=3,
    help='Number of epochs to train the model.',
    type=int)
training_args.add_argument(
    '--batch_sz',
    default=32,
    help='Batch size.',
    type=int)
training_args.add_argument(
    '--lr',
    default=0.001,
    help='Learning rate.',
    type=float
)
training_args.add_argument(
    '--clip_norm',
    default=None,
    help='Clip norm value to pass in the optimizer.',
    type=float
)
training_args.add_argument(
    '--clip_value',
    default=None,
    help='Clip value to pass in the optimizer.',
    type=float
)
training_args.add_argument(
    '--dropout_rate',
    default=0.1,
    help='Dropout rate to apply.',
    type=float
)
training_args.add_argument(
    '--l2_reg',
    default=None,
    help='L2 regularization value',
    type=float
)
training_args.add_argument(
    '--optimizer',
    default='sgd',
    help='Optimizer name. See Keras docs for available options.',
    choices=[
        'sgd', 'rmsprop', 'adam',
        'adamw', 'adagrad', 'adamax',
        'adafactor', 'nadam', 'ftrl']
)
training_args.add_argument(
    '--masked_value',
    default=None,
    help='Optionally, you can ignore certain values during loss '
         'calculation. For example, if your training data have variable '
         'forecast horizon, you can "pad" the decoder time-steps with a '
         'fixed value and mask the padded time-steps during loss calculation.',
    type=float
)
eval_args = parser.add_argument_group(
    title='Model evaluation arguments',
    description='Define model evaluation process'
)
eval_args.add_argument(
    '--n_samples_to_plot',
    default=50,
    type=int,
    help='Number of examples to plot for verification.'
)
eval_args.add_argument(
    '--plot_attn_weights',
    default=False,
    type=bool,
    help='Include in the plot a heatmap with the attention weights.'
)
callback_args = parser.add_argument_group(
    title='Callback arguments',
    description='Define callback configuration'
)

callback_args.add_argument('--log_dir', default=None, help='Log dir for TensorBoard callback.')

args = parser.parse_args()


def save_weights_and_inference_model(model, config):
    base_out_dir = define_outputs_dir(dataset_name=config.dataset,
                                      model_version=config.model_version)
    model_weights_dir = os.path.join(base_out_dir,
                                     'model_weights',
                                     f'{config.model_version}/model_weights')
    export_dir = os.path.join(base_out_dir,
                              'exported_model',
                              str(config.model_version))
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
        weights_fp = weights_path+'/model_weights'
        print(f'Loading model weights from: {weights_fp}')
        model.load_weights(weights_fp)

    callbacks = []
    if config.log_dir is not None:
        callbacks.append(tf.keras.callbacks.TensorBoard(
            os.path.join(config.log_dir, config.dataset),
            histogram_freq=1))
    n_targets = len(dset.targ_idx)
    quantile_loss = QuantileLoss(
        quantiles=config.quantiles,
        n_targets=n_targets,
        mask_val=config.masked_value).quantile_loss

    losses = {
        'y': quantile_loss,
        'attn_w': zeros_loss_fn,
        'h_w': zeros_loss_fn,
        'f_w': zeros_loss_fn,
        's_w': zeros_loss_fn
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
        'ftrl': tf.optimizers.Ftrl
    }
    optimizer = optimizer_factory[config.optimizer](
        learning_rate=config.lr,
        clipnorm=config.clip_norm,
        clipvalue=config.clip_value)
    model.compile(optimizer=optimizer, loss=losses, sample_weight_mode='temporal')

    model.summary()
    return callbacks, model


def main():
    config = Config(args)

    dataset_name = config.dataset
    model_version = config.model_version
    dset = datasets_factory[dataset_name](ts_len=config.ts_len,
                                          n_enc_steps=config.n_enc_steps,
                                          sample_sz=config.sample_sz)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = dset.extract_xy_pairs(
        ts_len=config.ts_len,
        n_dec_steps=config.n_dec_steps)

    callbacks, model = build_model(config, dset, x_train)

    history = model.fit(x_train, y_train,
                        batch_size=config.batch_sz,
                        epochs=config.epochs,
                        validation_data=(x_val, y_val),
                        shuffle=True,
                        callbacks=callbacks)

    plot_history(history, dataset_name, model_version)

    save_weights_and_inference_model(model, config=config)

    quick_evaluation(model,
                     x_test, x_train,
                     x_val, y_test,
                     y_train, y_val,
                     config)

    plot_examples(model, x_train, y_train,
                  quantiles=config.quantiles,
                  dataset_name=dataset_name,
                  tag='train',
                  plot_n_samples=config.n_samples_to_plot,
                  plot_attn_weights=config.plot_attn_weights)
    plot_examples(model, x_val, y_val,
                  quantiles=config.quantiles,
                  dataset_name=dataset_name,
                  tag='val',
                  plot_n_samples=config.n_samples_to_plot,
                  plot_attn_weights=config.plot_attn_weights)
    plot_examples(model, x_test, y_test,
                  quantiles=config.quantiles,
                  dataset_name=dataset_name,
                  tag='test',
                  plot_n_samples=config.n_samples_to_plot,
                  plot_attn_weights=config.plot_attn_weights)


if __name__ == '__main__':
    main()
