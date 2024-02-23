import argparse

from src.config import Config
from src.datasets.managers import datasets_factory
from src.eval import quick_evaluation
from src.plots import plot_history, plot_examples
from src.train_utils import build_model, save_weights_and_inference_model

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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

callback_args.add_argument('--log_dir', default=None,
                           help='Log dir for TensorBoard callback.')

args = parser.parse_args()


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
