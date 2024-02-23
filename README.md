# Temporal Fusion Transformer (TFT) - Keras implementation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rohanmohapatra/tft-transformer-keras/blob/master/TFT_Transformer_Keras.ipynb)

This is a refactoring of TFT transformer implementation
(see https://github.com/greatwhiz/tft_tf2), with minor modifications,
using Keras, Tensorflow >= 2.13.0 and python >= 3.10.

Goal of this repository is to provide a proof of concept, which you
might adapt to make it working for your dataset. The model instance
can be used as it is, but you need to handle the pre-processing and
post-processing steps.

See also:

- Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting --> https://arxiv.org/pdf/1912.09363.pdf
- https://github.com/google-research/google-research/blob/master/tft/libs/tft_model.py

## Inputs

Model expects as input a matrix with input shape
(`batch_size`, `n_time_steps`, `n_features`), where the total number
of time steps includes both the number of encoder and decoder steps.
It is expected that at least one input feature extends in the future,
until the end of the series. If no such input exist, you might create
one during the preprocessing phase, e.g. a time-step index value or
a fixed numeric value.

## Outputs

Output of the model is a dictionary with the following keys:

- `y`: Predicted values (size: `n_dec_steps`, `n_targets` \* `n_quantiles`)
- `attn_w`: Attention weights (size: `ts_len`, `ts_len`)
- `h_w`: Historical features weights (size: `n_enc_steps`, 1, `n_hist_feat`)
- `f_w`: Future features weights (size: `n_dec_steps`, 1, `n_future_feat`)
- `s_w`: Static features weights (size: `n_stat_features`)

## Model weights and export

Model weights are saved in the `outputs` directory, so that you can
resume training in the future, by initializing the model and loading its weights
(see `--load_model_weights` cli argument).

For model inference, you can use the exported model and run inference using the
`.serve` method (see https://www.tensorflow.org/guide/keras/serialization_and_saving#simple_exporting_with_export).

## Requirements

- python >= 3.10
- tensorflow >= 2.13.0

## Running examples

You can try the model using the dataset below. The electicity and traffic
datasets will be downloaded from UCI ML repository. For the favorita dataset
you need to manually download the data from Kaggle
(https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data)
and placed them under `data/favorita`.

1. Hourly electricity consumption dataset (source UCI ML repository)
   `python train.py --dataset electricity --epochs 50 --n_enc_steps 168
--n_dec_steps 24 --d_model 8 --log_dir logs --clip_value 2 --batch_sz 64 
--lr 0.001 --dropout_rate 0.2 --optimizer adamw
`

2. Run traffic dataset (source UCI ML repository - https://archive.ics.uci.edu/dataset/204/pems+sf)
   `python train.py --dataset traffic --epochs 50 --n_enc_steps 252 --n_dec_steps 25
--d_model 16 --log_dir logs --clip_value 2 --batch_sz 64
--lr 0.001 --optimizer adamw`

3. Run favorita dataset
   `python train.py --dataset favorita --epochs 50 --n_enc_steps 60 --n_dec_steps 20 
--d_model 64 --log_dir logs --clip_value 2 --batch_sz 64 --lr 0.001 --optimizer adamw`

## CLI Arguments

General configuration:

- `--dataset` Dataset name. Is used to select appropriate data and define output paths. (default: synthetic)
- `--model_version` Version of the model. Is used to define output paths (default: 1)

Model definition params:

- `--n_enc_steps` Number of time steps in the encoder (number historical input time steps). (default: None)
- `--n_dec_steps` Number of time steps in the decoder (number of time steps to forecast). (default: None)
- `--d_model` Depth of the model. Defines dimension of hidden layers. (default: 16)
- `--load_model_weights` If a path is provided, model will be initialized with these weights. Use this option to resume training. (default: None)

Model training arguments:
Define training configuration.

- `--sample_sz` Number of examples to sample. Rather than running the full dataset, select a fraction of it. Make sure it is greater than the batch size. If set to 0 the full dataset will be used. (default: 0)
- `--epochs` Number of epochs to train the model. (default: 3)
- `--batch_sz` Batch size. (default: 32)
- `--lr` Learning rate. (default: 0.001)
- `--clip_norm` Clip norm value to pass in the optimizer. (default: None)
- `--clip_value` Clip value to pass in the optimizer. (default: None)
- `--dropout_rate` Dropout rate to apply. (default: 0.1)
- `--l2_reg` L2 regularization value (default: None)
- `--optimizer` `{sgd,rmsprop,adam,adamw,adagrad,adamax,adafactor,nadam,ftrl}`
  Optimizer name. See Keras docs for available options. (default: `sgd`)
- `--masked_value` Optionally, you can ignore certain values during loss
  calculation. For example, if your training data have variable
  forecast horizon, you can "pad" the decoder time steps with a fixed
  value and mask the padded time-steps during loss calculation.

Model evaluation arguments:

- `--n_samples_to_plot` Number of examples to plot for verification. (default: 50)
- `--plot_attn_weights` Include in the plot a heatmap with the attention weights.

Callback arguments:

- `--log_dir` Log dir for TensorBoard callback. (default: None)
