import os

import numpy as np
import seaborn as sns  # type: ignore
from matplotlib import pyplot as plt

from src.paths import define_outputs_dir


def plot_history(history, dataset_name, model_version):
    fig, ax = plt.subplots()
    epoch = np.array(history.epoch) + 1
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    ax.plot(epoch, loss, label='loss', color='blue')
    ax.plot(epoch, val_loss, label='val loss', color='orange')
    ax.set_title('Loss over epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    plt.tight_layout()
    out_dir = define_outputs_dir(dataset_name, model_version)
    plt.savefig(os.path.join(out_dir, 'loss_over_epochs.png'))
    plt.show()


def plot_examples(model, x, y_real, quantiles, dataset_name: str, tag: str,
                  plot_n_samples: int = 50,
                  plot_attn_weights: bool = False):
    if plot_n_samples > x.shape[0]:
        plot_n_samples = x.shape[0]

    plot_out_dir = f'plots/{dataset_name}/{tag}/'
    print(f'Creating plots for {tag} -> saved under {plot_out_dir}')
    model_output = model.predict(x[:plot_n_samples])
    if not os.path.exists(plot_out_dir):
        os.makedirs(plot_out_dir)

    for sample_idx in range(plot_n_samples):
        if plot_attn_weights:
            fig, axes = plt.subplots(2, 1, figsize=(12, 12), height_ratios=[1, 3])
            _plot_ts_data(ax=axes[0],
                          quantiles=quantiles,
                          sample_idx=sample_idx,
                          x=x,
                          y_real=y_real,
                          model_output=model_output)
            _plot_attn_weights(ax=axes[1],
                               model_output=model_output,
                               sample_idx=sample_idx)
        else:
            fig, ax = plt.subplots(figsize=(12, 5))
            _plot_ts_data(ax=ax,
                          quantiles=quantiles,
                          sample_idx=sample_idx,
                          x=x,
                          y_real=y_real,
                          model_output=model_output)
        plt.savefig(os.path.join(plot_out_dir, f'{sample_idx}.png'))
        plt.close()


def _plot_attn_weights(ax, model_output, sample_idx):
    attn_mat = model_output['attn_w'][sample_idx]
    ts_len = attn_mat.shape[0]
    y_pred = model_output['y'][sample_idx]
    n_dec_steps = y_pred.shape[0]
    sns.heatmap(attn_mat, ax=ax, cmap='Blues')
    ax.hlines(y=(ts_len - n_dec_steps), xmin=0, xmax=ts_len, color='black')
    ax.vlines(x=(ts_len - n_dec_steps), ymin=0, ymax=ts_len, color='black')
    ax.set_title('Attention weights time step over time steps')
    plt.tight_layout()


def _plot_ts_data(ax, quantiles, sample_idx, x, y_real, model_output):
    ts_len = x[sample_idx].shape[0]
    n_dec_steps = y_real[sample_idx].shape[0]
    n_enc_steps = ts_len - n_dec_steps
    x_in = x[sample_idx][:n_enc_steps, 0]
    y_real = x[sample_idx][n_enc_steps:, 0].reshape(-1)
    y_pred = model_output['y'][sample_idx]
    ax.hlines(y=0, xmin=0, xmax=ts_len, linestyles='dashed', color='black')
    ax.plot([i for i in range(n_enc_steps)], x_in, color='blue', label='x_in')
    ax.plot([i + n_enc_steps for i in range(n_dec_steps)], y_real, color='green', label='y_real')
    line_styles = ['dashed', 'solid', 'dotted']
    for j, q in enumerate(quantiles):
        ax.plot([i + n_enc_steps for i in range(n_dec_steps)], y_pred[:, j],
                linestyle=line_styles[j % 3], color='orange', label=f'y_pred q={q}')
    ax.legend()
    ax.set_xlabel('time-steps')
    ax.set_ylabel('y value (normalized)')
    ax.set_title('y ~ time')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
