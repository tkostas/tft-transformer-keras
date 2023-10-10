import tensorflow as tf  # type: ignore


class QuantileLoss:
    def __init__(self, quantiles: list, n_targets: int, mask_val: float | None = None):
        self._quantiles = quantiles
        self._n_targets = n_targets
        self._mask_val = mask_val

    def quantile_loss(self, y_true, y_pred):
        """
        Calculate quantile loss. Here, y_true.shape = batch_sz, n_dec_steps, n_targets
        whereas y_pred.shape = batch_sz, n_dec_steps, n_targets * n_quantiles.

        So, in the output you expect sequences of n_targets columns for each quantile.
        """
        loss = 0.0
        for i, q in enumerate(self._quantiles):
            loss += self._q_loss(
                y_true,
                y_pred[..., self._n_targets * i:self._n_targets * (i+1)],
                q
            )

        return loss

    def _q_loss(self, y_true, y_pred, quantile: float):
        """
        For a given quantile, calculate error for each target parameter.
        Optionally, you can mask and omit certain values from the
        loss calculation, as you would do with pad token in NLP.
        """
        assert 0 < quantile < 1
        err = y_true - y_pred

        if self._mask_val is not None:
            mask = tf.cast(tf.math.not_equal(y_true, self._mask_val), tf.float32)
            err *= mask

        # a * (y - y_hat)           for y_hat <= y
        # (1 - a) * (y_hat - y)     for y_hat > y
        q_loss = (quantile * tf.maximum(err, 0.0) +
                  (1.0 - quantile) * tf.maximum(-err, 0.0))

        return tf.reduce_sum(input_tensor=q_loss, axis=-1)


def zeros_loss_fn(y_true, y_pred):
    return tf.reduce_mean(y_pred * 0.0)
