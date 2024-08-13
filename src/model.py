from typing import List

import tensorflow as tf  # type: ignore


class TFTEmbeddingsLayer(tf.keras.layers.Layer):
    def __init__(self, ts_len, d_model, targ_idx,
                 k_num_idx, k_cat_idx, k_cat_vocab_sz,
                 unk_num_idx, unk_cat_idx, unk_cat_vocab_sz,
                 stat_num_idx, stat_cat_idx, stat_cat_vocab_sz):
        super().__init__()
        self.ts_len = ts_len
        self.d_model = d_model
        self.targ_idx = targ_idx
        self.k_num_idx = k_num_idx
        self.k_cat_idx = k_cat_idx
        self.unk_num_idx = unk_num_idx
        self.unk_cat_idx = unk_cat_idx
        self.stat_num_idx = stat_num_idx
        self.stat_cat_idx = stat_cat_idx

        self.k_cat_emb = self.cat_emb_layer(k_cat_vocab_sz)
        self.unk_cat_emb = self.cat_emb_layer(unk_cat_vocab_sz)
        self.stat_cat_emb = self.cat_emb_layer(stat_cat_vocab_sz)
        self.real_to_emb = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.d_model))

    def call(self, inputs, *args, **kwargs):
        unk = self._ts_values(x=inputs,
                              num_inpt=self.unk_num_idx,
                              cat_inpt=self.unk_cat_idx,
                              embeddings=self.unk_cat_emb)
        known = self._ts_values(x=inputs,
                                num_inpt=self.k_num_idx,
                                cat_inpt=self.k_cat_idx,
                                embeddings=self.k_cat_emb)
        targ = self._embed_target_params(x=inputs)
        stat = self._static_emb(x=inputs)

        return targ, unk, known, stat

    def _embed_target_params(self, x):
        return tf.stack([
            self.real_to_emb(x[..., i:i + 1])
            for i in self.targ_idx], axis=-1)

    def cat_emb_layer(self, n_cat):
        """
        Create a categorical embeddings layer for the given number of categories.

        Parameters
        ----------
        n_cat: list
            A list specifying the number of discrete categories,
            per categorical column.

        Returns
        -------
        list
            A list of embeddings layers where each element corresponds to
            one categorical column.
        """
        emb = []
        if n_cat is not None and len(n_cat) > 0:
            for n_cls in n_cat:
                em = tf.keras.Sequential([
                    tf.keras.layers.InputLayer([self.ts_len]),
                    tf.keras.layers.Embedding(
                        n_cls,
                        self.d_model,
                        input_length=self.ts_len,
                        dtype=tf.float32)
                ])
                emb.append(em)
        return emb

    def _ts_values(self, x, num_inpt, cat_inpt=None, embeddings=None):
        """
        Extract values from known/unknown inputs
        and concatenate to a 2d array.
        """
        ts_emb = []
        if cat_inpt is not None:
            for i, idx in enumerate(cat_inpt):
                emb = embeddings[i](x[..., idx:idx + 1])
                ts_emb.append(emb)

        if num_inpt is not None:
            for i in num_inpt:
                emb = self.real_to_emb(x[..., i:i + 1])
                ts_emb.append(emb)

        if len(ts_emb) > 0:
            out = tf.stack(ts_emb, axis=-1)
        else:
            out = None

        # shape batch_sz, ts_len, d_model, n_inpt or None
        return out

    def _static_emb(self, x):
        stat_emb = []
        if self.stat_cat_idx:
            for i, c_i in enumerate(self.stat_cat_idx):
                emb = self.stat_cat_emb[i](x[:, 0, c_i:c_i+1])
                stat_emb.append(emb)

        if self.stat_num_idx:
            for i in self.stat_num_idx:
                emb = self.real_to_emb(
                    tf.expand_dims(x[:, 0, i:i+1], axis=1))
                stat_emb.append(emb)

        if len(stat_emb) > 0:
            out = tf.concat(stat_emb, axis=1)
        else:
            out = None

        return out


class StemLayer(tf.keras.layers.Layer):
    """Extracts historical, future and static features."""

    def __init__(self, *, ts_len, n_dec_steps, d_model, targ_idx,
                 k_num_idx, k_cat_idx, k_cat_vocab_sz,
                 unk_num_idx, unk_cat_idx, unk_cat_vocab_sz,
                 stat_num_idx, stat_cat_idx, stat_cat_vocab_sz):
        super().__init__()
        self.embeddings = TFTEmbeddingsLayer(
            ts_len=ts_len,
            d_model=d_model,
            targ_idx=targ_idx,
            k_num_idx=k_num_idx,
            k_cat_idx=k_cat_idx,
            k_cat_vocab_sz=k_cat_vocab_sz,
            unk_num_idx=unk_num_idx,
            unk_cat_idx=unk_cat_idx,
            unk_cat_vocab_sz=unk_cat_vocab_sz,
            stat_num_idx=stat_num_idx,
            stat_cat_idx=stat_cat_idx,
            stat_cat_vocab_sz=stat_cat_vocab_sz)
        self.ts_len = ts_len
        self.n_dec_steps = n_dec_steps

    def call(self, inputs, *args, **kwargs):
        targ, unk, knowns, stat = self.embeddings(inputs)

        historical_inpt = self._hist_inputs(unk, knowns, targ)
        future_inpt = self._fut_inputs(knowns)

        return historical_inpt, future_inpt, stat

    def _hist_inputs(self, unk, knowns, targ):
        n_enc_steps = self.ts_len - self.n_dec_steps
        # you have only inputs with known values in the past and future
        if unk is None and knowns is not None:
            hist_inpt = tf.concat([
                knowns[:, :n_enc_steps, :],
                targ[:, :n_enc_steps, :]
            ], axis=-1)
        # you have only inputs with known values until the point of forecast
        elif unk is not None and knowns is None:
            hist_inpt = tf.concat([
                unk[:, :n_enc_steps, :],
                targ[:, :n_enc_steps, :]
            ], axis=-1)
        # you only know values of the properties you try to predict
        elif unk is None and knowns is None:
            hist_inpt = targ[:, :n_enc_steps, :]
        # all possible combinations
        else:
            hist_inpt = tf.concat([
                unk[:, :n_enc_steps, :],
                knowns[:, :n_enc_steps, :],
                targ[:, :n_enc_steps, :]
            ], axis=-1)

        return hist_inpt

    def _fut_inputs(self, knowns):
        # you have only inputs with known values in the past and future
        if knowns is not None:
            return knowns[:, -self.n_dec_steps:, :]


def linear_layer(size, activation=None, time_distributed=False, use_bias=True):
    lin_layer = tf.keras.layers.Dense(
        size, activation=activation, use_bias=use_bias)
    if time_distributed:
        lin_layer = tf.keras.layers.TimeDistributed(lin_layer)
    return lin_layer


class GRNLayer(tf.keras.layers.Layer):
    """Gated residual network layer."""

    def __init__(self, d_model, output_sz=None, dropout_rate=None,
                 time_distributed=True):
        super().__init__()
        if output_sz is None:
            output_sz = d_model
            self._pass_through_dense = False
        else:
            self._pass_through_dense = True

        if time_distributed:
            self.dense1 = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(output_sz))
        else:
            self.dense1 = tf.keras.layers.Dense(output_sz)

        self.lin_layer1 = linear_layer(d_model,
                                       activation=None,
                                       time_distributed=time_distributed)
        self.lin_add_context = linear_layer(d_model,
                                            activation=None,
                                            time_distributed=time_distributed,
                                            use_bias=False)
        self.lin_layer2 = linear_layer(d_model,
                                       activation=None,
                                       time_distributed=time_distributed)
        self.gating_layer = GLULayer(output_sz,
                                     dropout_rate=dropout_rate,
                                     time_distributed=time_distributed)
        self.elu_activ = tf.keras.layers.Activation('elu')
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs, additional_context=None, **kwargs):
        if not self._pass_through_dense:
            skip = inputs
        else:
            skip = self.dense1(inputs)

        x = self.lin_layer1(inputs)

        if additional_context is not None:
            add_context = self.lin_add_context(additional_context)
            x = tf.keras.layers.Add()([x, add_context])

        x = self.elu_activ(x)
        x = self.lin_layer2(x)
        gated_x = self.gating_layer(x)
        out = tf.keras.layers.Add()([skip, gated_x])
        out = self.layer_norm(out)
        return out


class GLULayer(tf.keras.layers.Layer):
    """
    Gated linear unit layer. See https://arxiv.org/abs/1612.08083
    """

    def __init__(self, d_model, dropout_rate=0.0, time_distributed=True, activation=None):
        super().__init__()
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        if time_distributed:
            self.activation_layer = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(d_model, activation=activation))
            self.gate = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(d_model, activation='sigmoid'))
        else:
            self.activation_layer = tf.keras.layers.Dense(
                d_model, activation=activation)
            self.gate = tf.keras.layers.Dense(d_model, activation='sigmoid')

    def call(self, x, **kwargs):
        x = self.dropout_layer(x)
        activ_out = self.activation_layer(x)
        gate_out = self.gate(x)

        return tf.keras.layers.Multiply()([activ_out, gate_out])


class LSTMBlock(tf.keras.layers.Layer):
    """
    Performs variable selection and passes values
    through LSTM layer(s).
    """

    def __init__(self, *, n_feat, d_model=64, dropout_rate=0.1):
        super().__init__()
        self.grn1 = GRNLayer(d_model=d_model,
                             output_sz=n_feat,
                             dropout_rate=dropout_rate,
                             time_distributed=True)
        self.emb_grns = [GRNLayer(d_model=d_model,
                                  dropout_rate=dropout_rate,
                                  time_distributed=True) for _ in range(n_feat)]
        self.lstm = tf.keras.layers.LSTM(d_model,
                                         return_sequences=True,
                                         return_state=True)

    def call(self, inputs, additional_context=None, state_h=None, state_c=None, **kwargs):
        _, time_steps, emb_dim, n_inpt = inputs.shape
        flatten = tf.reshape(inputs, [-1, time_steps, emb_dim * n_inpt])

        emb_list = []
        for i in range(n_inpt):
            grn_out = self.emb_grns[i](inputs[..., i])
            emb_list.append(grn_out)
        transf_emb = tf.stack(emb_list, axis=-1)
        # transf_emb.shape -> batch_sz, n_timesteps, d_model, n_feat

        if additional_context is not None:
            static_context = tf.expand_dims(additional_context, axis=1)
        else:
            static_context = None

        x = self.grn1(flatten, additional_context=static_context)

        # importance scores
        x_weights = tf.keras.layers.Activation('softmax')(x)
        x_weights = tf.expand_dims(x_weights, axis=2)
        # x_weights.shape -> batch_sz, n_timesteps, 1, n_feat

        combined = tf.keras.layers.Multiply()([x_weights, transf_emb])
        temporal_ctx = tf.keras.backend.sum(combined, axis=-1)
        # temporal_ctx.shape -> batch_sz, n_timesteps, d_model

        # pass through LSTM
        if state_h is None or state_c is None:
            lstm_out, new_st_h, new_st_c = self.lstm(temporal_ctx)
        else:
            lstm_out, new_st_h, new_st_c = self.lstm(
                temporal_ctx, initial_state=[state_h, state_c])

        # lstm_out.shape -> batch_sz, n_timesteps, d_model
        # temporal_ctx -> batch_sz, n_timesteps, d_model
        # x_weights -> batch_sz, n_timesteps, 1, n_features
        # new_st_h -> batch_sz, d_model
        # new_st_c -> batch_sz, d_model
        return lstm_out, temporal_ctx, x_weights, new_st_h, new_st_c


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, *, d_model, ts_len, num_heads=4, dropout_rate=0.1):
        super().__init__()
        self.temporal_feat_layer_norm = tf.keras.layers.LayerNormalization()
        self.enriched_grn = GRNLayer(
            d_model, dropout_rate=dropout_rate, time_distributed=True)
        self.attn_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=ts_len, dropout=dropout_rate)
        self.transf_layer_norm = tf.keras.layers.LayerNormalization()
        self.gating_layer = GLULayer(
            d_model=d_model, dropout_rate=dropout_rate, activation=None)

    def call(self, inputs, static_context=None, **kwargs):
        inputs = self.temporal_feat_layer_norm(inputs)
        # inputs.shape = batch_sz, ts_len, d_model

        if static_context is not None:
            static_context = tf.expand_dims(static_context, axis=1)

        enriched = self.enriched_grn(inputs, additional_context=static_context)
        # enriched.shape -> batch_sz, ts_len, d_model

        mask = self._decoder_mask(enriched)
        # mask.shape -> batch_sz, ts_len, ts_len
        x, self_attn = self.attn_layer(query=enriched,
                                       value=enriched,
                                       key=enriched,
                                       attention_mask=mask,
                                       return_attention_scores=True,
                                       use_causal_mask=True)
        # x.shape -> batch_sz, ts_len, d_model
        # self_attn.shape -> batch_sz, attn_heads, ts_len, ts_len
        attn_weights = tf.keras.backend.mean(self_attn, axis=1)
        # attn_weights.shape -> batch_sz, ts_len, ts_len

        x = self.gating_layer(x)
        x = tf.keras.layers.Add()([x, enriched])
        # x.shape -> batch_sz, ts_len, d_model

        return x, enriched, attn_weights

    @staticmethod
    def _decoder_mask(x):
        """Return causal mask to apply in self-attention layer."""
        len_s = tf.shape(input=x)[1]
        bs = tf.shape(input=x)[:1]
        mask = tf.cumsum(tf.eye(len_s, batch_shape=bs), axis=1)
        return mask


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, dropout_rate=0.1):
        super().__init__()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.layer_norm2 = tf.keras.layers.LayerNormalization()
        self.grn = GRNLayer(
            d_model, dropout_rate=dropout_rate, time_distributed=True)
        self.glu = GLULayer(
            d_model=d_model, dropout_rate=dropout_rate, activation=None)

    def call(self, x, enriched, **kwargs):
        x = self.layer_norm(x)
        x = self.grn(x)
        x = self.glu(x)
        x = tf.keras.layers.Add()([x, enriched])
        x = self.layer_norm2(x)
        # x.shape -> batch_sz, ts_len, d_model

        return x


class StaticBlock(tf.keras.layers.Layer):
    def __init__(self, *, d_model, n_static_feat, dropout_rate=0.1):
        super().__init__()
        self.grn = GRNLayer(d_model=d_model,
                            output_sz=n_static_feat,
                            dropout_rate=dropout_rate,
                            time_distributed=False)
        self.stat_emb_grns = [
            GRNLayer(d_model=d_model,
                     dropout_rate=dropout_rate,
                     time_distributed=False) for _ in range(n_static_feat)
        ]

        self.stat_context_var_sel_grn = GRNLayer(
            d_model=d_model,
            dropout_rate=dropout_rate,
            time_distributed=False)
        self.stat_context_enrich_grn = GRNLayer(
            d_model=d_model,
            dropout_rate=dropout_rate,
            time_distributed=False)
        self.state_h_grn = GRNLayer(
            d_model=d_model,
            dropout_rate=dropout_rate,
            time_distributed=False)
        self.state_c_grn = GRNLayer(
            d_model=d_model,
            dropout_rate=dropout_rate,
            time_distributed=False)

    def call(self, inputs, **kwargs):
        s_weights, s_context, s_context_enrich, st_h, st_c = None, None, None, None, None
        if inputs is not None:
            static_vec, s_weights = self._stat_combine_and_mask(inputs)
            s_context = self.stat_context_var_sel_grn(static_vec)
            s_context_enrich = self.stat_context_enrich_grn(static_vec)
            st_h = self.state_h_grn(static_vec)
            st_c = self.state_c_grn(static_vec)

        # s_weights.shape -> batch_sz, n_stat_feat, 1
        # s_context.shape -> batch_sz, d_model
        # s_context_enrich.shape -> batch_sz, d_model
        # st_h.shape -> batch_sz, d_model
        # st_c.shape -> batch_sz, d_model

        return s_weights, s_context, s_context_enrich, st_h, st_c

    def _stat_combine_and_mask(self, inputs):
        """
        Applies variable selection network on static inputs.

        Parameters
        ----------
        inputs:
            Static inputs matrix.
        """
        _, n_stat, _ = inputs.get_shape().as_list()
        x = tf.keras.layers.Flatten()(inputs)
        x = self.grn(x)
        static_w = tf.keras.layers.Activation('softmax')(x)
        static_w = tf.expand_dims(static_w, axis=-1)
        # sparse_w.shape -> batch_sz, n_static_feat, 1

        emb_list = []
        for i in range(n_stat):
            emb = self.stat_emb_grns[i](inputs[:, i:i+1, :])
            # emb.shape -> batch_sz, 1, d_model

            emb_list.append(emb)

        embeddings = tf.concat(emb_list, axis=1)
        # embeddings.shape -> batch_sz, n_static_feat, d_model

        x = tf.keras.layers.Multiply()([static_w, embeddings])
        # x.shape -> batch_sz, n_static_feat, d_model

        static_vec = tf.keras.backend.sum(x, axis=1)
        # static_vec.shape -> batch_sz, d_model

        return static_vec, static_w


class LearnerLayer(tf.keras.layers.Layer):
    def __init__(self, *, ts_len: int, n_dec_steps: int, d_model: int,
                 n_static_feat: int, n_hist_feat: int, n_fut_feat: int,
                 n_heads=4, dropout_rate=0.1):
        """

        Parameters
        ----------
        ts_len
        n_dec_steps
        d_model: int
            Depth of the model
        n_hist_feat: int
            Number of ts inputs with known values until the time
            of prediction.
        n_fut_feat: int
            Number of ts inputs with known values in the future.
        """
        super().__init__()
        self.ts_len = ts_len
        self.n_dec_steps = n_dec_steps

        self.static_block = StaticBlock(d_model=d_model,
                                        n_static_feat=n_static_feat,
                                        dropout_rate=dropout_rate)

        self.h_lstm_block = LSTMBlock(n_feat=n_hist_feat,
                                      d_model=d_model,
                                      dropout_rate=dropout_rate)
        self.f_lstm_block = LSTMBlock(n_feat=n_fut_feat,
                                      d_model=d_model,
                                      dropout_rate=dropout_rate)
        self.gating_layer = GLULayer(d_model=d_model,
                                     dropout_rate=dropout_rate,
                                     activation=None)

        self.transformer_block = TransformerBlock(
            d_model=d_model, ts_len=ts_len, num_heads=n_heads,
            dropout_rate=dropout_rate)
        self.decoder_block = DecoderBlock(
            d_model=d_model, dropout_rate=dropout_rate)

    def call(self, inputs, **kwargs):
        hist_inputs, future_inputs, static_feat = inputs

        # --- extract static context, if applicable
        stat_w, stat_context, stat_context_enrich, stat_st_h, stat_st_c = (
            self.static_block(static_feat))

        # --- process historical and future features separately
        h_lstm_out, h_feat, h_flags, st_h, st_c = (
            self.h_lstm_block(inputs=hist_inputs,
                              additional_context=stat_context,
                              state_h=stat_st_h,
                              state_c=stat_st_c))
        f_lstm_out, f_feat, f_flags, _, _ = (
            self.f_lstm_block(inputs=future_inputs,
                              additional_context=stat_context,
                              state_h=st_h,
                              state_c=st_c))

        # --- join features together
        x = tf.concat([h_lstm_out, f_lstm_out], axis=1)
        # lstm_out.shape -> batch_sz, ts_len, d_model

        x_emb = tf.concat([h_feat, f_feat], axis=1)
        # inpt_emb.shape -> batch_sz, ts_len, d_model

        x = self.gating_layer(x)
        x = tf.keras.layers.Add()([x, x_emb])

        # --- pass through transformer block
        x, enriched, attn_weights = self.transformer_block(
            inputs=x, static_context=stat_context_enrich)

        x = self.decoder_block(x=x, enriched=enriched)

        return x, attn_weights, stat_w, h_flags, f_flags


class ForecasterLayer(tf.keras.layers.Layer):
    def __init__(self, n_dec_steps: int,
                 n_targets: int,
                 quantiles: List[float],
                 l2_reg: float | None = None):
        """
        Forecast layer.

        Parameters
        ----------
        n_dec_steps: int
            Number of time steps to predict.
        n_targets: int
            Number of target values.
        quantiles: list
            A list of quantiles to predict. Even if are interested
            in the mean value pass as argument [0.5].
        """
        super().__init__()
        self.n_dec_steps = n_dec_steps
        self.quantiles = quantiles

        # this is with quantile loss (original implementation)
        if l2_reg is not None:
            regularizer = tf.keras.regularizers.l2(l2_reg)
        else:
            regularizer = None
        self.multi_point_forcast_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(n_targets * len(self.quantiles), kernel_regularizer=regularizer))

    def call(self, inputs, **kwargs):
        return self.multi_point_forcast_layer(inputs[..., -self.n_dec_steps:, :])


class TFTTransformer(tf.keras.Model):
    def __init__(self, *, ts_len, n_dec_steps, d_model, targ_idx,
                 k_num_idx=None, k_cat_idx=None, k_cat_vocab_sz=None,
                 unk_num_idx=None, unk_cat_idx=None, unk_cat_vocab_sz=None,
                 stat_num_idx=None, stat_cat_idx=None, stat_cat_vocab_sz=None,
                 quantiles=List[float], n_heads=4, dropout_rate=0.1,
                 l2_reg: float | None = None):
        super().__init__()
        """
        Initialize model. The model expects as input a matrix where 
        each column is an input feature and each row a time step and
        generates predictions for a fixed number of time steps ahead. 
        The total length (total number of time steps) includes the number
        of steps in the past (unknown numeric/categorical) and the number 
        of steps in the future to be predicted (known numeric/categorical).
        
        All arguments for unknown/known or static inputs are optional, so 
        you might omit them if no such inputs exist. However at least one
        input with known values in the future is needed (e.g. you could add
        a time-step index feature, if no values exist).
        
        Categorical inputs should be converted to integers, in a similar
        way as you would vectorize text. Consider adding an index value
        for unknown/other category. This conversion of categories to indices
        is expected to be done as part of preprocessing.  
        
        Parameters
        ----------
        ts_len: int
            Length of the time-series. Number of time steps. 
        n_dec_steps: int
            Number of decoder time-steps (number of time-steps ahead to predict).
        d_model: int
            Length of the model. Size of the latent dimension.
        targ_idx: list of integers
            Column index of the feature to be predicted.
        k_num_idx: list of integers
            Column indices of the numeric inputs with known value in the future.
        k_cat_idx: list of integers
            Column indices of the categorical inputs (converted to integers) with 
            known values in the future.
        k_cat_vocab_sz:  list of integers
            Number of unique categories per categorical input feature with known 
            values in the future. 
        unk_num_idx: list of integers
            Column indices for numerics inputs with known values until the time
            of prediction and unknown in the future.
        unk_cat_idx: list of integers
            Column indices for categorical inputs with known values until the time
            of prediction and unknown in the future.
        unk_cat_vocab_sz: list of integers
            Number of unique categories per categorical input feature with known
            values until the time of prediction and unknown in the future.
        stat_num_idx: list of integers
            Column indices for numeric inputs with fixed (static) value throughout
            the time-series length. 
        stat_cat_idx: list of integers
            Column indices for categorical inputs with fixed (static) value 
            throughout the time-series length.
        stat_cat_vocab_sz: list of integers
            Number of unique categories per categorical static input feature.
        quantiles: list of floats
            Quantiles to generate predictions for (e.g. [0.1, 0.5, 0.9]. 
            To predict only the mean value use [0.5].
        n_heads: int
            Number of attention heads. 
        dropout_rate: float
            Dropout rate to use throughout the model.
        l2_reg: float
            L2 regularization value to be used in the forecaster block of the model.
        """
        self.n_timesteps = ts_len
        self.n_dec_steps = n_dec_steps
        self.d_model = d_model

        # calculate number of ts inputs per category
        n_snum = len(stat_num_idx) if stat_num_idx is not None else 0
        n_scat = len(stat_cat_idx) if stat_cat_idx is not None else 0
        n_knum = len(k_num_idx) if k_num_idx is not None else 0
        n_kcat = len(k_cat_idx) if k_cat_idx is not None else 0
        n_unknum = len(unk_num_idx) if unk_num_idx is not None else 0
        n_unkcat = len(unk_cat_idx) if unk_cat_idx is not None else 0
        n_targ = len(targ_idx)

        # make sure that if categorical values exist, you have also the "vocab_size"
        # for each of them, so that you can create the embeddings layers.
        if stat_cat_idx is not None:
            assert len(stat_cat_idx) <= len(stat_cat_vocab_sz)
        if k_cat_idx is not None:
            assert len(k_cat_idx) <= len(k_cat_vocab_sz)
        if unk_cat_idx is not None:
            assert len(unk_cat_idx) <= len(unk_cat_vocab_sz)
        if k_num_idx is None and k_cat_idx is None:
            raise AttributeError('You need to pass at least one known numeric '
                                 'or categorical feature in the future. At minimum, '
                                 'create a dummy one, so that the forecast horizon '
                                 'is inferred from the length of the series. It can '
                                 'also be the timestep index ahead.')

        n_static_feat = n_snum + n_scat
        n_fut_feat = n_knum + n_kcat
        n_hist_feat = n_fut_feat + n_unknum + n_unkcat + n_targ

        # define macro components
        self.stem = StemLayer(
            ts_len=ts_len,
            n_dec_steps=n_dec_steps,
            d_model=d_model,
            targ_idx=targ_idx,
            k_num_idx=k_num_idx,
            k_cat_idx=k_cat_idx,
            k_cat_vocab_sz=k_cat_vocab_sz,
            unk_num_idx=unk_num_idx,
            unk_cat_idx=unk_cat_idx,
            unk_cat_vocab_sz=unk_cat_vocab_sz,
            stat_num_idx=stat_num_idx,
            stat_cat_idx=stat_cat_idx,
            stat_cat_vocab_sz=stat_cat_vocab_sz)

        self.learner = LearnerLayer(ts_len=ts_len,
                                    n_dec_steps=n_dec_steps,
                                    d_model=d_model,
                                    n_static_feat=n_static_feat,
                                    n_hist_feat=n_hist_feat,
                                    n_fut_feat=n_fut_feat,
                                    n_heads=n_heads,
                                    dropout_rate=dropout_rate)
        self.forecaster = ForecasterLayer(n_dec_steps=n_dec_steps,
                                          quantiles=quantiles,
                                          n_targets=n_targ,
                                          l2_reg=l2_reg)

    def call(self, inputs, **kwargs):
        hist_inputs, future_inputs, static_feat = self.stem(inputs)
        # hist_inputs.shape -> batch_sz, n_hist_time_steps, d_model, n_hist_feat
        # future_inputs.shape -> batch_sz, n_fut_time_steps, d_model, n_fut_feat

        x, attn_weights, static_vip, h_flags, f_flags = (
            self.learner((hist_inputs, future_inputs, static_feat)))
        # x.shape -> batch_sz, ts_len, d_model

        forecast = self.forecaster(x)
        # forecast.shape -> batch_sz, n_dec_steps, n_targets

        outputs = {
            'y': forecast,
            'attn_w': attn_weights,
            'h_w': h_flags,
            'f_w': f_flags
        }
        if static_vip is None:
            outputs['s_w'] = tf.constant([[0.0]])
        else:
            outputs['s_w'] = static_vip[..., 0]

        return outputs
