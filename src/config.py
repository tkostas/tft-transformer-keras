class Paths:
    data_base_dir = 'data'


class Config:
    """General configuration file."""
    def __init__(self, args):
        # general configuration
        self.dataset = args.dataset
        self.model_version = args.model_version
        # model initialization
        self.n_enc_steps = args.n_enc_steps
        self.n_dec_steps = args.n_dec_steps
        self.d_model = args.d_model
        self.quantiles = [0.1, 0.5, 0.9]
        self.ts_len = args.n_enc_steps + args.n_dec_steps
        self.load_model_weights = args.load_model_weights
        # training
        self.sample_sz = args.sample_sz if args.sample_sz > 0 else None
        self.epochs = args.epochs
        self.batch_sz = args.batch_sz
        self.lr = args.lr
        self.clip_norm = args.clip_norm
        self.clip_value = args.clip_value
        self.dropout_rate = args.dropout_rate
        self.l2_reg = args.l2_reg
        self.optimizer = args.optimizer
        self.masked_value = args.masked_value
        # evaluation
        self.n_samples_to_plot = args.n_samples_to_plot if args.n_samples_to_plot > 0 else 50
        self.plot_attn_weights = args.plot_attn_weights
        # callbacks
        self.log_dir = args.log_dir
