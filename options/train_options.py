from options.base_options import BaseOptions, parse_bool
import argparse

class TrainCompOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--num_layers', type=int, default=8, help='num_layers of transformer')
        self.parser.add_argument('--latent_dim', type=int, default=512, help='latent_dim of transformer')
        self.parser.add_argument('--diffusion_steps', type=int, default=1000, help='diffusion_steps of transformer')
        self.parser.add_argument('--no_clip', action='store_true', help='whether use clip pretrain')
        self.parser.add_argument('--no_eff', action='store_true', help='whether use efficient attention')

        self.parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
        self.parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
        self.parser.add_argument('--reset_lr', action='store_true', help='Reset the optimizer lr to args.lr after resume from a ckpt')
        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
        self.parser.add_argument('--times', type=int, default=1, help='times of dataset')
        self.parser.add_argument(
            '--persistent_workers', type=parse_bool, default=False,
            help='Keep DataLoader workers alive between epochs (recommended on Linux when workers > 0)',
        )
        self.parser.add_argument(
            '--prefetch_factor', type=int, default=2,
            help='Number of batches prefetched by each DataLoader worker',
        )
        self.parser.add_argument(
            '--non_blocking_transfer', type=parse_bool, default=True,
            help='Use asynchronous host-to-device copies for pinned DataLoader tensors',
        )

        self.parser.add_argument('--feat_bias', type=float, default=5, help='Scales for global motion features and foot contact')
        self.parser.add_argument('--flow_matching', action='store_true', help='Use flow matching instead of DDPM')
        self.parser.add_argument('--fm_sample_steps', type=int, default=50,
                                 help='Number of ODE integration steps for FM sampling (default: 50)')
        self.parser.add_argument('--fm_solver', type=str, choices=['euler', 'rk4'], default='rk4',
                                 help='ODE solver used by Flow Matching sampling')
        self.parser.add_argument('--fm_transition_blend', type=int, default=7,
                                 help='Frames used to blend from an inpainted prefix into newly generated motion')
        self.parser.add_argument(
            '--fm_expression_condition', choices=['auto', 'x0', 'velocity'], default='auto',
            help='Expression signal passed to the gesture branch; auto preserves legacy checkpoints and uses x0 for new training',
        )
        self.parser.add_argument('--grad_accum_steps', type=int, default=1,
                                 help='Number of micro-batches accumulated before each optimizer step')

        self.parser.add_argument('--resume', action="store_true", help='Is this trail continued from previous trail?')
        self.parser.add_argument('--allow_legacy_checkpoint', action='store_true', help='Allow resuming a checkpoint without a saved config (unsafe; legacy FM uses velocity conditioning)')
        self.parser.add_argument('--allow_partial_checkpoint', action='store_true', help='Allow missing/unexpected model keys while loading a checkpoint')

        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress (by iteration)')
        self.parser.add_argument('--save_every_e', type=int, default=5, help='Frequency of saving models (by epoch)')
        self.parser.add_argument(
            '--latest_every_e', type=int, default=1,
            help='Frequency of updating resumable latest.tar (by epoch)',
        )
        self.parser.add_argument('--eval_every_e', type=int, default=5, help='Frequency of animation results (by epoch)')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of saving models (by iteration)')

        
        self.parser.add_argument('--ckpt', type=str, default='latest.tar', help='choose which checkpoint to use')
        # self.parser.add_argument('--max_motion_length', type=int, default=200, help='max_motion_length')

        self.is_train = True
