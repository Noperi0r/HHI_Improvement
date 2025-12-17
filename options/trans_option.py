from options.base_option import BaseOptions
import argparse

class TrainTransOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batch_size', type=int, default=52, help='Batch size')
        self.parser.add_argument('--max_epoch', type=int, default=500, help='Maximum number of epoch for training')

        '''LR scheduler'''
        self.parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
        self.parser.add_argument('--step_unroll', type=float, default=1, help='Step Unroll masking factor')
        self.parser.add_argument('--interaction_mask_prob', type=float, default=0.2, help='Interaction Mask probability')
        self.parser.add_argument('--gamma', type=float, default=1/3, help='Learning rate schedule factor')

        '''Condition'''
        self.parser.add_argument('--cond_drop_prob', type=float, default=0.1, help='Drop ratio of condition, for classifier-free guidance')
        self.parser.add_argument("--seed", default=3407, type=int, help="Seed")

        self.parser.add_argument('--is_continue', action="store_true", help='Is this trial continuing previous state?')
        self.parser.add_argument('--gumbel_sample', action="store_true", help='Strategy for token sampling, True: Gumbel sampling, False: Categorical sampling')

        self.parser.add_argument('--eval_every_e', type=int, default=10, help='Frequency of animating eval results, (epoch)')
        self.parser.add_argument('--eval_start_epoch', type=int, default=0, help='Start epoch for evaluation (0 means from beginning)')

        '''eval'''
        self.parser.add_argument('--do_eval', action="store_true", help='Perform evaluations during training')
        self.parser.add_argument('--test_batch_size', default=96, type=int, help='batch size for evaluation')


        '''Multi GPU'''
        self.parser.add_argument('--distributed', action='store_true',
                                help='Enable DistributedDataParallel training')
        self.parser.add_argument('--num_workers', type=int, default=4,
                                help='DataLoader workers per process')
        self.parser.add_argument('--sync_bn', action='store_true',
                                help='Use SyncBatchNorm with DDP')
        self.parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                                help='Number of steps to accumulate gradients before updating')
        self.parser.add_argument('--use_mixed_precision', action='store_true',
                                help='Enable Mixed Precision (FP16) training for speed (default: False for reproducibility)')

        '''Wandb'''
        self.parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
        self.parser.add_argument('--wandb_entity', type=str, default='', help='Wandb entity (team name)')
        self.parser.add_argument('--wandb_project', type=str, default='', help='Wandb project name')
        self.parser.add_argument('--wandb_api_key', type=str, default='', help='Wandb API key for authentication')
        self.parser.add_argument('--wandb_name', type=str, default=None, help='Custom wandb run name (if not specified, uses default naming convention)')

        self.is_train = True
