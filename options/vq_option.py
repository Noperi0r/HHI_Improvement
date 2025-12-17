import argparse
import os
import torch
import distutils

def arg_parse(is_train=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader
    parser.add_argument('--dataset_name', type=str, default='interhuman', help='dataset directory')
    parser.add_argument('--motion_rep', type=str, default='smpl', help='how is the motion represented')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--window_size', type=int, default=64, help='training motion length')
    parser.add_argument('--window_stride', type=int, default=10, help='stride to sample the data in windows')
    parser.add_argument("--gpu_id", type=int, default=0, help='GPU id')
    parser.add_argument('--cache', type=distutils.util.strtobool, default=True, help='cache the dataset')

    ## Multi-GPU options
    parser.add_argument('--distributed', action='store_true',
                        help='Enable DistributedDataParallel training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers per process')
    parser.add_argument('--sync_bn', action='store_true',
                        help='Use SyncBatchNorm with DDP')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    ## optimization
    parser.add_argument('--max_epoch', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")

    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument('--recons_loss', type=str, default='l1_smooth', help='reconstruction loss')
    parser.add_argument('--loss_explicit', type=float, default=1, help='hyper-parameter for the explicit loss')#0.1
    parser.add_argument('--loss_vel', type=float, default=100, help='hyper-parameter for the velocity loss')
    parser.add_argument('--loss_bn', type=float, default=5, help='hyper-parameter for the bone length loss')
    parser.add_argument('--loss_geo', type=float, default=0.01, help='hyper-parameter for the geodesic loss') #0.001
    parser.add_argument('--loss_fc', type=float, default=500, help='hyper-parameter for the foot contact loss') #100
        

    ## vqvae arch
    parser.add_argument("--code_dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb_code", type=int, default=1024, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down_t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride_t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=2, help="num of resblocks for each res")
    parser.add_argument("--dilation_growth_rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output_emb_width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq_act', type=str, default='relu', choices=['relu', 'silu', 'gelu'],
                        help='dataset directory')
    parser.add_argument('--vq_norm', type=str, default=None, help='dataset directory')

    parser.add_argument('--num_quantizers', type=int, default=1, help='num_quantizers')
    parser.add_argument('--shared_codebook', action="store_true")
    parser.add_argument('--quantize_dropout_prob', type=float, default=0.2, help='quantize_dropout_prob')

    ## other
    parser.add_argument('--name', type=str, default="vq_default", help='Name of this trial')
    parser.add_argument('--is_continue', action="store_true", help='Name of this trial')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--save_every_e', default=1, type=int, help='save model every n epoch')
    parser.add_argument('--eval_every_e', default=1, type=int, help='save eval results every n epoch')
    parser.add_argument('--feat_bias', type=float, default=5, help='Layers of GRU')

    ## eval
    parser.add_argument('--do_eval', action="store_true", help='Perform evaluations during training')
    parser.add_argument('--test_batch_size', default=96, type=int, help='batch size for evaluation')

    ## wandb
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_entity', type=str, default='', help='Wandb entity (team name)')
    parser.add_argument('--wandb_project', type=str, default='', help='Wandb project name')
    parser.add_argument('--wandb_api_key', type=str, default='', help='Wandb API key for authentication')
    parser.add_argument('--wandb_name', type=str, default=None, help='Custom wandb run name (if not specified, uses default naming convention)')

    opt = parser.parse_args()

    # Multi-GPU 모드에서는 torchrun이 각 프로세스를 특정 GPU에 할당
    # Single GPU 모드에서만 수동으로 설정
    if not getattr(opt, 'distributed', False):
        torch.cuda.set_device(opt.gpu_id)

    args = vars(opt)

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    opt.is_train = is_train
    if is_train:
    # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name, opt.name)
        os.makedirs(expr_dir, exist_ok=True)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
    return opt