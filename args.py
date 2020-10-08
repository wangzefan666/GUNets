import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ppi',
                        help='Data set.')
    parser.add_argument('--warm_batch_size', type=int, default=256,
                        help='Warm up batch size')
    parser.add_argument('--warm_batch_num', type=int, default=0,
                        help='Warm up batch number')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Number of nodes in a batch.')
    parser.add_argument('--output_batch', type=int, default=1,
                        help='Report result every how many batchs.')
    parser.add_argument('--train_seed', type=int, default=42, help='Training Random seed.')
    parser.add_argument('--pre_seed', type=int, default=10, help='Preprocessing Random seed.')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--n_jobs', type=int, default=46,
                        help='Max multiprocessing number.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--if_output', action='store_true', default=False,
                        help='Whether output the accuracies')
    parser.add_argument('--patience', type=int, default=60,
                        help='Number of max patience count.')
    parser.add_argument('--name', type=str, default='warm_tr',
                        help='Data set.')
    parser.add_argument('--recompute', action='store_true', default=False,
                        help='Whether recompute the feature of nodes.')

    # Unfolding
    parser.add_argument('--samp_pare_num', type=int, default=10,
                        help='Number of sampling parents.')
    parser.add_argument('--samp_times', type=int, default=1,
                        help='Total sampling number.')
    parser.add_argument('--samp_num', type=int, default=300,
                        help='The number of nodes in an sampling filter.')
    parser.add_argument('--un_layer', type=int, default=2,
                        help='number of unfolding layers.')
    parser.add_argument('--max_degree', type=int, default=50,
                        help='Max multiprocessing number.')
    parser.add_argument('--max_samp_nei', type=int, default=256,
                        help='Max sampling neighbor number.')
    parser.add_argument('--if_sampling', action='store_true', default=False,
                        help='Whether use Sampling.')
    parser.add_argument('--if_normalized', action='store_true', default=False,
                        help='Whether use raw normalization.')
    parser.add_argument('--degree_normalized', action='store_true', default=False,
                        help='Whether use degree normalization.')
    parser.add_argument('--if_sort', action='store_true', default=False,
                        help='Whether sorted the neighbors by its degree.')
    parser.add_argument('--if_self_loop', action='store_true', default=False,
                        help='Whether use self loop.')
    parser.add_argument('--weight', type=str, default='rw',
                        choices=['same', 'rw'],
                        help='Node aggregation Function.')

    # Model
    parser.add_argument('--emb_size', type=int, default=256,
                        help='Size of embedding layer.')
    parser.add_argument('--if_trans_bn', action='store_true', default=False,
                        help='Whether using bn after the trans')
    parser.add_argument('--if_mlp_bn', action='store_true', default=False,
                        help='Whether using bn after in MLP')
    parser.add_argument('--if_trans_share', action='store_true', default=False,
                        help='Whether using the same trans_layer for each step.')
    parser.add_argument('--if_bn_share', action='store_true', default=False,
                        help='Whether using the same bn layer.')
    parser.add_argument('--trans_act', type=str, default='leaky',
                        choices=['relu', 'leaky', 'sigmoid', 'none', 'tanh'],
                        help='Trans Layer Aggregation Method.')
    parser.add_argument('--mlp_act', type=str, default='leaky',
                        choices=['relu', 'leaky', 'sigmoid', 'none', 'tanh'],
                        help='MLP Layer Aggregation Method.')
    parser.add_argument('--trans_init', type=str, default='xavier',
                        choices=['none', 'xavier', 'kaiming'],
                        help='Trans Layer Aggregation Method.')
    parser.add_argument('--mlp_init', type=str, default='xavier',
                        choices=['none', 'xavier', 'kaiming'],
                        help='Trans Layer Aggregation Method.')
    parser.add_argument('--mlp_size', type=int, default=256,
                        help='Size of mlp layer.')
    parser.add_argument('--mlp_layer', type=int, default=2,
                        help='Number of mlp layer.')
    parser.add_argument('--if_bias', action='store_true', default=False,
                        help='Whether use bias in trans and map.')
    parser.add_argument('--drop_rate', type=float, default=0.2,
                        help='Drop Rate.')
    parser.add_argument('--bn_mom', type=float, default=0.1,
                        help='Batch normalization momentum.')
    parser.add_argument('--run_times', type=int, default=3,
                        help='Size of running times.')
    parser.add_argument('--pre_load', type=str, default='',
                        help='Preloading File')

    args, _ = parser.parse_known_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.run_times > 1:
        args.if_multi = True
    else:
        args.if_multi = False

    if args.samp_times > 1:
        args.if_sampling = True

    if args.dataset in ['ppi', 'yelp', 'amazon']:
        args.if_multi_label = True
    else:
        args.if_multi_label = False

    if args.if_output:
        print('\n'.join([(str(_) + ':' + str(vars(args)[_])) for _ in vars(args).keys()]))

    return args
