import argparse

def get_args():
    argp = argparse.ArgumentParser(description='Transformer',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argp.add_argument('--debug', action='store_true', default=False)
    argp.add_argument('--debug_num', type=int, default=100)

    # directories
    argp.add_argument('--fr_dir', type=str, default='./data/europarl-v7.fr-en.fr')
    argp.add_argument('--en_dir', type=str, default='./data/europarl-v7.fr-en.en')
    argp.add_argument('--out_dir', type=str, default='./out/')

    # preprocess
    argp.add_argument('--data_num', type=int, default=15000)
    argp.add_argument('--seed', type=int, default=1)

    # model
    argp.add_argument('--emb_dim', type=int, default=512)
    argp.add_argument('--N', type=int, default=6)
    argp.add_argument('--headnum', type=int, default=8)  # emb_dim % headnum == 0
    argp.add_argument('--enc_dropout', type=float, default=0.1)
    argp.add_argument('--dec_dropout', type=float, default=0.1)
    argp.add_argument('--reset_model', action='store_true', default=False)
    argp.add_argument('--model_name', type=str, default='transformer2')

    # train
    argp.add_argument('--epoch_num', type=int, default=100)
    argp.add_argument('--batchsize', type=int, default=50)
    argp.add_argument('--learning_rate', type=float, default=1e-5)
    argp.add_argument('--use_dynamic_lr', action='store_true', default=False)
    argp.add_argument('--warmup_steps', type=int, default=4000)   # for dynamic lr
    argp.add_argument('--lr_scale', type=int, default=1e+5)   # for dynamic lr
    argp.add_argument('--print_step', type=int, default=10)
    argp.add_argument('--save_step', type=int, default=10)
    argp.add_argument('--test_step', type=int, default=5)
    argp.add_argument('--hard_labels', action='store_true', default=False)  # to turn off label smoothing
    argp.add_argument('--smooth_rate', type=float, default=0.1)  # for label smoothing

    return argp.parse_args()