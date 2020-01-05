import argparse

def get_args():
    argp = argparse.ArgumentParser(description='Transformer',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argp.add_argument('--debug', action='store_true', default=False)
    argp.add_argument('--debug_num', type=int, default=10)

    # directories
    argp.add_argument('--fr_dir', type=str, default='./data/europarl-v7.fr-en.fr')
    argp.add_argument('--en_dir', type=str, default='./data/europarl-v7.fr-en.en')
    argp.add_argument('--out_dir', type=str, default='./out/')

    argp.add_argument('--seed', type=int, default=1)



    return argp.parse_args()