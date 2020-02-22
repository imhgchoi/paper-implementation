import argparse

def get_args():
    argp = argparse.ArgumentParser(description='Graph Convolutional Network',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argp.add_argument('--debug', action='store_true', default=False)

    # directories
    argp.add_argument('--datadir', type=str, default='./data/')

    # data
    argp.add_argument('--reset_data', action='store_true', default=False)
    argp.add_argument('--datatype', type=str, default='cora', choices=['cora','citeseer','pubmed'])

    # model
    argp.add_argument('--reset_model', action='store_true', default=False)
    argp.add_argument('--use_pyg', action='store_true', default=False)
    argp.add_argument('--use_base', action='store_true', default=False)
    argp.add_argument('--conv_layers', nargs='+', default=[32,32])
    argp.add_argument('--fc_layers', nargs='+', default=[])
    argp.add_argument('--save_step', type=int, default=5)

    # training
    argp.add_argument('--model_name', type=str, default='basic')
    argp.add_argument('--epoch_num', type=int, default=1000)
    argp.add_argument('--batchsize', type=int, default=1)
    argp.add_argument('--lr', type=float, default=0.001)
    argp.add_argument('--dropout', type=float, default=0.1)
    argp.add_argument('--plot_step', type=int, default=5)

    return argp.parse_args()