import os
import datetime

import torch
from argparse import ArgumentParser

from training import Training

parser = ArgumentParser()

parser.add_argument("--dataset", type=str, default="8gaussians", help="Training dataset [8gaussians/mnist/cityscapes]")
parser.add_argument("--nn_type", type=str, default="mlp", help="Type of NN in the coupling layers [mlp/densenet]")
parser.add_argument("--densenet_depth", type=int, default=1, help="If type of NN is 'densenet': number of dense blocks")

parser.add_argument("--n_hidden_nn", type=int, default=256, help="Number of hidden units in coupling layers' NN")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
parser.add_argument("--net_epochs", type=int, default=100,
                    help="Number of epochs to train the NN in coupling or splitprior layer")
parser.add_argument("--prior_epochs", type=int, default=30, help="Number of epochs to train the prior")
parser.add_argument("--k_sort", type=int, help="Parameter k in the discrete denoising coupling layer")


parser.add_argument("--with_splitprior", type=bool, default=True, help="Whether to train the model with splitpriors")
parser.add_argument("--save_model", type=bool, default=True, help="Whether to save the trained model")
parser.add_argument("--save_plots", type=bool, default=True, help="Whether to save the model outputs as plots")

args = parser.parse_args()

if __name__ == "__main__":
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # save results
    args.results_dir = 'results'
    run_dir = os.path.join('results', datetime.datetime.now().strftime("%m%d%Y_%H%M%S"))
    args.model_dir = os.path.join(run_dir, 'model')
    args.plots_dir = os.path.join(run_dir, 'plots')
    for dir_ in [args.results_dir, run_dir, args.model_dir, args.plots_dir]:
        if not os.path.isdir(dir_):
            os.mkdir(dir_)

    if args.dataset == '8gaussians':
        args.input_size = [2, 1, 1]
        args.num_classes = 91
        args.nn_type = 'mlp'

        training = Training(args)
        training.train_2D(args)

    elif args.dataset == 'mnist':
        args.input_size = [1, 28, 28]
        args.num_classes = 2
        args.num_building_blocks = 2
        args.nn_type = 'densenet'
        args.k_sort = 2

        training = Training(args)
        training.train_images(args)

    elif args.dataset == 'cityscapes':
        args.input_size = [1, 32, 64]
        args.num_classes = 8
        args.num_building_blocks = 4
        args.nn_type = 'densenet'

        training = Training(args)
        training.train_images(args)

    else:
        raise ValueError
