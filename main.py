import os
import datetime
import logging
import numpy as np

import torch
from argparse import ArgumentParser

import time
from training import Training
import pickle
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

parser.add_argument("--kfolds", type=int, default=1, help="The number of kfolds to test")
parser.add_argument("--num_coupling", type=int, default=1, help="The number of coupling layers")


parser.add_argument("--with_splitprior", type=bool, default=False, help="Whether to train the model with splitpriors")
parser.add_argument("--save_model", type=bool, default=False, help="Whether to save the trained model")
parser.add_argument("--save_plots", type=bool, default=False, help="Whether to save the model outputs as plots")

args = parser.parse_args()

if __name__ == "__main__":
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if(torch.cuda.is_available()):
        logger.info("Using device: cuda")
    else:
        logger.info("Using device: CPU")
    # save results
    args.results_dir = 'results'
    run_dir = os.path.join('results', datetime.datetime.now().strftime("%m%d%Y_%H%M%S"))
    args.model_dir = os.path.join(run_dir, 'model')
    args.plots_dir = os.path.join(run_dir, 'plots')
    for dir_ in [args.results_dir, run_dir, args.model_dir, args.plots_dir]:
        if not os.path.isdir(dir_):
            os.mkdir(dir_)
    args.DEBUG = False
    
    if args.dataset == '8gaussians':
        args.input_size = [2, 1, 1]
        args.num_classes = 91
        args.nn_type = 'mlp'
        args.test_losses = []
        args.train_times = []
        args.test_times = []
        args.total_params = []

        for i in range(0,args.kfolds):
            print("processing fold number "+str(i))
            args.current_fold = i
            training = Training(args)
            training.train_2D(args)
        args.test_losses = np.array(args.test_losses)
        args.train_times = np.array(args.train_times)
        args.test_times = np.array(args.test_times)
        
        logger.info('The average num of params %d folds is %.2f $pm %.2f',args.kfolds,np.mean(args.total_params),np.std(args.total_params)) 
        logger.info('The average NLL across %d folds is %.2f +/- %.2f',args.kfolds,np.mean(args.test_losses),np.std(args.test_losses))
        logger.info('The average train time across %d folds is %.2f +/- %.2f',args.kfolds,np.mean(args.train_times),np.std(args.train_times))
        logger.info('The average test time across %d folds is %.2f +/- %.2f',args.kfolds,np.mean(args.test_times),np.std(args.test_times))
        
        logger.info('%.4f $\pm$ %.4f & %.4f $\pm$ %.4f & %.4f & %.4f $\pm$ %.4f ',np.mean(args.test_losses),np.std(args.test_losses),np.mean(args.train_times),np.std(args.train_times),np.mean(args.total_params),np.mean(args.test_times),np.std(args.test_times)) 
        
        with open('./fold_results/'+args.dataset+'_kfolds_'+str(args.kfolds)+'_epochs_'+str(args.net_epochs)+'_n_hidden_nn_'+str(args.n_hidden_nn)+'_num_coupling_'+str(args.num_coupling)+'.Obj', 'wb') as f:
            pickle.dump(args, f)
    elif args.dataset == 'mnist':
        args.input_size = [1, 28, 28]
        args.num_classes = 2
        args.num_building_blocks = 2
        args.nn_type = 'densenet'
        args.k_sort = 2
        start_time = time.time()
        training = Training(args)
        training.train_images(args)
        end_time = time.time()
        print("training took "+str(end_time-start_time)+" seconds")
    
    elif args.dataset == 'mnist_folds':
        args.input_size = [784, 1, 1]
        args.num_classes = 2
        args.k_sort = 2
        args.test_losses = []
        args.train_times = []
        args.test_times = []
        args.total_params = []

        for i in range(0,args.kfolds):
            args.current_fold = i
            training = Training(args)
            training.train_2D(args)
        args.test_losses = np.array(args.test_losses)
        args.train_times = np.array(args.train_times)
        args.test_times = np.array(args.test_times)
        logger.info('The average num of params %d folds is %.2f $pm %.2f',args.kfolds,np.mean(args.total_params),np.std(args.total_params)) 
        logger.info('The average NLL across %d folds is %.2f +/- %.2f',args.kfolds,np.mean(args.test_losses),np.std(args.test_losses))
        logger.info('The average train time across %d folds is %.2f +/- %.2f',args.kfolds,np.mean(args.train_times),np.std(args.train_times))
        logger.info('The average test time across %d folds is %.2f +/- %.2f',args.kfolds,np.mean(args.test_times),np.std(args.test_times))
        
        logger.info('%.4f $\pm$ %.4f & %.4f $\pm$ %.4f & %.4f & %.4f $\pm$ %.4f ',np.mean(args.test_losses),np.std(args.test_losses),np.mean(args.train_times),np.std(args.train_times),np.mean(args.total_params),np.mean(args.test_times),np.std(args.test_times)) 
        with open('./fold_results/'+args.dataset+'_kfolds_'+str(args.kfolds)+'_epochs_'+str(args.net_epochs)+'_n_hidden_nn_'+str(args.n_hidden_nn)+'_num_coupling_'+str(args.num_coupling)+'.Obj', 'wb') as f:
            pickle.dump(args, f)
    
    elif args.dataset == 'genetic_folds':
        args.input_size = [805, 1, 1]
        args.num_classes = 2
        args.k_sort = 2
        args.total_params = []
        args.test_losses = []
        args.train_times = []
        args.test_times = []

        for i in range(0,args.kfolds):
            args.current_fold = i
            training = Training(args)
            training.train_2D(args)
        
        args.test_losses = np.array(args.test_losses)
        args.train_times = np.array(args.train_times)
        args.test_times = np.array(args.test_times)
        logger.info('The average num of params %d folds is %.2f $pm %.2f',args.kfolds,np.mean(args.total_params),np.std(args.total_params)) 
        logger.info('The average NLL across %d folds is %.2f +/- %.2f',args.kfolds,np.mean(args.test_losses),np.std(args.test_losses))
        logger.info('The average train time across %d folds is %.2f +/- %.2f',args.kfolds,np.mean(args.train_times),np.std(args.train_times))
        logger.info('The average test time across %d folds is %.2f +/- %.2f',args.kfolds,np.mean(args.test_times),np.std(args.test_times))
        logger.info('%.4f $\pm$ %.4f & %.4f $\pm$ %.4f & %.4f & %.4f $\pm$ %.4f ',np.mean(args.test_losses),np.std(args.test_losses),np.mean(args.train_times),np.std(args.train_times),np.mean(args.total_params),np.mean(args.test_times),np.std(args.test_times)) 
        with open('./fold_results/'+args.dataset+'_kfolds_'+str(args.kfolds)+'_epochs_'+str(args.net_epochs)+'_n_hidden_nn_'+str(args.n_hidden_nn)+'_num_coupling_'+str(args.num_coupling)+'.Obj', 'wb') as f:
            pickle.dump(args, f)
    
    elif args.dataset == 'mushroom_folds':
        
        args.input_size = [21, 1, 1]
        args.num_classes = 12
        args.k_sort = 12
        args.total_params = []
        args.test_losses = []
        args.train_times = []
        args.test_times = []

        for i in range(0,args.kfolds):
            args.current_fold = i
            training = Training(args)
            training.train_2D(args)
        
        args.test_losses = np.array(args.test_losses)
        args.train_times = np.array(args.train_times)
        args.test_times = np.array(args.test_times)
        logger.info('The average num of params %d folds is %.2f $pm %.2f',args.kfolds,np.mean(args.total_params),np.std(args.total_params)) 
        logger.info('The average NLL across %d folds is %.2f +/- %.2f',args.kfolds,np.mean(args.test_losses),np.std(args.test_losses))
        logger.info('The average train time across %d folds is %.2f +/- %.2f',args.kfolds,np.mean(args.train_times),np.std(args.train_times))
        logger.info('The average test time across %d folds is %.2f +/- %.2f',args.kfolds,np.mean(args.test_times),np.std(args.test_times))
        logger.info('%.4f $\pm$ %.4f & %.4f $\pm$ %.4f & %.4f & %.4f $\pm$ %.4f ',np.mean(args.test_losses),np.std(args.test_losses),np.mean(args.train_times),np.std(args.train_times),np.mean(args.total_params),np.mean(args.test_times),np.std(args.test_times)) 
        with open('./fold_results/'+args.dataset+'_kfolds_'+str(args.kfolds)+'_epochs_'+str(args.net_epochs)+'_n_hidden_nn_'+str(args.n_hidden_nn)+'_num_coupling_'+str(args.num_coupling)+'.Obj', 'wb') as f:
            pickle.dump(args, f)
    
    elif args.dataset == 'coph' or args.dataset == 'copm' or args.dataset == 'copw' or args.dataset == 'copn':
        args.input_size = [4, 1, 1]
        args.num_classes = 2
        args.k_sort = 2
        args.total_params = []
        args.test_losses = []
        args.train_times = []
        args.test_times = []

        for i in range(0,args.kfolds):
            args.current_fold = i
            training = Training(args)
            training.train_2D(args)
        
        args.test_losses = np.array(args.test_losses)
        args.train_times = np.array(args.train_times)
        args.test_times = np.array(args.test_times)
        logger.info('The average num of params %d folds is %.2f $pm %.2f',args.kfolds,np.mean(args.total_params),np.std(args.total_params)) 
        logger.info('The average NLL across %d folds is %.2f +/- %.2f',args.kfolds,np.mean(args.test_losses),np.std(args.test_losses))
        logger.info('The average train time across %d folds is %.2f +/- %.2f',args.kfolds,np.mean(args.train_times),np.std(args.train_times))
        logger.info('The average test time across %d folds is %.2f +/- %.2f',args.kfolds,np.mean(args.test_times),np.std(args.test_times))
        logger.info('%.4f $\pm$ %.4f & %.4f $\pm$ %.4f & %.4f & %.4f $\pm$ %.4f ',np.mean(args.test_losses),np.std(args.test_losses),np.mean(args.train_times),np.std(args.train_times),np.mean(args.total_params),np.mean(args.test_times),np.std(args.test_times)) 
        converted_args = vars(args)
        print(converted_args)
        with open('./fold_results/'+args.dataset+'_kfolds_'+str(args.kfolds)+'_epochs_'+str(args.net_epochs)+'_n_hidden_nn_'+str(args.n_hidden_nn)+'_num_coupling_'+str(args.num_coupling)+'.Obj', 'wb') as f:
            pickle.dump(args, f)
        
       
    elif args.dataset == 'cityscapes':
        args.input_size = [1, 32, 64]
        args.num_classes = 8
        args.num_building_blocks = 4
        args.nn_type = 'densenet'
        start_time = time.time()
        training = Training(args)
        training.train_images(args)
        end_time = time.time()
        print("training took "+str(end_time-start_time)+" seconds")

    else:
        raise ValueError
