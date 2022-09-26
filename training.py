import logging
import os.path

import torch
import numpy as np
import time

from data.cityscapes.cityscapes import get as get_cityscapes
from data.eightgaussians import get_eightgaussians
from data.mnist import get_mnist_loaders, get_binary_mnist_loaders, get_snp_loaders,get_mushroom_loaders, get_cop_loaders
from model.categorical_prior import CategoricalPrior, CategoricalSplitPrior, log_prior
from model.flow import Flow
from model.flow_layers import Squeeze, Permutation, Coupling
from model.model import Model
from model.network import NN
from visualization import save_grid_image, plot_2D_samples,plot_mnist_samples

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.manual_seed(42)
#torch.set_num_threads(1)

class Training:
    """
    Handles the training loop.
    """

    def __init__(self, args):

        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.k_sort = args.k_sort

        self.save_model = args.save_model
        self.model_dir = args.model_dir
        self.save_plots = args.save_plots
        self.plots_dir = args.plots_dir

        self.device = args.device

        self.input_size = args.input_size
        self.n_channels, self.height, self.width = args.input_size  # this will be updated

        self.flow = Flow(args).to(args.device)  # start with empty flow

        self.build_dataloader(args)

    def build_dataloader(self, args):
        """
        Load data
        """
        if args.dataset == 'mnist':
            self.train_loader, self.test_loader = get_mnist_loaders(args.batch_size)
            self.dimensionality = 4
        elif args.dataset == 'mnist_folds':
            self.train_loader, self.test_loader = get_binary_mnist_loaders(args.batch_size, args.kfolds, args.current_fold)
            self.dimensionality = 2
        elif args.dataset == 'genetic_folds':
            self.train_loader, self.test_loader = get_snp_loaders(args.batch_size, args.kfolds, args.current_fold)
            self.dimensionality = 2
        elif args.dataset == 'mushroom_folds':
            self.train_loader, self.test_loader = get_mushroom_loaders(args.batch_size, args.kfolds, args.current_fold)
            self.dimensionality = 2
        elif args.dataset == 'coph' or args.dataset == 'copm' or args.dataset == 'copw' or args.dataset == 'copn':
            self.train_loader, self.test_loader = get_cop_loaders(args.dataset,args.batch_size, args.kfolds, args.current_fold, args.device)
            self.dimensionality = 2
        elif args.dataset == 'cityscapes':
            self.train_loader, _, self.test_loader = get_cityscapes(args.batch_size)
            self.dimensionality = 4
        elif args.dataset == '8gaussians':
            self.train_loader, self.test_loader = get_eightgaussians(args.num_classes, args.batch_size, args.kfolds, args.current_fold)
            
            self.dimensionality = 2
        else:
            raise ValueError

    def train_splitprior(self, args):
        """
        Train one splitprior layer network.
        """
        splitprior = CategoricalSplitPrior(self.n_channels, args)
        optimizer = torch.optim.Adam(splitprior.parameters(), lr=args.lr)

        for epoch in range(args.net_epochs):

            running_loss = 0.0

            for data_batch in self.train_loader:

                if self.dataset == 'mnist':
                    (data_batch, _) = data_batch

                data_batch, _, _ = self.flow(data_batch.to(args.device))

                py, y, z = splitprior(data_batch)
                log_py = splitprior.log_prior(y, py)

                loss = - log_py.mean()
                bpd = loss / np.prod(self.input_size) / np.log(2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += bpd.item()

            logger.info('[%d] Loss: %.3f' % (epoch + 1, running_loss / len(self.train_loader)))
            running_loss = 0.0

        return splitprior

    def train_net(self, args):
        """
        Train one coupling layer network
        """
        split_idx = self.n_channels - (self.n_channels // 2)

        net = NN(args,
                 c_in=split_idx * self.num_classes,
                 c_out=(self.n_channels - split_idx) * self.num_classes,
                 nn_type=args.nn_type)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

        for epoch in range(args.net_epochs):

            running_loss = 0.0
            for data_batch in self.train_loader:

                if self.dataset == 'mnist':
                    (data_batch, _) = data_batch

                if(args.DEBUG):
                    print("train_net:data batch is")
                    print(data_batch)
                x, _, _ = self.flow(data_batch.to(args.device))
                if(args.DEBUG):
                    print("train_net:x is (after applying flow to data batch, which is a permutation layer)")
                    print(x)
                
                x1 = x[:, :split_idx]
                x2 = x[:, split_idx:]

                p_x2_given_x1 = net(x1)
                              
                loss = criterion(p_x2_given_x1, x2.long())

                # backward + update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            logger.info('[%d] Loss: %.3f' % (epoch + 1, running_loss / len(self.train_loader)))
            running_loss = 0.0

        logger.info('#' * 20 + '\n')
        return net

    def train_prior(self, args):
        """
        Train categorical prior distribution
        """
        
        logger.info("train prior")
        prior = CategoricalPrior([self.n_channels, self.height, self.width], self.num_classes, self.dimensionality).to(
            args.device)

        optimizer = torch.optim.Adam(prior.parameters(), lr=.01)

        for epoch in range(args.prior_epochs):

            running_loss = 0.0
            running_loss_2 = 0.0
            for data_batch in self.train_loader:

                if self.dataset == 'mnist':
                    (data_batch, _) = data_batch

                z, pys, ys = self.flow(data_batch.to(args.device))
 
                log_pz = prior.log_prior(z)

                for py, y in zip(pys, ys):
                    log_py = log_prior(y, py, self.num_classes)
                    log_pz += log_py

                loss = -torch.mean(log_pz)
                bpd = loss / np.prod(self.input_size) / np.log(2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += bpd.item()
                running_loss_2 += loss.item()

            logger.info('[%d] Loss: %.3f BPD %.3f NLL' % (epoch + 1, running_loss / len(self.train_loader), running_loss_2 / len(self.train_loader)))
            running_loss = 0.0
            running_loss_2 = 0.0
        logger.info('#' * 20 + '\n')
        return prior

    def add_coupling_layer(self, args):
        """
        Train a new coupling layer on top of the current flow and then add it to the flow
        """
        # add permutation
        logger.info("add coupling")
        self.flow.add_layer(Permutation(self.n_channels))
        net = self.train_net(args)
        self.flow.add_layer(Coupling(self.num_classes, self.k_sort, self.n_channels, net, self.dimensionality))

    def add_squeeze_layer(self):
        """
        Add a squeeze layer to the flow
        """
        logger.info("add squeeze")
        self.flow.add_layer(Squeeze())
        self.n_channels *= 4
        self.height //= 2
        self.width //= 2

    def add_splitprior_layer(self, args):
        """
        Train a new splitprior layer on top of the current flow and then add it to the flow
        """
        logger.info("add splitprior")
        splitprior = self.train_splitprior(args)
        self.flow.add_layer(splitprior)
        self.n_channels //= 2

    def evaluate(self, prior, args):
        """
        Evaluate the model (i.e. flow + prior) on the test data set
        """
        logger.info("evaluate on test set")
        with torch.no_grad():

            self.flow.eval()
            prior.eval()

            running_loss = 0.0
            running_loss_2 = 0.0
            
            for data_batch in self.test_loader:

                if self.dataset == 'mnist':
                    (data_batch, _) = data_batch

                z, pys, ys = self.flow(data_batch.to(args.device))

                log_pz = prior.log_prior(z)

                for py, y in zip(pys, ys):
                    log_py = log_prior(y, py, self.num_classes)
                    log_pz += log_py

                loss = -torch.mean(log_pz)
                bpd = loss / np.prod(self.input_size) / np.log(2)

                running_loss += bpd.item()
                running_loss_2 += loss.item()

            logger.info('EVALUATION: %.3f BPD %.3f NLL' % (running_loss / len(self.test_loader), running_loss_2 / len(self.test_loader)))
            args.test_losses.append(running_loss_2 / len(self.test_loader))

        return

    def train_2D(self, args):
        """
        Train a single coupling layer on two dimensional toy data
        """
       
        total_train_time = 0
        total_test_time = 0
        
        for i in range(0,args.num_coupling):
            logger.info('Adding coupling %d',(i+1))
            start_time = time.time()
            self.add_coupling_layer(args)
            end_time = time.time()
            train_time_coupling = end_time - start_time
            logger.info('Training Network took %.3f seconds',(train_time_coupling))
            
            # train prior to get current BPD
            start_time = time.time()
            prior = self.train_prior(args)
            end_time = time.time()
            train_time_prior = end_time - start_time 

            logger.info('Training Prior took %.3f seconds',(train_time_prior))
            total_train_time += train_time_coupling + train_time_prior
            logger.info('Total training time took %.3f seconds',(train_time_coupling+train_time_prior))

            start_time = time.time()
            self.evaluate(prior, args)
            end_time = time.time()
            total_test_time = end_time - start_time
            logger.info('Evaluating took %.3f seconds',(end_time-start_time))

            model = Model(self.flow, prior)      
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)   
        args.train_times.append(total_train_time)
        args.test_times.append(total_test_time)
        args.total_params.append(pytorch_total_params)        
        
        if self.save_plots and self.dataset=='8_gaussians':
            plot_2D_samples(os.path.join(self.plots_dir, "8_gaussians_df.png"), model, 5000, 91)
        elif self.save_plots and self.dataset=='mnist_folds':
            plot_mnist_samples(os.path.join(self.plots_dir, "mnist_folds.png"), model, 25, 2)
        
        if self.save_model:
            logger.info("make model")
            torch.save(model, os.path.join(self.model_dir, 'model_final'))

    def train_images(self, args):
        """
        Train a multilayer architecture on image data
        """
        for i in range(args.num_building_blocks):

            self.add_squeeze_layer()

            # Coupling
            self.add_coupling_layer(args)
            prior = self.train_prior(args)
            self.evaluate(prior, args)

            # Splitprior
            if args.with_splitprior:
                self.add_splitprior_layer(args)
                prior = self.train_prior(args)
                self.evaluate(prior, args)

            # Coupling
            self.add_coupling_layer(args)
            prior = self.train_prior(args)
            self.evaluate(prior, args)

            # Splitprior
            if args.with_splitprior:
                self.add_splitprior_layer(args)
                prior = self.train_prior(args)
                self.evaluate(prior, args)

            model = Model(self.flow, prior)
            if self.save_plots:
                save_grid_image(model, os.path.join(self.plots_dir, "model_samples_") + str(i), h=self.input_size[1], w=self.input_size[2],
                                dataset=args.dataset)

            if i == 1 and self.dataset == 'cityscapes':
                args.k_sort = 8

        if self.save_model:
            logger.info("make model")
            torch.save(model, os.path.join(self.model_dir, "model_final"))
