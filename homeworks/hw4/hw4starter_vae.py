'''
Starter Code for Problem 2 of HW4 Tufts COMP 150 Bayesian Deep Learning Fall 2019

Instructions: https://www.cs.tufts.edu/comp/150BDL/2019f/hw4.html

Author: Mike Hughes

Sample Usage
------------
To train a variational autoencoder (VAE) on MNIST for 3 epochs with learning rate 0.001

$ python hw4starter_vae.py --n_epochs 3 --lr 0.001


Further Help
------------
To display a full help message with all possible keyword arguments:

$ python hw4starter_ae.py --help

'''

from __future__ import print_function
import argparse
import numpy as np
import pandas as pd

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from collections import defaultdict

class VariationalAutoencoder(nn.Module):
    def __init__(
            self,
            q_sigma=0.2,
            n_dims_code=2,
            n_dims_data=784,
            hidden_layer_sizes=[32]):
        super(VariationalAutoencoder, self).__init__()
        self.n_dims_data = n_dims_data
        self.n_dims_code = n_dims_code
        self.q_sigma = torch.Tensor([float(q_sigma)])
        encoder_layer_sizes = (
            [n_dims_data] + hidden_layer_sizes + [n_dims_code]
            )
        self.n_layers = len(encoder_layer_sizes) - 1
        # Create the encoder, layer by layer
        self.encoder_activations = list()
        self.encoder_params = nn.ModuleList()
        for layer_id, (n_in, n_out) in enumerate(zip(
                encoder_layer_sizes[:-1], encoder_layer_sizes[1:])):
            self.encoder_params.append(nn.Linear(n_in, n_out))
            self.encoder_activations.append(F.relu)
        self.encoder_activations[-1] = lambda a: a

        self.decoder_activations = list()
        self.decoder_params = nn.ModuleList()
        decoder_layer_sizes = [a for a in reversed(encoder_layer_sizes)]
        for (n_in, n_out) in zip(
                decoder_layer_sizes[:-1], decoder_layer_sizes[1:]):
            self.decoder_params.append(nn.Linear(n_in, n_out))
            self.decoder_activations.append(F.relu)
        self.decoder_activations[-1] = torch.sigmoid

    def forward(self, x_ND):
        """ Run entire probabilistic autoencoder on input (encode then decode)

        Returns
        -------
        xproba_ND : 1D array, size of x_ND
        """
        mu_NC = self.encode(x_ND)
        z_NC = self.draw_sample_from_q(mu_NC)
        return self.decode(z_NC), mu_NC

    def draw_sample_from_q(self, mu_NC):
        ''' Draw sample from the probabilistic encoder q(z|mu(x), \sigma)

        We assume that "q" is Normal with:
        * mean mu (argument of this function)
        * stddev q_sigma (attribute of this class, use self.q_sigma)

        Args
        ----
        mu_NC : tensor-like, N x C
            Mean of the encoding for each of the N images in minibatch.

        Returns
        -------
        z_NC : tensor-like, N x C
            Exactly one sample vector for each of the N images in minibatch.
        '''
        N = mu_NC.shape[0]
        C = self.n_dims_code
        if self.training:
            # Draw standard normal samples "epsilon"
            eps_NC = torch.randn(N, C)
            ## TODO
            # Using reparameterization trick,
            # Write a procedure here to make z_NC a valid draw from q 
            z_NC = 1.0 * eps_NC # <-- TODO fix me
            return z_NC
        else:
            # For evaluations, we always just use the mean
            return mu_NC


    def encode(self, x_ND):
        cur_arr = x_ND
        for ll in range(self.n_layers):
            linear_func = self.encoder_params[ll]
            a_func = self.encoder_activations[ll]
            cur_arr = a_func(linear_func(cur_arr))
        mu_NC = cur_arr
        return mu_NC

    def decode(self, z_NC):
        cur_arr = z_NC
        for ll in range(self.n_layers):
            linear_func = self.decoder_params[ll]
            a_func = self.decoder_activations[ll]
            cur_arr = a_func(linear_func(cur_arr))
        xproba_ND = cur_arr
        return xproba_ND

    def calc_vi_loss(self, x_ND, n_mc_samples=1):
        ''' Compute VI loss (negative ELBO) for given data

        Returns
        -------
        loss : scalar Pytorch tensor
        sample_xproba_ND : tensor shaped like xbin_ND, values within unit interval (0,1)
            Sampled from the probabilistic decoder
        '''
        total_loss = 0.0
        mu_NC = self.encode(x_ND)
        for ss in range(n_mc_samples):
            sample_z_NC = self.draw_sample_from_q(mu_NC)
            sample_xproba_ND = self.decode(sample_z_NC)
            sample_bce_loss = torch.tensor(np.nan)       # <- TODO fix me

            # KL divergence from q(mu, sigma) to prior (std normal)
            # see Appendix B from VAE paper
            # https://arxiv.org/pdf/1312.6114.pdf
            kl = torch.tensor(np.nan)                    # <- TODO fix me
            total_loss += sample_bce_loss + kl
        return total_loss / float(n_mc_samples), sample_xproba_ND


def train_for_one_epoch_of_gradient_update_steps(
        model, optimizer, train_loader, epoch, args):
    ''' Perform exactly one epoch of gradient updates on provided model & data.

    Steps through one full pass of dataset, one minibatch at a time.
    At each minibatch, we compute the gradient and step in that direction.

    Returns
    -------
    model : Pytorch NN Module object
        Updated version of input model.
    '''
    model.train()
    train_loss_val = 0.0
    n_seen = 0
    n_batch_total = len(train_loader)
    report_at_frac_seen = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0]
    for batch_idx, (batch_data, _) in enumerate(train_loader):
        # Reshape the data from n_images x 28x28 to n_images x 784 (NxD)
        batch_x_ND = batch_data.to(device).view(-1, model.n_dims_data)

        # Zero out any stored gradients attached to the optimizer
        optimizer.zero_grad()

        # Compute the loss (and the required reconstruction as well)
        batch_loss_tensor, batch_xproba_ND = model.calc_vi_loss(batch_x_ND, n_mc_samples=args.n_mc_samples)
        batch_loss_val = batch_loss_tensor.item()
        if np.isnan(batch_loss_val):
            # Special case for starter code
            print("WARNING: NaN loss. No gradient update happening. This is expected for unedited starter code.")
        else:
            # Increment the total loss (over all batches)
            train_loss_val += batch_loss_val

            # Compute the gradient of the loss wrt model parameters
            # (gradients are stored as attributes of parameters of 'model')
            batch_loss_tensor.backward()

            # Take an optimization step (gradient descent step)
            optimizer.step()

        # Done with this batch. Write a progress update to stdout and move on.
        n_seen += batch_x_ND.shape[0]
        frac_seen = (1+batch_idx) / n_batch_total
        if frac_seen >= report_at_frac_seen[0]:
            l1_dist = torch.mean(torch.abs(batch_x_ND - batch_xproba_ND))
            print("  epoch %3d | frac_seen %.3f | total loss %.3e | batch loss % .3e | batch l1 % .3f" % (
                epoch, frac_seen,
                train_loss_val / float(n_seen),
                batch_loss_val / float(batch_x_ND.shape[0]),
                l1_dist,
                ))
            report_at_frac_seen = report_at_frac_seen[1:] # set next level of progress to reach before report
    return model


def eval_model_on_data(
        model, data_loader, device, args):
    ''' Evaluate common performance metrics for probabilistic model of images.

    Returns
    -------
    vi_loss_per_pixel : scalar float
        Negative ELBO value
    l1_loss_per_pixel : scalar float
        L1 loss (mean absolute error)
    bce_loss_per_pixel : scalar float
        Binary cross entropy
    '''
    model.eval()
    total_vi_loss = 0.0
    total_l1 = 0.0
    total_bce = 0.0
    n_seen = 0
    total_1pix = 0.0
    for batch_idx, (batch_data, _) in enumerate(data_loader):
        batch_x_ND = batch_data.to(device).view(-1, model.n_dims_data)
        total_1pix += torch.sum(batch_x_ND)
        loss, _ = model.calc_vi_loss(batch_x_ND, n_mc_samples=args.n_mc_samples)
        total_vi_loss += loss.item()

        # Use deterministic reconstruction to evaluate bce and l1 terms
        batch_xproba_ND = model.decode(model.encode(batch_x_ND))
        total_l1 += torch.sum(torch.abs(batch_x_ND - batch_xproba_ND))
        total_bce += F.binary_cross_entropy(batch_xproba_ND, batch_x_ND, reduction='sum')
        n_seen += batch_x_ND.shape[0]
    print("Total images %d. Total on pixels: %d. Frac pixels on: %.3f" % (
        n_seen, total_1pix, total_1pix / float(n_seen*784)))

    vi_loss_per_pixel = total_vi_loss / float(n_seen * model.n_dims_data)
    l1_per_pixel = total_l1 / float(n_seen * model.n_dims_data)
    bce_per_pixel = total_bce / float(n_seen * model.n_dims_data) 
    return float(vi_loss_per_pixel), float(l1_per_pixel), float(bce_per_pixel)

def print_dataset_info(data_loader, name='Training Set', device='cpu'):
    n_img, n_row, n_col = data_loader.dataset.data.shape
    print("%s: %d images, each of size %d x %d pixels" % (name, n_img, n_row, n_col))
    uval_ct_map = defaultdict(int)
    for batch_idx, (batch_data, _) in enumerate(data_loader):
        batch_x_ND = batch_data.to(device).view(-1, 784)
        uvals_U, counts_U = np.unique(batch_x_ND.flatten().numpy(), return_counts=True)
        for uu in range(uvals_U.size):
            uval_ct_map[uvals_U[uu]] += counts_U[uu]
    uvals = np.fromiter(uval_ct_map.keys(), dtype=float)
    counts = np.fromiter(uval_ct_map.values(), dtype=float)
    print("%d unique values in 'x'" % (
        len(uvals)))
    S = np.sum(counts)
    for uval, ct in zip(uvals, counts):
        print("  uval %s : fraction %.3f" % (uval, ct/S))

def parse_args_from_stdin_using_common_format(
        description="",
        nickname=""):
    ''' Parse arguments from standard input (when running this as a script).

    Provides a common reuseable interface, so that AEs and VAEs have same defaults.

    Returns
    -------
    args : Arguments object from argparse module
    '''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--nickname', type=str, default='VAE',
        help='Useful short name to describe this run')
    ## Model hyperparameters: Architecture size
    parser.add_argument(
        '--hidden_layer_sizes', type=str, default='32',
        help='Comma-separated list of size values (default: "32")')
    ## Optimization settings
    parser.add_argument(
        '--n_epochs', type=int, default=1,
        help="Number of epochs, aka complete passes through the training set (default: 1)")
    parser.add_argument(
        '--batch_size', type=int, default=1024,
        help='Number of images in each batch (default: 1024)')
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate for gradient descent (default: 0.001)')
    parser.add_argument(  
        '--seed', type=int, default=8675309,
        help='Random seed controls parameter initialization (default: 8675309)')
    ## Control where things get saved on disk
    parser.add_argument(
        '--output_filename_prefix', type=str, default='{{nickname}}-arch={{hidden_layer_sizes}}-lr={{lr}}',
        help="Prefix for the filename used when results are written to disk.\nAny string like {{key}} will be replaced with value of variable named 'key' in workspace.")
    parser.add_argument(
        '--data_dir', type=str, default='data/',
        help="Path to directory where MNIST data can be downloaded and stored")
    ## Special hyperparameters for the posterior approximation
    parser.add_argument(
        '--q_sigma', type=float, default=0.1,
        help='Fixed variance of approximate posterior (default: 0.1)')
    parser.add_argument(
       '--n_mc_samples', type=int, default=1,
       help='Number of Monte Carlo samples (default: 1)')

    ## Read in the required arguments from stdin
    args = parser.parse_args()

    ## Transform hidden layer sizes from string '10,2,3' to a list [10, 2, 3]
    args.hidden_layer_sizes = [int(s) for s in args.hidden_layer_sizes.split(',')]

    ## Setup filename_prefix for results
    for key, val in args.__dict__.items():
        args.output_filename_prefix = args.output_filename_prefix.replace('{{%s}}' % key, str(val))

    print("Provided keyword arguments")
    for key in sorted(args.__dict__.keys()):
        print("--%s : %s" % (key, args.__dict__[key]))

    return args


if __name__ == "__main__":
    args = parse_args_from_stdin_using_common_format(
        nickname="VAE",
        description="Variational Inference for a Probabilistic Autoencoder: Example on MNIST data")
    ## Use the local cpu
    device = torch.device("cpu")
    ## Set random seed
    torch.manual_seed(args.seed)

    ## Create generators for grabbing batches of train or test data
    # Each loader will produce **binary** data arrays (using transforms defined below)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_dir, train=True, download=True,
            transform=transforms.Compose([transforms.ToTensor(), torch.round])),    
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_dir, train=False,
            transform=transforms.Compose([transforms.ToTensor(), torch.round])),
        batch_size=args.batch_size, shuffle=True)

    print("Dataset Description: MNIST digits")
    print_dataset_info(train_loader, "Training Set")
    print_dataset_info(test_loader, "Test Set")

    ## Create VAE model by calling its constructor
    model = VariationalAutoencoder(q_sigma=args.q_sigma, hidden_layer_sizes=args.hidden_layer_sizes)
    model = model.to(device)

    ## Create an optimizer linked to the model parameters
    # Given gradients computed by pytorch, this optimizer handle update steps to params
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ## Training loop that repeats for each epoch:
    #  -- perform minibatch training updates (one epoch = one pass thru dataset)
    #  -- for latest model, compute performance metrics on training set
    #  -- for latest model, compute performance metrics on test set
    for epoch in range(args.n_epochs + 1):
        if epoch > 0:
            model = train_for_one_epoch_of_gradient_update_steps(
                model, optimizer, train_loader, epoch, args)

        ## Only perform extensive validation and save results for specific epochs
        # Namely, 0,1,2,3,4,5 and 10,20,30,...
        if epoch > 5 and epoch % 10 != 0:
            continue

        print('==== evaluation after epoch %d' % (epoch))
        ## For evaluation, we'll use functions available in the 'VAE' class 
        ## This makes it easy to compare AEs and VAEs using one consistent set of eval functions

        ## Compute VI loss (bce + kl), bce alone, and l1 alone
        tr_loss, tr_l1, tr_bce = eval_model_on_data(model, train_loader, device, args)
        print('  epoch %3d  train loss %.3f  bce %.3f  l1 %.3f' % (epoch, tr_loss, tr_bce, tr_l1))
        te_loss, te_l1, te_bce = eval_model_on_data(model, test_loader, device, args)
        print('  epoch %3d  test  loss %.3f  bce %.3f  l1 %.3f' % (epoch, te_loss, te_bce, te_l1))

        ## Write perf metrics to CSV string (so we can easily plot later)
        # Create str repr of architecture size list: [20,30] becomes '[20;30]'
        arch_str = '[' + ';'.join(map(str,args.hidden_layer_sizes)) + ']'
        row_df = pd.DataFrame([[
                epoch,
                tr_loss, tr_l1, tr_bce,
                te_loss, te_l1, te_bce,
                arch_str, args.lr, args.q_sigma, args.n_mc_samples]],
            columns=[
                'epoch',
                'tr_vi_loss', 'tr_l1_error', 'tr_bce_error',
                'te_vi_loss', 'te_l1_error', 'te_bce_error',
                'arch_str', 'lr', 'q_sigma', 'n_mc_samples'])
        csv_str = row_df.to_csv(
            None,
            float_format='%.8f',
            index=False,
            header=False if epoch > 0 else True,
            )

        ## Output 1/2: Write to CSV (one row per recorded epoch)
        if epoch == 0:
            # At start, write to a clean file with mode 'w'
            with open('%s_perf_metrics.csv' % args.output_filename_prefix, 'w') as f:
                f.write(csv_str)
        else:
            # Append to existing file with mode 'a'
            with open('%s_perf_metrics.csv' % args.output_filename_prefix, 'a') as f:
                f.write(csv_str)

        ## Output 2/2: Make pretty plots of random samples in code space decoding into data space
        with torch.no_grad():
            P = int(np.sqrt(model.n_dims_data))
            sample = torch.randn(25, model.n_dims_code).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(25, 1, P, P), nrow=5, padding=4,
                       filename='%s_sample_images_epoch=%03d.png' % (args.output_filename_prefix, epoch))

        print("====  done with eval at epoch %d" % epoch)

