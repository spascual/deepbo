import matplotlib.pyplot as plt
import numpy as np
import GPy

# from .context import gpr
from context import dgpr
from context import gpr
from context import ei
from context import lbfgs_search
from context import base_task

np.random.seed(42)
import pdb

# TODO: ADD DIFFERENT KERNEL FUNCTIONS FROM GPY
# USE : DPG_sampling jupyter no
        

class DGP_sampler(object):
    """Sampling from a multiple layer Deep Gaussian Process"""
    def __init__(self, X, kern='RBF', kern_vars=np.array([1,0.9]),kern_lens=np.array([1,0.5])):
        super(DGP_sampler, self).__init__()
        self.X = X
        self.N = X.shape[0]
        self.input_dim = X[:,None].shape[1]
        self.kern_vars = kern_vars
        self.kern_lens = kern_lens
        self.kern = kern
        self.n_hidden_layers = kern_vars.shape[0]-1

    def sample_1D(self, out_var=[1e-2], layer_noise=True, layer_var= [1e-4], layer_plots=True): 
        self.out_var= out_var 
        self.layer_var = layer_var
        mu = np.zeros(self.N)
        H_in = self.X

        for l in range(self.n_hidden_layers): 
            kernel = GPy.kern.RBF(input_dim=self.input_dim,   variance=self.kern_vars[l], lengthscale=self.kern_lens[l])

            # Insert noise into hidden layers to increase eigenvalues in diagonal matrix
            if layer_noise==True:
                layer_cov = kernel.K(H_in) + self.layer_var[0]*np.eye(self.N)
            else: 
                layer_cov = kernel.K(H_in)
            H_out = np.random.multivariate_normal(mu, layer_cov).reshape(-1,1)

            if layer_plots==True: 
                plt.figure()
                plt.plot(H_in[:],H_out[:], 'b+')
                plt.title('From ' + str(l) + ' to ' + str(l+1))
                plt.show()
            # Update
            H_in = H_out

        kernel_out = GPy.kern.RBF(input_dim=self.input_dim,
                                variance=self.kern_vars[self.n_hidden_layers], 
                                lengthscale= self.kern_lens[self.n_hidden_layers])

        out_cov = kernel_out.K(H_in)
        
        self.f = np.random.multivariate_normal(mu, out_cov).reshape(-1,1)
        self.out_noise = np.random.multivariate_normal(mu, self.out_var*np.eye(self.N)).reshape(-1,1)
        self.y = self.f + self.out_noise
        if layer_plots==True: 
            plt.plot(H_in[:],self.y[:], 'b+')
            plt.plot(H_in[:],self.f[:], 'r+')
            plt.title('From ' + str(self.n_hidden_layers) + ' to y')
            plt.show()
        plt.figure()
        plt.plot(self.X[:],self.y[:], 'b+')
        plt.plot(self.X[:],self.f[:], 'r+')
        plt.title('From x to y')
        plt.show()
        return self.y 
