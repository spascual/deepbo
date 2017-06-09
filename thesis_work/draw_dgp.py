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

class dgpr_sample(object):
    """docstring for dgpr_sample"""
    def __init__(self, X, n_hidden_layers=1, 
                    layer_vars=np.array([1,0.9]),
                    layer_lens=np.array([1,0.5]),
                    kern='RBF'):
        super(dgpr_sample, self).__init__()
        self.X = X[:,None]
        self.N = X.shape[0]
        self.input_dim = X.shape[1]
        self.n_hidden_layers = n_hidden_layers 
        self.layer_vars = layer_vars
        self.layer_lens = layer_lens
        self.kern = 'RBF'

    def sample_1D(self, noise_var=0.05, layer_noise=False):
        self.X_max = X[np.argmax(X)]
        self.X_min = X[np.argmax(X)]
        self.noise_var = noise_var

        mu = np.zeros(self.N)
        H_in = self.X

        for l in range(self.n_hidden_layers): 
            if self.kern == 'RBF': 
                kernel = GPy.kern.RBF(input_dim=self.input_dim,
                                    variance=self.layer_vars[l], 
                                    lengthscale= self.layer_vars[l])
            if layer_noise==True:
                layer_cov = kernel.K(H_in) + np.eye(self.N)*np.sqrt(self.noise_var).reshape(-1,1)
            else: 
                layer_cov = kernel.K(H_in)
            H_out = np.random.multivariate_normal(mu, layer_cov).reshape(-1,1)

            plt.figure()
            plt.plot(self.H_in[:],self.H_out[:], 'b+')
            plt.title('From' + str(l) + 'to' + str(l+1))
            plt.show()
            # Update
            H_in = H_out

        kernel_out = GPy.kern.RBF(input_dim=self.input_dim,
                                    variance=self.layer_vars[self.n_hidden_layers], 
                                    lengthscale= self.layer_vars[self.n_hidden_layers])
        out_cov = kernel.K(H_in) + np.eye(self.N)*np.sqrt(self.noise_var).reshape(-1,1)
        self.y = np.random.multivariate_normal(mu, out_cov).reshape(-1,1)
        return self.y 

    def plot(self):
        plt.figure()
        plt.plot(self.X[:],self.y[:], 'b+')
        plt.show()
        # y = DGP.sample_1D(noise_var=0.01)
        X =  np.random.uniform(0,10.,1000)[:,None]
        

        DGP = dgpr_sample(X)
        #DGP = dgpr_sample(X, n_hidden_layers=0, layer_lens=[0.5], layer_vars=[.9])
        y = DGP.sample_1D()

        # AEP deepgeepees
        model_aep = dgpr.DGPR(input_dim=DGP.input_dim)
        model_aep.train(X,y)

        m,v = model_aep.predict(X_test, grad=False)
        top = m + 2*np.sqrt(v)
        bottom = m - 2*np.sqrt(v)
        #pdb.set_trace()

        # Full GPs

        model_gp = gpr.GPR(input_dim=DGP.input_dim)
        model_gp.train(X,y)

        m2,v2 = model_aep.predict(X_test, grad=False)
        top2 = m2 + 2*np.sqrt(v2)
        bottom2 = m2 - 2*np.sqrt(v2)


        plt.figure()
        plt.plot(DGP.X[:],DGP.y[:], 'b+')

        plt.plot(X_test[:], top[:],'k')
        plt.plot(X_test[:], bottom[:],'k')
        # plt.plot(X_test[:], top2[:],'y')
        # plt.plot(X_test[:], bottom2[:],'y')

        plt.plot(X_test[:],m[:], 'r')
        # plt.plot(X_test[:],m2[:], 'g')
        plt.show()










        