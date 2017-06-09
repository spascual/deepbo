import matplotlib.pyplot as plt
import numpy as np
import GPy

# from .context import gpr
from context import dgpr
from context import gpr
from context import ei
from context import lbfgs_search
from context import base_task

from dgp_sampler import DGP_sampler

np.random.seed(42)
import pdb

X = np.linspace(0,10.,1000)[:,None]
DGP = DGP_sampler(X,kern_vars=np.array([1,0.9]),kern_lens=np.array([1,0.5]))
y = DGP.sample_1D(out_var=[1e-2], layer_noise=True, layer_var= [1e-4], layer_plots=False)

