import matplotlib.pyplot as plt
import numpy as np
import GPy

# from .context import gpr
from .context import dgpr
from .context import gpr
from .context import ei
from .context import pool_search
from .context import base_task
from draw_dgp import dgpr_sample

X_pool =  np.random.uniform(0,10.,1000)[:,None]

DGP = dgpr_sample(X_pool)
#DGP = dgpr_sample(X, n_hidden_layers=0, layer_lens=[0.5], layer_vars=[.9])
y_pool = DGP.sample()


model = dgpr.DGPR(input_dim=DGP.input_dim, noise_var=1e-3)

# The acquisition function that we optimize in order to pick a new x
acquisition_func = ei.EI(model)

# Set the method that we will use to optimize the acquisition function
maximiser = pool_search.PoolSearch(acquisition_func)

# Draw one random point and evaluate it to initialize BO
X = np.array([X_pool[0]])
Y = np.array([y_pool[0]])

# This is the main Bayesian optimization loop
for i in xrange(100):

    # Fit the model on the data we observed so far
    model.train(X, Y)

    # Update the acquisition function model with the retrained model
    acquisition_func.update(model)

    # Optimize the acquisition function to obtain a new point
    best, new_x, obj_val = maximiser.maximise(X_pool)

    # Evaluate the point and add the new observation to our set of previous seen points
    new_y = y_pool[best]
    X = np.append(X, new_x, axis=0)
    Y = np.append(Y, new_y, axis=0)

plt.figure()
plt.plot(DGP.X_pool[:],DGP.y_pool[:], 'b+')
plt.show()
