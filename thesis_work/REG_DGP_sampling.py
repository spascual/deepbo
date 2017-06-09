import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import GPy
from dgp_sampler import DGP_sampler

# from .context import gpr
from context import dgpr
from context import gpr
from context import ei
from context import lbfgs_search
from context import base_task
from context import metrics
import sys
import pdb
# from sklearn.metrics import mean_squared_error as mse


# Read arguments from terminal 
DGP_data_file_train = 'thesis_work/data_DGPS/'+ sys.argv[1] + '_train.txt'
DGP_data_file_test = 'thesis_work/data_DGPS/'+ sys.argv[1] + '_test.txt'
training_size = int(sys.argv[2])

# LOAD dgp sampled training and testing data separatery generated from REG_DGP_sampling notebook 

X,y = np.loadtxt(DGP_data_file_train, delimiter=',', unpack=True)
N = X.shape[0] # 3000
list_indx = np.random.choice(range(N), training_size)
X_train = X[list_indx,None]
y_train = y[list_indx,None]

X_test, y_test = np.loadtxt(DGP_data_file_test, delimiter=',', unpack=True)

# # Make a plot to see that everythign has been imported accordingly
# plt.plot(X_train,y_train, 'k+')
# plt.show()
# pdb.set_trace()

# Full GPs
print 'Training full GP using GPy from models/gpr'
model_gp = gpr.GPR(input_dim=X_train.shape[1])
model_gp.train(X_train,y_train)

# PLOTTING
# X_test = np.linspace(0,10., 1000)[:,None]
# m2,v2 = model_gp.predict(X_test, grad=False)
# top2 = m2 + 2*np.sqrt(v2)
# bottom2 = m2 - 2*np.sqrt(v2)
# plt.plot(X_test[:], top2[:],'y')
# plt.plot(X_test[:], bottom2[:],'y')
# plt.plot(X_test[:],m2[:], 'g')
# plt.plot(DGP.X[:],DGP.y[:], 'b+')
# plt.show()

# AEP deepgeepees
print 'Training AEP DGP with no_epochs=2000 from models/dgpr'
model_aep = dgpr.DGPR(input_dim=X_train.shape[1])
model_aep.train(X_train,y_train)

# PLOTTING
# X_test = np.linspace(0,10., 1000)[:,None]
# m,v = model_aep.predict(X_test, grad=False)
# top = m + 2*np.sqrt(v)
# bottom = m - 2*np.sqrt(v)
# plt.figure()
# plt.plot(DGP.X[:],DGP.y[:], 'b+')
# plt.plot(X_test[:], top[:],'k')
# plt.plot(X_test[:], bottom[:],'k')
# plt.plot(X_test[:],m[:], 'r')
# plt.show()

# PREDICTIONS IN THE TEST SET: mean and variance
m_pred_gp, var_pred_gp = model_gp.predict(X_test[:,None], grad=False)
m_pred_aep, var_pred_aep = model_aep.predict(X_test[:,None], grad=False)

## Scoring both models:
results_gp = metrics.METRICS(y_test, m_pred_gp, var_pred_gp)
mse_gp , nll_gp = results_gp.mse(), results_gp.nll()

results_aep = metrics.METRICS(y_test, m_pred_aep, var_pred_aep)
mse_aep , nll_aep = results_aep.mse(), results_aep.nll()


# OLD STUFF: mse metric on train and full sets
# m_pred_gp = model_gp.predict(X_train, grad=False)[0]
# y_pred_aep = model_aep.predict(X_train, grad=False)[0]
# score_gp = np.sqrt(mse(y_true=y_train, y_pred=y_pred_gp))
# score_aep = np.sqrt(mse(y_true=y_train, y_pred=y_pred_aep))

# print 'If we consider the full DGP sampled datasets'
# y_pred_gp_full = model_gp.predict(X[:,None], grad=False)[0]
# y_pred_aep_full = model_aep.predict(X[:,None], grad=False)[0]
# score_gp_full = np.sqrt(mse(y_true=y, y_pred=y_pred_gp_full))
# score_aep_full = np.sqrt(mse(y_true=y, y_pred=y_pred_aep_full))

print 'GPs scores with',training_size, mse_gp , nll_gp
print 'AEP scores with',training_size, mse_aep , nll_aep

results_file = 'thesis_work/result_DGPS/' + sys.argv[1] + '.txt'
array_scores = np.array([training_size, mse_gp, mse_aep, nll_gp, nll_aep]).reshape(1,-1) #1x5 array

with open(results_file,'a') as temp: 
    np.savetxt(temp, array_scores, delimiter=',')
