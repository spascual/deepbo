import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from deepbo.acquisition.bo import ei
from deepbo.acquisition.active import exploration
from deepbo.models import dgpr
from deepbo.models import gpr
from deepbo.optimisers import lbfgs_search
from deepbo.optimisers import pool_search

from deepbo.tasks import base_task

import geepee.aep_models as aep
import geepee.vfe_models as vfe
import geepee.ep_models as ep
from geepee.kernels import compute_kernel, compute_psi_weave
import geepee.config as config
# from geepee.aep_models import SGPLVM, SGPR, SDGPR

from thesis_work import metrics

