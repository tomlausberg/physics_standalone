import gt4py.storage as gt_storage
import numpy as np
import os
from radiation.python.config import npts

from radiation.python.trial.stencil_gpu_v_cpu import test_masking

backend = os.getenv("BACKEND")
field_shape = (128,128,79)
npts_shape = (npts,)


infield1_np = np.ones(npts_shape, dtype=np.float64)
infield2_np = np.full(npts_shape, 2,dtype=np.float64)
infield1 = gt_storage.create_storage_from_array(infield1_np, backend, npts_shape, np.float64)
infield2 = gt_storage.create_storage_from_array(infield2_np, backend, npts_shape, np.float64)

outfield = gt_storage.create_storage_zeros(backend, field_shape, np.float64)

test_masking(infield1, infield2, outfield)

