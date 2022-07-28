from stencil_gpu_v_cpu import test_masking
import gt4py.storage as gt_storage
import numpy as np
import os
import sys

sys.path.append('../../python')

from config import DTYPE_BOOL, DTYPE_FLT
from util import numpy_dict_to_gt4py_dict

npts = 24

backend = os.getenv("BACKEND")
field_shape = (128, 128, 79)
npts_shape = (npts,)

storage_vars = {
    "mask": {"shape": (npts,), "type": DTYPE_BOOL},
    "infield1": {"shape": (npts,), "type": DTYPE_FLT},
    "infield2": {"shape": (npts,), "type": DTYPE_FLT},
    "outfield": {"shape": (npts,), "type": DTYPE_FLT},
}

np_dict = {
    "mask": np.array([True if i < npts / 2 else False for i in range(npts)]),
    "infield1": np.full(npts_shape, 1, dtype=np.float64),
    "infield2": np.full(npts_shape, 2, dtype=np.float64),
    "outfield": np.full(npts_shape, 0, dtype=np.float64),
}

storage_dict = numpy_dict_to_gt4py_dict(np_dict, storage_vars)

for k, v in storage_dict.items():
    try:
        print(f"{k}: {storage_dict[k]._sync_state.state}")
    except:
        print(f"{k}: No sync state")
    v.synchronize()

for k, v in storage_dict.items():
    try:
        print(f"{k}: {storage_dict[k]._sync_state.state}")
    except:
        print(f"{k}: No sync state")




# mask_np = np.array([True if i < npts / 2 else False for i in range(npts)])
# infield1_np = np.ones(npts_shape, dtype=np.float64)
# infield2_np = np.full(npts_shape, 2, dtype=np.float64)

# mask = gt_storage.from_array(
#     mask_np, shape=npts_shape, backend=backend, default_origin=(0,), dtype=np.float64)
# infield1 = gt_storage.from_array(
#     infield1_np, shape=npts_shape, backend=backend, default_origin=(0,), dtype=np.float64)
# infield2 = gt_storage.from_array(
#     infield2_np, shape=npts_shape, backend=backend, default_origin=(0,), dtype=np.float64)

# outfield = gt_storage.zeros(
#     backend=backend, shape=npts_shape, default_origin=(0,), dtype=np.float64)

test_masking(storage_dict["mask"], storage_dict["infield1"],
             storage_dict["infield2"], storage_dict["outfield"])

for k, v in storage_dict.items():
    try:
        print(f"{k}: {storage_dict[k]._sync_state.state}")
        v.synchronize()
    except:
        print(f"{k}: No sync state")


infield1_np = np_dict["infield1"]
infield1_gt = np.squeeze(storage_dict["infield1"].view(np.ndarray))
outfield_np = np.squeeze(storage_dict["outfield"].view(np.ndarray))
mask = np_dict["mask"]
for i in range(npts):
    print(f"{i}:\t{infield1_np[i]}\t{infield1_gt[i]}\t{mask[i]}\t{outfield_np[i]}")
