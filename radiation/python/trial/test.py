from gt4py import backend
from test_stencil import test_normal_16x, FIELD_TYPE, backend
import test_stencil
import gt4py.storage as gt_storage
import numpy as np

field_shape = (100,100,100)

ret_field = gt_storage.zeros(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field01 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field02 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field03 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field04 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field05 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field06 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field07 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field08 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field09 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field10 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field11 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field12 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field13 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field14 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field15 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field16 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))

test_normal_16x(
    ret_field,
    field01,
    field02,
    field03,
    field04,
    field05,
    field06,
    field07,
    field08,
    field09,
    field10,
    field11,
    field12,
    field13,
    field14,
    field15,
    field16
)

print(ret_field)