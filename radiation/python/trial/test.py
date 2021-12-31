from test_stencil import test_multidim_16x, test_multidim_32x, test_normal_16x, FIELD_TYPE, MULTIDIM_TYPE, test_normal_32x, band_type
import test_stencil
import gt4py.storage as gt_storage
import numpy as np
import os
print("Imports done")

backend = os.getenv("GT4PY_BACKEND")

field_shape = (192,192,79)
print("Fieldshape:")
print(field_shape)

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
field17 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field18 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field19 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field20 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field21 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field22 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field23 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field24 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field25 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field26 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field27 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field28 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field29 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field30 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field31 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
field32 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))



multidim_ret_field = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type)
multidim_field01 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type)
multidim_field02 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field03 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field04 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field05 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field06 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field07 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field08 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field09 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field10 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field11 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field12 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field13 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field14 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field15 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field16 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field17 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field18 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field19 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field20 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field21 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field22 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field23 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field24 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field25 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field26 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field27 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field28 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field29 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field30 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field31 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 
multidim_field32 = gt_storage.zeros(backend=backend, default_origin=(0,0,0), shape=field_shape, dtype=band_type) 


print("Fields initialized")

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

test_normal_32x(
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
    field16,
    field17,
    field18,
    field19,
    field20,
    field21,
    field22,
    field23,
    field24,
    field25,
    field26,
    field27,
    field28,
    field29,
    field30,
    field31,
    field32
)
print(ret_field)

test_multidim_16x(
    multidim_ret_field,
    multidim_field01,
    multidim_field02,
    multidim_field03,
    multidim_field04,
    multidim_field05,
    multidim_field06,
    multidim_field07,
    multidim_field08,
    multidim_field09,
    multidim_field10,
    multidim_field11,
    multidim_field12,
    multidim_field13,
    multidim_field14,
    multidim_field15,
    multidim_field16
)
print(multidim_ret_field)

test_multidim_32x(
    multidim_ret_field,
    multidim_field01,
    multidim_field02,
    multidim_field03,
    multidim_field04,
    multidim_field05,
    multidim_field06,
    multidim_field07,
    multidim_field08,
    multidim_field09,
    multidim_field10,
    multidim_field11,
    multidim_field12,
    multidim_field13,
    multidim_field14,
    multidim_field15,
    multidim_field16,
    multidim_field17,
    multidim_field18,
    multidim_field19,
    multidim_field20,
    multidim_field21,
    multidim_field22,
    multidim_field23,
    multidim_field24,
    multidim_field25,
    multidim_field26,
    multidim_field27,
    multidim_field28,
    multidim_field29,
    multidim_field30,
    multidim_field31,
    multidim_field32
)

print(multidim_ret_field)



