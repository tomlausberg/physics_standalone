from test_stencil import test_multidim_16x, test_multidim_32x, test_normal_16x, FIELD_TYPE, MULTIDIM_TYPE, test_normal_32x, band_type, test_taugb03
import test_stencil
import gt4py.storage as gt_storage
import numpy as np
import os


backend = os.getenv("BACKEND")
field_shape = (128,128,79)

test_normal = True
test_multidim = False

print("Imports done")
print(f"Backend: {backend}")
print("Fieldshape:")
print(field_shape)

if (test_normal):
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
    field33 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field34 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field35 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field36 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field37 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field38 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field39 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field40 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field41 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field42 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field43 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field44 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field45 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field46 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field47 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field48 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field49 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field50 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field51 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field52 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field53 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field54 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field55 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field56 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field57 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field58 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field59 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field60 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field61 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field62 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field63 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field64 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field65 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field66 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field67 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field68 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))
    field69 = gt_storage.ones(backend=backend, shape=field_shape, dtype=np.float64, default_origin=(0, 0, 0))

if(test_multidim):
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

if (test_normal):
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

    test_taugb03(
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
        field32,
        field33,
        field34,
        field35,
        field36,
        field37,
        field38,
        field39,
        field40,
        field41,
        field42,
        field43,
        field44,
        field45,
        field46,
        field47,
        field48,
        # field49,
        # field50,
        # field51,
        # field52,
        # field53,
        # field54,
        # field55,
        # field56,
        # field57,
        # field58,
        # field59,
        # field60,
        # field61,
        # field62,
        # field63,
        # field64,
        # field65,
        # field66,
        # field67,
        # field68,
        # field69,
    )
    print(ret_field)

if(test_multidim):
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



