from gt4py.gtscript import (
    FORWARD,
    IJK,
    interval,
    stencil,
    computation,
    Field,
    IJ
)
import numpy as np
import os

band_type = (np.float64, (16,))
FIELD_TYPE = Field[np.float64]
MULTIDIM_TYPE = Field[IJ,band_type]

backend = os.getenv("GT4PY_BACKEND")

    
@stencil(backend=backend)
def test_normal_16x(
    ret_field: FIELD_TYPE,
    field01: FIELD_TYPE,
    field02: FIELD_TYPE,
    field03: FIELD_TYPE,
    field04: FIELD_TYPE,
    field05: FIELD_TYPE,
    field06: FIELD_TYPE,
    field07: FIELD_TYPE,
    field08: FIELD_TYPE,
    field09: FIELD_TYPE,
    field10: FIELD_TYPE,
    field11: FIELD_TYPE,
    field12: FIELD_TYPE,
    field13: FIELD_TYPE,
    field14: FIELD_TYPE,
    field15: FIELD_TYPE,
    field16: FIELD_TYPE
    ):
    with computation(FORWARD), interval(1,None):
        field01[0,0,0]= 2
        field02[0,0,0]= 2
        field03[0,0,0]= 2
        field04[0,0,0]= 2
        field05[0,0,0]= 2
        field06[0,0,0]= 2
        field07[0,0,0]= 2
        field08[0,0,0]= 2
        field09[0,0,0]= 2
        field10[0,0,0]= 2
        field11[0,0,0]= 2
        field12[0,0,0]= 2
        field13[0,0,0]= 2
        field14[0,0,0]= 2
        field15[0,0,0]= 2
        field16[0,0,0]= 2
    
    with computation(FORWARD), interval(1,None):
        ret_field[0,0,0] = field01[0,0,0] \
            + field02[0,0,0] \
            + field03[0,0,0] \
            + field04[0,0,0] \
            + field05[0,0,0] \
            + field06[0,0,0] \
            + field07[0,0,0] \
            + field08[0,0,0] \
            + field09[0,0,0] \
            + field10[0,0,0] \
            + field11[0,0,0] \
            + field12[0,0,0] \
            + field13[0,0,0] \
            + field14[0,0,0] \
            + field15[0,0,0] \
            + field16[0,0,0]

@stencil(backend=backend)
def test_normal_32x(
    ret_field: FIELD_TYPE,
    field01: FIELD_TYPE,
    field02: FIELD_TYPE,
    field03: FIELD_TYPE,
    field04: FIELD_TYPE,
    field05: FIELD_TYPE,
    field06: FIELD_TYPE,
    field07: FIELD_TYPE,
    field08: FIELD_TYPE,
    field09: FIELD_TYPE,
    field10: FIELD_TYPE,
    field11: FIELD_TYPE,
    field12: FIELD_TYPE,
    field13: FIELD_TYPE,
    field14: FIELD_TYPE,
    field15: FIELD_TYPE,
    field16: FIELD_TYPE,
    field17: FIELD_TYPE,
    field18: FIELD_TYPE,
    field19: FIELD_TYPE,
    field20: FIELD_TYPE,
    field21: FIELD_TYPE,
    field22: FIELD_TYPE,
    field23: FIELD_TYPE,
    field24: FIELD_TYPE,
    field25: FIELD_TYPE,
    field26: FIELD_TYPE,
    field27: FIELD_TYPE,
    field28: FIELD_TYPE,
    field29: FIELD_TYPE,
    field30: FIELD_TYPE,
    field31: FIELD_TYPE,
    field32: FIELD_TYPE,
    ):
    with computation(FORWARD), interval(1,None):
        field01[0,0,0]= 2
        field02[0,0,0]= 2
        field03[0,0,0]= 2
        field04[0,0,0]= 2
        field05[0,0,0]= 2
        field06[0,0,0]= 2
        field07[0,0,0]= 2
        field08[0,0,0]= 2
        field09[0,0,0]= 2
        field10[0,0,0]= 2
        field11[0,0,0]= 2
        field12[0,0,0]= 2
        field13[0,0,0]= 2
        field14[0,0,0]= 2
        field15[0,0,0]= 2
        field16[0,0,0]= 2
        field17[0,0,0]= 2
        field18[0,0,0]= 2
        field19[0,0,0]= 2
        field20[0,0,0]= 2
        field21[0,0,0]= 2
        field22[0,0,0]= 2
        field23[0,0,0]= 2
        field24[0,0,0]= 2
        field25[0,0,0]= 2
        field26[0,0,0]= 2
        field27[0,0,0]= 2
        field28[0,0,0]= 2
        field29[0,0,0]= 2
        field30[0,0,0]= 2
        field31[0,0,0]= 2
        field32[0,0,0]= 2
    
    with computation(FORWARD), interval(1,None):
        ret_field[0,0,0] = field01[0,0,0] \
            + field02[0,0,0] \
            + field03[0,0,0] \
            + field04[0,0,0] \
            + field05[0,0,0] \
            + field06[0,0,0] \
            + field07[0,0,0] \
            + field08[0,0,0] \
            + field09[0,0,0] \
            + field10[0,0,0] \
            + field11[0,0,0] \
            + field12[0,0,0] \
            + field13[0,0,0] \
            + field14[0,0,0] \
            + field15[0,0,0] \
            + field16[0,0,0] \
            + field17[0,0,0] \
            + field18[0,0,0] \
            + field19[0,0,0] \
            + field20[0,0,0] \
            + field21[0,0,0] \
            + field22[0,0,0] \
            + field23[0,0,0] \
            + field24[0,0,0] \
            + field25[0,0,0] \
            + field26[0,0,0] \
            + field27[0,0,0] \
            + field28[0,0,0] \
            + field29[0,0,0] \
            + field30[0,0,0] \
            + field31[0,0,0] \
            + field32[0,0,0]

@stencil(backend=backend)
def test_multidim_16x(
    ret_field: MULTIDIM_TYPE,
    field01: MULTIDIM_TYPE,
    field02: MULTIDIM_TYPE,
    field03: MULTIDIM_TYPE,
    field04: MULTIDIM_TYPE,
    field05: MULTIDIM_TYPE,
    field06: MULTIDIM_TYPE,
    field07: MULTIDIM_TYPE,
    field08: MULTIDIM_TYPE,
    field09: MULTIDIM_TYPE,
    field10: MULTIDIM_TYPE,
    field11: MULTIDIM_TYPE,
    field12: MULTIDIM_TYPE,
    field13: MULTIDIM_TYPE,
    field14: MULTIDIM_TYPE,
    field15: MULTIDIM_TYPE,
    field16: MULTIDIM_TYPE
    ):
    with computation(FORWARD), interval(1,None):
        field01[0,0][0]= 2
        field02[0,0][0]= 2
        field03[0,0][0]= 2
        field04[0,0][0]= 2
        field05[0,0][0]= 2
        field06[0,0][0]= 2
        field07[0,0][0]= 2
        field08[0,0][0]= 2
        field09[0,0][0]= 2
        field10[0,0][0]= 2
        field11[0,0][0]= 2
        field12[0,0][0]= 2
        field13[0,0][0]= 2
        field14[0,0][0]= 2
        field15[0,0][0]= 2
        field16[0,0][0]= 2
    
    with computation(FORWARD), interval(1,None):
        ret_field[0,0][0] = field01[0,0][0] \
            + field02[0,0][0] \
            + field03[0,0][0] \
            + field04[0,0][0] \
            + field05[0,0][0] \
            + field06[0,0][0] \
            + field07[0,0][0] \
            + field08[0,0][0] \
            + field09[0,0][0] \
            + field10[0,0][0] \
            + field11[0,0][0] \
            + field12[0,0][0] \
            + field13[0,0][0] \
            + field14[0,0][0] \
            + field15[0,0][0] \
            + field16[0,0][0]

@stencil(backend=backend)
def test_multidim_32x(
    ret_field: MULTIDIM_TYPE,
    field01: MULTIDIM_TYPE,
    field02: MULTIDIM_TYPE,
    field03: MULTIDIM_TYPE,
    field04: MULTIDIM_TYPE,
    field05: MULTIDIM_TYPE,
    field06: MULTIDIM_TYPE,
    field07: MULTIDIM_TYPE,
    field08: MULTIDIM_TYPE,
    field09: MULTIDIM_TYPE,
    field10: MULTIDIM_TYPE,
    field11: MULTIDIM_TYPE,
    field12: MULTIDIM_TYPE,
    field13: MULTIDIM_TYPE,
    field14: MULTIDIM_TYPE,
    field15: MULTIDIM_TYPE,
    field16: MULTIDIM_TYPE,
    field17: MULTIDIM_TYPE,
    field18: MULTIDIM_TYPE,
    field19: MULTIDIM_TYPE,
    field20: MULTIDIM_TYPE,
    field21: MULTIDIM_TYPE,
    field22: MULTIDIM_TYPE,
    field23: MULTIDIM_TYPE,
    field24: MULTIDIM_TYPE,
    field25: MULTIDIM_TYPE,
    field26: MULTIDIM_TYPE,
    field27: MULTIDIM_TYPE,
    field28: MULTIDIM_TYPE,
    field29: MULTIDIM_TYPE,
    field30: MULTIDIM_TYPE,
    field31: MULTIDIM_TYPE,
    field32: MULTIDIM_TYPE,
    ):
    with computation(FORWARD), interval(1,None):
        field01[0,0][0]= 2
        field02[0,0][0]= 2
        field03[0,0][0]= 2
        field04[0,0][0]= 2
        field05[0,0][0]= 2
        field06[0,0][0]= 2
        field07[0,0][0]= 2
        field08[0,0][0]= 2
        field09[0,0][0]= 2
        field10[0,0][0]= 2
        field11[0,0][0]= 2
        field12[0,0][0]= 2
        field13[0,0][0]= 2
        field14[0,0][0]= 2
        field15[0,0][0]= 2
        field16[0,0][0]= 2
        field17[0,0][0]= 2
        field18[0,0][0]= 2
        field19[0,0][0]= 2
        field20[0,0][0]= 2
        field21[0,0][0]= 2
        field22[0,0][0]= 2
        field23[0,0][0]= 2
        field24[0,0][0]= 2
        field25[0,0][0]= 2
        field26[0,0][0]= 2
        field27[0,0][0]= 2
        field28[0,0][0]= 2
        field29[0,0][0]= 2
        field30[0,0][0]= 2
        field31[0,0][0]= 2
        field32[0,0][0]= 2
    
    with computation(FORWARD), interval(1,None):
        ret_field[0,0] = field01[0,0][0] \
            + field02[0,0][0] \
            + field03[0,0][0] \
            + field04[0,0][0] \
            + field05[0,0][0] \
            + field06[0,0][0] \
            + field07[0,0][0] \
            + field08[0,0][0] \
            + field09[0,0][0] \
            + field10[0,0][0] \
            + field11[0,0][0] \
            + field12[0,0][0] \
            + field13[0,0][0] \
            + field14[0,0][0] \
            + field15[0,0][0] \
            + field16[0,0][0] \
            + field17[0,0][0] \
            + field18[0,0][0] \
            + field19[0,0][0] \
            + field20[0,0][0] \
            + field21[0,0][0] \
            + field22[0,0][0] \
            + field23[0,0][0] \
            + field24[0,0][0] \
            + field25[0,0][0] \
            + field26[0,0][0] \
            + field27[0,0][0] \
            + field28[0,0][0] \
            + field29[0,0][0] \
            + field30[0,0][0] \
            + field31[0,0][0] \
            + field32[0,0][0]