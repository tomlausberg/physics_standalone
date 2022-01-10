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
MULTIDIM_TYPE = Field[band_type]

backend = os.getenv("BACKEND")

    
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
        for i in range(16):
            field01[0,0,0][i]= i
            field02[0,0,0][i]= i
            field03[0,0,0][i]= i
            field04[0,0,0][i]= i
            field05[0,0,0][i]= i
            field06[0,0,0][i]= i
            field07[0,0,0][i]= i
            field08[0,0,0][i]= i
            field09[0,0,0][i]= i
            field10[0,0,0][i]= i
            field11[0,0,0][i]= i
            field12[0,0,0][i]= i
            field13[0,0,0][i]= i
            field14[0,0,0][i]= i
            field15[0,0,0][i]= i
            field16[0,0,0][i]= i
    
    with computation(FORWARD), interval(1,None):
        for band in range(16):
            ret_field[0,0,0][band] = field01[0,0,0][band] \
                + field02[0,0,0][band] \
                + field03[0,0,0][band] \
                + field04[0,0,0][band] \
                + field05[0,0,0][band] \
                + field06[0,0,0][band] \
                + field07[0,0,0][band] \
                + field08[0,0,0][band] \
                + field09[0,0,0][band] \
                + field10[0,0,0][band] \
                + field11[0,0,0][band] \
                + field12[0,0,0][band] \
                + field13[0,0,0][band] \
                + field14[0,0,0][band] \
                + field15[0,0,0][band] \
                + field16[0,0,0][band]

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
        for i in range(16):
            field01[0,0,0][i]= 2*i
            field02[0,0,0][i]= 2*i
            field03[0,0,0][i]= 2*i
            field04[0,0,0][i]= 2*i
            field05[0,0,0][i]= 2*i
            field06[0,0,0][i]= 2*i
            field07[0,0,0][i]= 2*i
            field08[0,0,0][i]= 2*i
            field09[0,0,0][i]= 2*i
            field10[0,0,0][i]= 2*i
            field11[0,0,0][i]= 2*i
            field12[0,0,0][i]= 2*i
            field13[0,0,0][i]= 2*i
            field14[0,0,0][i]= 2*i
            field15[0,0,0][i]= 2*i
            field16[0,0,0][i]= 2*i
            field17[0,0,0][i]= 2*i
            field18[0,0,0][i]= 2*i
            field19[0,0,0][i]= 2*i
            field20[0,0,0][i]= 2*i
            field21[0,0,0][i]= 2*i
            field22[0,0,0][i]= 2*i
            field23[0,0,0][i]= 2*i
            field24[0,0,0][i]= 2*i
            field25[0,0,0][i]= 2*i
            field26[0,0,0][i]= 2*i
            field27[0,0,0][i]= 2*i
            field28[0,0,0][i]= 2*i
            field29[0,0,0][i]= 2*i
            field30[0,0,0][i]= 2*i
            field31[0,0,0][i]= 2*i
            field32[0,0,0][i]= 2*i
    
    with computation(FORWARD), interval(1,None):
        for band in range(16):
            ret_field[0,0,0][band] = field01[0,0,0][band] \
                + field02[0,0,0][band] \
                + field03[0,0,0][band] \
                + field04[0,0,0][band] \
                + field05[0,0,0][band] \
                + field06[0,0,0][band] \
                + field07[0,0,0][band] \
                + field08[0,0,0][band] \
                + field09[0,0,0][band] \
                + field10[0,0,0][band] \
                + field11[0,0,0][band] \
                + field12[0,0,0][band] \
                + field13[0,0,0][band] \
                + field14[0,0,0][band] \
                + field15[0,0,0][band] \
                + field16[0,0,0][band] \
                + field17[0,0,0][band] \
                + field18[0,0,0][band] \
                + field19[0,0,0][band] \
                + field20[0,0,0][band] \
                + field21[0,0,0][band] \
                + field22[0,0,0][band] \
                + field23[0,0,0][band] \
                + field24[0,0,0][band] \
                + field25[0,0,0][band] \
                + field26[0,0,0][band] \
                + field27[0,0,0][band] \
                + field28[0,0,0][band] \
                + field29[0,0,0][band] \
                + field30[0,0,0][band] \
                + field31[0,0,0][band] \
                + field32[0,0,0][band]

@stencil(backend=backend)
def test_taugb03(
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
    field33: FIELD_TYPE,
    field34: FIELD_TYPE,
    field35: FIELD_TYPE,
    field36: FIELD_TYPE,
    field37: FIELD_TYPE,
    field38: FIELD_TYPE,
    field39: FIELD_TYPE,
    field40: FIELD_TYPE,
    field41: FIELD_TYPE,
    field42: FIELD_TYPE,
    field43: FIELD_TYPE,
    field44: FIELD_TYPE,
    field45: FIELD_TYPE,
    field46: FIELD_TYPE,
    field47: FIELD_TYPE,
    field48: FIELD_TYPE,
    field49: FIELD_TYPE,
    # field50: FIELD_TYPE,
    # field51: FIELD_TYPE,
    # field52: FIELD_TYPE,
    # field53: FIELD_TYPE,
    # field54: FIELD_TYPE,
    # field55: FIELD_TYPE,
    # field56: FIELD_TYPE,
    # field57: FIELD_TYPE,
    # field58: FIELD_TYPE,
    # field59: FIELD_TYPE,
    # field60: FIELD_TYPE,
    # field61: FIELD_TYPE,
    # field62: FIELD_TYPE,
    # field63: FIELD_TYPE,
    # field64: FIELD_TYPE,
    # field65: FIELD_TYPE,
    # field66: FIELD_TYPE,
    # field67: FIELD_TYPE,
    # field68: FIELD_TYPE,
    # field69: FIELD_TYPE
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
        field33[0,0,0]= 2
        field34[0,0,0]= 2
        field35[0,0,0]= 2
        field36[0,0,0]= 2
        field37[0,0,0]= 2
        field38[0,0,0]= 2
        field39[0,0,0]= 2
        field40[0,0,0]= 2
        field41[0,0,0]= 2
        field42[0,0,0]= 2
        field43[0,0,0]= 2
        field44[0,0,0]= 2
        field45[0,0,0]= 2
        field46[0,0,0]= 2
        field47[0,0,0]= 2
        field48[0,0,0]= 2
        field49[0,0,0]= 2
        # field50[0,0,0]= 2
        # field51[0,0,0]= 2
        # field52[0,0,0]= 2
        # field53[0,0,0]= 2
        # field54[0,0,0]= 2
        # field55[0,0,0]= 2
        # field56[0,0,0]= 2
        # field57[0,0,0]= 2
        # field58[0,0,0]= 2
        # field59[0,0,0]= 2
        # field60[0,0,0]= 2
        # field61[0,0,0]= 2
        # field62[0,0,0]= 2
        # field63[0,0,0]= 2
        # field64[0,0,0]= 2
        # field65[0,0,0]= 2
        # field66[0,0,0]= 2
        # field67[0,0,0]= 2
        # field68[0,0,0]= 2
        # field69[0,0,0]= 2
    
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
            + field32[0,0,0] \
            + field33[0,0,0] \
            + field34[0,0,0] \
            + field35[0,0,0] \
            + field36[0,0,0] \
            + field37[0,0,0] \
            + field38[0,0,0] \
            + field39[0,0,0] \
            + field40[0,0,0] \
            + field41[0,0,0] \
            + field42[0,0,0] \
            + field43[0,0,0] \
            + field44[0,0,0] \
            + field45[0,0,0] \
            + field46[0,0,0] \
            + field47[0,0,0] \
            + field48[0,0,0] \
            + field49[0,0,0] \
            # + field50[0,0,0] \
            # + field51[0,0,0] \
            # + field52[0,0,0] \
            # + field53[0,0,0] \
            # + field54[0,0,0] \
            # + field55[0,0,0] \
            # + field56[0,0,0] \
            # + field57[0,0,0] \
            # + field58[0,0,0] \
            # + field59[0,0,0] \
            # + field60[0,0,0] \
            # + field61[0,0,0] \
            # + field62[0,0,0] \
            # + field63[0,0,0] \
            # + field64[0,0,0] \
            # + field65[0,0,0] \
            # + field66[0,0,0] \
            # + field67[0,0,0] \
            # + field68[0,0,0] \
            # + field69[0,0,0] \

# Field Normal
# 50

# Field [float, (7,)]
# Field [float, (6,2)]
# 2x Field [float, (140,)]
# Field[float,(16,585)]
# Field[float,(16,1175)]
# Field[float,(16,10)]
# Field[float,(16,4)]
# Field[float,(16,9)]
# Field[float,(16,5)]
# 2x Field[float,(16,9,19)]
# Field[float,(7,59)]

