from gt4py.gtscript import (
    FORWARD,
    interval,
    stencil,
    computation,
    Field,
    IJ
)
import numpy as np

FIELD_TYPE = Field[np.float64]
backend = "gtc:gt:cpu_ifirst"


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
    
        