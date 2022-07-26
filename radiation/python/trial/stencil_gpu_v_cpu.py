from gt4py.gtscript import (
    FORWARD,
    interval,
    stencil,
    computation,
    Field
)

import numpy as np
import os

from radiation.python.config import FIELD_2D, FIELD_2DBOOL

backend = os.getenv("BACKEND")

@stencil(backend=backend)
def test_masking(
    mask: FIELD_2DBOOL,
    in_field1: FIELD_2D,
    in_field2: FIELD_2D,
    out_field: FIELD_2D
):
    with computation(FORWARD):
        with interval(0,1):
            if mask:
                out_field = in_field1
        with interval(1, None):
            if mask:
                out_field = in_field2

        