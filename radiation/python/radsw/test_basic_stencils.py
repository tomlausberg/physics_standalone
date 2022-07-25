import numpy as np
import sys
import os

IS_TEST = (os.getenv("IS_TEST") == "True") if ("IS_TEST" in os.environ) else False

IS_DAINT = (os.getenv("IS_DAINT") == "True") if ("IS_DAINT" in os.environ) else False
IS_DOCKER = (os.getenv("IS_DOCKER") == "True") if ("IS_DOCKER" in os.environ) else False

if IS_TEST:
    sys.path.insert(0, "/deployed/radiation/python")
elif IS_DAINT:
    sys.path.insert(0, "/users/tlausber/toml_standalone/radiation/python")
elif IS_DOCKER:
    sys.path.insert(0, "/work/radiation/python")
else:
    sys.path.insert(0, "/home/tom/semester_thesis/code/toml_standalone/radiation/python")


from config import *
print(f"Running basic test with {backend} backend")

from radsw.radsw_main_gt4py import RadSWClass
from radphysparam import icldflg

import serialbox as ser

me = 0
iovrsw = 1
isubcsw = 2


rsw = RadSWClass(me, iovrsw, isubcsw, icldflg)

rsw.create_input_data(me)



for rank in range(2):
    serializer = ser.Serializer(
        ser.OpenModeKind.Read,
        os.path.join(FORTRANDATA_DIR, "SW"),
        "Generator_rank" + str(rank),
    )

    nday = serializer.read("nday", serializer.savepoint["swrad-in-000000"])[0]

    rsw = RadSWClass(rank, iovrsw, isubcsw, icldflg)
    if nday > 0:
        rsw.create_input_data(rank)
        rsw.swrad_split(rank)
