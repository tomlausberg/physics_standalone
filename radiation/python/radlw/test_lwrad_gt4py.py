import numpy as np
import sys
import os
print(sys.path)
IS_TEST = (os.getenv("IS_TEST") == "True") if ("IS_TEST" in os.environ) else False

IS_DAINT = (os.getenv("IS_DAINT") == "True") if ("IS_DAINT" in os.environ) else False

if IS_TEST:
    sys.path.insert(0, "/deployed/radiation/python")
elif IS_DAINT:
    sys.path.insert(0, "/users/tlausber/physics_standalone/radiation/python")
else:
    sys.path.insert(0, "/work/radiation/python")

print(sys.path)
from radlw.radlw_main_gt4py import RadLWClass
#from radlw_main_gt4py import RadLWCLass
me = 0
iovrlw = 1
isubclw = 2

for rank in range(6):
    rlw = RadLWClass(rank, iovrlw, isubclw)
    rlw.create_input_data(rank)
    #rlw.lwrad(rank, do_subtest=False)
    rlw.time_lwrad(rank)
