import numpy as np
import sys
import os
import json

IS_TEST = (os.getenv("IS_TEST") == "True") if ("IS_TEST" in os.environ) else False

IS_DAINT = (os.getenv("IS_DAINT") == "True") if ("IS_DAINT" in os.environ) else False
IS_DOCKER = (os.getenv("IS_DOCKER") == "True") if ("IS_DOCKER" in os.environ) else False

if IS_TEST:
    sys.path.insert(0, "/deployed/radiation/python")
elif IS_DAINT:
    sys.path.insert(0, "/users/tlausber/physics_standalone/radiation/python")
elif IS_DOCKER:
    sys.path.insert(0, "/work/radiation/python")
else:
    sys.path.insert(0, "/home/tom/semester_thesis/code/toml_standalone/radiation/python")


print(sys.path)
from radlw.radlw_main_gt4py import RadLWClass
#from radlw_main_gt4py import RadLWCLass
me = 0
iovrlw = 1
isubclw = 2


number_of_runs = 20
timings = {}
for run in range(number_of_runs):
    print(f"Run {run} of {number_of_runs}")
    for rank in range(6):
        rlw = RadLWClass(rank, iovrlw, isubclw)
        rlw.create_input_data(rank)
        current_timings = rlw.time_lwrad(rank)
        for key, val in current_timings.items():
            timings[key] = timings.get(key,0) + val


for key, val in timings.items():
    timings[key] /= 6*number_of_runs

with open("timings.json","w+") as timings_file:
    json.dump(timings,timings_file)
