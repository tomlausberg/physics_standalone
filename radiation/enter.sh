source ~/env_new.daint.sh
export PYTHONPATH=/users/tlausber/gt4py/install:$PYTHONPATH
export BACKEND="gtc:gt:gpu"
export RADIATION_INPUT="/users/tlausber/toml_standalone/radiation/"
export IS_DAINT=True
export IS_DOCKER=False
export HDF5_USE_FILE_LOCKING=FALSE
source ~/venv_new/bin/activate

