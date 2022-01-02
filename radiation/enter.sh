source ~/venv/bin/activate
source ~/env.daint.sh
export PYTHONPATH=/users/tlausber/gt4py/install:$PYTHONPATH
export BACKEND="gtc:gt:gpu"
export RADIATION_INPUT="~/toml_standalone/radiation/"
export IS_DAINT=True
