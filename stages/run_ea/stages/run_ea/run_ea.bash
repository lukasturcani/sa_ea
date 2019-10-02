# Update stk and get the correct commit.
cd /setup_environment/stk
git fetch ea
git checkout c92818290728b577856df51c7360e0c70271f767

cd -
source activate sa_ea_run_ea
PYTHONPATH=/setup_environment/stk/src python -m stk.ea $1
