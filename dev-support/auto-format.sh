set -euxo pipefail

FWDIR="$(cd "$(dirname "$0")"; pwd)"
cd "$FWDIR"
cd ../

# Sort imports
python3 -m isort downstream_tasks/ upstream_models/ utils/ *.py --profile black
# Autoformat code
python3 -m black downstream_tasks/ upstream_models/ utils/ *.py

set +euxo pipefail
