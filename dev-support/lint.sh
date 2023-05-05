set -euxo pipefail

FWDIR="$(cd "$(dirname "$0")"; pwd)"
cd "$FWDIR"
cd ../

# Check imports
python3 -m isort -c downstream_tasks/ upstream_models/ --profile black
# Check code format
python3 -m black downstream_tasks/ upstream_models/
# Check lint: Not checking the code in website

set +euxo pipefail
