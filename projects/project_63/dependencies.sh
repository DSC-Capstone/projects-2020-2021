# brick builder tool
# TODO : https://github.com/gtfierro/brick-builder
git clone https://github.com/devanshuDesai/brick-builder --quiet
mv brick-builder ../brick-builder
pip install -r ../brick-builder/requirements.txt --quiet

# reconciliation API
# TODO: https://github.com/BrickSchema/reconciliation-api
git clone https://github.com/Advitya17/reconciliation-api ../reconciliation-api --quiet
cp ../reconciliation-api/abbrmap.py .
