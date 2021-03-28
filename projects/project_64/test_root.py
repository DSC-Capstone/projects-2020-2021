import uproot

h = uproot.create("testdata/test.root")

h["deepntuplizer/tree"] = {}

with open("config/data-params.json") as fh:
    params = json.load(fh)

num_lines = 3

for feat in params["features"]:
    h["deepntuplizer/tree"][feat] = [0] * num_lines
    
for label in params["labels"]:
    h["deepntuplizer/tree"][label] = [1] * num_lines

for spec in params["spectators"]:
    h["deepntuplizer/tree"][spec] = [1] * num_lines