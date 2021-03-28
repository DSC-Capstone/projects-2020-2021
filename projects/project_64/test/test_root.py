import uproot
import json
import numpy as np
import awkward


# h = uproot.open("/teams/DSC180A_FA20_A00/b06particlephysics/train/ntuple_merged_10.root")



h = uproot.recreate("testdata/test.root", compression = None)
print(h)

#h["deepntuplizer/tree"] = 

with open("../config/test-data-params.json") as fh:
    params = json.load(fh)
    
with open("../config/test-compare-params.json") as fh:
    compare = json.load(fh)

num_lines = 3


newtree = {}
extender = {}

for feat in params["features"]:
    newtree[feat] = uproot.newbranch(np.int32)
    extender[feat] = np.tile([0], num_lines)
    
for label in params["labels"]:
    newtree[label] = uproot.newbranch(np.int32)
    extender[label] = np.tile([1], num_lines)

for spec in params["spectators"]:
    newtree[spec] = uproot.newbranch(np.int32)
    extender[spec] = np.tile([1], num_lines)
    
for jet in compare["jet_features"]:
    newtree[jet] = uproot.newbranch(np.dtype(">i8"))
    extender[jet] = awkward.array.jagged.JaggedArray.fromiter([[250.51,250.51],[250.51,250.51,250.51], [250.51,250.51,250.51]])
                                    #np.tile([250.51], num_lines)
    
for t in compare["track_features"]:
    newtree[t] = uproot.newbranch(np.dtype(">i8"))
    extender[t] = awkward.array.jagged.JaggedArray.fromiter([[15.5,15.5,15.5],[15.5,15.5], [15.5,15.5,15.5]])
                                  #np.tile([15.5], num_lines)
    #array.jagged.JaggedArray.
for sv in compare["sv_features"]:
    newtree[sv] = uproot.newbranch(np.dtype(">i8"))
    extender[sv] = awkward.array.jagged.JaggedArray.fromiter([[3000.5,3000.5,3000.5],[3000.5,3000.5,3000.5], [3000.5,3000.5]]) #np.tile([3000.5], num_lines)
    
print(newtree)
print(extender)
    
h['deepntuplizertree'] = uproot.newtree(newtree)
h['deepntuplizertree'].extend(extender)