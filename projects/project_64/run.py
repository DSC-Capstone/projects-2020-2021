import sys
import os
import pandas as pd

import json



paths = os.getcwd() + "/" 
sys.path.insert(0, paths + "/src")
import calculator

def main(targets):
  



    if (("test" in targets) or (targets == "test")):
        t0 = json.load(open('config/room101.json'))
        print("In " + t0["rm"] + " given " + str(t0["n_occupants"]) + " occupants " + t0["activity"] + " and " + t0["expiratory_activity"]
             + " for " + str(t0["time"]) + " hours: ")
        calculator.infection_risk(t0["time"], t0["rm"], t0["n_occupants"], t0["activity"], t0["expiratory_activity"], paths + "/data/rm.csv", "max")
        print()
        t0 = json.load(open('config/rm2.json'))
        print("In " + t0["rm"] + " given " + str(t0["n_occupants"]) + " occupants " + t0["activity"] + " and " + t0["expiratory_activity"]
             + " for " + str(t0["time"]) + " hours: ")
        calculator.infection_risk(t0["time"], t0["rm"], t0["n_occupants"], t0["activity"], t0["expiratory_activity"], paths + "/data/rm.csv", "max")
        print()
        t0 = json.load(open('config/rm3.json'))
        print("In " + t0["rm"] + " given " + str(t0["n_occupants"]) + " occupants " + t0["activity"] + " and " + t0["expiratory_activity"]
             + " for " + str(t0["time"]) + " hours: ")
        calculator.infection_risk(t0["time"], t0["rm"], t0["n_occupants"], t0["activity"], t0["expiratory_activity"], paths + "/data/rm.csv", "max")
        print()
        t0 = json.load(open('config/rm4.json'))
        print("In " + t0["rm"] + " given " + str(t0["n_occupants"]) + " occupants " + t0["activity"] + " and " + t0["expiratory_activity"]
             + " for " + str(t0["time"]) + " hours: ")
        calculator.infection_risk(t0["time"], t0["rm"], t0["n_occupants"], t0["activity"], t0["expiratory_activity"], paths + "/data/rm.csv", "min")
        print()
        t0 = json.load(open('config/rm5.json'))
        print("In " + t0["rm"] + " given " + str(t0["n_occupants"]) + " occupants " + t0["activity"] + " and " + t0["expiratory_activity"]
             + " for " + str(t0["time"]) + " hours: ")
        calculator.infection_risk(t0["time"], t0["rm"], t0["n_occupants"], t0["activity"], t0["expiratory_activity"], paths + "/data/rm.csv", "min")
if __name__ == '__main__':
    
    targets = sys.argv[1:]
    
    main(targets)

    
    
    
    