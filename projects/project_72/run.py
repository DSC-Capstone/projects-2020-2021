#!/usr/bin/env python
from src.data.etl import fetch_and_save,transform_data,save_zip_file
from src.model.model import loss_p,loss_s,loss_ps
from src.model.ManHoleGraph import ManHoleGraph
from src.analysis.analysis import showLoss, makePlot
import pandas as pd
import sys
import json
import os

def main(targets):
    if 'data' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)

        # make the data target
        os.mkdir("data")
        save_zip_file(data_cfg["links"][1])
    if 'graph' in targets: 
        graph = ManHoleGraph()
        graph.buildGraph()
        graph.exportCSV()
    
    if 'test' in targets: 
        with open('config/model-params.json') as fh:
            corr_cfg = json.load(fh)
        covid_series = pd.read_csv('test/testdata/test_series.csv')
        offsetDict = showLoss(covid_series["cases_specimen"],
                covid_series["cases_reported"],
                corr_cfg["correlations"][0],
                corr_cfg["correlations"][1],
                [loss_ps,loss_p,loss_s])
        covid_series["cases_specimen"] = covid_series["cases_specimen"] / 10000
        makePlot(offsetDict["sp"], offsetDict["p"], offsetDict["s"], "correlationsTestPlot.png", covid_series)
    return


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
