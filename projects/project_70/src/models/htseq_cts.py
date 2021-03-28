import HTSeq
import numpy as np
import itertools
import pandas as pd
from matplotlib import pyplot 
import os

def cts(htseq_output_path):
    # read the GFF reader
    gtf_file = HTSeq.GFF_Reader(
        "/teams/DSC180A_FA20_A00/b04genetics/group_4/opioids-od-genome-analysis/data/external/gencode.v24.annotation.gtf", end_included=True)   
    exons = HTSeq.GenomicArrayOfSets( "auto", stranded=False )
    for feature in gtf_file:
        if feature.type == "exon":
            exons[ feature.iv ] += feature.name
    iv = HTSeq.GenomicInterval("III", 23850, 23950, ".")
    # get the counts as the row names
    counts = {}
    for feature in gtf_file:
        if feature.type == "exon":
            counts[feature.name] = 0
    df = pd.DataFrame(counts.keys())
    df.columns = [['target_id']]
    file_name = os.popen('ls /teams/DSC180A_FA20_A00/b04genetics/group_4/opioids-od-genome-analysis/data/processed/duplicates_removed/').read()
    file_name = file_name.split('\n')
    for i in file_name:
        counts_i = counts
        df_i = df
        path = "/teams/DSC180A_FA20_A00/b04genetics/group_4/opioids-od-genome-analysis/data/processed/duplicates_removed/" + i
        bam_file = HTSeq.BAM_Reader(path)
        for alnmt in bam_file:
            if alnmt.aligned:
                iset = None
                for iv2, step_set in exons[alnmt.iv].steps():
                    if iset is None:
                        iset = step_set.copy()
                    else:
                        iset.intersection_update(step_set)
                if len(iset) == 1:
                    counts_i[list(iset)[0]] += 1
        df_i[i] = counts_i.values()
        name = i[:-3]+"csv"
        df_i.to_csv(htseq_output_path + name)
        print("finished ", i)
          
                
                
                
                
                
                