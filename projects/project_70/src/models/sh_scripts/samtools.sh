#! /usr/bin/bash

cd /teams/DSC180A_FA20_A00/b04genetics/group_4/opioids-od-genome-analysis/data/external/bam

counter=0
datadir=/teams/DSC180A_FA20_A00/b04genetics/group_4/opioids-od-genome-analysis/data

for entry in `ls`; do
    if [[ $entry == SRR* && $counter -gt 11 ]];
    then
      echo $counter
      echo $entry
      /opt/samtools-1.10/samtools rmdup $entry $datadir/processed/duplicates_removed/$entry
      /opt/samtools-1.10/samtools sort -n -O BAM ./data/processed/duplicates_removed/$entry -o ./data/processed/sorted/$entry
      /opt/samtools-1.10/samtools view -q 10 -b ./data/processed/sorted/$entry -o ./data/processed/final/$entry
    fi
    ((counter+=1))
done