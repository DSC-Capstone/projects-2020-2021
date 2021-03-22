import os
import json
import glob
import numpy as np 
import pandas as pd
import subprocess

def folder_manager():
    if not os.path.isdir('data'):
        os.system('mkdir data')
    if not os.path.isdir('data/fastqc_results'):
        os.system('mkdir data/fastqc_results')
        os.system('mkdir data/fastqc_results/esbl')
        os.system('mkdir data/fastqc_results/ctrl')
    if not os.path.isdir('data/cutadapt'):
        os.system('mkdir data/cutadapt')
        os.system('mkdir data/cutadapt/esbl')
        os.system('mkdir data/cutadapt/ctrl')
    if not os.path.isdir('data/samfiles'):
        os.system('mkdir data/samfiles')
    if not os.path.isdir('data/bamfiles'):
        os.system('mkdir data/bamfiles')
    if not os.path.isdir('data/gatk'):
        os.system('mkdir data/gatk')
    return 

def clean():
    if os.path.isdir('data'):
        os.system('rm -R data')
    return

def fastqc_helper(grp,dictionary):
    
    group = glob.glob(dictionary[(grp + '_1')])
    
    for sample in group:
        s1 = sample
        s2 = sample.replace("_1.","_2.")
        command1 = f"/opt/FastQC/fastqc {s1} --outdir=data/fastqc_results/{grp}/"
        command2 = f"/opt/FastQC/fastqc {s2} --outdir=data/fastqc_results/{grp}/"

        os.system(command1)
        os.system(command2)
        
    zipped = glob.glob(f"data/fastqc_results/{grp}/*.zip")
    
    for file in zipped:
        os.system(f"unzip {file} -d data/fastqc_results/{grp}/unzipped")
        
    return

def fastqc(dictionary):
    fastqc_helper('esbl',dictionary)
    fastqc_helper('ctrl',dictionary)
    return

def cutadapt_helper(grp,dictionary):
    
    group = glob.glob(dictionary[(grp + '_1')])
    
    for sample in group:
        s1 = sample
        f1 = s1.split('/')[-1]

        s2 = sample.replace("_1.","_2.")
        f2 = s2.split('/')[-1]

        command1 = f"cutadapt -j 4 -a {dictionary['adapter_sequence']} -o data/cutadapt/{grp}/{f1} {s1}"
        command2 = f"cutadapt -j 4 -a {dictionary['adapter_sequence']} -o data/cutadapt/{grp}/{f2} {s2}"

        os.system(command1)
        os.system(command2)
        
    return

def cutadapt(dictionary):
    cutadapt_helper('esbl',dictionary)
    cutadapt_helper('ctrl',dictionary)
    return

def bowtie2_helper(grp,dictionary):
    
    group = glob.glob(dictionary[(grp + '_1')])
    
    for sample in group:
        s1 = sample
        s2 = sample.replace("_1.","_2.")
        s = s1.split("/")[-1].split('_1.')[0]
        
        command = f"bowtie2 --threads 4 -x {dictionary['idx']} -1 {s1} -2 {s2} -S data/samfiles/{s}.sam"
        os.system(command)
        
    
def sam_converter():
    samfiles = glob.glob("data/samfiles/*.sam")
    
    for sam in samfiles:

        bamfile = sam.split('/')[-1].split('.sam')[0]
        convert = f"samtools view -S -b {sam} > data/bamfiles/{bamfile}.bam"
        sort = f"samtools sort data/bamfiles/{bamfile}.bam -o data/bamfiles/{bamfile}_sorted.bam"
        #idx = f"samtools index /home/myhaider/antibiotic-resistance/data/bamfiles/{bamfile}_sorted.bam"

        os.system(convert)
        os.system(sort)
        #os.system(idx)
    return

def bowtie2(dictionary):
    bowtie2_helper('esbl',dictionary)
    bowtie2_helper('ctrl',dictionary)
    return
    
def picard(dictionary):
    bamfiles = glob.glob("data/bamfiles/*_sorted.bam")
    
    for bam in bamfiles:
        filename = bam.split("_")[0].split('/')[-1]
        command = f"java -jar /opt/picard-tools-1.88/AddOrReplaceReadGroups.jar I={bam} O=data/bamfiles/{filename}_cleaned.bam RGID=4 RGLB=lib1 RGPL=ILLUMINA RGPU=unit1 RGSM=20"
        os.system(command)
    return
    
def gatk(dictionary):
    
    bamfiles = glob.glob("data/bamfiles/*_cleaned.bam")
    for bam in bamfiles:
        idx = f"samtools index {bam}"
        b = bam.split("/")[-1].split('_cleaned.')[0]
        command = f"gatk --java-options '-Xmx4g' HaplotypeCaller --native-pair-hmm-threads 128 -R {dictionary['idx']}.fasta -I {bam} -O data/gatk/{b}.g.vcf.gz -ERC GVCF"
        os.system(idx)
        os.system(command)
    return

def snpEff(dictionary):   
    vcffile = dictionary['vcf']
    rename = f"cp {vcffile} data/snp/final.vcf"
    replace = f"sed 's/{dictionary['Chromosome']}/Chromosome/g' 'data/snp/final.vcf' > data/snp/final.edited.vcf"
    command = f"java -jar /opt/snpEff/snpEff.jar Escherichia_coli data/snp/final.edited.vcf > data/snp/final.snpeff.vcf"
    os.system(rename)
    os.system(replace)
    os.system(command)
    return

def snpSift(dictionary):
    cc = 'SnpSift caseControl "++++++++++++++++++++++++++++++++++++------------------------------------" data/snp/final.snpeff.vcf > data/snp/final.snpeff.cc.vcf'
    filter1 = "cat  data/snp/final.snpeff.vcf |SnpSift filter '(ANN[*].IMPACT has 'HIGH') | (ANN[*].IMPACT has 'MODERATE')' >  data/snp/final.snpeff.imp.vcf"
    filter2 = "cat  data/snp/final.snpeff.vcf |SnpSift filter '(GEN[0].GT has 1 & GEN[1].GT has 1 & GEN[2].GT has 1 & GEN[3].GT has 1 & GEN[4].GT has 1 & GEN[5].GT has 1 & GEN[6].GT has 1 & GEN[7].GT has 1 & GEN[8].GT has 1 & GEN[9].GT has 1 & GEN[10].GT has 1 & GEN[11].GT has 1 & GEN[12].GT has 1 & GEN[13].GT has 1 & GEN[14].GT has 1 & GEN[15].GT has 1 & GEN[16].GT has 1 & GEN[17].GT has 1 & GEN[18].GT has 1 & GEN[19].GT has 1 & GEN[20].GT has 1 & GEN[21].GT has 1 & GEN[22].GT has 1 & GEN[23].GT has 1 & GEN[24].GT has 1 & GEN[25].GT has 1 & GEN[26].GT has 1 & GEN[27].GT has 1 & GEN[28].GT has 1 & GEN[29].GT has 1 & GEN[30].GT has 1 & GEN[31].GT has 1 & GEN[32].GT has 1 & GEN[33].GT has 1 & GEN[34].GT has 1 & GEN[35].GT has 1) & (GEN[36].GT has 0 | GEN[37].GT has 0 | GEN[38].GT has 0 | GEN[39].GT has 0 | GEN[40].GT has 0 | GEN[41].GT has 0 | GEN[42].GT has 0 | GEN[43].GT has 0 | GEN[44].GT has 0 | GEN[45].GT has 0 | GEN[46].GT has 0 | GEN[47].GT has 0 | GEN[48].GT has 0 | GEN[49].GT has 0 | GEN[50].GT has 0 | GEN[51].GT has 0 | GEN[52].GT has 0 | GEN[53].GT has 0 | GEN[54].GT has 0 | GEN[55].GT has 0 | GEN[56].GT has 0 | GEN[57].GT has 0 | GEN[58].GT has 0 | GEN[59].GT has 0 | GEN[60].GT has 0 | GEN[61].GT has 0 | GEN[62].GT has 0 | GEN[63].GT has 0 | GEN[64].GT has 0 | GEN[65].GT has 0 | GEN[66].GT has 0 | GEN[67].GT has 0 | GEN[68].GT has 0 | GEN[69].GT has 0 | GEN[70].GT has 0 | GEN[71].GT has 0)' >  data/snp/final.snpeff.fff.vcf"
    ext1 = 'SnpSift extractFields data/snp/final.snpeff.fff.vcf "CHROM" "POS" "ANN[*].ALLELE" > data/snp/ale.txt'
    ext2 = 'SnpSift extractFields data/snp/final.snpeff.fff.vcf "CHROM" "POS" "ANN[*].EFFECT" > data/snp/eff.txt'
    ext3 = 'SnpSift extractFields data/snp/final.snpeff.fff.vcf "CHROM" "POS" "ID" "ANN[*].GENE" > data/snp/gene.txt'
    ext4 = 'SnpSift extractFields data/snp/final.snpeff.fff.vcf "CHROM" "POS" "ID" "ANN[*].GENEID" > data/snp/gid.txt'
    ext5 = 'SnpSift extractFields data/snp/final.snpeff.fff.vcf "CHROM" "POS" "ID" "ANN[*].FEATUREID" > data/snp/ft.txt'
    ext6 = 'SnpSift extractFields data/snp/final.snpeff.fff.vcf "CHROM" "POS" "ID" "REF" "ALT" "FILTER" > data/snp/ba.txt'
    #os.system(cc)
    #os.system(filter1)
    os.system(filter2)
    os.system(ext1)
    os.system(ext2)
    os.system(ext3)
    os.system(ext4)
    os.system(ext5)
    os.system(ext6)
        
        