import os 
import json
import sys
sys.path.append('src')
from etl import *
import subprocess

def main():
    
    args = sys.argv
    
    if (len(args) > 1):
        if (args[1] == "test"):
            
            # Test project
            dictionary = json.load(open("config/test_params.json"))
            folder_manager()
            fastqc(dictionary)
            cutadapt(dictionary)
            bowtie2(dictionary)
            sam_converter()
            picard(dictionary)
            gatk(dictionary)
            return
        
        elif (args[1] == "clean"):
            print("cleaning...")
            clean()
            print("cleaned!")
            return
    else:
        dictionary = json.load(open("config/params.json"))
    
        #Full Project
        folder_manager()
        fastqc(dictionary)
        cutadapt(dictionary)
        bowtie2(dictionary)
        sam_converter()
        gatk(dictionary)
        snpEff(dictionary)
        snpSift(dictionary)
        return
    
if __name__ == '__main__':
    main()