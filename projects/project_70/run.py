import os
import json
import sys

# import the modules under src
import src.data.import_data as import_data
import src.features.build_features as build_features
import src.models.build_model as build_model
import src.models.htseq_cts as htseq_cts

with open("./config/data_config.json") as f:
    config = json.load(f)
data_path = config["download"]["data_path"]
kallisto_idx_input = config["kallisto_index"]["input"]
kallisto_idx_output = config["kallisto_index"]["output"]
kallisto_output = config["kallisto_align"]["output"]

with open("./config/feature_config.json") as f:
    feature_config = json.load(f)
cts_dir = feature_config["cts_dir"]

with open("./config/test_config.json") as f:
    test_config = json.load(f)
test_data_path = test_config["data"]
test_kallisto_output = test_config["output"]

with open("./config/model_config.json") as f:
    model_config = json.load(f)
samtools_sh_path = model_config["samtools_sh"]
htseq_output = model_config["htseq_output"]
deseq = model_config["deseq"]
wgcna = model_config["wgcna"]

def main(target):
    if target == "run.py":
        # download the data from NCBI
        # import_data.download_seq(data_path)

        # Convert the Kallisto index
        # import_data.convert_idx(kallisto_idx_input, kallisto_idx_output)

        # Kallisto Alignment
        # import_data.align_kallisto(kallisto_idx_output, data_path, kallisto_output)

        # Read counts of Kallisto Alignments
        # build_features.make_cts(kallisto_output, cts_dir)

        # Samtools preprocessing
        # build_model.run_shell(samtools_sh_path)

        # HTSeq counts rendering
        # htseq_cts.cts(htseq_output)
        
        # DESeq2
        build_model.run_R(deseq)

        # WGCNA
        build_model.run_R(wgcna)
        print("finished the whole pipeline")
        

    if target == "test":
        import_data.test_download_seq(test_data_path)
        import_data.convert_idx(kallisto_idx_input, kallisto_idx_output)
        import_data.test_align_kallisto(kallisto_idx_output, test_data_path, test_kallisto_output)
        build_features.test_make_cts(test_kallisto_output, cts_dir)


if __name__ == "__main__":
    target = sys.argv[-1]
    main(target)
