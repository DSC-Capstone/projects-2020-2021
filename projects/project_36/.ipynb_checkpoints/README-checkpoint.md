# DSC180b_project

## 3.8.2021 updates - Alex
- wrote EDA notebook that is callable from command line
    - Run EDA with the following command line parameter: `-eda`
    - EDA can be run with the following parameters: `time` and `limit`
        - `python run.py -eda -time` will run the EDA and print the time to run it on completion
- Cleaned old code and adding documentation
- To do: 
    - Clean up parameters in `config/params.json` and delete unused parameters
    - Remove unused methods
    - update dockerfile with `nbconvert` and `pandoc` to run `EDA.ipynb` from command line
    - Run EDA on 1000 apps

## 3.5.2021 updates - Alex
- added argument `-log` for the `<redirect_std_out>` (save console output to log file) parameter 
- Moved SHNE_code to `src` directory

## 3.2.2021 updates - Alex

### `run.py` has been updated to include more command line arguments
- `-t`, `-test`, `-Test`: Run on test set 
- `-node2vec`, `-n2v`: Run with node2vec instead of word2vec
- `--skip-embeddings`: Skip the word embeddings stage 
- `--skip-shne`: Skip SHNE model creation final step
- `-p`, `-parse`: Only create node dictionaries `dict_A.json`, `dict_B.json`, `dict_P.json`, `dict_I.json`, `api_calls.json`, and `naming_key.json`
- `-o`, `-overwrite`: Overwrite previous node dictionaries created when parsing
- `--save-out`: Save console output to file 
- `-time`: time how long to run `main.py`

### Updated params config file. All parameters used are now found in `config/params.json`.
- All outputs will be saved under the values for `<out_path>` and `<test_out_path>`
    - Subdirectories to save configured in respective dictionary. 
        - For instance word2vec embeddings will be saved under the path `<save_dir>` in the `<word2vec-params>` dictionary int `config/params.json`
- All filenames parameterizable 
    