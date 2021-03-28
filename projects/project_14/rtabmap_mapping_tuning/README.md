# rtabmap_mapping_tuning

Tuning pipeline for the RTABMAP algorithm

**Table of contents**
- [What does it do](#what-does-it-do)
- [Targets](#targets)
- [Usage Instructions](#usage-instructions)
- [References](#references)

## What does it do

Takes a bunch of configuration files and pose data provided by the user and returns the best out of them using popular metric ATE. It loads it into a RTABMAP base launch file that can then be used by the user after changing topics to their own car

## Targets

1. data:

  This target allows you to move data into the main repository. For formatting instructions go to test/testdata and see how the data inputted should be arranged
  
2. eda:

  This target runs an eda on the data. Currently it's setup to run on test data
  
3. tuning:

  This takes all the configurations and their respective pose data, evaluates them by generating metrics for each configuration and stores the results in a text file
  
4. generate:

  This target takes the best configuration file from earlier and attaches it to a base mapping launch file
  
5. test:

  This target runs all previous targets on test data and is mainly used to ensure the repository is still working as intended as well as give a demo of what the current targets look like
  
6. all:

  This target works similarly to the test data except it runs on the data inputs you specified in data-params.json under config

Example call: ```python run.py data eda tuning generate```


## Usage Instructions

1. Clone this repository
   ```
   git clone https://github.com/sisaha9/rtabmap_mapping_tuning.git
   ```
   Once cloned, switch directories to inside this repository

2. Build the Docker image
   ```
   docker build -t rm_tune
   docker run --rm -it rm_tune /bin/bash.
   ```

3. Modify target parameters by going to config/

4. Once you have made all the changes to the configs (you really only need to change the data inputs) run the following command
   ```
   python run.py all
   ```
   If you want to see a test run
    ```
    python run.py test
    ```
5. Once you are done copy the best mapping configuration file and launch file outside the container. Refer to this StackOverflow thread if you are unsure: https://stackoverflow.com/questions/22049212/docker-copying-files-from-docker-container-to-host?rq=1. Once done you can exit and use the base file + config in your own car

## References

- This work heavily relies on RTABMAP algorithm: http://introlab.github.io/rtabmap/

- The evaluation file and metrics is based off : https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/

- utils.py and creation of notebook for EDA taken from Aaron Fraenkel: https://github.com/afraenkel
