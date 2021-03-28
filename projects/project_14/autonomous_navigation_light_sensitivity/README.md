# autonomous_navigation_light_sensitivity

Evaluation of Intel Realsense D455 Depth camera's sensitivity to light. Focuses on the tuning work of camera configuration settings with ROS rqt plugin by using SSIM and MSE as a metric to find the best parameter combination that displays high structural similarity to the image under default setting conditions (non-bright).

To run the repository, run ```python run.py test```

**Table of Contents**
- [What does it do](#what-does-it-do)
- [Targets](#Targets)
- [Usage Instructions](#usage-instructions)
- [References](#References)


## What does it do

Takes single image file (default vs. tuned) and set of image files (default vs. tuned) as test data and returns metric results (SSIM, MSE) within visualization in /results. Generates a .txt file with runtime performance stats and similarity level between sets of image files (default vs. tuned).  

## Targets

1. data:

Allows you to move data into the main repository.

2. eda:

Runs a basic EDA on the data.

3. comparison:

Performs and displays comparative metric results between our baseline (non-bright vs. bright under default configuration) and tuned (non-bright under default settings vs. bright under tuned configuration) data.

4. evaluate:

Implements and measures runtime performance of our best-tuned configuration's ability to offset light during the car's dynamic movement or runtime. Evaluated through comparing the similarity level runtime performance of default set of images (non-bright conditions) vs. tuned set of images (bright conditions).


5. test:

Runs all the previous targets on test data to confirm that all the results we output is shown in the main repository.

## Usage Instructions

1. Clone this repository.

   ```git clone https://github.com/dys525/autonomous_navigation_light_sensitivity.git```
   
2. Build the Docker image.

   ```docker build -t camera_tuning ```
   ```docker run --rm -it camera_tuning /bin/bash.```

3. Modify target parameters by going to config/

4. Once you have made all the changes to the configs (you only need to change the data inputs) run the following command

   ```python run.py all```

   If you want to see a test run, 

   ```python run.py test```
   
5. Once you are done, confirm in /results the data inputs that bring about the highest SSIM and lowest MSE, as well as a significantly improved similarity level in runtime performance evaluation (> 80%). You can then use those image data inputs and tuned configuration for the car.
   
   
## References

* utils.py and creation of notebook for EDA taken from Aaron Fraenkel: https://github.com/afraenkel

* The tuning guideline for Intel Realsense D455 Camera is based off: https://www.intel.com/content/dam/support/us/en/documents/emerging-technologies/intel-realsense-technology/BKMs_Tuning_RealSense_D4xx_Cam.pdf

