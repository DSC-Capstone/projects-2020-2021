# DSC180B Face Mask Detection

This repository focuses on the creation of a Face Mask Detection report.
Link to website: https://athena112233.github.io/DSC180B_Project_Webpage/

-----------------------------------------------------------------------------------------------------------------

### Introduction
* This repo is about training an Convolutional Neunral Network(CNN) image classification model on MaskedFace-Net. MaskedFace-Net is a dataset that contains more than 60,000 images of person either wearing a mask not. For images that contain a person wearing a mask, the dataset is further splited into either a person is wearing a mask properly or not. In this repo, we've trained a model on this dataset and also implemented a Grad-CAM algorithm on the model.

##### config 
* This folder contains the parameters for running each target. (Make sure the paths to the model and image are correct!)

##### model
* This folder contains a trained model parameters(model.pt)

##### my_image
* This folder contains all the custom images that you want the model to test on. This folder will only be created when a custom image path is provided in /config

##### notebook
* This folder contains the exploratory data analysis(EDA) of the MaskedFace-Net.

##### result
* This folder contains the images that display the result of model prediction, Grad-CAM algorithm, and Integrated Gradient.

##### src
* This folder contains the .py files for model architecture, training procedure, testing procedure, Integrated Gradient, and Grad-CAM algorithm.

##### run.py
* This `run.py` file will the specified target.

##### submission.json
* `submission.json` contains the general structure of this repo.

### How to run this repo with explanation:
*  Please visit the `EDA.ipynb` inside the `notebook folder` to understand the MaskedFace-Net. Once you understand the daatset, then you can process to run the repo

To run this repo on GPU (highly recommended), run the following lines in a terminal

```
launch-scipy-ml-gpu.sh -i j0e2r1r0/face-mask-detection -c 4 -m 8
git clone https://github.com/gatran/DSC180B-Face-Mask-Detection
cd DSC180B-Face-Mask-Detection
```

OR

To run this repo on CPU, run the following lines in a terminal

```
launch.sh -i j0e2r1r0/face-mask-detection -c 4 -m 8
git clone https://github.com/gatran/DSC180B-Face-Mask-Detection
cd DSC180B-Face-Mask-Detection
```

Then you can start to run the various targets we've provided in ```src/``` folder

To train a model on your own, run the following line in a terminal

```
python run.py training
```

To test the model, run the following line in a terminal

```
python run.py testing
```

To implement the Grad-Cam algorithm on the model, run the following in a terminal

```
python run.py gradcam
```

To implement the Integrated Gradient algorithm on the model, run the following in a terminal

```
python run.py ig
```

##### Contributions
* Gavin Tran: train the model and generate output
* Che-Wei Lin: implement gradcam and generate output
* Athena Liu: create a report based on those outputs
