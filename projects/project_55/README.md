# StockMarket_explainableAI
Contributing Members: 
- Sohyun Lee
- Shin Ehara
- Jou-Ying Lee

## Abstract
Deep learning architectures are now publicly recognized and repeatedly proven to be powerful in a wide range of high-level prediction tasks. While these algorithms’ modeling generally have beyond satisfactory performances with apposite tuning, the long-troubling issue of this specific learning lies in the un-explainability of model learning and predicting. This interpretability of “how” machines learn is often times even more important than ensuring machines outputting “correct” predictions. Especially in the field of finance, users’ ability to dissect how and why an algorithm reached a conclusion from a business standpoint is integral for later applications of i.e., to be incorporated for business decision making, etc. This project studies similar prior work done on image recognition in the financial market and takes a step further on explaining predictions outputted by the Convolutional Neural Network by applying the Grad-CAM algorithm. 

Project Website at: https://connielee99.github.io/Explainable-AI-in-Finance/

## Instructions on Runing Project
* This project aims to apply the Grad-CAM technique to a CNN model trained on images that represent closing prices during the first hour of market exchange. 
* **To engineer data and create a CNN model**, you would need to run each notebook in `notebooks` folder in the following order:
	* **1. Run every cell in `Data Processing.ipynb`**
		* This notebooke is preprocessing the raw data by extracting closing prices during first hour after market open and labeling depends on prices increasing or decreasing
		* **Input:** `raw_NIFTY100.csv`
		* **output:** `first_combined.csv` contains closing prices during the first hour of market exchange
	* **2. Run every cell in `Image Conversion.ipynb`**
		* This notebook is for an image conversion with `first_combined.csv` data. We will converse data into image with Gramian Angular Algorithm.
		*  **Input:**`first_combined.csv`
		*  **output** `.png` images in `imgs` folder
	* **3. Run every cell in `CNN.ipynb`**
		* This notebook uses FastAI, a PyTorch-based deep learning library, to build the neural network, which is able to figure out the relationship between input features and find hidden relationship with them. The input data is an image dataset with labels, which is converted from time series with Gramian Angular Field algorithm as described in the previous sections.

* **To run Grad-CAM**: 
	- Clone the Grad-CAM submodule we have included in repo homepage.
	- Navigate to <i>StockMarket_explainableAI/test</i> and put <i>test_imgs</i> folder inside this cloned submodule folder.
	- Set your directory to be in this submodule, and run the following command (feel free to modify the last part in the code for specific images):
		* python3 main.py demo1 -a resnet34 -t layer4 -i test_imgs/2017-01-03.png -k 1

## Directory Structure
* **config**</br>
	This folder contains json files for main and testing parameters
	* `data_params.json`</br>contains parameters for running main on all data
	* `test_params.json`</br>contains parameters for running main on test data
* **data**</br>
	This folder contains all stock data from time series to image representation</br>
	**imgs**</br>
	* This folder contains all images converted from time series. ex) 2017-01-02.png
	
	**raw data**</br>
	* `raw_NIFTY100.csv`</br>contains raw stoack market data; time series data

	**processed data**</br>
	* `first_combined.csv`</br>contains closing prices during the first hour of market exchange
	* `gramian_df.csv`</br>contains data after implementing gramian angular algorithm
	* `label_dir_2.csv`</br>contains data with label Whether the price goes up or down that day
* **gradcam_submodule @ fd10ff7**</br>
	This folder is the submodule for gradcam
	
* **notebooks**</br>
	This folder is the notebook directory
	
	* `CNN + Grad-CAM.ipynb`</br>is the development notebook for CNN and GradCam implementation
	* `Data Processing.ipynb`</br>is the notebook that wraps together data cleaning to feature engineering
	* `EDA.ipynb`</br>is the notebook with eda work demonstration
	* `Image Conversion.ipynb`</br>is the notebook with image conversion work done
* **references**</br>
	This folder contains additional information/references in regards to our project
	
	**report_img**</br>
	* This folder contains images extracted from coded notebooks and included in the written report

* **src**</br>
	This folder contains library codes extracted from notebooks
	
	**features**</br>
	* `build_features.py`</br>scripts to build features from merged data
	* `build_labels.py`</br>scripts to create labels for image classification
	* `build_images.py`</br>scripts to convert and save time series data to images
	
	**model**</br>
	* `gradcam.py`</br>scripts to implement gradcam

* **test**</br>
      This folder contains test results and test images
      		
* **`Dockerfile`**</br>
	This is the dockerfile necessary to build the environment for this project development
* **`run.py`**</br>
	This is the main python file to execute our program
