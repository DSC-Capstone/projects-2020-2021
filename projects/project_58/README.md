# FaceMaskDetection
In an attempt to bring more transparency in artificial intelligence in a high stakes situation such as the Coronavirus pandemic, our aim was to create a model that would be able to determine if an individual was wearing a mask correctly, incorrectly, or not at all. Utilizing a subsection of the dataset MaskedFace-Net, we were able to train a model with the Inception Resnet V1 model. Moreover, as this dataset further breaks down incorrect mask usage into why, such as uncovered chin, mouth, or nose area, we aimed to apply GradCAM in order to build transparency and trust, and ultimately ensure that our model was coming to the conclusion for the right reasons.


## Project Stucture

#### Config files

These files contain important links and file paths to images and our dataset that are used by our run files 

#### Presentation

These contain brief snippets of our notebook to give you an idea of how our model was built and the underlying code and output for different features of our project. Our notebook GradCam EDA looks at implementing an algorithm that can identify what our neural network would look at to identify whether a mask would be work correctly and this is displayed in the gradcam presentation

#### run.py

This is our main file run file that calls in methods given in gradcam.py and etl.py. To run on a given image type the command  ```python run.py test``` and in order to edit the file_path for another image, run the command  ```python run.py run_grad``` and give the file path in the data_input.py file to test it out. 

### src
The src folder contains information on the functions used to train the model and our config folder contains parameter information that simplifies the working of run.py. 

## Usage of GradCam

In order to run out code on a given input path, type the command ```python run.py test``` from the main directory. 

This calls the etl.py function which presents a list of stats for our images in our dataset as well as invokes the gradcam class defined in **gradcam.py** 

Based on a predefinied path, Gradcam will be applied to the image rendering a heatmap of what the netowrk looked at to make our prediction. As seen below here are examples of what GradCam looked at to make a prediction regarding the correct wearing of FaceMaks. This increases one trust in the Neural Netowrk as it bceomes more Explainable to the Human Eye. 

![image](https://drive.google.com/uc?export=view&id=10EIantVsmZLYXwfyJI6VtpXCQ1fwNJhS)
![image](https://drive.google.com/uc?export=view&id=1kqw8QJYPR7vOBCco7p4XcVZ7xQKexdIR)

Looking at these images, neural networks make a lot more sense intuitively as we know why our network made that prediction.


# Results and Discussion

The result of our model was an accuracy of 96% in being able to classify between the three classes: improper face mask usage, no mask, and proper face mask usage. In terms of Grad-CAM, the implementation was successful in building trust and transparency within our model: the model was looking at the correct areas to determine the face mask usage.

# Website and Frontend links
Checkout our website and demo: https://elizabethmkim.github.io/FaceMaskDetection/

Checkout our frontend repo:  https://github.com/elizabethmkim/FaceMaskDetection 
