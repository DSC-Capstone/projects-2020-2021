The purpose of this code is generating captions from an image and creating attention maps to help explain the model’s reasoning for the captions generated.
In order to test the robustness of the model we also use counterfactual images to see how the model's prediction changes when certain object are removed. 
Using this infomation we can also generate an object importance map to show which object in an image are most important to the caption generation process.

This project has six targets: data, train, evaluate_model, generate_viz, counterfactual_production, and explain_model. 
  - **data**: This target loads in the COCO dataset and prepares it for our image captioning model. 
  - **train**: This target builds the encoder and decoder in our image captioning model and trains it with the COCO dataset. 
  - **evaluate_model**: This target evaluates the trained model using beam search caption generation and BLEU score. 
  - **generate_viz**: This target generates a visualization of the attention maps at each stage of the caption generation process.
  - **counterfactual_production**: This target creates all of the files necessary to generate the counterfactuals (such as masks) and 
                                   then produces the counterfactual images.
  - **explain_model**: This target takes all of the counterfactual images and generates caption based on the new counterfactuals. 
                       Then compares the caption change from the original caption to generate a visualization to explain object 
                       importance using BERT similarity score.

To run the four targets, clone our repo to the dsmlp server and execute the command ‘python run.py all’ to run all the targets in sequence or 
'python run.py <target>' to run a single target. To run on a small set of test data execute: ‘python run.py test’. The output images will be saved 
 to the same directory as run.py.


Docker Repo: https://hub.docker.com/layers/140345085/afosado/capstone_project/final_docker/images/sha256-198c698d15e7a67d1bba8180a30c21dbf00dfd5e839189a94c06e6ffe96f9fac?context=explore

Demo Website: https://afosado.github.io/180b_capstone_xai/index.html
