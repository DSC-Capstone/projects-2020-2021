# Snake NeuralBackedDecisionTrees
DSC180B Group 6 Snake Classification using Neural Backed Decision Trees

Team Members:

Nikolas Racelis-Russell - A15193225

Weihua (Cedric) Zhao - A14684029 

Rui Zheng - A15046475

## Abstract

Our project focuses on building explanable image classification models on snake images from https://www.aicrowd.com/challenges/snakeclef2021-snake-species-identification-challenge/dataset_files. Our plan is to apply gradcam to images of different snake species, construct a Densenet, and transform them into decision trees to visualize the classification process. 

### Demo

Link to our website:  https://nikolettuce.github.io/DSC180B_06_NeuralBackedDecisionTrees/

#### GradCAM

Our approach, called Gradient-weighted Class Activation Mapping (Grad-CAM), uses the class-specific gradient information flowing into the final convolutional layer of a CNN to produce a coarse localization map of the important regions in the image. [1]

#### Neural Backed Decision Tree

Nowadays, machine learning has been applied in multifaceted areas of our life. While its prominence grows, its interpretabilty leaves people insecure because of the fact that people hardly see through the classfication decision process. Many attempts to solve this problem either ends up with the cost of interpretability or the cost of accuracy. Intended to avoid this dilemma in our snake classification process, we applied Neural-Backed Decision Trees (NBDTs). NBDTs replace a neural network's final linear layer with a differentiable sequence of decisions and a surrogate loss. This forces the model to learn high-level concepts and lessens reliance on highly-uncertain decisions, yielding (1) accuracy: NBDTs match or outperform modern neural networks on CIFAR, ImageNet and better generalize to unseen classes by up to 16%. Furthermore, our surrogate loss improves the original model's accuracy by up to 2%. NBDTs also afford (2) interpretability: improving human trustby clearly identifying model mistakes and assisting in dataset debugging.[2] 

##### Induced Hierarchy

When creating the hierarchy tree, one thing to note is that "[it] requires pre-trained model weights". We took row vectors wk : k ∈ [1, K], each representing a class, from the fully- connected layer weights W; then, we ran hierarchical agglomerative clustering on the normalized class representatives wk/kwkk2. Last but not least, we built the leaf nodes based on the weights. [2]

##### Loss Conversion

### Practical Use

Our project has valid applications in real life. Wild snakes are prevalent on mountains, and hikers have high possibilities to encounter them. A genuine snake classifier would provide useful information to hikers about whether the snake is venomous or not; thus, they can avoid the snake if a certain dangerous species emerge.

### Methods
Our classiciation starts with a baseline Densenet model with 5 epochs. In terms of performance, it reached a F-score of 0.495 on validation data and accuracy of 0.66 on validation data. We then managed to improve the model later with higher accuracy.

Then we applied Grad-CAM to five different snake pictures with different features. The first category includes pictures where snakes blend in with the background. The second category includes pictures where snakes differ from the background. The third category includes pictures where snakes appear with other objects, like hand. Grad-CAM performs well on localizing the target object. [3]

Last, but not least, we are currently working on transforming the CNN models to decision trees. For now, we have a general hierarchy tree where you can see each decision. Though the decision is not clear for now, we will manage to elucidate them in the upcoming weeks.

When creating the hierarchy tree, one thing to note is that "[it] requires pre-trained model weights". We took row vectors wk : k ∈ [1, K], each representing a class, from the fully- connected layer weights W; then, we ran hierarchical agglomerative clustering on the normalized class representatives wk/kwkk2. Last but not least, we built the leaf nodes based on the weights. [4]
### Results

#### Heatmaps

<img src="https://github.com/nikolettuce/DSC180B_06_NeuralBackedDecisionTrees/blob/reputation/0a00cdd2b8.jpg" width="200"/> <img src="https://github.com/nikolettuce/DSC180B_06_NeuralBackedDecisionTrees/blob/reputation/cam%201.jpg" width="200"/> <img src="https://github.com/nikolettuce/DSC180B_06_NeuralBackedDecisionTrees/blob/reputation/cam_gb%201.jpg" width="200"> <img src="https://github.com/nikolettuce/DSC180B_06_NeuralBackedDecisionTrees/blob/reputation/gb%201.jpg" width="200"> 


<img src="https://github.com/nikolettuce/DSC180B_06_NeuralBackedDecisionTrees/blob/reputation/0a7eded849.jpg" width="200"/> <img src="https://github.com/nikolettuce/DSC180B_06_NeuralBackedDecisionTrees/blob/reputation/cam%204.jpg" width="200"/> <img src="https://github.com/nikolettuce/DSC180B_06_NeuralBackedDecisionTrees/blob/reputation/cam_gb%204.jpg" width="200"> <img src="https://github.com/nikolettuce/DSC180B_06_NeuralBackedDecisionTrees/blob/reputation/gb%204.jpg" width="200"> 

<img src="https://github.com/nikolettuce/DSC180B_06_NeuralBackedDecisionTrees/blob/reputation/0a54501d6d.jpg" width="200"/> <img src="https://github.com/nikolettuce/DSC180B_06_NeuralBackedDecisionTrees/blob/reputation/cam%207.jpg" width="200"/> <img src="https://github.com/nikolettuce/DSC180B_06_NeuralBackedDecisionTrees/blob/reputation/cam_gb%207.jpg" width="200"> <img src="https://github.com/nikolettuce/DSC180B_06_NeuralBackedDecisionTrees/blob/reputation/gb%207.jpg" width="200"> 

#### Hierarchy Trees

<img src="https://github.com/nikolettuce/DSC180B_06_NeuralBackedDecisionTrees/blob/reputation/Screen%20Shot%202021-02-07%20at%205.32.34%20PM.png">

### Conclusions

## Installation

To use this project, please run build.sh and allocate at least 30 GB of hard drive space to install the data.

Then run python run.py data to first process the data before using the test target

To run the CNN and test on the snake dataset, run python run.py test

## Resources

1. Grad-CAM Paper https://arxiv.org/pdf/1610.02391v1.pdf
2. @misc{wan2021nbdt, title={NBDT: Neural-Backed Decision Trees}, author={Alvin Wan and Lisa Dunlap and Daniel Ho and Jihan Yin and Scott Lee and Henry Jin and Suzanne Petryk and Sarah Adel Bargal and Joseph E. Gonzalez}, year={2021}, eprint={2004.00221}, archivePrefix= {arXiv}, primaryClass={cs.CV} }

3. Grad-CAM code  https://github.com/jacobgil/pytorch-grad-cam

4. NBDT: https://github.com/alvinwan/neural-backed-decision-trees


