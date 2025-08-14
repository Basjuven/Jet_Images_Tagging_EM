# Ensemble Model

This is the code repository for our paper titled **[Jet Image Tagging Using Deep Learning: An Ensemble Model](arxiv)**.


## Model Architecture

The Ensemble model (EM) use two parallel outputs from both [*ResNet50*](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) and [*InceptionV3*](https://docs.pytorch.org/vision/main/models/inception.html) models for classification of jets as described in the following image.</br>

<p align="center">
  <img width="400" src="https://github.com/Basjuven/Jet_Images_Tagging_EM/blob/main/Images/Ensemble_arch.jpg" />
</p>


The model integrates features extracted from *ResNet50* and *InceptionV3*, where a $1024$-dimensional feature vector from *ResNet50* and a $2048$-dimensional feature vector from *InceptionV3* are concatenated, forming a $3072$-dimensional representation. The fusion layer reduces the dimensionality to $512$ before passing it to a softmax classifier.


## Results

The ROC curves plots presented below provide a class-by-class visualization of the models performance for the five jet categories. They were generated for each class in a one-vs-rest approach and help to visualize how well each model discriminates among each class from the rest, which is particularly useful in cases where classes share similar jet substructures.


From the individual standpoints, *ResNet50* exhibits relatively sharper ROC profiles for classes with compact and localized radiation patterns, such as $q$-jet (class 1). This aligns with its architecture design, which favors depth and localized feature extraction through residual learning. However, its curves for more complex jet structures show slightly diminished separability, likely due to the limited multi-scale processing capacity of the network. 

**ROC curve for *ResNet50***
![alt text](https://github.com/Basjuven/Jet_Images_Tagging_EM/blob/main/Images/ROC-Resnet_5types.png "Title")

*InceptionV3*, in contrast, produces more balanced ROC curves across all classes, particularly improving separability for classes like $t$-jet (class 2) and bosonic jets (classes 3 and 4), which exhibit more distributed substructure, this can be attributed to its architectural use of parallel convolutions with varying receptive fields. 

**ROC curve for *InceptionV3***
![alt text](https://github.com/Basjuven/Jet_Images_Tagging_EM/blob/main/Images/ROC-Inception_5types.png "Title")


With the EM architecture we get a more nuanced ROC curve, the statistical and other advantages are discussed in the main paper.

**ROC curve for *Ensemble Model***
![alt text](https://github.com/Basjuven/Jet_Images_Tagging_EM/blob/main/Images/ROC-EM_5types.png "Title")


## File structure overview

We have used [JetNet](https://zenodo.org/record/6975118) dataset for training and testing our models on $5$ different types of jets. </br></br>

 - **Jet2image:** This folder contains the code to generate images in the required format for training and testing.</br></br>

 - **Binary Classification:** Contains the codes for training and testing *ResNet50*, *InceptionV3* as well as the *Ensemble Model* for binary classification between different jets.</br></br>

 - **Multi-class Classification:** Contains the codes for training and testing *ResNet50*, *InceptionV3* as well as the *Ensemble Model* for all 5 class classification between different jets.</br></br>



## License and Citation

![Creative Commons License](https://mirrors.creativecommons.org/presskit/buttons/80x15/png/by.png)

This work is licensed under a [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

We encourage use of these codes \& data in derivative works. If you use the material provided here, please cite the paper using the reference:

```
Still to get one
```

