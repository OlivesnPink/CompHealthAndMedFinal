# Convolutional Neural Networks for Eye Disease Detection

Laura Cahill, Olivia Jones-Martin, Roberto Mercado, Zuriel Pagan

## Project Overview

Our goal was to set up and train convolution neural networks that can detect eye diseases within fundus images. The dataset we used is the [Retinal Fundus Multi-disease Image Dataset (RFMiD)](https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid), which contained 46 classes (1 healthy and 45 disease classes). However, we noticed that the original dataset had a severe class imbalance during exploratory data analysis, so we worked with the version used in the RIADD Challenge that contained 29 classes, where diseases with less than 10 examples were labeled as "other." We worked with this dataset on University of Massachusetts Lowell's GPU servers, which allowed us to quickly experiment with neural networks. However, we had to adhere to the GPU server's disk quota, so we compressed the images to 1024x1024 before uploading them to the servers.

We trained and evaluated ResNet-18 and ResNet-34 as our baseline models, Inception v3 as our primary model, and EfficientNet B1 as our refined model. For each of the models we've worked on, we set up the training and testing logic for multi-label classification, loaded the metadata associated with the training, validation, testing sets included with RFMiD, preprocessed the images from our dataset to fit within the input layers of our models, and executed our training and testing algorithms on those sets. For our loss function, we worked with binary class entropy with logits, which uses the weights we calculated to place more emphasis on the rare class examples. In addition, we used softmax to predict disease labels whose probabilities were greater than 50%. Each of us experimented with different optimizer, data loading, and training setups to squeeze out as much performance as we can from our individual models. For more information, please see EyeDiseaseDetectionFinalSubmission.ipynb for more information.

## Requirements

Conda is highly recommended to download the following dependencies into a single environment. You may also use pip create a virtual environment with the following dependencies as well.

* Python 3.12
* ipython
* jupyter
* ipykernel
* matplotlib
* numpy
* scikit-learn
* torch
* torchvision
* torchmetrics
* pandas
* imageio
* seaborn
* scipy
* Pillow

Note that if you are executing EyeDiseaseDetectionFinalSubmission.ipynb on the GPU servers, we recommend using the following command to install PyTorch into your virtual environment. We have encountered issues where PyTorch could not recognize the GPUs connected to the servers due to incompatible versioning. The following command fixed that problem for us.

```
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir
```

## Expected Results

The EyeDiseaseDetectionFinalSubmission.ipynb file contains our expected results, which was gathered from the individual notebooks we have worked on for each of our models. In general, our models have exceed our expectations in terms of accuracy. However, our models fell short of our F1 score goal, which was most likely caused by the severe class imbalance. We were researching various methods that can help us address this, like K-fold cross validation, thresholding, and bagging for example. Despite our efforts, the highest F1 scores we have received was 44% from our EfficientNet B1 model on 29 classes and 38% from our Inception v3 model on 46 classes. Should we have more time, we believe we would be able to find ways to increase this metric for our models.

## Execution

The cells in EyeDiseaseDetectionFinalSubmission.ipynb are set up to be executed in sequential order. We recommend executing the Jupyter Notebook in Visual Studio Code using the environment setup we have discussed earlier. The Jupyter notebook also contains a section on the packages we've used. We've also included the expected outputs from the notebook versions we worked on individually in case the code in EyeDiseaseDetectionFinalSubmission.ipynb breaks for whatever reason.

## Team Contributions

### Laura Cahill

* Preprocessing Lead
* Final model research
* Final presentation
* ResNet-18 model training and evaluation

### Olivia Jones-Martin

* Model Developer
* EfficientNet B1 model training and evaluation

### Roberto Mercado

* Team Lead
* Model documentation
* Inception v3 model training and evaluation

### Zuriel Pagan

* Evaluations and Metrics Lead
* Presentation and Visualization Lead
* ResNet-34 model training and evaluation

## Known Limitations

EyeDiseaseDetectionFinalSubmission.ipynb tends to run slowly, even on the GPU servers. For the code involving Inception v3, it will take roughly 18 minutes to reproduce the results shown in the notebook. EfficientNet B1 takes the longest amount of time out of all our models due to the bagging method used.

## Resources

1. [RFMiD: Retinal Image Analysis for multi-Disease Detection challenge](https://www.sciencedirect.com/science/article/pii/S1361841524002901)