# Bias in Face Recognition
### Project Documentation

</br>

### Project-Members
- Anna Stöhrer da Silva (ann.stoehrerdasil.23@heilbronn.dhbw.de)
- Bernice Fabich (ber.fabich.23@heilbronn.dhbw.de)
- Jan Schneeberg (jan.schneeberg.23@heilbronn.dhbw.de)
- Niklas Elsaesser (nik.elsaesser.23@heilbronn.dhbw.de)

### University Lecturer
 - Hans Ramsl (hans@wandb.com)

</br></br>

## Abstract
Project for the Introduction2DataScience Course at the DHBW Heilbronn in the first semester of the first year. Analyzing the sex based bias when we train the algorithm on Pictures of men and woman, where woman smile all the time and men show neutral emotions.

## Table of Contents
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Materials and Methods](#materials-and-methods)
- [Implemented Code](#implemented-code)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction
Machine Learning Algorithms and Models improve day to day and especially facial based Algorithms are widely used in day to day applications. From unlocking phones to authentication at airports and beauty filter suggestions in Apps like Snapchat. However, this technology is often based on inherited biases, deeply hidden in its design. 

The goal of this project is to test and show this bias in the area of gender and emotion based face recognition.


### Problem:
Evaluating the Bias of a face recognition algorithm based on insufficient diversified input, when men are only shown as neutral and woman as emotional i.e. smiling.
</br>

### Hypothesis:
When we test the trained model on:
- male faces showing neutral facial expressions
- female faces showing happy facial expressions

A newly uploaded picture showing a men with happy facial expressions or a woman with neutral facial expressions, the model will classify the male picture as female and opposite.

## Materials and Methods
Hereby an overview to give a better understanding of the underlying tools and concepts used in this project.
### Google Colaboratory
Google Colaboratory is a Cloud based, Python executing, Jupyter Notebook running interactive development environment. Its big advantage is the free access to Graphics Processing Units (GPU) and Tensor Processing Units (TPU) which allow for an increased computing power when it comes to machine learning, compared to regular computers.[3]

### Wands & Biases
Wands & Biases is a tracking and visualisation platform when doing machine learning experiments. It allows the logging of various parameters and metricks when tracking machine learning trainings. It furthermore allows to visualize and compare results to improve the model and its parameters for better results. To do all of this its integration into, in this case, Google Colab is seamless and easy.[4]

### Open CV
Open Source Computer Vision Library (OpenCV) is an open source computer vision machine learning library. The algorithm can be used to augment pictures, detect and recognice faces, which is the reason why it was chosen in this project.[5]

## Implemented Code
Collecting pictures of 2 woman and 2 men with regular Smartphones. Additionally the collected pictures got augmented to increase and diversify the dataset.

Taking multiple pictures of two Woman:
- 20 Pictures of Anna
- 71 Pictures of Bernice

in which they show happy facial expressions. Furthermore taking multiple Pictures of two Men:
- 62 Pictures of Jan
- 66 Pictures of Niklas

in which they show neutral facial expressions.

<figure align="middle" alt="hfe">
  <img src="Faces/20231015_220611.jpg" width="100" />
  <img src="Faces/IMG_9388.JPG" width="167" />
  <figcaption align="middle">Happy Facial Expressions</figcaption>
</figure>
<figure align="middle">
    <img src="Faces/IMG_4256.jpg" width="100" />
    <img src="Faces/IMG_6485.jpeg" width="100" />
    <figcaption align="middle">Neutral Facial Expressions</figcaption>
</figure>

The pictures are stored on a Google Drive for easier collaboration between the project members and the seamless integration with Google Colab. 

To increase the dataset, the pictures were augmented in the following parameters:
- **Flip**: Horizontal flipping of the picture to mirror it.[5]
- **Scaling**: Reducing the size of the image by resampling pixels and therefore resulting in a lower quality.[5]
- **Translation**: Moving the picture along its X and Y Axis.[5]
- **Noise**: Projecting a Matrix of random values, drawn from a Gaussian distribution.[1]
- **Contrast**: Increasing or decreasing *beta* value will add/substract a constant value to every pixel, resulting in a brightened or darkened picture.[8]
- **Brightness**: The *alpha* parameter decreases the contrast if *alpha* < 1 and increase the contrast if *alpha* > 1.[8]

To review the code used for the augmentation, check [augmenting_faces.ipynb](augmenting_faces.ipynb)

Augmentation of pictures is used to avoid overfitting. Overfitting describes a problem when ML-Models know their training data too well and achieve poor results on new unknown data. Data Augmentaition is used in the case of this project, to increase the availbale data and improve the overall quality of the available data. [1] 

<figure align="middle" alt="hfe">
  <img src="Faces/adjusted_IMG_9447_0.jpg" width="125" />
  <img src="Faces/20231015_220611.jpg" width="75" />
  <figcaption align="middle">Augmented Happy Facial Expressions</figcaption>
</figure>
<figure align="middle">
    <img src="Faces/adjusted_IMG_4268_0.jpg" width="100" />
    <img src="Faces/noisy_IMG_6505_2.jpg" width="100" />
    <figcaption align="middle">Augmented Neutral Facial Expressions</figcaption>
</figure>


### Preproccessing & Labeling
Since the labeling is done via the file structure and not with a labeling tool, the manual work is reduced. Below a picture of the implemented file structure.
<img src="drawio/Unbenanntes%20Diagramm.drawio-2.png" alt="Description of the image" height="300">

In the [FaceBias](FaceBias.ipynb) file under step 4 is the actual code, used to label the pictures.

To load and prepare the data [FaceBias](FaceBias.ipynb) in step 4 get pictures get read into the code and resized for an uniform data format.

#### TensorFlow
Created by the Google Brain Team TensorFlow ended up as an open source project. By now it is one of the most famous libraries in the machine learning community.[6] It is important to remark that pure TensorFlow is usually not used anymore, but rather used in combination with Keras. Keras uses code that relies on data from several family surveys to determine the risk of delivery.  [7]

 **Tensors**, the building blocks of TensorFlow are, per definition by Googles TensorFlow team:

> A tensor is a generalization of vectors and matrices to poten- tially higher dimensions. Internally, TensorFlow represents tensors as n-dimensional arrays of base datatypes.

In short, a Tensor is a multidimensional array with some dynamic properties.

A **Flow** describes an underlying graph computation framework which uses tensors for its execution.[6]

#### Convolutional Neural Network
Convolutional Neural Networks (CNNs) are a special class of neural networks that specialize in processing grid data, like image and video. CNNs consist of three distinct layers, as shown in the figure below:

- Convolutional
- Pooling
- Fully connected

The **Convolutional** layer applies one or more filters to an input (image). The filter is a matrix of integers used on a subset of the input, the same size as the filter.[6] The convolutional layer can futhermore extract local features and needs less parameters than a dense layer.[9]
![CNN architecture](assets/CNNs.png)[7]

The **Pooling** layer reduces the dimensionality of the input features and therefore the complexity of the model and follows each convolutional layer. Additionally the pooling layer is used to reduce the convolutional layers output to avoid overfitting. The pooling layer is also shown in the picture above.[6] [9]

In the **Fully Connected** (FC) layer, at least one, the neurons have complete connection to all activations from the previous layers.[6] The result is that every portion of the output is linked to the input via the *Convolutional* layer.[7] The FC layer aggregates the global information and returns a vector in the size of the number of categories. [6]

### Training
In the [FaceBias.ipynb](FaceBias.ipynb) under step 6 the model is trained to recognize faces and match the labels (emotion, gender).\
The algorithm uses a sequential model, where each layer has one input and one output layer. It uses 2 *2D-Convolutional* layers to be able to learn more complex and abstract features. The first *2D-Convolutional* layer captures basic patterns an features from the 48X48 scaled, gray picture. The second layer captures the more complex features with 128 filters with a size of 3x3. Stacking multiple *2D-Convolutional* layers improves the models ability to understand the underlying data and therefore leads to a better performance. [6] [9]

Each *2D-Convolutional* layer is followed by a max-pooling layer. A max-pooling layer reduces overfitting and improves computational efficency.[6] They help to capture the most at different levels of abstraction, by reducing the spatial dimensions it allows the model to focus on the more prominent features. It furthermore reduces the spatial dimensions of the feature maps and therefore the parameters in the following layer, shortening the training process and decreasing the complexity of the training process.[9]

To track the whole training process a callback to Weights&Biases (WandB) is made. WandB allows for comprehensive data review to improve the training process if needed. In case of this project, various Epoch sizes made a significant differenz in computing power, not recognizable without WandB.

### Testing
[bias_testing.ipynb](bias_testing.ipynb)


### Results


## Conclusion


## Citations
[1] Khoshgoftaar, Taghi M., "A survey on Image Data Augmentation for Deep Learning", [doi](https://doi.org/10.1186/s40537-019-0197-0),  2019.[Journal]\
[2] Mainzer, Klaus, "Quantencomputer: von der Quantenwelt zur Künstlichen Intelligenz", 2020.[Book]\
[3]Google, "Colaboratoy FAQ", [Site](https://research.google.com/colaboratory/faq.html), 2023.[FAQ]\
[4]WandB, "Website", [Weights&Biases](https://wandb.ai/site), 2023.[Site]\
[5]OpenCV, "About", [OpenCV](https://opencv.org/about/), 2023.[Site]\
[6]@misc {noauthor2020LearnTensorFlow,
title = {Learn TensorFlow 2.0: implement machine learning and deep learning models with Python},
editor = {Singh, Pramod [Verfasser/in] and Manure, Avinash [Verfasser/in]},
series = {Springer eBook Collection},
address = {Berkeley, CA},
publisher = {Apress},
year = {2020},
isbn = {978-1-4842-5558-2},
isbn = {9781484255605 (Druck-Ausgabe)},
isbn = {9781484255575, 9781484255599, 9781484255605 (Sekundärausgabe)},
language = {Englisch},
keywords = {TensorFlow. Maschinelles Lernen. Deep learning. Künstliche Intelligenz. Python / Programmiersprache. Programmiersprache. Programmierung},
note = {1 Online-Ressource (XVI, 164 p. 126 illus.)},
abstract = {Chapter 1: Introduction to TensorFlow 2.0 -- Chapter 2: Supervised Learning with TensorFlow 2.0 -- Chapter 3: Neural Networks and Deep Learning with TensorFlow 2.0 -- Chapter 4: Images with TensorFlow 2.0 -- Chapter 5: NLP Modeling with TensorFlow 2.0 -- Chapter 6: TensorFlow Models in Production. .},
abstract = {Learn how to use TensorFlow 2.0 to build machine learning and deep learning models with complete examples. The book begins with introducing TensorFlow 2.0 framework and the major changes from its last release. Next, it focuses on building Supervised Machine Learning models using TensorFlow 2.0. It also demonstrates how to build models using customer estimators. Further, it explains how to use TensorFlow 2.0 API to build machine learning and deep learning models for image classification using the standard as well as custom parameters. You'll review sequence predictions, saving, serving, deploying, and standardized datasets, and then deploy these models to production. All the code presented in the book will be available in the form of executable scripts at Github which allows you to try out the examples and extend them in interesting ways. You will: Review the new features of TensorFlow 2.0 Use TensorFlow 2.0 to build machine learning and deep learning models Perform sequence predictions using TensorFlow 2.0 Deploy TensorFlow 2.0 models with practical examples.},
}\
[7]@misc {Mukhopadhyay2023AdvancedData,
author = {Mukhopadhyay, Sayan},
title = {Advanced Data Analytics Using Python: With Architectural Patterns, Text and Image Classification, and Optimization Techniques},
editor = {Samanta, Pratip [Verfasser/in]},
address = {Berkeley, CA},
publisher = {Apress},
year = {2023},
edition = {2nd ed. 2023.},
isbn = {978-1-4842-8005-8},
isbn = {9781484280041, 9781484280065 (Sekundärausgabe)},
language = {Englisch},
keywords = {Artificial intelligence—Data processing.. Machine learning.. Python (Computer program language).. Artificial intelligence.},
note = {1 Online-Ressource(XVII, 249 p. 32 illus.)},
abstract = {Chapter 1: Overview of Python Language -- Chapter 2: ETL with Python -- Chapter 3: Supervised Learning and Unsupervised Learning with Python -- Chapter 4: Clustering with Python -- Chapter 5: Deep Learning & Neural Networks -- Chapter 6: Time Series Analysis -- Chapter 7: Analytics in Scale.},
abstract = {Understand advanced data analytics concepts such as time series and principal component analysis with ETL, supervised learning, and PySpark using Python. This book covers architectural patterns in data analytics, text and image classification, optimization techniques, natural language processing, and computer vision in the cloud environment. Generic design patterns in Python programming is clearly explained, emphasizing architectural practices such as hot potato anti-patterns. You'll review recent advances in databases such as Neo4j, Elasticsearch, and MongoDB. You'll then study feature engineering in images and texts with implementing business logic and see how to build machine learning and deep learning models using transfer learning. Advanced Analytics with Python, 2nd edition features a chapter on clustering with a neural network, regularization techniques, and algorithmic design patterns in data analytics with reinforcement learning. Finally, the recommender system in PySpark explains how to optimize models for a specific application. You will: Build intelligent systems for enterprise Review time series analysis, classifications, regression, and clustering Explore supervised learning, unsupervised learning, reinforcement learning, and transfer learning Use cloud platforms like GCP and AWS in data analytics Understand Covers design patterns in Python .},
}\
[8]OpenCV, "Changing the contrast and brightness of an image!", [OpenCV](https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html), 2023.[Site]\
[9]@misc {noauthor2022TensorComputation,
title = {Tensor Computation for Data Analysis},
editor = {Liu, Yipeng [Verfasser/in] and Liu, Jiani [Verfasser/in] and Long, Zhen [Verfasser/in] and Zhu, Ce [Verfasser/in]},
address = {Cham},
publisher = {Springer International Publishing},
year = {2022},
edition = {1st ed. 2022.},
isbn = {978-3-0307-4386-4},
isbn = {9783030743857, 9783030743871, 9783030743888 (Sekundärausgabe)},
language = {Englisch},
keywords = {Image processing.. Speech processing systems.. Computer engineering.. Internet of things.. Embedded computer systems.. Electronic circuits.. Signal processing.. Cooperating objects (Computer systems).},
note = {1 Online-Ressource(XX, 338 p. 132 illus., 119 illus. in color.)},
abstract = {1- Tensor Computation -- 2-Tensor Decomposition -- 3-Tensor Dictionary Learning -- 4-Low Rank Tensor Recovery -- 5-Coupled Tensor for Data Analysis -- 6-Robust Principal Tensor Component Analysis -- 7-Tensor Regression -- 8-Statistical Tensor Classification -- 9-Tensor Subspace Cluster -- 10-Tensor Decomposition in Deep Networks -- 11-Deep Networks for Tensor Approximation -- 12-Tensor-based Gaussian Graphical Model -- 13-Tensor Sketch. .},
abstract = {Tensor is a natural representation for multi-dimensional data, and tensor computation can avoid possible multi-linear data structure loss in classical matrix computation-based data analysis. This book is intended to provide non-specialists an overall understanding of tensor computation and its applications in data analysis, and benefits researchers, engineers, and students with theoretical, computational, technical and experimental details. It presents a systematic and up-to-date overview of tensor decompositions from the engineer's point of view, and comprehensive coverage of tensor computation based data analysis techniques. In addition, some practical examples in machine learning, signal processing, data mining, computer vision, remote sensing, and biomedical engineering are also presented for easy understanding and implementation. These data analysis techniques may be further applied in other applications on neuroscience, communication, psychometrics, chemometrics, biometrics, quantum physics, quantum chemistry, etc. The discussion begins with basic coverage of notations, preliminary operations in tensor computations, main tensor decompositions and their properties. Based on them, a series of tensor-based data analysis techniques are presented as the tensor extensions of their classical matrix counterparts, including tensor dictionary learning, low rank tensor recovery, tensor completion, coupled tensor analysis, robust principal tensor component analysis, tensor regression, logistical tensor regression, support tensor machine, multilinear discriminate analysis, tensor subspace clustering, tensor-based deep learning, tensor graphical model and tensor sketch. The discussion also includes a number of typical applications with experimental results, such as image reconstruction, image enhancement, data fusion, signal recovery, recommendation system, knowledge graph acquisition, traffic flow prediction, link prediction, environmental prediction, weather forecasting, background extraction, human pose estimation, cognitive state classification from fMRI, infrared small target detection, heterogeneous information networks clustering, multi-view image clustering, and deep neural network compression.},
}