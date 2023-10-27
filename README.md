# Bias in Face Recognition
### Project Documentation

</br>

### Project-Members
- Anna Stöhrer da Silva 
- Bernice Fabich
- Jan Schneeberg
- Niklas Elsaesser

### University Lecturer
 - Hans Ramsl (hans@wandb.com)

</br></br>

## Abstract
Project for the Introduction2DataScience Course at the DHBW Heilbronn with Face Recognition. Analyzing the sex based bias when we train the algorithm on Pictures of men and women, where women smile all the time and men show neutral emotions.

## Table of Contents
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Materials and Methods](#materials-and-methods)
- [Implemented Code](#implemented-code)
- [Training](#training)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction
Machine Learning Algorithms and Models improve day to day and especially facial based Algorithms are widely used in day to day applications. From unlocking phones to authentication at airports and beauty filter suggestions in Apps like Snapchat. However, this technology is often based on inherited biases, deeply hidden in its design. 



 but humanbased input is still key for good training.

 If the provided Data or the labeling process were insufficient, the outcome will be lackluster.


### Problem:
Evaluating the Bias of a face recognition algorithm based on insufficient diversified input, when men are only shown as neutral and women as emotional i.e. smiling.
</br>

### Hypothesis:
When we test the trained model on:
- male faces showing neutral facial expressions
- female faces showing happy facial expressions

A newly uploaded picture showing a men with happy facial expressions or a woman with neutral facial expressions, the model will classify the male picture as female and opposite.

## Materials and Methods
*Ausgehend von der Aufgabenstellung ist der derzeitige Stand der Technik für die Lösungsfindung zu beschreiben. Es sind z.B. die Vor- und Nachteile bisheriger Lösungen bzw. fundamentaler Lösungsprinzipien fundiert von und ggf. anderen Quellen darzulegen.*

***Description of:***

### Google Colab

### Wands & Biases

### Open CV

### Numpy

### Tensorflow / Keras @Niklas

#### Convolutional Neural Network (CNN)


## Implemented Code
*Collecting*

Taking multiple Pictures of two Women:
- 20 Pictures of Anna
- 71 Pictures of Bernice

in which they smile (happy facial expressons). Furthermore taking multiple Pictures of two Men
- 62 Pictures of Jan
- 66 Pictures of Niklas

in which they look neutral (neutral facial expressions).

<figure align="middle" alt="hfe">
  <img src="Faces/20231015_220611.jpg" width="60" />
  <img src="Faces/IMG_9388.JPG" width="100" />
  <figcaption align="middle">Happy Facial Expressions</figcaption>
</figure>
<figure align="middle">
    <img src="Faces/IMG_4256.jpg" width="100" />
    <img src="Faces/IMG_6485.jpeg" width="100" />
    <figcaption align="middle">Neutral Facial Expressions</figcaption>
</figure>

The pictures are stored on a 
<figure align="middle">
    <img src="drawio/Unbenanntes%20Diagramm.drawio-2.png" width="200" alt="folderstruct"/>
    <figcaption align="middle">Folder Structure</figcaption>
</figure>

### Training
The collected Data, consisting of pictures, must be labeled so the algorithm can learn  which characteristics in the pictures are relevant. The chosen software to label the pictures is labelImg, which was an OpenSource Projec (now part of Label Studio).

Setting up a Google Colab Environment to train the 

*labeling*\
*preprocessing*\
*training*\
*testing*

### Results

## Conclusion