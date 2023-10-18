# FaceRecognition Bias Project Documentation

## Team Members
- Anna Stöhrer da Silva
- Bernice Fabich
- Jan Schneeberg
- Niklas Elsaesser

## Abstract
Project for the I2DataScience Course at the DHBW Heilbronn with FaceRecognition and analyzing the sex based bias.

## Table of Contents
- [To Dos](#to-dos)
- [Introduction](#introduction)
- [Documentation](#documentation)

## To Dos:
- Collecting Pictures of all participants in a Google Drive     https://drive.google.com/drive/folders/18tse-vAMD6Yn75w7iMLegNBR_w2ZuyLW?usp=sharing <span style="color:blue">
@everybody
</span>
- Labeling the Pictures (Woman only Happy, Men only Sad?) -> <span style="color:blue">
@?
</span>
- Training the Algorithm -> <span style="color:blue">
@?
</span>
- Reviewing and evaluating the results -> <span style="color:blue">
@?
</span>

## Introduction

Problem:\
Evaluating the Bias of a face recognition algorithm based on insufficient diversified input, when men are only shown as neutral and women as emotional.

## Documentation
### Collecting
Taking multiple Pictures of two Women in which they smile (show happy emotions) and taking multiple Pictures of tow Men in which they look neutral (show no emotions).

### Labeling
The collected Data, consisting of pictures, must be labeled so the algorithm can learn  which characteristics in the pictures are relevant. The chosen software to label the pictures is labelImg, which was an OpenSource Projec (now part of Label Studio).





Labeling the Data in labelImg, splitting it into a 80/20 train and test Dataset. 
![Alt text](Pictures/labelImgEx.png "LabelImg Example")

Setting up a Google Colab Environment to train the 
