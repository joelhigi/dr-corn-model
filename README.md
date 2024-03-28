# Dr. Corn: Model & Server

## 0. Introduction
This repo houses the model training Python script and the server script for the Dr. Corn project. It is important to understand that, as it is, the server was designed to run on a Raspberry Pi 3 with 4GB of RAM. However, a Raspberry Pi 4 would be better suited for the task due to its advanced specs.

To get started, install the dependencies that are needed to run the project. They can be found in the requirements.txt file.

```python
pip install -r requirements.txt
```

This project relies on [Smaranjit Ghose][https://www.kaggle.com/smaranjitghose]'s [Maize Leaf Disease Dataset][https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset] from [Kaggle][https://www.kaggle.com]. To train a model, the images must be downloaded and stored inside the dataset directory under the appropriate sub-directories.

## 1. Training the model
Navigate to the project directory in your terminal, then run

```python
python custom_disease_trainer.py
```

to start the training process. Note that there are some absolute directory paths that must be changed beforehand. The training process can be lengthy, depending on the specifications of your machine. I recommend carrying it out online if you can afford good GPUs.

## 2. Classifying images
The command

```python
python DrCornServer.py
```
will kickstart the simple Python server, which will then listen for socket connections. It receives images as bytestreams, reconstructs them, stores them in the "images" directory, then uses the model to classify them. The server sends back a string that contains the findings of the model, then loops back into a listening status.