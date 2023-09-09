# Fine-tuning-the-Vision-Transformer-ViT-on-Fashion_Mnist-for-Image-Classification


## Introduction
In this project, we use the Vision Tranformer (ViT) model to classify the images in the Fashion_Mnist dataset.
This code is inspired by the HuggingFace demo notebook linked in: (https://huggingface.co/docs/transformers/model_doc/vit)

## Vision Tranformer (ViT)
ViT is a transformer-based deep learning model for computer vision tasks. It represents a big jump in the field of computer vision, since this architeure does not use any convolutional layers. 
In this model, the input image is split into a sequences of fixed-size patches (16x16 or 32x32). These patches are embedded with their absolute position into a vector which is then fed to a multi-layer transformer architecture. 
Similar to other transformer models, ViT is pre-trained on large corpus of images and then fine-tune for different computer vision tasks like image classification and object detection. Vit also outperforms traditional CNN models on multiple benchmarking datasets.

![alt text](https://github.com/AymanELS/Fine-tuning-the-Vision-Transformer-ViT-on-Fashion_Mnist-for-Image-Classification/blob/main/ViT.png)

Source: This model was introduced by the Google Brain Research Team in the followin paper: (An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale)[https://arxiv.org/abs/2010.11929]

## Dataset
The dataset used in this project is the Fashon_Mnist which consists of 60,000 images for training and 10,000 images for testing. Each image is a 28x28 grayscale image, with a label from 10 different classes. 
Source: (https://huggingface.co/datasets/fashion_mnist)

