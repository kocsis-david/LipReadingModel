### Team: Ágnes Márkó, Kocsis Dávid

# Phantom Project

Currently more than 1.5 billion people live with hearing loss, which is nearly 20% of the global population and it is expected that by 2050, there could be over 2.5 billion people to have some degree of hearing loss. That is a reason why lip reading has received increasing attention in recent years. Visual Speech Recognition (VSR), also known as Automatic Lip-Reading (ALR) is a process, which aims to recognize the content of speech based on lip movements. In recent years many deep learning based methods have been proposed, besides traditional machine learning to work on the problem of ALR.

In this project we would like to present an approach for determining spoken words from a visual dataset of known words. As for now we would like to build a model that can recognize a fixed number of words presented as sequences of lip movements, which can be improved over time. Rather than relying on traditional machine learning models, we focus on neural network architectures, mainly on Convolutional Neural Networks (CNN), which we think are a great way to analyze, process and extract features not just from images but videos too. 
The dataset used in this project (LRW) consists short videos of hundreds of people speaking different words.

Vision transformers (ViTs) are another popular way for visual representation learning. That’s why we would like to consider a way to take advantage of the priors of both models. 
In order to achieve a better performance we plan detecting the lips in the frames of the videos before training the model. Based on our plans we would like to be inspired from similar architectures, but to use our own ideas and improve it to achieve a similar accuracy. 

## Dataset information:
[LRW dataset](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html)
