# Digit Recogniser

In this project, I wanted to see how a neuromorphic algorithm from [1] compares to simple convolutional neural network and multilayer perceptron. I also created a little frontend component where new hand-written digits can be predicted.

### 

As commonly said, neural networks are inspired by the brain. However, the main learning paradigm used in neural network, backpropagation, is not biologically feasible. This is because brain uses local updates rule, while backpropagation (with hidden layers) uses infromation from layers above to figure out weight updates. 

Another thing is that newborns learn mostly in an unsupervised way, while most of the deep learning architectures that are changing our world require a vast amount of data (supervised learning). 

Authors try to tackle these two problems by coming up with an unsupervised algorithm that makes use of local update rules to extract features from the data. They then use a single layer (without hidden units, therefore still with local updates) trained in a classical, supervised way for MNIST and CIFAR-100. For more details check [1] and [2].

### 

This is not a 1-1 replication of the paper, I have changed a few things around and played with parameters a little bit. My simple NMF network got around 88\% compared to high 90s from CNN and MLP on test data. However, the network has around 100 trainable parameters (excluding the pre-trained synapses) and is much simpler compared to the other two.

# GIF

![](https://github.com/mkacki98/digit-recogniser-app/blob/main/digit-recogniser-demo.gif)

References:

[1] https://www.pnas.org/doi/full/10.1073/pnas.1820458116
[2] https://www.youtube.com/watch?v=4lY-oAY0aQU