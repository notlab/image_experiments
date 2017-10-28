## Style Transfer

An implementation of the style transfer algorithm described in https://arxiv.org/pdf/1508.06576.pdf. 

- The error function for the reconstruction is derived in part from the MSE between different activations of features in
VGG net (the VGG architecture is described in https://arxiv.org/pdf/1409.1556.pdf). 

- Here we use the VGG author's pre-trained weights (converted from their original Caffe format by Davi Frossard, see https://www.cs.toronto.edu/~frossard/post/vgg16/).
