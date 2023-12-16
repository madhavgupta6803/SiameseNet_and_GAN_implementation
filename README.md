# Siamese Network for Face Verification & GAN for Face Generation

## Project Overview
This project implements a Siamese neural network to verify whether two face images are of the same person using a metric learning scheme. The project uses the Labeled Faces in the Wild (LFW) dataset and includes a Generative Adversarial Network (GAN) for generating new face images.

## Objectives
1. Use a pre-trained network on ImageNet for feature extraction.
2. Train a Siamese network with image augmentation and regularization techniques.
3. Experiment with learning rate schedulers and different optimizers.
4. Test the model performance on the LFW dataset and personal images.
5. Generate new face images using a GAN.

## Dataset
The LFW dataset was downloaded from [LFW Dataset](http://vis-www.cs.umass.edu/lfw/). The dataset was split by person into training, validation, and testing sets to ensure that the same person's images are not mixed between these sets.

## Model Training
### Siamese Network
- Pre-trained network: [ResNet-18]
- Cropping and resizing details: [Details about the image pre-processing steps]
- Image augmentation: [ColorJitter, RandomRotation, RandomHorizontalFlip, RandomResizedCrop]
- Regularization technique: [Elastic-Net Regularization]
- Metric learning scheme: [Cosine similarity or Euclidean distance, and loss function]
- Learning rate schedulers: [ReduceLROnPlateau, StepLR]
- Optimizers: [Adam, Stochastic Gradient Descent]

## DCGAN Architecture

### Generator
The generator of the DCGAN takes a latent noise vector as input and generates synthetic images. It is built with a series of five `ConvTranspose2d` layers, each followed by a batch normalization (`BatchNorm2d`) except for the final layer, and LeakyReLU activation functions. The output is passed through a `Tanh` function to produce a color image with pixel values in the range [-1, 1]. The architecture specifics are as follows:

- First layer: Transforms the latent vector into a feature map with dimensions `(num_feat_maps_gen*8) x 4 x 4`.
- Second layer: Upscales to `(num_feat_maps_gen*4) x 8 x 8`.
- Third layer: Further upscales to `(num_feat_maps_gen*2) x 16 x 16`.
- Fourth layer: Upscales to `num_feat_maps_gen x 32 x 32`.
- Final layer: Outputs a `color_channels x 64 x 64` image, which corresponds to the size of the generated image.

### Discriminator
The discriminator takes an image as input and outputs the likelihood of the image being real. Its architecture mirrors the generator but with `Conv2d` layers, which reduce the spatial dimensions while increasing the depth of feature maps. It includes the following layers:

- First layer: Reduces the `color_channels x 64 x 64` image to `num_feat_maps_dis x 32 x 32`.
- Second layer: Reduces further to `(num_feat_maps_dis*2) x 16 x 16`.
- Third layer: Reduces to `(num_feat_maps_dis*4) x 8 x 8`.
- Fourth layer: Reduces to `(num_feat_maps_dis*8) x 4 x 4`.
- Final layer: Condenses the features to a single value that represents the authenticity of the input image.

Both generator and discriminator utilize LeakyReLU activation functions to add non-linearity to the model, allowing it to learn complex patterns. The generator uses transposed convolutions to upscale the image, while the discriminator uses standard convolutions to downscale it. Batch normalization is applied in both models to stabilize training and improve convergence.

## Video link for project walkthrough
#### https://drive.google.com/file/d/1mOAbq8lyKsO687MXr5jJw54Q96pqwSeA/view?usp=sharing
