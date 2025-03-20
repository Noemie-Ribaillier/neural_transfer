# Neural style transfer

## Project description
This project aims at learning about neural style transfer. Neural style transfer (NST) aims at generating an image (G) from a content image (C) and a style image (S). It creates a new image by applying the style of the image S to the content of the image C. We use an optimization process that helps the generated image to gradually mix content from the content image and style from the style image to minimize the total loss function.

In this project, we will combine the picture of a cat (content image C) with the sandstone style (style image S) to generate a new image (cat picture with sandstone style).


## Model used
We use a previously trained CNN and we build on top of that, this is called transfer learning (training a network on a task and applying it to a new [different but not too different] task).
For this project, we use the VGG-19, which is a 19-layers version of the VGG network. VGG-19 works well because its convolutional layers are highly effective at capturing hierarchical representations of images. This allows it to capture both low-level features (like textures) and high-level features (like object structures). VGG-19 contains the following layers:
* 16 convolution layers, mostly using a small 3x3 kernel with a stride of 1. Some of these convolutional layers are followed by a maxpool layer to reduce the spatial dimensions (height and width) and increase the depth (number of channels or feature maps)
* 3 fully-connected layers at the end, which output class probabilities for image classification tasks (these layers are not used in NST since we don't aim at classifying, we only aim at extracting features)
This model has already been trained on the very large ImageNet database (which contains millions of labelled images across 1000 classes), and has learnt to recognize a variety of low level features (such as edges, textures and simple patterns at the shallower layers) and high level features (such as objects, shapes and parts of objects at the deeper layers).


## Optimization problem
The overall goal of NST is to minimize the difference in content between the content image and the generated image (content loss) while also minimizing the difference in style between the style image and the generated image (style loss).
So the overall goal is to minimize the total loss which is the weighted sum of the content loss and style loss with:
* content loss: it ensures that the generated image keeps the content (ie the structure or objects) of the content image
* style loss: it ensures that the generated image resembles the style of the style image

In NST, the optimization algorithm updates the pixel values rather than the neural network's parameters (usually a model updates the NN parameters) using gradient descent to minimize the total loss (content loss + style loss).

The generated image is initialized as the content image with random noise, then adjusted during the optimization problem.


### Content loss
The content loss represents the distance in content between the content image and the generated image. It is computed by comparing the feature maps (activations) of the content image and of the generated image at a specific layer (here we decide block5_conv4) of a pre-trained VGG-19 model. We decide to choose the last convolutional layer because the deepest the layer is, the highest-level the features extracted are (such as objects or parts of objects).
Steps to compute the content loss:
1. Content image and generated image are both put through the VGG-19 model and the feature map (activation) from block5_conv4 layer are kept (called a_C and a_G)
2. a_C and a_G are unrolled to get 2D matrices
3. The content loss is the mean squared error (MSE) between the feature maps of the content image (a_C) and the generated image (a_G). It is normalized by the dimensions of the tensor to ensure the error is scaled correctly


### Gram matrix
The Gram matrix is a way to grasp the style of an image. It quantifies the style of an image by capturing the correlations between the feature maps of a CNN layer.

A feature map shows how strongly a particular feature (such as an edge, corner, or texture) is present at various locations in the image. A CNN typically uses multiple filters, and each filter produces a separate feature map. So after passing through a convolutional layer, the network will have multiple feature maps. For example, if there are 32 filters in a layer, we would get 32 feature maps, forming an output tensor of shape (height, width, 32). If a filter detects a feature, the corresponding pixel in the feature map will have a higher value (stronger activation), indicating that the feature was detected there. After the convolution operation, an activation function (like ReLU) is applied to introduce non-linearity, which helps the network to learn more complex patterns. This results in the final activated feature map.

These activations are spatially structured and represent the presence of certain features (like edges, textures, etc.) at specific locations.
However, to capture style, we don't care about the spatial structure (locations) of these features but rather how the features themselves correlate across the entire image. This is because style is about patterns and textures, not the exact locations of features.

Steps to compute the Gram matrix for a feature map:
* Extract the feature map: after passing an image through a convolutional layer in a pre-trained CNN, we get a feature map. Suppose the feature map has shape (height, width, num_channels), num_channels is the depth of the feature map.
* Flatten the feature map: it is reshaped into a 2D matrix where each row represents one feature (channel) across the spatial dimensons. So, we transform the shape (height, width, num_channels) into a 2D matrix (num_channels, height*width)
* Compute the Gram matrix: G is computed as the dot product of the flattened matrix with its transpose. It captures the correlation between different channels of the feature map. The Gram matrix effectively captures the style of the image by encoding how strongly different channels of the feature map are correlated with each other. If 2 channels are highly correlated the Gram matrix will reflect that.

Gram matrix represents style because:
* The style of an image can be thought as the distribution of textures and patterns across the image. These textures often involve correlations between features at different spatial locations
* The Gram matrix captures these correlations at each layer in the CNN. High correlation between channels means that certain patterns (like brush strokes, textures or other stylistic elements) are consistent across the image
* Style loss is computed by comparing the Gram matrix of the feature maps from the style image with the Gram matrix of the feature maps from the generated image (which is a combination of the content and style). By minimizing this style loss, the network ensures that the generated image has similar style characteristics to the style image


### Style loss
To compute the style loss in NST, we use the difference between Gram matrices of the style image and the generated image. We compute it for a specific layer by normalizing the squared difference between the Gram matrix of the style image at this specific layer and the Gram matrix of the generated image at this specific layer.

The total style loss is typically the sum of the style losses from multiple layers of the network (aiming at capturing different types of style information). Indeed, we get better results if we combine the style loss from several layers (proportional sum of the style loss of each layer). Each layer should get a weight that reflects how much this layer contributes to the style:
* If we want the generated image to softly follow the style image, we should choose larger weights for deeper layers and smaller weights for the first layers
* If we want the generated image to strongly follow the style image, we should choose smaller weights for deeper layers and larger weights for the first layers

In our case we choose the following layers block1_conv1, block2_conv1, block3_conv1, block4_conv1, block5_conv1 because:
* We are only interested in convolutional layers since those are the ones that learn and extract features from an image (so we don't take the maxpool layers)
* They strike a balance between capturing both fine-grained textures (from shallow layers) and high-level structures (from deeper layers)
* Reducing the number of layers involved in style loss calculation helps save computational resources


## Script explanation, step by step
1. Load the content image 
2. Load the style image
3. Randomly initialize the image to be generated (as content with a bit of noise)
4. Load the pre-trained VGG-19 model
5. Build and compute the content loss: it represents the distance in content between the content image and the content of the generated image. We take the activated feature maps of the (content) selected layer for the content image and generated image and we compute a normalized MSE between them
6. Build and compute the style loss: it represents the distance in style between the style image and the style of the generated image. We take the activated feature maps of the (style) selected layers for the style image and generated image and we compute a normalized MSE between their Gram matrix
7. Compute the total loss (using the content loss and style loss weighted by hyperparameters)
8. Define the optimizer (and learning rate) and train the model using gradient descent to minimize the total loss by iteratively adjusting the pixel values of the generated image.


## To continue
Several steps can be done to go further with this project:
* run the optimization algorithm longer (more epochs, for example 20000)
* use a smaller learning rate (for example 0.001)
* select different layers to represent the style (redefine STYLE_LAYERS)
* alter the relative weight of content versus style (with changing alpha and beta values)


## References
This script is coming from the Deep Learning Specialization course. I enriched it to this new version.
