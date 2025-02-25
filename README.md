# Neural style transfer
This project aims at learning about neural style transfer. Neural Style Learning aims at generating an image (G) from a content image (C) and a style image (S). It creates a new image by applying the style of an the image S to the content of the image C.


## Project description
In this project, we will combine the Louvre museum in Paris (content image C) with the impressionist style of Claude Monet (style image S) to generate a new image (Louvre picture with Monet style).


## Model used
We use a previously trained CNN and we build on top of that, this is called transfer learning (training a network on a  different task and applying it to a new task).
For this project, we use the VGG-19, which is a 19-layers version of the VGG network.
This model has already been trained on the very large ImageNet database, and has learned to recognize a variety of low level features (at the shallower layers) and high level features (at the deeper layers).

NST aims at building a model in which the optimization algorithm updates the pixel values rather than the neural network's parameters (usually a model updates the NN parameters). 

### Layers
When performing NST, the goal is for the content in generated image G to match the content of image C. To do so, we need the importance and use of the different layers:
* The shallower layers of a CNN tend to detect lower-level features such as edges and simple textures.
* The deeper layers tend to detect higher-level features such as more complex textures and object classes. 
We want the generated image G to have similar content as the input image C. In practice, we will usually get the most visually pleasing results if we choose a layer from somewhere in the middle of the network (neither too shallow nor too deep so that the network detects both higher-level and lower-level features).

So if we want the generated image to softly follow the style image, we should choose larger weights for deeper layers and smaller weights for the first layers. On contrary, if we want the generated image to strongly follow the style image, we should choose smaller weights for deeper layers and larger weights for the first layers.

In our case we choose these layers block1_conv1, block2_conv1, block3_conv1, block4_conv1, block5_conv1 because:
* we are only interested in convolutional layers since those are the ones that learn and extract features from an image (so we don;t take the maxpool layers)
* these layers capture low-level texture and style information that is crucial for defining the artistic style of the image. So using these layers reduces the risk of mixing content information into the style loss
* reducing the number of layers involved in style loss calculation helps save computational resources

## Gram matrix
The Gram matrix is a way to grasp a style of an image and the measure of correlation between features after each layer. 
The style matrix is also called a "Gram matrix". In linear algebra, the Gram matrix G of a set of vectors v_1,... ,v_n is the matrix of dot products, G_{ij} = v_{i}^T v_{j} = np.dot(v_{i}, v_{j}). So G_{ij} compares how similar v_i is to v_j (if they are highly similar, you would expect them to have a large dot product, and thus for G_{ij} to be large).
We want to minimize the distance between the Gram matrix of the "style" image S and the Gram matrix of the "generated" image G. 


## Script explanation, step by step
1. Load the content image 
2. Load the style image
3. Randomly initialize the image to be generated 
4. Load the VGG19 model
5. Build and compute the content loss
It represents the distance in content between the content image and the content of the generated image.
6. Build and compute the style loss
It represents the distance in style between the style image and the style of the generated image.
7. Compute the total cost (using the content cost and style cost)
8. Define the optimizer (and learning rate) and train the model


## To continue
Several steps can be done to go further with this project:
* run the optimization algorithm longer (more epochs, for example 20000)
* use a smaller learning rate (for example 0.001)
* select different layers to represent the style (redefine STYLE_LAYERS)
* alter the relative weight of content versus style (with changing alpha and beta values)