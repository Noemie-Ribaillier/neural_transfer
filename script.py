####################################################################################################
#####                                                                                          #####
#####                        DEEP LEARNING & ART: NEURAL STYLE TRANSFER                        #####
#####                                  Created on: 2025-02-20                                  #####
#####                                  Updated on: 2025-03-18                                  #####
#####                                                                                          #####
####################################################################################################

####################################################################################################
#####                                         PACKAGES                                         #####
####################################################################################################

# Clear the global environment
globals().clear()

# Load the libraries
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import tensorflow as tf
import pprint

# Set up the right directory
import os
os.chdir('C:/Users/Admin/Documents/Python Projects/neural_transfer')

# Set up the seed to have reproducible results
tf.random.set_seed(272)


####################################################################################################
#####                                     TRANSFER LEARNING                                    #####
####################################################################################################

# Size of the image
img_size = 400

# Load the VGG19 model using the weights from the model pre-trained on ImageNet
vgg = tf.keras.applications.VGG19(
    # Whether to include the 3 FC layers at the top/last of the network (here we don't do classification so we don't need the last layers, we just need the layers which extract information/features)
    include_top = False,
    # To be specified since include_top is False (the 3 comes from RGB colors)
    input_shape = (img_size, img_size, 3),
    weights='imagenet')

# Freeze the weights of the VGG model (to do transfer learning)
vgg.trainable = False

# Get the model's architecture (including the layer names, types, parameters, outputs, etc.) in a table format
vgg.summary()


####################################################################################################
#####                                       CONTENT COST                                       #####
####################################################################################################

# Create the function to compute the content cost
def compute_content_cost(content_output, generated_output):
    """
    Compute the content cost
    
    Inputs:
    content_output -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    generated_output -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- content cost (scalar)
    """
    # Get the last element from both inputs
    a_C = content_output[-1]
    a_G = generated_output[-1]
        
    # Retrieve dimensions from a_G
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Unroll (reshape) a_C and a_G (from (1,n_H,n_W,n_C) to (1, n_H * n_W, n_C))
    a_C_unrolled = tf.reshape(a_C, shape=[1, n_H * n_W, n_C]) 
    a_G_unrolled = tf.reshape(a_G, shape=[1, n_H * n_W, n_C]) 
    
    # Compute the cost (kind of MSE between a_C_unrolled and a_G_unrolled normalized by the dimensions of the tensor to ensure the error is scaled correctly)
    J_content = (1/(4*n_H*n_W*n_C))*tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled)))

    return J_content


####################################################################################################
#####                          GRAM MATRIX (ALSO CALLED STYLE MATRIX)                          #####
####################################################################################################

# Create a function to compute the Gram matrix
def gram_matrix(A):
    """
    Compute the Gram matrix

    Inputs:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    # Formula of GA = AA^T
    GA = tf.linalg.matmul(A,tf.transpose(A))

    return GA


####################################################################################################
#####                                        STYLE COST                                        #####
####################################################################################################

# Create the function to compute the style cost for a single layer
def compute_layer_style_cost(a_S, a_G):
    """
    Create the style cost for a single layer

    Inputs:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost
    """
    # Retrieve dimensions from the hidden layer a_G
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Unroll (reshape) the tensors from (1, n_H, n_W, n_C) to (n_C, n_H * n_W)
    a_S = tf.transpose(tf.reshape(a_S,shape=[n_H * n_W,n_C]))
    a_G = tf.transpose(tf.reshape(a_G,shape=[n_H * n_W,n_C]))

    # Computing Gram_matrices for both images S and G
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss (style cost)
    J_style_layer = (1/(4*((n_H*n_W)**2)*(n_C**2)))*tf.reduce_sum(tf.square(tf.subtract(GS,GG)))

    return J_style_layer


####################################################################################################
#####                                       STYLE WEIGHTS                                      #####
####################################################################################################

# We get better results if we "merge" style costs from several layers (proportional sum of the style cost of each layer)
# (each layer gets a weight that reflects how much this layer contributes to the style)

# List the layer names
for layer in vgg.layers:
    print(layer.name)

# Get a look at the output of a layer, for example 'block5_conv4'
vgg.get_layer('block5_conv4').output

# Choose layers to represent the style of the image and assign style costs 
# (by default we assign equal weight to each layer, and weights add up to 1)
STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)]

# Create the function to compute the overall style cost (from several chosen layers)
def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    style_image_output -- our tensorflow model
    generated_image_output --
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value
    """
    
    # Initialize the overall style cost
    J_style = 0

    # Set a_S to be the hidden layer activation from the layer we have selected.
    # The last element of the array contains the content layer image, which must not be used.
    a_S = style_image_output[:-1]

    # Set a_G to be the output of the choosen hidden layers.
    # The last element of the list contains the content layer image which must not be used.
    a_G = generated_image_output[:-1]

    # For each layer:
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):  

        # Get the style of the style image "S" from the current layer
        a_S_i = a_S[i]

        # Get the style of the generated image "G" from the current layer
        a_G_i = a_G[i]

        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S_i, a_G_i)

        # Add weight * J_style_layer of this layer to overall style cost (add the weighted style cost to the overall style cost)
        J_style += weight[1] * J_style_layer

    # Return the overall style cost 
    return J_style


####################################################################################################
#####                           DEFINING THE TOTAL COST TO OPTIMIZE                            #####
####################################################################################################

# Create the total cost function (that minimizes both the style and the content cost)
# Total cost is linear combination of the content cost and the style cost
# alpha and beta are hyperparameters that control the relative weighting between content and style

# To convert a Python function into a TF graph operation (which can lead to faster and more efficient execution compared to eager execution)
# It will significantly improve the performance of our code by enabling TF to optimize and parallelize the computations
@tf.function()
def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost
    J_style -- style cost
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """

    J = alpha*J_content+beta*J_style
    
    return J


####################################################################################################
#####                                  LOAD THE CONTENT IMAGE                                  #####
####################################################################################################

# Load and reshape the "content" image C
content_image = np.array(Image.open("images/persian_cat.jpg").resize((img_size, img_size)))
content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))

# Show the content image C and its shape
print(content_image.shape)
imshow(content_image[0])
plt.axis('off')
plt.show()


####################################################################################################
#####                                   LOAD THE STYLE IMAGE                                   #####
####################################################################################################

# Load and reshape the "style" image S
style_image =  np.array(Image.open("images/monet.jpg").resize((img_size, img_size)))
style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))

# Show the style image S and its shape
print(style_image.shape)
imshow(style_image[0])
plt.axis('off')
plt.show()


####################################################################################################
#####                       RANDOMLY INITIALIZE THE IMAGE TO BE GENERATED                      #####
####################################################################################################

# Set up generated image to content image (Variable to be mutuable and float32 to normalize)
generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))

# Create uniform noise with shape of the generated image
noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)

# Add this noise to the generated image, to create a noisy image of the content image (being the generated image)
generated_image = tf.add(generated_image, noise)

# Set up the values between 0 and 1
generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)

print(generated_image.shape)
imshow(generated_image.numpy()[0])
plt.show()


####################################################################################################
#####                               LOAD PRE-TRAINED VGG19 MODEL                               #####
####################################################################################################

# Define a function which loads the model and returns a list of the outputs for the middle layers
def get_layer_outputs(vgg, layer_names):

    """ Creates a vgg model that returns a list of intermediate output values."""
    
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

# Define the content layer and build the model
content_layer = [('block5_conv4', 1)]
vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)

# Save the outputs for the content and style layers in separate variables.
content_target = vgg_model_outputs(content_image)  # Content encoder
style_targets = vgg_model_outputs(style_image)     # Style encoder

 
####################################################################################################
#####                                   COMPUTE CONTENT COST                                   #####
####################################################################################################

# To compute the content cost, we encode (a_C) our content image using the appropriate hidden layer activations. 

# Assign the content image to be the input of the VGG model.  
# Set a_C to be the hidden layer activation from the layer we have selected
preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
a_C = vgg_model_outputs(preprocessed_content)


####################################################################################################
#####                                    COMPUTE STYLE COST                                    #####
####################################################################################################

# Compute the style image encoding
# a_S sets to be the tensor giving the hidden layer activation for `STYLE_LAYERS` using our style image.

# Assign the input of the model to be the "style" image 
preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
a_S = vgg_model_outputs(preprocessed_style)

# To convert a given tensor into a PIL image
def tensor_to_image(tensor):
    """
    Converts the given tensor into a PIL image
    
    Arguments:
    tensor -- Tensor
    
    Returns:
    Image: A PIL image
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    # If it has 4 dimensions (because added dimension for the batch)
    if np.ndim(tensor) > 3:
        # We check that indeed the 1st dimension (batch dimension) is equal to 1 (batch of only 1 image). Otherwise it will raise an error
        assert tensor.shape[0] == 1
        # In that case, we get the 1st image, meaning we remove the batch dimension
        tensor = tensor[0]
    
    # Converts the input NumPy array (tensor) into a PIL Image object
    return Image.fromarray(tensor)


####################################################################################################
#####                                      TRAIN THE MODEL                                     #####
####################################################################################################

# Use Adam optimizer to minimize the total cost with a 0.01 learning rate
# If we increase the learning rate we can speed up the style transfer, but often at the cost of quality.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# We implement the train_step() function for transfer learning
@tf.function()
def train_step(generated_image):

    # Perform a step of gradient-based optimization where generated_image is updated to minimize the loss function J
    # Records all operations performed on tensors inside the with block so that the gradients can be computed later
    with tf.GradientTape() as tape:

        # Compute a_G as the vgg_model_outputs for the current generated image
        a_G = vgg_model_outputs(generated_image)
        
        # Compute the style cost
        J_style = compute_style_cost(a_S, a_G, STYLE_LAYERS=STYLE_LAYERS)
        
        # Compute the content cost
        J_content = compute_content_cost(a_C, a_G)

        # Compute the total cost
        J = total_cost(J_content, J_style, alpha = 10, beta = 40)

    # Computes the gradient of the loss function J with respect to the variable generated_image
    grad = tape.gradient(J, generated_image)

    # Applies the computed gradient to generated_image, updating it in the direction that reduces the loss function J
    optimizer.apply_gradients([(grad, generated_image)])
    
    # generated_image.assign(clip_0_1(generated_image))
    # Set up balues between 0 and 1
    generated_image.assign(tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0))
    
    return J

# Use tf.Variable to make the generated_image mutable (so that it can be modified during the training or optimization process)
# The variable will track its gradients, allowing TensorFlow to adjust its values based on some loss function or optimization algorithm
generated_image = tf.Variable(generated_image)

# Choose 2501 epochs (we run 2501 times the model)
epochs = 501

# Show the generated image every 250 (meaning the script stops to run until we don't close the temp generated image)
# Save the images generated every 250 epochs to keep track of the generated image
for i in range(epochs):
    # print(i)
    train_step(generated_image)
    if i % 250 == 0:
        print(f"Epoch {i} ")
    if i % 250 == 0:
        image = tensor_to_image(generated_image)
        imshow(image)
        image.save(f"output/cat_image_{i}.jpg")
        plt.show() 

# Show the 3 images at the same time: the content image, the style image and the generated image (content mixed with style image)
fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 3, 1)
imshow(content_image[0])
ax.title.set_text('Content image')
ax = fig.add_subplot(1, 3, 2)
imshow(style_image[0])
ax.title.set_text('Style image')
ax = fig.add_subplot(1, 3, 3)
imshow(generated_image[0])
ax.title.set_text('Generated image')
plt.show()
