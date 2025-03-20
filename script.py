####################################################################################################
#####                                                                                          #####
#####                        DEEP LEARNING & ART: NEURAL STYLE TRANSFER                        #####
#####                                  Created on: 2025-02-20                                  #####
#####                                  Updated on: 2025-03-20                                  #####
#####                                                                                          #####
####################################################################################################

####################################################################################################
#####                                         PACKAGES                                         #####
####################################################################################################

# Clear the global environment
globals().clear()

# Load the libraries
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import tensorflow as tf

# Set up the right directory
import os
os.chdir('C:/Users/Admin/Documents/Python Projects/neural_transfer')

# Set up the seed to have reproducible results
tf.random.set_seed(272)


####################################################################################################
#####                                     TRANSFER LEARNING                                    #####
####################################################################################################

# Determine the size of the image (to have same size for the content and the style images)
img_size = 400

# Load the VGG-19 model using the weights from the model pre-trained on ImageNet
vgg = tf.keras.applications.VGG19(
    # Whether to include the 3 FC layers at the top/last of the network (here we don't do classification so we don't need the last layers, we just need the layers which extract information/features)
    include_top = False,
    # To be specified since include_top is False (the 3 comes from RGB colors)
    input_shape = (img_size, img_size, 3),
    # Weights from the pre-trained model on ImageNet
    weights='imagenet')

# Freeze the weights of the VGG model (to do transfer learning)
vgg.trainable = False

# Get the model's architecture (including the layer names, types, parameters, outputs, etc.)
vgg.summary()


####################################################################################################
#####                                       CONTENT COST                                       #####
####################################################################################################

# Create the function to compute the content cost
def compute_content_cost(content_output, generated_output):
    """
    Compute the content cost
    
    Inputs:
    content_output -- list containing several tensors (each tensor of dimension (1, n_H, n_W, n_C), feature map (activation) of the content image at a particular layer in the NN)
    generated_output -- list containing several tensors (each tensor of dimension (1, n_H, n_W, n_C), feature map (activation) of the generated image at the same layer in the NN)
    
    Returns: 
    J_content -- content cost (scalar)
    """
    # Get the last element from both inputs (to use the layer we decided to use for content image [since it will come at last in the layer list later in the script])
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
#####                                        GRAM MATRIX                                       #####
####################################################################################################

# Create a function to compute the Gram matrix (will be used later to compute the style cost)
def gram_matrix(A):
    """
    Compute the Gram matrix (useful to compute the style cost)

    Inputs:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    # Compute the Gram matrix: formula is GA = AA^T
    GA = tf.linalg.matmul(A,tf.transpose(A))

    return GA


####################################################################################################
#####                                 STYLE COST (FOR 1 LAYER)                                 #####
####################################################################################################

# Create the function to compute the style cost for a single layer
def compute_layer_style_cost(a_S, a_G):
    """
    Compute the style cost for a single layer

    Inputs:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), feature map (activation) of the style image at a particular layer in the NN
    a_G -- tensor of dimension (1, n_H, n_W, n_C), feature map (activation) of the generated image at a particular layer in the NN
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost (for this specific layer)
    """
    # Retrieve dimensions from the hidden layer a_G
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Unroll (reshape) the tensors from (1, n_H, n_W, n_C) to (n_C, n_H * n_W)
    a_S_unrolled = tf.transpose(tf.reshape(a_S,shape=[n_H * n_W,n_C]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G,shape=[n_H * n_W,n_C]))

    # Compute Gram matrices for both images S and G, shape will be n_C*n_C
    GS = gram_matrix(a_S_unrolled)
    GG = gram_matrix(a_G_unrolled)

    # Compute the loss (style cost) for this specific layer
    J_style_layer = (1/(4*((n_H*n_W)**2)*(n_C**2)))*tf.reduce_sum(tf.square(tf.subtract(GS,GG)))

    return J_style_layer


####################################################################################################
#####                        STYLE WEIGHTS & STYLE COST (FOR ALL LAYERS)                       #####
####################################################################################################

# List the layer names
for layer in vgg.layers:
    print(layer.name)

# Choose layers to represent the style of the image (by default we assign equal weight to each layer, and weights add up to 1)
STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)]

# Create the function to compute the overall style cost (from several chosen layers)
def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
    """
    Compute the overall style cost from several chosen layers
    
    Inputs:
    style_image_output -- list containing several tensors (each tensor of dimension (1, n_H, n_W, n_C), feature map (activation) of the style image at a particular layer in the NN)
    generated_image_output -- list containing several tensors (each tensor of dimension (1, n_H, n_W, n_C), feature map (activation) of the generated image at the same layer in the NN)
    STYLE_LAYERS -- a Python list containing several tuples (name of the layers we would like to extract style from, its weight)
    
    Returns: 
    J_style -- tensor representing a scalar , total style cost
    """
    # Initialize the overall style cost
    J_style = 0

    # Set a_S to be the hidden layer activation from the layers we have selected
    # The last element of the array contains the content layer image, which must not be used
    a_S = style_image_output[:-1]

    # Set a_G to be the hidden layer activation from the layers we have selected
    # The last element of the list contains the content layer image which must not be used
    a_G = generated_image_output[:-1]

    # Iterate on each layer (selected for the style cost)
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):  

        # Get the style of the style image "S" from the specific layer (being the feature map)
        a_S_i = a_S[i]

        # Get the style of the generated image "G" from the specific layer (being the feature map)
        a_G_i = a_G[i]

        # Compute style_cost for the specific layer
        J_style_layer = compute_layer_style_cost(a_S_i, a_G_i)

        # Add weighted style cost for this specific layer to the overall style cost 
        J_style += weight[1] * J_style_layer

    # Return the overall style cost 
    return J_style


####################################################################################################
#####                                        TOTAL COST                                        #####
####################################################################################################

# Create the total cost function (weighted sum of the style and the content cost)

# Improve the performance of our code by enabling TF to optimize and parallelize the computations
@tf.function()
def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Compute the total cost function
    
    Inputs:
    J_content -- content cost
    J_style -- style cost
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost 
    """
    # Compute the total cost
    J = alpha * J_content + beta * J_style
    
    return J


####################################################################################################
#####                                  LOAD THE CONTENT IMAGE                                  #####
####################################################################################################

# Load, resize and transform to array the content image C
content_image = np.array(Image.open("images/persian_cat.jpg").resize((img_size, img_size)))
# Reshape the content image (to have a batch dimension) and transform it to an immutable tensor (to be processed with TF)
content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))

# Show the content image C and its shape
print(content_image.shape)
imshow(content_image[0])
plt.axis('off')
plt.show()


####################################################################################################
#####                                   LOAD THE STYLE IMAGE                                   #####
####################################################################################################

# Load, resize and transform to array the style" image S
style_image = np.array(Image.open("images/sandstone.jpg").resize((img_size, img_size)))
# Reshape the style image (to have a batch dimension) and transform it to an immutable tensor (to be processed with TF)
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

# Clip the values between 0 and 1
generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)

# Show the 1st version of the generated image and its shape
print(generated_image.shape)
imshow(generated_image.numpy()[0])
plt.show()


####################################################################################################
#####                         GENERATE THE MODEL FOR THE CHOSEN LAYERS                         #####
####################################################################################################

# Define a function which loads the model and returns the model only for the specified layers
def get_layer_outputs(vgg, layer_names):
    """ 
    Create a VGG model that returns a list of intermediate output values
    
    Inputs:
    vgg -- VGG model
    layer_names -- name of the layers we want to keep

    Returns:
    model -- VGG model (with only the layers we specified)
    """
    # Create the output of the model (taking only the layer of interest)
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]
    
    # Create the model
    model = tf.keras.Model([vgg.input], outputs)

    return model


# Define the content layer (choose the last layer of VGG-19 model because it focuses on high-level features)
content_layer = [('block5_conv4', 1)]
# Create the model with the layers selected for the style cost and the layer selected for the content cost
vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)


####################################################################################################
#####               GENERATE CONTENT AND STYLE OUTPUT LINKED TO THE CHOSEN LAYERS              #####
####################################################################################################

# Transform content_image to mutable and normalized tensor
preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
# Set a_C to be the hidden layer activation from the layers we have selected (using content image as input of the VGG model)
a_C = vgg_model_outputs(preprocessed_content)

# Transform style_image to mutable and normalized tensor
preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
# Set a_S to be the hidden layer activation from the layers we have selected (using style image as input of the VGG model)
a_S = vgg_model_outputs(preprocessed_style)


####################################################################################################
#####                                      TRAIN THE MODEL                                     #####
####################################################################################################

# Use Adam optimizer to minimize the total cost with a 0.01 learning rate
# If we increase the learning rate we can speed up the style transfer, but often at the cost of quality
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


# Create the train_step() function to update the generated_image while minimizing the total cost
@tf.function()
def train_step(generated_image):
    '''
    Update the generated_image while minimizing the total cost

    Inputs:
    generated_image -- generated image

    Returns:
    J -- total cost
    '''
    # Perform a step of gradient-based optimization where generated_image is updated to minimize the loss function J
    # Records all operations performed on tensors inside the with block so that the gradients can be computed later
    with tf.GradientTape() as tape:

        # Compute a_G as the vgg_model_outputs for the current generated image (which used only the chosen layers)
        a_G = vgg_model_outputs(generated_image)
        
        # Compute the style cost
        J_style = compute_style_cost(a_S, a_G, STYLE_LAYERS=STYLE_LAYERS)
        
        # Compute the content cost
        J_content = compute_content_cost(a_C, a_G)

        # Compute the total cost
        J = total_cost(J_content, J_style, alpha = 10, beta = 40)

    # Compute the gradient of the loss function J with respect to the variable generated_image
    grad = tape.gradient(J, generated_image)

    # Apply the computed gradient to generated_image, updating it (and saving) in the direction that reduces the loss function J
    optimizer.apply_gradients([(grad, generated_image)])
    
    # Clip values between 0 and 1
    generated_image.assign(tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0))
    
    return J


# Create the function to convert a given tensor into a PIL image
def tensor_to_image(tensor):
    """
    Convert the given tensor into a PIL image
    
    Inputs:
    tensor -- tensor
    
    Returns:
    image: a PIL image
    """
    # Transform the values of tensor from [0.0,1.0] to [0,255]
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)

    # If it has 4 dimensions (because added dimension for the batch)
    if np.ndim(tensor) > 3:
        # Check that indeed the 1st dimension (batch dimension) is equal to 1 (batch of only 1 image). Otherwise it will raise an error
        assert tensor.shape[0] == 1
        # Get the 1st image (meaning we remove the batch dimension)
        tensor = tensor[0]
    
    # Convert the tensor into a PIL Image object
    image = Image.fromarray(tensor)

    return image


# Use tf.Variable to make the generated_image mutable (so that it can be modified during the training/optimization process)
# The variable will track its gradients, allowing TF to adjust its values based on some loss function/optimization algorithm
generated_image = tf.Variable(generated_image)

# Choose 2501 epochs (we run 2501 times the model)
epochs = 501

# Update the generated_image while minimizing the total cost during 2501 epochs. Save it every 250 epochs to keep track
# Iterate on the number of epochs
for i in range(epochs):
    # Run the train/optimization algorithm
    train_step(generated_image)
    # Print the number of epochs every 250 epochs to keep track 
    if i % 250 == 0:
        print(f"Epoch {i} ")
    # Save the generated image every 250 epochs to keep track of the generated image
    if i % 250 == 0:
        image = tensor_to_image(generated_image)
        imshow(image)
        image.save(f"output/cat_image_sandstone_{i}.jpg")
        plt.show() 


# Show the 3 images at the same time: the content image, the style image and the generated image (content mixed with style image)
fig = plt.figure(figsize=(16, 4))
# Plot the content image
ax = fig.add_subplot(1, 3, 1)
imshow(content_image[0])
ax.title.set_text('Content image')
# Plot the style image
ax = fig.add_subplot(1, 3, 2)
imshow(style_image[0])
ax.title.set_text('Style image')
# Plot the generated image (mix between content and style images)
ax = fig.add_subplot(1, 3, 3)
imshow(generated_image[0])
ax.title.set_text('Generated image')
plt.show()
