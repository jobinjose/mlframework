import tensorflow as tf
import os
import glob
import cv2
import csv
import numpy as np
from time import gmtime, strftime
import sys

np.set_printoptions(threshold='nan')

# set the variables first
learning_rate = 0.001
training_iterations = 100

# Labels for each class
classes = ['class_1', 'class_2', 'class_3']

# Number of classes
n_classes = len(classes)

# Different hidden layer combinations to brute force
n_layer_nodes_shape_per_iteration = [[512, 1024]]

# Number of nodes in each of the hidden layers
n_layer_nodes = [0, 0]

# Number of hidden layers
n_hidden_layers = len(n_layer_nodes)

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 1

# image dimensions
img_size = 64

# Number of preprocessing steps done on the image, which are concatenated to form the final image
num_preprocesses = 0

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels * (1+num_preprocesses)

# Number of input nodes per images
n_input = img_size_flat

# Number of training images to use for each class
n_training_images_per_class = 1000

# Number of validation images to use for each class
n_validation_images_per_class = 120

# Name of file the model is saved to
model_file_name = ''

def createWeights(n_inputs, n_layer_nodes, n_outputs):
    weights = [tf.Variable(tf.truncated_normal(shape = [n_inputs, n_layer_nodes[0]], stddev=0.05), name='nn_weights_0')]
    
    print "\nWeights"
    print weights[0].shape
    for i in range(0, len(n_layer_nodes)-1):
        wt = tf.Variable(tf.truncated_normal(shape=[n_layer_nodes[i], n_layer_nodes[i+1]], stddev=0.05), name='nn_weights_'+str(i+1))
        print wt.shape
        weights.append(wt)
    
    out_wt = tf.Variable(tf.truncated_normal(shape=[n_layer_nodes[-1], n_outputs], stddev=0.05), name='nn_output_weights')
    print out_wt.shape
    weights.append(out_wt)
    return weights

def createBiases(n_layer_nodes, n_outputs):
    biases = []
    
    print "\nBiases"
    for i in range(0, len(n_layer_nodes)):
        bias = tf.Variable(tf.constant(0.05, shape=[n_layer_nodes[i]]), name='nn_biases_' + str(i))
        print bias.shape
        biases.append(bias)
    
    out_bias = tf.Variable(tf.constant(0.05, shape=[n_outputs]), name='output_biases')
    print out_bias.shape
    biases.append(out_bias)
    return biases

def load_known_set(train_path, image_size, classes, limit):
    images = []
    labels = []
    
    for fld in classes:
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(train_path, fld, '*g')
        files = glob.glob(path)
        
        l = 0;
        for fl in files:
            if '_feature' in fl:
                continue
            # Read the image
            image = cv2.imread(fl)
            
            # Resize the image into image_size * image_size
            image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
            
            # Make image grayscale
            image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            """image_gradient = cv2.Sobel(image_grayscale,cv2.CV_64F,1,0,ksize=5)
            image_threshold = cv2.adaptiveThreshold(image_grayscale,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
            
            image = np.concatenate((image_grayscale, image_threshold), axis = 0)
                """

            # Add the image to list of images
            images.append(image_grayscale)

            # Create the label for the image
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            
            # if Limit is reached, break out of loop
            l += 1
            if l == limit:
                break

    images = np.array(images)
    labels = np.array(labels)

    # flatten the image
    images = images.reshape(len(images), img_size_flat)
    return images, labels

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.
    
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    
    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)
    
    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    layer = tf.add(layer, biases)
    
    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    layer = tf.nn.relu(layer)
    
    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()
    
    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]
    
    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])
    
    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]
    
    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

def createNN():
    # Create the x and y placeholders
    x = tf.placeholder(tf.float32, [None, n_input], name="X_PLACEHOLDER")
    x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
    
    y = tf.placeholder(tf.float32, [None, n_classes], name="Y_PLACEHOLDER")

    # Convolutional Layer 1
    conv_layer_1, conv1_weights = new_conv_layer(x_image, num_channels, 3, 32, False)
    
    # Convolutional Layer 2
    conv_layer_2, conv2_weights = new_conv_layer(conv_layer_1, 32, 3, 32)
    
    # List of all the layers in the fully connected NN, including input and output layers
    flattend_input_layer, num_input_features = flatten_layer(conv_layer_2)

    # Create the weights and biases
    weights = createWeights(num_input_features, n_layer_nodes, n_classes)
    biases = createBiases(n_layer_nodes, n_classes)

    fc_layers = [flattend_input_layer]

    # Create the consecutive layers and add it to the list of fully connected layers
    for i in range(0, n_hidden_layers):
        #print "i = {}, LEN: fc_layers = {}, weights = {}, biases = {}".format(i, len(fc_layers), len(weights), len(biases))
        print "Shapes: fc_layer : {}".format(fc_layers[i])
        fcl = tf.add(tf.matmul(fc_layers[i], weights[i]), biases[i])
        fcl = tf.nn.relu(fcl)
        fc_layers.append(fcl)

    # The prediction
    #print "Shapes: fc_layer : {}, weights: {}, biases: {}".format(fc_layers[-1].shape, weights[-1].shape, biases[-1].shape)

    pred = tf.add(tf.matmul(fc_layers[-1], weights[-1]), biases[-1], name = 'prediction')
    #print "Pred shape: {}".format(pred.shape)
    # pred = tf.nn.relu(pred)

    # Using softmax to get a probability distribution
    pred_cls = tf.nn.softmax(pred, name='prediction_softmax')

    # Cost to use cross entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
    cost = tf.reduce_mean(cross_entropy)

    # TODO: Try out other optimizers?
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred_cls, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    return optimizer, accuracy, cost, x, y, weights, biases, conv1_weights, conv2_weights

def train_model():
    
    with tf.Session() as sess:
        
        # read the training images
        print "Reading training images..."
        x_inputs, y_outputs = load_known_set('training_data', img_size, classes, n_training_images_per_class)
        
        # Create the NN
        optimizer, accuracy, cost, x, y, weights, biases, conv1_weights, conv2_weights = createNN()
        
        # Initialize the tf Variables
        sess.run(tf.global_variables_initializer())
        
        # Create file to write output
        filename = str(n_layer_nodes) + strftime('_%H_%M_%S', gmtime()) + '_{}iterations_train{}_validate{}'.format(training_iterations, n_training_images_per_class, n_validation_images_per_class)
        file = open(filename + '.csv', 'w+')
        writer = csv.writer(file, delimiter=',')
        # Header row of the file is 'epoch', 'accuracy', 'loss'
        writer.writerow(['epoch', 'accuracy', 'loss'])
        
        # train
        for i in range(training_iterations):
            opt, acu, loss = sess.run([optimizer, accuracy, cost], feed_dict={x: x_inputs, y:y_outputs})
            if i % 10 == 0:
                print "epoch {}, accuracy: {}%, loss: {}".format(i+1, acu*100, loss)
            writer.writerow([i+1, acu*100, loss])

        file.close()
        # Get the validation inputs and truth
        print "Reading validation images..."
        validation_input, validation_truth = load_known_set('validation_data', img_size, classes, n_validation_images_per_class)

        # Test the validation accuracy
        acu = sess.run([accuracy], feed_dict={x: validation_input, y:validation_truth})
        print "Validation accuracy {}%".format(round(acu,3))

        # Write validation accuracy to file
        file = open(filename+'_accuracy.csv', 'w+')
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['accuracy'])
        writer.writerow([acu])
        file.close()

        # Write the model to file
        saver = tf.train.Saver()
        save_path = saver.save(sess, 'training_model')
        
        global model_file_name
        model_file_name = filename
        
        print("Model saved in file: %s" % save_path)

def restore_model(validate):
    sess = tf.Session()
    #print "model file name {}".format(model_file_name)
    #meta_file = model_file_name + '.meta'
    saver = tf.train.import_meta_graph('training_model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    print("Model restored.")
        
    graph = tf.get_default_graph()
    accuracy = graph.get_tensor_by_name('accuracy:0')
    
    if validate == False:
        return graph, sess
    
    validation_input, validation_truth = load_known_set('validation_data', img_size, classes, n_validation_images_per_class)
    acu = sess.run(accuracy, feed_dict={'X_PLACEHOLDER:0': validation_input, 'Y_PLACEHOLDER:0':validation_truth})
    print "Validation accuracy from loaded model: {}%".format(round(acu*100,3))
    
    return graph, sess

def classify_image():
    # Get the graph and the session
    graph, session = restore_model()
    
    # Get the prediction node in the graph which we need to use
    prediction = graph.get_tensor_by_name('prediction_softmax:0')
    
    # Get the image to test. TODO make this image the parameter for this method
    test_input, test_result = load_known_set('validation_data', img_size, classes, 1)
    
    # Get the prediction
    print session.run(prediction, feed_dict={'X_PLACEHOLDER:0': test_input})

print 'arguments are {}'.format(sys.argv)

num_args = len(sys.argv)

if num_args == 2:
    if sys.argv[1] == 'train':
        print 'training...'
        for i in range(len(n_layer_nodes_shape_per_iteration)):
            global n_layer_nodes
            n_layer_nodes = n_layer_nodes_shape_per_iteration[i]
            train_model()
    elif sys.argv[1] == 'validate':
        restore_model(validate = True)

elif num_args == 3:
    if sys.argv[1] == 'classify':
        print 'classifying...'
        img_name = sys.argv[2]
        # Read the image
        image = cv2.imread(img_name)
            
        # Resize the image into image_size * image_size
        image = cv2.resize(image, (img_size, img_size), cv2.INTER_LINEAR)
            
        # Make image grayscale
        image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        images = []
        images.append(image_grayscale)
        images = np.array(images)
        images = images.reshape(len(images), img_size_flat)

        graph, session = restore_model(validate = False)

        # Get the prediction node in the graph which we need to use
        prediction = graph.get_tensor_by_name('prediction_softmax:0')
        preds = session.run(prediction, {'X_PLACEHOLDER:0': images})
        preds = preds[0]
        
        print '\n\nPredictions:\nMeningioma: {}% \nGlioma: {}% \nPituitary Gland tumour: {}%'.format(round(preds[0]*100,2), round(preds[1]*100,2), round(preds[2]*100,2))
#classify_image()
