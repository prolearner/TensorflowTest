import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#symbolic varible creation, "None" means that the dimension can be of any length
#This is a 2-D tensor of floating-point numbers reprensenting images flattened into a 784-dimensional vector.
x = tf.placeholder(tf.float32, [None, 784])

#A Variable is a modifiable tensor that lives in TensorFlow's graph of interacting operations:

# Weigth variable: 2-D tensor, 784 are the number of dimension (pixels) of the image and 10 are the numer of
# classes, in the MINST case the digits from 0 to 9.
W = tf.Variable(tf.zeros([784, 10]))

#Bias for each class, indipendent from the input (so it as a dimension of 10).
b = tf.Variable(tf.zeros([10]))

#Model implementation: only one line to define the model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# --- TRAINING ----

# using the cross-entropy function to se how bad the trained model is in comparison with the training set

#Correct answers: a one-hot vector [0,0,0,1,0,0,0,0,0,0] (only a one and all the rest are 0)
y_ = tf.placeholder(tf.float32, [None, 10])

#cross entropy implementation
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#Optimzation function to minimize the cross_entropy function, using backpropagation algorithm
#to efficiently determine how your variables affect the cost you ask it minimize, 0.01 is the learning rate
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# there are many other optimization algorithms,
# see https://www.tensorflow.org/versions/master/api_docs/python/train.html#optimizers

#variables initialization
init = tf.initialize_all_variables()

# open the session, and run the initializations
sess = tf.Session()
sess.run(init)

#run the training 1000 times using stochastic training (stochastic gradient descent)
for i in range(1000):
  #get a batch of 100 random data points(vector representing images), cheaper than using all of it
  batch_xs, batch_ys = mnist.train.next_batch(100)
  # feeding the trianing data to the train step
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# ---- EVALUATION ----

#tells if the prediction matches the truth, tf.argmax returns the index of the highest entry in a tensor along some axis
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#model accuracy, cast to float and mean of the vector correct_prediction, that has one value for image examined
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# print the accuracy on test data
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

