import os
import tensorflow as tf

# gets rid of warnings from tf.Session().run() about a possibly faster run time if switch settings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

sess = tf.Session()


# constants don' take input
def constants():
    node1 = tf.constant(3.0, dtype=tf.float32)
    # defaults at dtype=tf.float32
    node2 = tf.constant(4.0)
    node3 = tf.add(node1, node2)

    print("node1:", node1)
    print("node2:", node2)
    print("node3:", node3)
    print("sess.run(node3):", sess.run(node3))


# placeholders promise to provide a value later
def placeholders():
    a = tf.placeholder(tf.float64)
    b = tf.placeholder(tf.float64)
    adder_node = a + b
    add_and_triple = adder_node * 3

    print("sess.run(adder_node, {a: 3, b: 4}):", sess.run(adder_node, {a: 3, b: 4.5}))
    print("sess.run(adder_node, {a: [1, 3], b: [2, 4]}):", sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
    print("sess.run(add_and_triple, {a: 3, b: 4.5}):", sess.run(add_and_triple, {a: 3, b: 4.5}))


# variables allows us to add trainable parameters to a graph
def variables():
    # variables are only initialized when you call tf.Variable
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b

    # special operation to initialize all variables
    init = tf.global_variables_initializer()
    sess.run(init)

    print(sess.run(linear_model, {x: [1, 2, 3, 4]}))


def loss_function():
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    linear_model = W * x + b
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)

    init = tf.global_variables_initializer()
    sess.run(init)

    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))


# constants()
# placeholders()
# variables()
loss_function()