import tensorflow as tf

g = tf.Graph()

with g.as_default():
    input_value = tf.constant(1.0, name = "input")
    weight = tf.Variable(6.9, name = "weight")
    output_value = tf.add(input_value, weight, name="output")
    assert input_value.graph is g
    assert weight.graph, output_value.graph is g


sess = tf.Session()
tf.global_variables_initializer()

summary_writer = tf.summary.FileWriter("./try", g)




