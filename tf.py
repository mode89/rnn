import tensorflow as tf
import numpy

BATCH_SIZE = 1
EPOCHS = 3000

def generate_training_data():

    X = numpy.array([
        [ 0, 0 ],
        [ 0, 1 ],
        [ 1, 0 ],
        [ 1, 1 ],
    ])

    Y = numpy.array([
        [ 0 ],
        [ 1 ],
        [ 1 ],
        [ 0 ],
    ])

    return X, Y

def main():

    tf.set_random_seed(1)

    X, Y = generate_training_data()

    x = tf.placeholder(tf.float32, (None, 2))
    y = tf.placeholder(tf.float32, (None, 1))

    logits = tf.layers.dense(x, 4, activation=tf.tanh)
    logits = tf.layers.dense(logits, 1, activation=tf.sigmoid)
    lossOp = tf.losses.mean_squared_error(y, logits)
    optimizer = tf.train.AdamOptimizer()
    trainOp = optimizer.minimize(lossOp)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for i in range(1000):
            _, loss = session.run([trainOp, lossOp], {x: X, y: Y})
            print(loss)

if __name__ == "__main__":
    main()
