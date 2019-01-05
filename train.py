import argparse
import numpy
import os
import tensorflow as tf

SEQUENCE_SIZE = 20
BATCH_SIZE = 1

class Model:

    def __init__(self):
        self.parse_arguments()

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("directory")
        args = parser.parse_args()
        self.directory = args.directory

    def load_samples(self):
        print("Loading samples ...")
        samplesPath = os.path.join(self.directory, "samples.npz")
        data = numpy.load(samplesPath)
        inputs = data["inputs"]
        outputs = data["outputs"]
        sampleNum = inputs.shape[0]
        self.featureSize = inputs.shape[2]
        self.classNum = outputs.shape[2]
        self.samples = list()
        for i in range(sampleNum):
            class Sample: pass
            sample = Sample()
            sample.inputs = inputs[i]
            sample.outputs = outputs[i]
            self.samples.append(sample)

    def build(self):
        self.load_samples()
        print("Building model ...")

        tf.set_random_seed(0)

        self.inputs = tf.placeholder(tf.float32,
            (None, SEQUENCE_SIZE, self.featureSize), name="inputs")
        self.outputs = tf.placeholder(tf.float32,
            (None, SEQUENCE_SIZE, self.classNum), name="outputs")

        lstmCell = tf.nn.rnn_cell.LSTMCell(num_units=128)
        zeroState = lstmCell.zero_state(BATCH_SIZE, tf.float32)
        initialState = tf.nn.rnn_cell.LSTMStateTuple(
            c=tf.placeholder(tf.float32, (None, lstmCell.state_size.c)),
            h=tf.placeholder(tf.float32, (None, lstmCell.state_size.h)))
        lstmOutputs, finalState = tf.nn.dynamic_rnn(
            cell=lstmCell,
            inputs=self.inputs,
            sequence_length=[SEQUENCE_SIZE],
            initial_state=initialState)
        logits = list()
        denseLayer = tf.layers.Dense(self.classNum, tf.sigmoid)
        for i in range(SEQUENCE_SIZE):
            tempLogits = denseLayer(lstmOutputs[:, i, :])
            logits.append(tempLogits)
        logits = tf.reshape(logits, tf.shape(self.outputs))
        lossOp = tf.losses.softmax_cross_entropy(self.outputs, logits)
        optimizer = tf.train.AdamOptimizer()
        trainOp = optimizer.minimize(lossOp)
        logitsSoftmax = tf.nn.softmax(logits)
        mseOp = tf.losses.mean_squared_error(self.outputs, logits)
        diffOp = tf.losses.absolute_difference(self.outputs, logits)

if __name__ == "__main__":
    model = Model()
    model.build()
