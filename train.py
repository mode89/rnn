import argparse
import numpy
import os
import tensorflow as tf

NUM_UNITS = 128
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

        lstmCell = tf.nn.rnn_cell.LSTMCell(num_units=NUM_UNITS)
        self.zeroState = lstmCell.zero_state(BATCH_SIZE, tf.float32)
        self.initialState = tf.nn.rnn_cell.LSTMStateTuple(
            c=tf.placeholder(tf.float32, (None, lstmCell.state_size.c)),
            h=tf.placeholder(tf.float32, (None, lstmCell.state_size.h)))
        lstmOutputs, self.finalState = tf.nn.dynamic_rnn(
            cell=lstmCell,
            inputs=self.inputs,
            sequence_length=[SEQUENCE_SIZE],
            initial_state=self.initialState)
        logits = list()
        denseLayer = tf.layers.Dense(self.classNum, tf.sigmoid)
        for i in range(SEQUENCE_SIZE):
            tempLogits = denseLayer(lstmOutputs[:, i, :])
            logits.append(tempLogits)
        self.logits = tf.reshape(logits, tf.shape(self.outputs))
        self.lossOp = tf.losses.softmax_cross_entropy(
            self.outputs, self.logits)
        optimizer = tf.train.AdamOptimizer()
        # optimizer = tf.train.RMSPropOptimizer(0.001)
        self.trainOp = optimizer.minimize(self.lossOp)
        self.logitsSoftmax = tf.nn.softmax(self.logits)
        self.mseOp = tf.losses.mean_squared_error(
            self.outputs, self.logits)
        self.diffOp = tf.losses.absolute_difference(
            self.outputs, self.logits)

    def train(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as self.session:
            self.session.run(tf.global_variables_initializer())
            epoch = 0
            try:
                while True:
                    print("Epoch {} ...".format(epoch))
                    self.epoch()
                    epoch += 1
            except KeyboardInterrupt:
                pass

    def epoch(self):
        for sample in self.samples:
            mseMax = None
            lossMax = None
            diffMax = None
            state = self.session.run(self.zeroState)
            lossSum = 0
            for batch in self.batches(sample):
                _, state, loss, mse, logitsValues = self.session.run(
                    [
                        self.trainOp,
                        self.finalState,
                        self.lossOp,
                        self.mseOp,
                        self.logits
                    ],
                    {
                        self.inputs: batch.inputs,
                        self.outputs: batch.outputs,
                        self.initialState.c: state.c,
                        self.initialState.h: state.h
                    })
                lossSum += loss
                lossMax = max(loss, lossMax)
                mseMax = max(mse, mseMax)
                diff = numpy.max(
                    numpy.abs(logitsValues - batch.outputs))
                diffMax = max(diff, diffMax)
            print(lossSum / self.batchNum, lossMax, mseMax, diffMax)

    def batches(self, sample):
        self.batchNum = sample.inputs.shape[0] / SEQUENCE_SIZE
        for i in range(self.batchNum):
            firstTimeStep = i * SEQUENCE_SIZE
            lastTimeStep = firstTimeStep + SEQUENCE_SIZE
            class Batch: pass
            batch = Batch()
            batch.inputs = numpy.reshape(
                sample.inputs[firstTimeStep:lastTimeStep,:],
                (BATCH_SIZE, SEQUENCE_SIZE, self.featureSize))
            batch.outputs = numpy.reshape(
                sample.outputs[firstTimeStep:lastTimeStep,:],
                (BATCH_SIZE, SEQUENCE_SIZE, self.classNum))
            yield batch

if __name__ == "__main__":
    model = Model()
    model.build()
    model.train()
