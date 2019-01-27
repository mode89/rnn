import argparse
from collections import namedtuple
import numpy
import os
import random
from sklearn.preprocessing import Normalizer, StandardScaler, scale
import tensorflow as tf
from tensorflow.nn.rnn_cell import GRUCell, LSTMCell, LSTMStateTuple, MultiRNNCell
from tqdm import tqdm

numpy.set_printoptions(suppress=True, precision=3)

NUM_UNITS = 64
SAMPLE_NUM = 300
SAMPLE_LENGTH = 100
SEQUENCE_SIZE = SAMPLE_LENGTH
CLASS_NUM = 11
BATCH_SIZE = 60
BATCH_NUM = int(SAMPLE_NUM / BATCH_SIZE)
INPUT_NOISE = 0.0
INITIAL_NOISE = 0.0

Sample = namedtuple("Sample", ["inputs", "outputs"])
Batch = namedtuple("Batch", ["inputs", "outputs"])

class Model:

    def __init__(self):
        self.featureSize = 1

    def build(self):
        print("Building model ...")

        tf.set_random_seed(0)

        self.inputs = tf.placeholder(tf.float32,
            (None, SEQUENCE_SIZE, self.featureSize), name="inputs")
        self.logitsReference = tf.placeholder(tf.float32,
            (None, CLASS_NUM), name="logitsReference")

        rnn = GRUCell(NUM_UNITS)
        # self.initialState = tf.placeholder(
        #     tf.float32, (None, rnn.state_size), name="initialState")
        # rnn = LSTMCell(NUM_UNITS)
        # self.initialState = tf.placeholder(
        #     tf.float32, (None, rnn.state_size), name="initialState")
        # rnn = MultiRNNCell([
        #     GRUCell(NUM_UNITS),
        #     GRUCell(NUM_UNITS)
        # ])
        self.initialState = rnn.zero_state(BATCH_SIZE, tf.float32)

        denseLayer = tf.layers.Dense(CLASS_NUM, tf.sigmoid)
        state = self.initialState
        for i in range(SAMPLE_LENGTH):
            inputs = self.inputs[:, i, :]
            outputs, state = rnn(inputs, state)
        self.logits = denseLayer(outputs)
        losses = list()
        for batchIndex in range(BATCH_SIZE):
            # loss = tf.losses.softmax_cross_entropy(
            #     self.logitsReference[i,:], self.logits[i,:])
            loss = tf.keras.backend.categorical_crossentropy(
                self.logitsReference[batchIndex,:],
                self.logits[batchIndex,:],
                from_logits=True)
            print(loss)
            losses.append(loss)
        self.loss = tf.reduce_mean(losses)
        # self.loss = tf.losses.softmax_cross_entropy(
        #     self.logitsReference, self.logits)
        # self.loss = tf.losses.mean_squared_error(
        #     self.logitsReference, self.logits)

        self.finalState = state
        self.argmax = tf.argmax(self.logits, 1)
        self.oneHot = tf.one_hot(self.argmax, CLASS_NUM)

        optimizer = tf.train.AdamOptimizer()
        self.trainOp = optimizer.minimize(self.loss)

        self.logitsSoftmax = tf.nn.softmax(self.logits)
        self.mseOp = tf.losses.mean_squared_error(
            self.logitsReference, self.logits)
        self.diffOp = tf.losses.absolute_difference(
            self.logitsReference, self.logits)

    def train(self, samples):
        samples = list(samples)
        random.shuffle(samples)
        trainSamplesNum = int(len(samples) * 0.8)
        trainSamples = samples[:trainSamplesNum]
        for sample in trainSamples:
            print(sample.outputs)
        testSamples = samples[trainSamplesNum:]
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as self.session:
            self.session.run(tf.global_variables_initializer())
            epoch = 0
            try:
                while True:
                    print("Epoch {} ...".format(epoch))
                    self.epoch(trainSamples)
                    self.validate(testSamples)
                    epoch += 1
            except KeyboardInterrupt:
                pass

    def epoch(self, samples):
        oneHotDiffSum = 0.0
        lossSum = 0.0
        random.shuffle(samples)
        for batch in self.batches(samples):
            results = self.session.run(
                {
                    "trainOp": self.trainOp,
                    "oneHot": self.oneHot,
                    "logits": self.logits,
                    "loss": self.loss,
                    "finalState": self.finalState,
                },
                {
                    self.inputs: batch.inputs,
                    # self.initialState: initialState,
                    self.logitsReference: batch.outputs,
                })
            # initialState = results["finalState"]
            oneHotDiff = numpy.sum(numpy.abs(
                results["oneHot"] - batch.outputs))
            oneHotDiffSum += oneHotDiff
            lossSum += results["loss"]
            # print(results["finalState"])
            # print(results["logits"])
        print(
            oneHotDiffSum / len(samples),
            lossSum / len(samples) * BATCH_SIZE,
        )

    def validate(self, samples):
        oneHotDiffSum = 0.0
        lossSum = 0.0
        for batch in self.batches(samples):
            results = self.session.run(
                {
                    "oneHot": self.oneHot,
                    "loss": self.loss,
                    "finalState": self.finalState,
                },
                {
                    self.inputs: batch.inputs,
                    self.logitsReference: batch.outputs,
                    # self.initialState: initialState,
                })
            # initialState = results["finalState"]
            oneHotDiff = numpy.sum(numpy.abs(
                results["oneHot"] - batch.outputs))
            oneHotDiffSum += oneHotDiff
            lossSum += results["loss"]
        print(
            oneHotDiffSum / len(samples),
            lossSum / len(samples) * BATCH_SIZE,
        )

    def initial_state(self):
        return numpy.random.uniform(
            -INITIAL_NOISE, INITIAL_NOISE, (BATCH_SIZE, NUM_UNITS))
        # return LSTMStateTuple(
        #     numpy.zeros((BATCH_SIZE, NUM_UNITS)),
        #     numpy.zeros((BATCH_SIZE, NUM_UNITS))
        # )
        # return [
        #     [
        #         numpy.zeros((BATCH_SIZE, NUM_UNITS)),
        #         numpy.zeros((BATCH_SIZE, NUM_UNITS)),
        #     ],
        #     [
        #         numpy.zeros((BATCH_SIZE, NUM_UNITS)),
        #         numpy.zeros((BATCH_SIZE, NUM_UNITS)),
        #     ],
        # ]

    def batches(self, samples):
        batchNum = len(samples) / BATCH_SIZE
        for batchIndex in tqdm(range(batchNum), ascii=True):
            firstSample = batchIndex * BATCH_SIZE
            lastSample = firstSample + BATCH_SIZE
            inputs = list()
            outputs = list()
            for sample in samples[firstSample:lastSample]:
                inputs.append(sample.inputs)
                outputs.append(sample.outputs)
            class Batch: pass
            batch = Batch()
            batch.inputs = numpy.reshape(
                inputs,
                (BATCH_SIZE, SEQUENCE_SIZE, self.featureSize))
            batch.outputs = numpy.reshape(
                outputs,
                (BATCH_SIZE, CLASS_NUM))
            yield batch

def generate_samples():
    for sampleIndex in range(SAMPLE_NUM):
        maxDelta = SAMPLE_LENGTH
        delta = random.randrange(maxDelta)
        second = SAMPLE_LENGTH - 1
        first = second - delta
        inputs = numpy.ones((SAMPLE_LENGTH, 1)) * -1.0
        inputs[first, 0] = 1.0
        inputs[second, 0] = 1.0
        inputNoise = numpy.random.uniform(
            -INPUT_NOISE, INPUT_NOISE, (SAMPLE_LENGTH, 1))
        inputs += inputNoise
        deltaPerCategory = maxDelta / 15.0
        category = int(delta / deltaPerCategory)
        outputs = numpy.zeros(CLASS_NUM)
        if category > 0 and category < CLASS_NUM:
            outputs[category] = 1.0
        else:
            outputs[0] = 1.0
        sample = Sample(inputs, outputs)
        print(category, inputs.transpose()[0], outputs)
        yield sample

def loss_weights(outputs):
    weights = numpy.full(outputs.shape[0], 1.0)
    activated = outputs[:, 0] == 0.0
    weights[activated] = 10.0
    return weights

if __name__ == "__main__":
    samples = generate_samples()
    model = Model()
    model.build()
    model.train(samples)
