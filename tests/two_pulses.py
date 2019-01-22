import argparse
import numpy
import os
import random
from sklearn.preprocessing import Normalizer, StandardScaler, scale
import tensorflow as tf

numpy.set_printoptions(suppress=True, precision=3)

NUM_UNITS = 64
SAMPLE_LENGTH = 20
SEQUENCE_SIZE = 20
CLASS_NUM = 11
BATCH_SIZE = 1
INPUT_NOISE = 0.4

class Model:

    def __init__(self):
        self.featureSize = 1
        self.classNum = 11

    def build(self):
        print("Building model ...")

        tf.set_random_seed(0)

        self.inputs = tf.placeholder(tf.float32,
            (None, SEQUENCE_SIZE, self.featureSize), name="inputs")
        self.logitsReference = tf.placeholder(tf.float32,
            (None, SEQUENCE_SIZE, self.classNum), name="logitsReference")
        self.stepInputs = list()
        self.lossWeights = tf.placeholder(tf.float32, (None, SEQUENCE_SIZE))

        rnn = tf.nn.rnn_cell.GRUCell(num_units=NUM_UNITS)
        self.initialState = tf.placeholder(
            tf.float32, (None, rnn.state_size), name="initialState")

        denseLayer = tf.layers.Dense(self.classNum, tf.sigmoid)
        logits = list()
        losses = list()
        softmaxes = list()
        state = self.initialState
        for i in range(SEQUENCE_SIZE):
            stepInputs = self.inputs[:, i, :]
            self.stepInputs.append(stepInputs)
            outputs, state = rnn(stepInputs, state)
            outputs = denseLayer(outputs)
            logits.append(outputs)
            loss = tf.losses.softmax_cross_entropy(
                self.logitsReference[:, i, :], outputs)
            losses.append(loss)
            softmax = tf.nn.softmax(outputs)
            softmaxes.append(softmax)
        self.stepInputs = tf.concat(self.stepInputs, 0, name="stepInputs")
        self.losses = tf.stack(losses) * self.lossWeights
        self.finalState = state
        self.logits = tf.reshape(logits, tf.shape(self.logitsReference))
        self.argmax = tf.argmax(self.logits, 2)
        self.oneHot = tf.one_hot(self.argmax, CLASS_NUM)
        self.softmaxes = tf.concat(softmaxes, 0)

        self.lossOp = tf.reduce_sum(self.losses)

        optimizer = tf.train.AdamOptimizer()
        self.trainOp = optimizer.minimize(self.lossOp)
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
        random.shuffle(samples)
        for sample in samples:
            initialState = numpy.random.uniform(
                -0.5, 0.5, (BATCH_SIZE, NUM_UNITS))
            for batch in self.batches(sample):
                results = self.session.run(
                    {
                        "trainOp": self.trainOp,
                        "oneHot": self.oneHot,
                        "finalState": self.finalState,
                    },
                    {
                        self.inputs: batch.inputs,
                        self.initialState: initialState,
                        self.logitsReference: batch.outputs,
                        self.lossWeights: batch.lossWeights,
                    })
                initialState = results["finalState"]
                oneHotDiff = numpy.sum(numpy.abs(
                    results["oneHot"] - batch.outputs))
                oneHotDiffSum += oneHotDiff
        print(oneHotDiffSum / len(samples))

    def validate(self, samples):
        oneHotDiffSum = 0.0
        for sample in samples:
            initialState = numpy.random.uniform(
                -0.5, 0.5, (BATCH_SIZE, NUM_UNITS))
            for batch in self.batches(sample):
                results = self.session.run(
                    {
                        "oneHot": self.oneHot,
                        "finalState": self.finalState,
                    },
                    {
                        self.inputs: batch.inputs,
                        self.logitsReference: batch.outputs,
                        self.initialState: initialState,
                    })
                initialState = results["finalState"]
                oneHotDiff = numpy.sum(numpy.abs(
                    results["oneHot"] - batch.outputs))
                oneHotDiffSum += oneHotDiff
        print(oneHotDiffSum / len(samples))

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
            outputs = sample.outputs[firstTimeStep:lastTimeStep,:]
            batch.outputs = numpy.reshape(
                outputs,
                (BATCH_SIZE, SEQUENCE_SIZE, self.classNum))
            batch.lossWeights = numpy.reshape(loss_weights(outputs),
                (BATCH_SIZE, SEQUENCE_SIZE))
            yield batch

def generate_samples():
    for delta in range(0, SAMPLE_LENGTH - 5):
        for deltaSampleIndex in range(20):
            first = random.randrange(SAMPLE_LENGTH - delta - 4)
            second = first + delta
            inputs = numpy.ones((SAMPLE_LENGTH, 1)) * -1.0
            inputs[first, 0] = 1.0
            inputs[second, 0] = 1.0
            noise = numpy.random.uniform(
                -INPUT_NOISE, INPUT_NOISE, (SAMPLE_LENGTH, 1))
            inputs += noise
            outputs = numpy.zeros((SAMPLE_LENGTH, CLASS_NUM))
            outputs[:, 0] = 1.0
            if delta > 0 and delta < 11:
                pulseBegin = second + 1
                pulseEnd = second + 5
                outputs[pulseBegin:pulseEnd, 0] = 0.0
                outputs[pulseBegin:pulseEnd, delta] = 1.0
            class Sample: pass
            sample = Sample()
            sample.inputs = inputs
            sample.outputs = outputs
            yield sample

def loss_weights(outputs):
    weights = numpy.full(outputs.shape[0], 1.0)
    activated = outputs[:, 0] == 0.0
    weights[activated] = 20.0
    return weights

if __name__ == "__main__":
    samples = generate_samples()
    model = Model()
    model.build()
    model.train(samples)
