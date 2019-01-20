import argparse
import numpy
import os
import random
from sklearn.preprocessing import Normalizer, StandardScaler, scale
import tensorflow as tf

numpy.set_printoptions(suppress=True, precision=3)

NUM_UNITS = 32
SAMPLE_LENGTH = 20
SEQUENCE_SIZE = 20
CLASS_NUM = 11
BATCH_SIZE = 1

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

        rnn = tf.layers.Dense(NUM_UNITS, tf.tanh)
        state = tf.zeros((BATCH_SIZE, NUM_UNITS), tf.float32)

        denseLayer = tf.layers.Dense(self.classNum, tf.sigmoid)
        logits = list()
        losses = list()
        softmaxes = list()
        for i in range(SEQUENCE_SIZE):
            stepInputs = self.inputs[:, i, :]
            self.stepInputs.append(stepInputs)
            outputs = tf.concat((stepInputs, state), 1)
            outputs = rnn(outputs)
            state = outputs
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
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as self.session:
            self.session.run(tf.global_variables_initializer())
            epoch = 0
            try:
                while True:
                    print("Epoch {} ...".format(epoch))
                    self.epoch(samples)
                    epoch += 1
            except KeyboardInterrupt:
                pass

    def epoch(self, samples):
        for sample in samples:
            mseMax = None
            mseSum = 0
            lossMax = None
            lossSum = 0
            diffMax = None
            diffSum = 0
            for batch in self.batches(sample):
                _, state, loss, mse, logitsValues, losses, softmaxes, stepInputs = self.session.run(
                    [
                        self.trainOp,
                        self.finalState,
                        self.lossOp,
                        self.mseOp,
                        self.logits,
                        self.losses,
                        self.softmaxes,
                        self.stepInputs,
                    ],
                    {
                        self.inputs: batch.inputs,
                        self.logitsReference: batch.outputs,
                        self.lossWeights: sample.lossWeights,
                    })
                lossSum += loss
                lossMax = max(loss, lossMax)
                mseMax = max(mse, mseMax)
                mseSum += mse
                diff = numpy.max(
                    numpy.abs(logitsValues - batch.outputs))
                diffMax = max(diff, diffMax)
                diffSum += diff
            print(
                lossSum / self.batchNum, lossMax,
                mse / self.batchNum, mseMax,
                diffSum / self.batchNum, diffMax)

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

def generate_samples():
    for delta in range(0, SAMPLE_LENGTH - 5):
        for deltaSampleIndex in range(20):
            first = random.randrange(SAMPLE_LENGTH - delta - 4)
            second = first + delta
            inputs = numpy.ones((SAMPLE_LENGTH, 1)) * -1.0
            inputs[first, 0] = 1.0
            inputs[second, 0] = 1.0
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

if __name__ == "__main__":
    samples = generate_samples()
    model = Model()
    model.build()
    model.train(samples)