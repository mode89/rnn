import argparse
import numpy
import os
import random
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.nn.rnn_cell import LSTMCell, GRUCell
from tqdm import tqdm

numpy.set_printoptions(suppress=True, precision=3)

NUM_UNITS = 32
SAMPLE_LENGTH = 300
BATCH_SIZE = 10

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
        self.classNum = outputs.shape[1]
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
            (None, SAMPLE_LENGTH, self.featureSize), name="inputs")
        self.logitsReference = tf.placeholder(tf.float32,
            (None, self.classNum), name="logitsReference")

        rnn = GRUCell(NUM_UNITS)

        denseLayer = tf.layers.Dense(self.classNum, tf.sigmoid)
        state = rnn.zero_state(BATCH_SIZE, tf.float32)
        for i in range(SAMPLE_LENGTH):
            inputs = self.inputs[:, i, :]
            outputs, state = rnn(inputs, state)
        self.logits = denseLayer(outputs)
        losses = list()
        for batchIndex in range(BATCH_SIZE):
            loss = tf.losses.softmax_cross_entropy(
                self.logitsReference[batchIndex,:],
                self.logits[batchIndex,:])
            losses.append(loss)
        self.loss = tf.reduce_mean(losses)

        self.argmax = tf.argmax(self.logits, 1)
        self.oneHot = tf.one_hot(self.argmax, self.classNum)

        optimizer = tf.train.AdamOptimizer()
        self.trainOp = optimizer.minimize(self.loss)

    def train(self, samples):
        trainSamplesNum = int(len(samples) * 0.8)
        trainSamples = samples[:trainSamplesNum]
        testSamples = samples[trainSamplesNum:]
        self.normalize_inputs()
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

    def normalize_inputs(self):
        print("Normalizing inputs ...")
        inputs = list()
        for sample in self.samples:
            inputs.append(sample.inputs)
        inputs = numpy.vstack(inputs)
        self.normalizer = StandardScaler().fit(inputs)

    def epoch(self, samples):
        truthCounter = 0
        lossSum = 0.0
        random.shuffle(samples)
        for batch in self.batches(samples, progress=True):
            results = self.session.run(
                {
                    "trainOp": self.trainOp,
                    "oneHot": self.oneHot,
                    "loss": self.loss,
                },
                {
                    self.inputs: batch.inputs,
                    self.logitsReference: batch.outputs,
                })
            for i in range(BATCH_SIZE):
                diff = numpy.sum(numpy.abs(
                    results["oneHot"][i] - batch.outputs[i]))
                if diff < 0.1:
                    truthCounter += 1
            lossSum += results["loss"]
        print(
            float(truthCounter) / len(samples),
            lossSum / len(samples) * BATCH_SIZE,
        )

    def validate(self, samples):
        truthCounter = 0
        lossSum = 0.0
        for batch in self.batches(samples, progress=False):
            results = self.session.run(
                {
                    "oneHot": self.oneHot,
                    "loss": self.loss,
                },
                {
                    self.inputs: batch.inputs,
                    self.logitsReference: batch.outputs,
                })
            for i in range(BATCH_SIZE):
                diff = numpy.sum(numpy.abs(
                    results["oneHot"][i] - batch.outputs[i]))
                if diff < 0.1:
                    truthCounter += 1
            lossSum += results["loss"]
        print(
            float(truthCounter) / len(samples),
            lossSum / len(samples) * BATCH_SIZE,
        )

    def batches(self, samples, progress):
        batchNum = len(samples) / BATCH_SIZE
        batchRange = range(batchNum)
        batchRange = tqdm(batchRange, ascii=True) \
            if progress else batchRange
        for batchIndex in batchRange:
            firstSample = batchIndex * BATCH_SIZE
            lastSample = firstSample + BATCH_SIZE
            inputs = list()
            outputs = list()
            for sample in samples[firstSample:lastSample]:
                inputsNormalized = self.normalizer.transform(sample.inputs)
                inputs.append(inputsNormalized)
                outputs.append(sample.outputs)
            class Batch: pass
            batch = Batch()
            batch.inputs = numpy.reshape(
                inputs,
                (BATCH_SIZE, SAMPLE_LENGTH, self.featureSize))
            batch.outputs = numpy.reshape(
                outputs,
                (BATCH_SIZE, self.classNum))
            yield batch

if __name__ == "__main__":
    model = Model()
    model.build()
    model.train(model.samples)
