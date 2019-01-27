import argparse
import json
import matplotlib.pyplot as plt
import numpy
import os
from record import VOCABULARY
from sklearn.preprocessing import MinMaxScaler

FRAME_RATE = 44100
SAMPLE_LENGTH = 3.0
ACTIVATION_LENGTH = 0.5

class TrainingData:

    def __init__(self):
        self.parse_arguments()

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("directory")
        args = parser.parse_args()
        self.directory = args.directory

    def prepare(self):
        self.load_features()
        self.load_labels()
        self.generate_samples()
        self.dump_samples()

    def generate_samples(self):
        print("Generating samples ...")
        sampleLength = \
            int(SAMPLE_LENGTH * FRAME_RATE / self.chunk_size)
        activationLength = \
            int(ACTIVATION_LENGTH * FRAME_RATE / self.chunk_size)
        self.inputs = list()
        self.outputs = list()
        scaler = MinMaxScaler()
        scaler = scaler.fit(self.features)
        print(numpy.amin(self.features, axis=0))
        print(numpy.amax(self.features, axis=0))
        for label in self.labels:
            utteranceBegin = int(label["begin"] / self.chunk_size)
            utteranceEnd = int(label["end"] / self.chunk_size)
            utteranceLength = utteranceEnd - utteranceBegin
            sampleBegin = utteranceEnd - sampleLength
            sampleEnd = utteranceEnd
            inputs = self.features[sampleBegin:sampleEnd,:]
            print(inputs)
            inputs = scaler.transform(inputs)
            print(inputs)
            plt.title(label["word"])
            plt.imshow(inputs.transpose(), cmap="jet")
            plt.show()
            self.inputs.append(inputs)
            outputs = single_output_from_label(label["word"])
            self.outputs.append(outputs)
        self.inputs = numpy.array(self.inputs)
        self.outputs = numpy.array(self.outputs)

    def dump_samples(self):
        samplesPath = os.path.join(self.directory, "samples.npz")
        print("Saving samples to {} ...".format(samplesPath))
        numpy.savez_compressed(samplesPath,
            inputs=self.inputs,
            outputs=self.outputs)

    def load_features(self):
        mfccPath = os.path.join(self.directory, "mfcc.npz")
        mfcc = numpy.load(mfccPath)
        self.chunk_size = mfcc["chunk_size"]
        self.features = mfcc["features"]

    def load_labels(self):
        labelsPath = os.path.join(self.directory, "labels.json")
        with open(labelsPath, "r") as labelsFile:
            self.labels = json.load(labelsFile)

def single_output_from_label(word):
    vocabulary = VOCABULARY + [None]
    output = numpy.zeros(len(vocabulary))
    index = vocabulary.index(word)
    output[index] = 1.0
    return output

if __name__ == "__main__":
    trainingData = TrainingData()
    trainingData.prepare()
