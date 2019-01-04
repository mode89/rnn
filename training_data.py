import argparse
import json
import numpy
import os
from record import VOCABULARY

FRAME_RATE = 44100
SAMPLE_LENGTH = 5.0
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
        sampleLength = \
            int(SAMPLE_LENGTH * FRAME_RATE / self.chunk_size)
        activationLength = \
            int(ACTIVATION_LENGTH * FRAME_RATE / self.chunk_size)
        for label in self.labels:
            utteranceBegin = label["begin"] / self.chunk_size
            utteranceEnd = label["end"] / self.chunk_size
            utteranceLength = utteranceEnd - utteranceBegin
            pauseLength = sampleLength - utteranceLength - activationLength
            halfPause = int(pauseLength / 2)
            sampleBegin = utteranceBegin - halfPause
            sampleEnd = sampleBegin + sampleLength
            inputs = self.features[sampleBegin:sampleEnd,:]

    def load_features(self):
        mfccPath = os.path.join(self.directory, "mfcc.npz")
        mfcc = numpy.load(mfccPath)
        self.chunk_size = mfcc["chunk_size"]
        self.features = mfcc["features"]

    def load_labels(self):
        labelsPath = os.path.join(self.directory, "labels.json")
        with open(labelsPath, "r") as labelsFile:
            self.labels = json.load(labelsFile)

if __name__ == "__main__":
    trainingData = TrainingData()
    trainingData.prepare()
