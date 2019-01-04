import argparse
import json
import numpy
import os

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
