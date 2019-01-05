import argparse
import numpy
import os

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
        class Samples: pass
        self.samples = Samples()
        self.samples.inputs = data["inputs"]
        self.samples.outputs = data["outputs"]

if __name__ == "__main__":
    model = Model()
