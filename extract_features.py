import argparse
import numpy
import os
import struct
from tqdm import tqdm
import wave
import python_speech_features

FRAME_CHUNK_SIZE = 441

class FeatureExtractor:

    def __init__(self):
        self.parse_arguments()

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("directory")
        args = parser.parse_args()
        self.directory = args.directory

    def extract_features(self):
        self.features = list()
        for frames in self.wave_file():
            features = python_speech_features.mfcc(
                frames, samplerate=self.framerate,
                winlen=0.01,
                winstep=0.01,
                nfft=FRAME_CHUNK_SIZE)
            self.features.append(features[0])
        self.features = numpy.array(self.features)
        mfccPath = os.path.join(self.directory, "mfcc.npz")
        numpy.savez_compressed(mfccPath,
            chunk_size=FRAME_CHUNK_SIZE,
            features=self.features)

    def wave_file(self):
        waveFilePath = os.path.join(self.directory, "record.wav")
        waveFile = wave.open(waveFilePath, "rb")
        self.framerate = waveFile.getframerate()
        frameNum = waveFile.getnframes()
        sampleWidth = waveFile.getsampwidth()
        for i in tqdm(range(0, frameNum, FRAME_CHUNK_SIZE), mininterval=1):
            frames = waveFile.readframes(FRAME_CHUNK_SIZE)
            if len(frames) == FRAME_CHUNK_SIZE * sampleWidth:
                frames = numpy.array(map(float,
                    struct.unpack("{}h".format(FRAME_CHUNK_SIZE), frames)))
                yield frames

if __name__  == "__main__":
    featureExtractor = FeatureExtractor()
    featureExtractor.extract_features()
