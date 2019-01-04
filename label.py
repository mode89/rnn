from __future__ import print_function
import argparse
import json
import os
import pyaudio
import time
import wave

WAV_FILE = "record.wav"
LABELS_FILE = "labels.json"

def main():
    args = parse_arguments()
    labelMaker = LabelMaker(args)
    labelMaker.run()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    return parser.parse_args()

class LabelMaker:

    def __init__(self, args):
        self.directory = args.directory
        self.labels = list()

    def run(self):
        self.start_playback()
        self.load_script()
        self.display_script()
        self.stop_playback()

    def start_playback(self):
        self.pyaudio = pyaudio.PyAudio()
        waveFilePath = os.path.join(self.directory, WAV_FILE)
        self.waveFile = wave.open(waveFilePath, "rb")
        sampleWidth = self.waveFile.getsampwidth()

        def callback(data, frameCount, timeInfo, status):
            if status != 0:
                print("Something went wrong during audio capturing")
            data = self.waveFile.readframes(frameCount)
            self.frameCount += frameCount
            return (data, pyaudio.paContinue)

        self.frameCount = 0
        self.stream = self.pyaudio.open(
            format=self.pyaudio.get_format_from_width(sampleWidth),
            channels=self.waveFile.getnchannels(),
            rate=self.waveFile.getframerate(),
            output=True,
            input_device_index=1,
            stream_callback=callback)
        self.stream.start_stream()

    def stop_playback(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()
        self.waveFile.close()

    def load_script(self):
        scriptPath = os.path.join(self.directory, "script.json")
        with open(scriptPath, "r") as scriptFile:
            self.script = json.load(scriptFile)

    def display_script(self):
        self.counter = 0
        for stamp in sorted(map(int, self.script)):
            self.wait_until(stamp)
            self.create_label(stamp)
            self.dump_labels()
            self.counter += 1

    def wait_until(self, stamp):
        while self.frameCount < stamp:
            time.sleep(0.001)

    def create_label(self, stamp):
        word = self.script[str(stamp)]
        print(self.counter, end=" ")
        raw_input(word)
        label = {
            "word": word,
            "begin": stamp,
            "end": self.frameCount
        }
        self.labels.append(label)

    def dump_labels(self):
        labelsPath = os.path.join(self.directory, LABELS_FILE)
        with open(labelsPath, "w") as labelsFile:
            json.dump(self.labels, labelsFile, indent=4)

if __name__ == "__main__":
    main()
