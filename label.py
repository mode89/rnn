import argparse
import json
import os
import pyaudio
import time
import wave

WAV_FILE = "record.wav"

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
        for stamp in sorted(map(int, self.script)):
            while self.frameCount < stamp:
                time.sleep(0.001)
            print(self.script[str(stamp)])

if __name__ == "__main__":
    main()
