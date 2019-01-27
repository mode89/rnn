import datetime
import json
import os
import pyaudio
import random
import time
import wave

FRAME_RATE = 44100
FRAMES_PER_BUFFER = 441
WORD_NUM = 100
WAV_FILE = "record.wav"
SCRIPT_FILE = "script.json"

VOCABULARY = [
    "cancel",
    "computer",
    "yes",
    "no",
    "ok",
]

class Recorder:

    def __init__(self):
        self.generate_random_list_of_words()
        self.script = dict()
        self.make_directory()

    def run(self):
        self.start_recording()
        self.print_words()
        self.stop_recording()

    def generate_random_list_of_words(self):
        wordNum = WORD_NUM / len(VOCABULARY)
        words = list()
        for word in VOCABULARY:
            words += [word] * wordNum
        random.shuffle(words)
        self.words = words

    def make_directory(self):
        now = datetime.datetime.now()
        self.directory = "{:%Y-%m-%d-%H-%M-%S}".format(now)
        print("Creating directory {}".format(self.directory))
        os.mkdir(self.directory)

    def start_recording(self):
        self.pyaudio = pyaudio.PyAudio()
        self.waveFile = wave.open(
            os.path.join(self.directory, WAV_FILE), "wb")
        self.waveFile.setnchannels(1)
        self.waveFile.setsampwidth(2)
        self.waveFile.setframerate(FRAME_RATE)

        def callback(data, frameCount, timeInfo, status):
            if status != 0:
                print("Something went wrong during audio capturing")
            else:
                self.frameCount += frameCount
                self.waveFile.writeframes(data)
            return (None, pyaudio.paContinue)

        self.frameCount = 0
        self.stream = self.pyaudio.open(
            format=self.pyaudio.get_format_from_width(2),
            channels=1,
            rate=FRAME_RATE,
            input=True,
            input_device_index=1,
            frames_per_buffer=FRAMES_PER_BUFFER,
            stream_callback=callback)
        self.stream.start_stream()

    def stop_recording(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()
        self.waveFile.close()

    def print_words(self):
        counter = 0
        self.wait_for(3)
        for word in self.words:
            print("{} {}".format(counter, word))
            self.script[self.frameCount] = word
            self.dump_script()
            counter += 1
            pause = random.uniform(4, 8)
            self.wait_for(pause)

    def dump_script(self):
        scriptPath = os.path.join(self.directory, SCRIPT_FILE)
        with open(scriptPath, "w") as scriptFile:
            json.dump(self.script, scriptFile, indent=4, sort_keys=True)

    def wait_for(self, seconds):
        time.sleep(seconds)

if __name__ == "__main__":
    recorder = Recorder()
    recorder.run()
