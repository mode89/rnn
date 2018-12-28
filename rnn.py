from keras.callbacks import Callback
from keras.layers import Activation, Dense, LSTM
from keras.models import Sequential
import math
import numpy

CLASSES = 32
SAMPLES = 4096
BATCH_SIZE = 1024
EPOCHS = 10000

def main():

    # numpy.random.seed(6)
    numpy.random.seed(0)
    sequence = numpy.eye(CLASSES)[
        numpy.random.choice(CLASSES, SAMPLES + 1)]
    print(sequence)

    inputs = sequence[:-1]
    inputs = numpy.reshape(inputs, (SAMPLES, 1, CLASSES))
    print(inputs)
    outputs = sequence[1:]
    outputs = numpy.reshape(outputs, (SAMPLES, CLASSES))
    print(outputs)

    model = Sequential()
    model.add(LSTM(
        units=1024,
        batch_input_shape=(BATCH_SIZE, 1, CLASSES),
        stateful=True))
    model.add(Dense(CLASSES, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"])

    class ResetStatesCallback(Callback):

        def on_batch_begin(self, batch, logs):
            if batch == (SAMPLES / BATCH_SIZE):
                model.reset_states()

    model.fit(inputs, outputs, callbacks=[ResetStatesCallback()],
        epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False)

    print(inputs)
    print(outputs)
    print(model.predict(inputs, batch_size=BATCH_SIZE))

if __name__ == "__main__":
    main()
