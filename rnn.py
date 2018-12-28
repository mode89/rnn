from keras.layers import Dense
from keras.models import Sequential
import math
import numpy

CLASSES = 3
SAMPLES = 10
EPOCHS = 100

def main():

    numpy.random.seed(0)
    sequence = numpy.eye(CLASSES)[
        numpy.random.choice(CLASSES, SAMPLES + 1)]
    print(sequence)

    inputs = sequence[:-1]
    print(inputs)
    outputs = sequence[1:]
    print(outputs)

    model = Sequential()
    model.add(Dense(128, activation="tanh", input_shape=(CLASSES,)))
    model.add(Dense(CLASSES, activation="sigmoid"))
    model.compile(optimizer="adam", loss="mse")
    model.fit(inputs, outputs, epochs=EPOCHS)

    print(inputs)
    print(outputs)
    print(model.predict(inputs))
    print(model.predict(inputs) - outputs)

if __name__ == "__main__":
    main()
