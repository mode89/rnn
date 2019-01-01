import numpy
import tensorflow as tf

numpy.set_printoptions(suppress=True)

CLASSES = 32
SAMPLES = 4096
BATCH_SIZE = 1
SEQUENCE_SIZE = 16

def main():

    numpy.random.seed(0)
    sequence = numpy.eye(CLASSES)[
        numpy.random.choice(CLASSES, SAMPLES + 1)]
    print(sequence)

    inputs = sequence[:-1]
    inputs = numpy.reshape(inputs, (SAMPLES, CLASSES))
    print(inputs)
    outputs = sequence[1:]
    outputs = numpy.reshape(outputs, (SAMPLES, CLASSES))
    print(outputs)

    tf.set_random_seed(0)

    phInputs = tf.placeholder(tf.float32, (None, SEQUENCE_SIZE, CLASSES),
    name="phInputs")
    phOutputs = tf.placeholder(tf.float32, (None, SEQUENCE_SIZE, CLASSES),
    name="phOutputs")

    lstmCell = tf.nn.rnn_cell.LSTMCell(num_units=1024)
    zeroState = lstmCell.zero_state(BATCH_SIZE, tf.float32)
    initialState = tf.nn.rnn_cell.LSTMStateTuple(
        c=tf.placeholder(tf.float32, (None, lstmCell.state_size.c)),
        h=tf.placeholder(tf.float32, (None, lstmCell.state_size.h)))
    lstmOutputs, finalState = tf.nn.dynamic_rnn(
        cell=lstmCell,
        inputs=phInputs,
        sequence_length=[SEQUENCE_SIZE],
        initial_state=initialState)
    logits = list()
    denseLayer = tf.layers.Dense(CLASSES, tf.sigmoid)
    for i in range(SEQUENCE_SIZE):
        tempLogits = denseLayer(lstmOutputs[:, i, :])
        logits.append(tempLogits)
    logits = tf.reshape(logits, tf.shape(phOutputs))
    lossOp = tf.losses.softmax_cross_entropy(phOutputs, logits)
    optimizer = tf.train.AdamOptimizer()
    trainOp = optimizer.minimize(lossOp)
    logitsSoftmax = tf.nn.softmax(logits)
    mseOp = tf.losses.mean_squared_error(phOutputs, logits)
    diffOp = tf.losses.absolute_difference(phOutputs, logits)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.optimizer_options.global_jit_level = \
        tf.OptimizerOptions.ON_1
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        epoch = 0
        try:
            while True:
                epoch += 1
                state = session.run(zeroState)
                mseMax = None
                lossMax = None
                diffMax = None
                for batch in batches(inputs, outputs):
                    _, state, loss, mse, logitsValues = session.run(
                        [trainOp, finalState, lossOp, mseOp, logits],
                        {
                            phInputs: batch.inputs,
                            phOutputs: batch.outputs,
                            initialState.c: state.c,
                            initialState.h: state.h
                        })
                    lossMax = max(loss, lossMax)
                    mseMax = max(mse, mseMax)
                    diff = numpy.max(
                        numpy.abs(logitsValues - batch.outputs))
                    diffMax = max(diff, diffMax)
                print(epoch, lossMax, mseMax, diffMax)
                if diffMax < 0.2:
                    break
        except KeyboardInterrupt:
            pass
        state = session.run(zeroState)
        for batch in batches(inputs, outputs):
            logitsValues, state = session.run([logits, finalState],
                {
                    phInputs: batch.inputs,
                    phOutputs: batch.outputs,
                    initialState.c: state.c,
                    initialState.h: state.h
                })
            print(batch.outputs - logitsValues)
        print(epoch)

class Batch:
    pass

def batches(inputs, outputs):
    batchNum = SAMPLES / SEQUENCE_SIZE
    for i in range(batchNum):
        firstSample = i * SEQUENCE_SIZE
        lastSample = firstSample + SEQUENCE_SIZE
        batch = Batch()
        batch.inputs = numpy.reshape(
            inputs[firstSample:lastSample,:],
            (BATCH_SIZE, SEQUENCE_SIZE, CLASSES))
        batch.outputs = numpy.reshape(
            outputs[firstSample:lastSample,:],
            (BATCH_SIZE, SEQUENCE_SIZE, CLASSES))
        yield batch

if __name__ == "__main__":
    main()
