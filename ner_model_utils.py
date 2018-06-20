import tensorflow as tf
import numpy as np


def reset_default_graph(seed=23):
    """helper function to reset the default graph."""
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def minibatches(data, minibatch_size):
    """ batch generator. yields x and y batch. """
    x_batch, y_batch = [], []
    for sentence in data:
        sentence_x = []
        sentence_y = []
        for (x, y) in sentence:
            if len(x_batch) == minibatch_size:
                yield x_batch, y_batch
                x_batch, y_batch = [], []

            sentence_x += [x]
            sentence_y += [y]
        x_batch += [sentence_x]
        y_batch += [sentence_y]

    if len(x_batch) != 0:
        for sentence in data:
            sentence_x = []
            sentence_y = []
            if len(x_batch) != minibatch_size:
                for (x, y) in sentence:
                    sentence_x.append(x)
                    sentence_y.append(y)
                x_batch.append(sentence_x)
                y_batch.append(sentence_y)
            else:
                break
        yield x_batch, y_batch
        x_batch, y_batch = [], []

def _pad_sequences(sequences, pad_tok, max_length):
    """
    Pads the given seqeunce to max_length with pad token.
    Args:
        sequences: Array.
        pad_tok: Integer. Index of the padding token.
        max_length: Integer. Length to pad sequence to.

    Returns:

    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length



def pad_sequences(sequences, pad_tok, chars=False):
    """pads the sentences, so that all sentences in a batch have the same length."""

    if chars:
        max_length_word = max([len(inds_word)
                               for seq in sequences
                               for inds_word in seq])

        sequence_padded, sequence_length = [], []

        for seq in sequences:
            sp, sl = _pad_sequences(seq,
                                    pad_tok,
                                    max_length_word)

            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(len(seq) for seq in sequences)
        sequence_padded, _ = _pad_sequences(sequence_padded,
                                            [pad_tok] * max_length_word,
                                            max_length_sentence)

        sequence_length, _ = _pad_sequences(sequence_length,
                                            0,
                                            max_length_sentence)


    else:
        max_length = max(len(seq) for seq in sequences)
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                                          pad_tok,
                                                          max_length)

    return sequence_padded, sequence_length
