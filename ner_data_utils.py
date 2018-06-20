import numpy as np
import nltk


def preprocess_data(file_path, lower = False, use_chars = False):
    """
    Creates input data for the model and most common words
    to create the lookup dicts.
    Optionally lower the words. Optionally same on char level.

    Args:
        file_path: String. Path to the data file.
        lower: Boolean.
        use_chars: Boolean.

    """
    with open(file_path, 'r') as f:
        data = f.readlines()

        tokenized = []
        for line in data:
            tokens = nltk.word_tokenize(line, language='german')
            tokenized.append(tokens)

        data = []
        entities = []
        words = []
        cats = ['B-OTH', 'I-OTH', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER']
        # cats = ['OTH', 'LOC', 'ORG', 'PER']

        if use_chars:
            unique_chars = set()

        # create tokenized list of words
        for row in tokenized:

            if len(row) == 4 and row[0] != '#':
                for cat in cats:
                    if cat in row[2]:
                        if lower:
                            if use_chars:
                                char_array = [ch for ch in row[1].lower()]
                                entities.append(((char_array, row[1].lower()), cat))
                                unique_chars.update(char_array)
                                char_array = []
                            else:
                                entities.append((row[1].lower(), cat))

                        else:
                            if use_chars:
                                char_array = [ch for ch in row[1]]
                                entities.append(((char_array, row[1]), cat))
                                unique_chars.update(char_array)
                                char_array = []
                            else:
                                entities.append((row[1], cat))

                if row[2] == 'O':
                    if lower:
                        if use_chars:
                            char_array = [ch for ch in row[1].lower()]
                            entities.append(((char_array, row[1].lower()), 'O'))
                            unique_chars.update(char_array)
                            char_array = []
                        else:
                            entities.append((row[1].lower(), 'O'))

                    else:
                        if use_chars:
                            char_array = [ch for ch in row[1]]
                            entities.append(((char_array, row[1]), 'O'))
                            unique_chars.update(char_array)
                            char_array = []
                        else:
                            entities.append((row[1], 'O'))
                if lower:
                    words.append(row[1].lower())
                else:
                    words.append(row[1])

            elif len(row) == 0 and entities!=[]:
                data.append(entities)
                entities = []

        print("File loaded: {} !".format(file_path))
        if use_chars:
            return data, np.array(words), unique_chars
        else:
            return data, np.array(words)


def create_lookup_dicts(tokens_counted, specials=None, chars=None):
    """
    Creates the lookup dicts.
    Args:
        tokens_counted: Array of tuples.
        specials: Array of special tokens to include or None.
        chars: Array of characters or none.

    Returns:
        word2ind: Lookup dictionary from words to index.
        ind2word: Lookup dictionary from index to words.
        + And vocab size.

    """
    missing_words = {}
    word2ind = {}
    ind2word = {}
    i = 0

    if specials is not None:
        for sp in specials:
            word2ind[sp] = i
            ind2word[i] = sp
            i += 1
    if chars is not None:
        for ch in chars:
            word2ind[ch] = i
            ind2word[i] = ch
            i += 1

    for (token, count) in tokens_counted:
        if token not in word2ind.keys():
            word2ind[token] = i
            ind2word[i] = token
            i += 1

    return word2ind, ind2word, len(word2ind)


def create_char_lookup_dicts(unique_chars):
    """
    Creates lookup dicts on character level.
    Args:
        unique_chars: Array of characters.

    Returns:

    """
    char2ind = {'<PAD>': 0, '<UNK>': 1}
    ind2char = {0: '<PAD>', 1: '<UNK>'}

    i = 1

    for ch in unique_chars:
        char2ind[ch] = i
        ind2char[i] = ch
        i += 1
    return char2ind, ind2char, len(char2ind)


def convert_to_inds(sentence, word2ind, char2ind=None, chars=False):
    """Converts given input to int values corresponding to given word2ind """
    inds = []
    unknown_words = []
    for (word, entity) in sentence:
        if chars:
            if word[1] in word2ind.keys():
                char_inds = [char2ind[ch] if ch in char2ind.keys() else char2ind['<UNK>'] for ch in word[0]]
                inds.append(((char_inds, word2ind[word[1]]), word2ind[entity]))

            else:
                char_inds = [char2ind[ch] if ch in char2ind.keys() else char2ind['<UNK>'] for ch in word[0]]
                inds.append(((char_inds, word2ind['<UNK>']), word2ind[entity]))
                unknown_words.append(word)
        else:
            if word in word2ind.keys():
                inds.append((word2ind[word], word2ind[entity]))

            else:
                inds.append((word2ind['<UNK>'], word2ind[entity]))
                unknown_words.append(word)

    return inds, unknown_words, len(unknown_words)


def convert_inputs_and_targets(inputs, word2ind, targets=None, char2ind=None, chars=False):
    """Converts inputs and targets to integers with help of lookup dicts."""
    converted_inputs = []
    all_unknown_words = set()

    if targets is not None:
        converted_targets = []
        # first the inputs
        for sentence in inputs:
            converted_input, unknown_words, _ = convert_to_inds(sentence, word2ind, char2ind, chars)
            converted_inputs.append(converted_input)
            all_unknown_words.update(unknown_words)

        # and the targets
        for sentence in targets:
            converted_target, unknown_words, _ = convert_to_inds(sentence, word2ind, char2ind, chars)
            converted_targets.append(converted_target)
            all_unknown_words.update(unknown_words)

        return converted_inputs, converted_targets, all_unknown_words, all_unknown_words

    else:
        for sentence in inputs:
            converted_input, unknown_words, _ = convert_to_inds(sentence, word2ind, char2ind, chars)
            converted_inputs.append(converted_input)
            all_unknown_words.update(unknown_words)
        return converted_inputs, all_unknown_words, all_unknown_words


def print_examples(input_sentences, preds, ind2word, chars=False):
    """
    Prints predictions side by side with the actual entities of a sentence.
    """
    counter = 0
    sentences = []
    actual_ents = []
    predicted_ents = []

    for sent in input_sentences:
        sentence_array = []
        actual_ents_array = []
        predicted_ents_array = []
        for word, ent in sent:
            if chars:
                word = word[1]

            sentence_array.append(ind2word[word])
            actual_ents_array.append(ind2word[ent])
            predicted_ents_array.append(ind2word[preds[counter]])
            counter += 1

        sentences.append(sentence_array)
        actual_ents.append(actual_ents_array)
        predicted_ents.append(predicted_ents_array)

    for i, (s, a, p) in enumerate(zip(sentences, actual_ents, predicted_ents)):
        print('\n\nSentence n.{}:\n{}\nActual entities:\n{}\nPredicted entites:\n{}\n'.format(i, s, a, p))
