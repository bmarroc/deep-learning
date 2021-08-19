import csv
import numpy as np
import emoji

def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        word_to_vec = {}
        word_to_vec['<eos>'] = np.zeros(shape=(50,), dtype='float32')
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            word_to_vec[curr_word] = np.array(line[1:], dtype='float32')
        
        word_to_idx = {}
        idx_to_word = {}
        i = 0
        for w in list(word_to_vec.keys()):
            word_to_idx[w] = i
            idx_to_word[i] = w
            i = i + 1

    return word_to_idx, idx_to_word, word_to_vec


def read_csv(filename):
    phrase = []
    emoji = []

    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            phrase.append(row[0])
            emoji.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(emoji, dtype=int)

    return X, Y


emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

def label_to_emoji(label):
    """
    Converts a label (int or string) into the corresponding emoji code (string) ready to be printed
    """
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)
    
    
