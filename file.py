import codecs
import csv
import itertools
import re,os,json
from random import random
import random
import torch as t

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import unicodedata
from opt_einsum.backends import torch

#printing data
corpus_name = "movie-corpus"
corpus = os.path.join("data")

def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

# printLines(os.path.join(corpus, "utterances.jsonl"))

# I /Create formatted data file

# Splits each line of the file to create lines and conversations
def loadLinesAndConversations(fileName):
    lines = {}
    conversations = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            lineJson = json.loads(line)
            # Extract fields for line object
            lineObj = {}
            lineObj["lineID"] = lineJson["id"]
            lineObj["characterID"] = lineJson["speaker"]
            lineObj["text"] = lineJson["text"]
            lines[lineObj['lineID']] = lineObj

            # Extract fields for conversation object
            if lineJson["conversation_id"] not in conversations:
                convObj = {}
                convObj["conversationID"] = lineJson["conversation_id"]
                convObj["movieID"] = lineJson["meta"]["movie_id"]
                convObj["lines"] = [lineObj]
            else:
                convObj = conversations[lineJson["conversation_id"]]
                convObj["lines"].insert(0, lineObj)
            conversations[convObj["conversationID"]] = convObj

    return lines, conversations


# Extracts pairs of sentences from conversations
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations.values():
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            #strip (delete white space )
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


# Define path to new file
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

delimiter = '\t'
# Unescape the delimiter
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

# Initialize lines dict and conversations dict
lines = {}
conversations = {}
# Load lines and conversations
print("\nProcessing corpus into lines and conversations...")
lines, conversations = loadLinesAndConversations(os.path.join(corpus, "utterances.jsonl"))

# Write new csv file
print("\nWriting newly formatted file...")
with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
    for pair in extractSentencePairs(conversations):
        writer.writerow(pair)

# Print a sample of lines
print("\nSample lines from file:")
# printLines(datafile)

# putting pairs into variable not a file
pairs=extractSentencePairs(conversations)
#print(pairs)


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.num_words = 4  # Count SOS, EOS, PAD, OUT

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.num_words = 4 # Count default tokens

        for word in keep_words:
            self.addWord(word)
MAX_LENGTH = 10  # Maximum sentence length to consider

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
#separates the base character and diacritical mark into separate characters.
# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    #removes diacritics from the string, and then lower and strip are
    # used to convert the string to lowercase and remove any leading/trailing
    # whitespace.
    s = s.lower()
    s = re.sub(r"i'm", "i am", s)
    s = re.sub(r"he's", "he is", s)
    s = re.sub(r"she's", "she is", s)
    s = re.sub(r"that's", "that is", s)
    s = re.sub(r"what's", "what is", s)
    s = re.sub(r"where's", "where is", s)
    s = re.sub(r"\'ll", " will", s)
    s = re.sub(r"\'ve", " have", s)
    s = re.sub(r"\'re", " are", s)
    s = re.sub(r"\'d", " would", s)
    s = re.sub(r"won't", "will not", s)
    s = re.sub(r"can't", "can not", s)
    s = re.sub(r"[^\w\s]", "", s)
    #This step adds a space before any punctuation marks (. ! ?) so
    # that they can be treated as separate tokens in the later stages
    # of the NLP pipeline
    s = re.sub(r"([.!?])", r" \1", s)
    #replaces any character that is not a letter, punctuation mark or
    # whitespace with a single space. This is done to simplify the text
    # and reduce the number of unique tokens.
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    #eplaces multiple spaces with a single space and removes any leading/trailing
    # whitespace that may have been introduced in the previous steps.
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# Read query/response pairs and return a voc object

for i in range(len(pairs)):
    pairs[i][0] = normalizeString(pairs[i][0])
    pairs[i][1] = normalizeString(pairs[i][1])
voc = Voc(corpus_name)

# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(voc , pairs ):
    print("Start preparing training data ...")

    print("Read {!s} sentence pairs".format(len(pairs)))
    #limiting for testing
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs


# Load/Assemble voc and pairs
save_dir = os.path.join("data", "save")
voc, pairs = loadPrepareData(voc,pairs)
#for testing we limit the pairs
pairs = pairs[:60000]
# Print some pairs to validate

MIN_COUNT = 3    # Minimum word count threshold for trimming
def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs
# Trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)

from keras.layers import *
from keras.models import *
from keras.utils import *
from keras.initializers import *
from keras.optimizers import Adam
import tensorflow as tf



"""tf_input = tf.convert_to_tensor(input_variable,dtype=tf.int32)
tf_target = tf.convert_to_tensor(target_variable,dtype=tf.int32)
decoder_target_data = np.zeros_like(tf_target)
decoder_target_data[:, :-1] = tf_target[:, 1:]"""

# Define the encoder

input_characters = set()
target_characters = set()
input_texts = []
target_texts = []
for input_text, target_text in pairs:
    target_text = "SOS " + target_text + " EOS"
    input_texts.append(input_text)
    target_texts.append(target_text)
    for token in input_text.split():
        if token not in input_characters:
            input_characters.add(token)
    for token in target_text.split():
        if token not in target_characters:
            target_characters.add(token)


input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])
MAX_LENGTH = MAX_LENGTH + 4 #for the added tokens

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

print('input',input_token_index)
print('output',target_token_index)

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)
"""encoder_input_data = np.zeros(
    (len(input_texts), MAX_LENGTH, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), MAX_LENGTH, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), MAX_LENGTH, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text.split()):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text.split(" ")):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1."""
tokens = ['PAD', 'EOS', 'OUT', 'SOS']
x = voc.num_words
for token in tokens:
    voc.word2index[token] = x
    x += 1
    voc.num_words+=1
voc.word2index['PAD'] = 0
print(voc.word2index)
encoder_inp = []
for line in input_texts:
    lst = []
    for word in line.split():
        if word not in voc.word2index:
            lst.append(voc.word2index["OUT"])
        else:
            lst.append(voc.word2index[word])

    encoder_inp.append(lst)

decoder_inp = []
for line in target_texts:
    lst = []
    for word in line.split():
        if word not in voc.word2index:
            lst.append(voc.word2index["OUT"])
        else:
            lst.append(voc.word2index[word])
    decoder_inp.append(lst)



### inv answers dict ###
voc.index2word = {w: v for v, w in voc.word2index.items()}
from tensorflow.keras.preprocessing.sequence import pad_sequences

encoder_inp = pad_sequences(encoder_inp, MAX_LENGTH, padding='post', truncating='post')
decoder_inp = pad_sequences(decoder_inp, MAX_LENGTH, padding='post', truncating='post')




decoder_final_output = []
for i in decoder_inp:
    decoder_final_output.append(i[1:])

decoder_final_output = pad_sequences(decoder_final_output, MAX_LENGTH, padding='post', truncating='post')



from tensorflow.keras.utils import to_categorical
decoder_final_output = to_categorical(decoder_final_output, voc.num_words)



print(decoder_final_output.shape)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input


enc_inp = Input(shape=(MAX_LENGTH, ))
dec_inp = Input(shape=(MAX_LENGTH, ))


VOCAB_SIZE = voc.num_words
embed = Embedding(VOCAB_SIZE+1, output_dim=50,
                  input_length=MAX_LENGTH,
                  trainable=True
                  )

enc_embed = embed(enc_inp)
enc_lstm = LSTM(400, return_sequences=True, return_state=True)
enc_op, h, c = enc_lstm(enc_embed)
enc_states = [h, c]


dec_embed = embed(dec_inp)
dec_lstm = LSTM(400, return_sequences=True, return_state=True)
dec_op, _, _ = dec_lstm(dec_embed, initial_state=enc_states)

dense = Dense(VOCAB_SIZE, activation='softmax')

dense_op = dense(dec_op)

model = Model([enc_inp, dec_inp], dense_op)




model.compile(loss='categorical_crossentropy',metrics=['acc'],optimizer='adam')

model.fit([encoder_inp, decoder_inp],decoder_final_output,epochs=50)
model.save('training_model_2000.h5')

def getTensorsFromIndex(index):
    input = []
    output = []
    for x in input_texts[index]:
        lst = []
        for y in x.split():
            try:
                lst.append(voc.word2index[y])
            except KeyError:
                lst.append(voc.word2index["OUT"])
        input.append(lst)

    for x in target_texts[index]:
        lst = []
        for y in x.split():
            try:
                lst.append(voc.word2index[y])
            except KeyError:
                lst.append(voc.word2index["OUT"])
        output.append(lst)

    ## txt = [[454]]
    input = pad_sequences(input, MAX_LENGTH, padding='post')
    output = pad_sequences(output, MAX_LENGTH, padding='post')
    return [input, output]

def getTensor(input):
    prepro1 = normalizeString(input)
    ## prepro1 = "hello"

    prepro = [prepro1]
    ## prepro1 = ["hello"]

    txt = []
    for x in prepro:
        # x = "hello"
        lst = []
        for y in x.split():
            ## y = "hello"
            try:
                lst.append(voc.word2index[y])
                ## vocab['hello'] = 454
            except KeyError:
                lst.append(voc.word2index["OUT"])
        txt.append(lst)

    print('text', txt)
    ## txt = [[454]]
    txt = pad_sequences(txt, MAX_LENGTH, padding='post')
    return txt


def generate_response(input):
    prepro1 = normalizeString(input)
    ## prepro1 = "hello"

    prepro = [prepro1]
    ## prepro1 = ["hello"]

    txt = []
    for x in prepro:
        # x = "hello"
        lst = []
        for y in x.split():
            ## y = "hello"
            try:
                lst.append(voc.word2index[y])
                ## vocab['hello'] = 454
            except KeyError:
                lst.append(voc.word2index["OUT"])
        txt.append(lst)

    print('text', txt)
    ## txt = [[454]]
    txt = pad_sequences(txt, MAX_LENGTH, padding='post')

    ## txt = [[454,0,0,0,.........13]]

    stat = enc_model.predict(txt)

    empty_target_seq = np.zeros((1, 1))
    ##   empty_target_seq = [0]

    empty_target_seq[0, 0] = voc.word2index['SOS']
    ##    empty_target_seq = [255]
    stop_condition = False
    decoded_translation = ''

    while not stop_condition:
        dec_outputs, h, c = dec_model.predict([empty_target_seq] + stat)
        decoder_concat_input = dense(dec_outputs)
        ## decoder_concat_input = [0.1, 0.2, .4, .0, ...............]

        sampled_word_index = np.argmax(decoder_concat_input[0, -1, :])
        ## sampled_word_index = [2]
        print("word index", sampled_word_index)

        sampled_word = voc.index2word[sampled_word_index]
        if sampled_word != 'EOS':
            if (decoded_translation != ''):
                sampled_word = ' ' + sampled_word
            decoded_translation += sampled_word

        if sampled_word == 'EOS' or len(decoded_translation.split()) > MAX_LENGTH:
            stop_condition = True

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        ## <SOS> - > hi
        ## hi --> <EOS>
        stat = [h, c]
        return decoded_translation
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K



#model = load_model('training_model_2000.h5')
enc_model = Model([enc_inp], enc_states)
decoder_state_input_h = Input(shape=(400,))
decoder_state_input_c = Input(shape=(400,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]


decoder_outputs, state_h, state_c = dec_lstm(dec_embed ,
                                    initial_state=decoder_states_inputs)


decoder_states = [state_h, state_c]


dec_model = Model([dec_inp]+ decoder_states_inputs,
                                      [decoder_outputs]+ decoder_states)
import tensorflow as tf
import numpy as np

import numpy as np
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences





print(voc.word2index['hello'])
print("##########################################")
print("#       start chatting ver. 1.0          #")
print("##########################################")
user_input = ""
while user_input != 'q':
    user_input = input("you : ")
    print("chatbot attention : ", generate_response(user_input))
    print("==============================================")


