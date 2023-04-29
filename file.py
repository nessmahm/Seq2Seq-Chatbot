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
    pairs = filterPairs(pairs)[:10000]
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
pairs = pairs[:10000]
# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)

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
"""def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

#The function returns a list of sequences, where each sequence
# is zero-padded to the length of the longest sequence in the input list.
def zeroPadding(l, fillvalue=PAD_token):

    #itertools.zip_longest function works by taking the elements of each
    # sequence in the input list in turn and grouping them together. If
    # the sequences are of different lengths, the missing values are filled
    # with the specified fillvalue. The * in zip_longest(*l) is used to
    # unpack the list of sequences, so that each sequence is passed as a
    # separate argument to the zip_longest function.
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = t.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = t.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = t.BoolTensor(mask)
    padVar = t.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


# Example for validation
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches


print(np.shape(input_variable))
print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)"""

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

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

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


print(voc.word2index['hello'])
print("##########################################")
print("#       start chatting ver. 1.0          #")
print("##########################################")


prepro1 = ""
while prepro1 != 'q':
    prepro1  = input("you : ")
    ## prepro1 = "Hello"

    prepro1 = normalizeString(prepro1)
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

    print('text',txt)
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
        print("word index",sampled_word_index)

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

    print("chatbot attention : ", decoded_translation)
    print("==============================================")
"""print(np.shape(tf_target))
print(np.shape(tf_input))
print(np.shape([tf_input, tf_target]))
print(np.shape([encoder_inputs, decoder_inputs]))
# Define an input sequence and process it.
#Dimensionality
dimensionality = 256
#The batch size and number of epochs
batch_size = 50
epochs = 20
latent_dim = 256
#Encoder
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_lstm = LSTM(dimensionality, return_state=True)
encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
encoder_states = [state_hidden, state_cell]
#Decoder
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
#Compiling
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode='temporal')
#Training
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size = batch_size, epochs = epochs, validation_split = 0.2)

model.save('training_model_2000.h5')
from keras.models import load_model
training_model = load_model('training_model_2000.h5')
encoder_inputs = training_model.input[0]
encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)
latent_dim = 256
decoder_state_input_hidden = Input(shape=(latent_dim,))
decoder_state_input_cell = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_hidden, state_cell]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)


# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(test_input):
    # Getting the output states to pass into the decoder
    states_value = encoder_model.predict(test_input)
    # Generating empty target sequence of length 1
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Setting the first token of target sequence with the start token
    target_seq[0, 0, target_token_index['\t']] = 1.

    # A variable to store our response word by word
    decoded_sentence = ''

    stop_condition = False

    while not stop_condition:
        # Predicting output tokens with probabilities and states
        output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value)
        # Choosing the one with highest probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_char_index[sampled_token_index]
        decoded_sentence += " " + sampled_token
        # Stop if hit max length or found the stop token
        if (sampled_token == '\n' or len(decoded_sentence) > MAX_LENGTH):
            stop_condition = True
        # Update the target sequence
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        # Update states
        states_value = [hidden_state, cell_state]
    return decoded_sentence


def string_to_matrix(user_input):
    tokens = normalizeString(user_input).split()[:MAX_LENGTH]
    user_input_matrix = np.zeros(
        (1, MAX_LENGTH, num_encoder_tokens),
        dtype='float32')
    for timestep, token in enumerate(tokens):
        if token in input_token_index:
            user_input_matrix[0, timestep, input_token_index[token]] = 1.
    return user_input_matrix
input_prompt = 'hello!'
generated_response = generate_response(input_prompt)
print(generated_response)
for seq_index in range(10):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = string_to_matrix(input_texts[seq_index])
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)

def chat():
    user_input = input('Hi, How can I help you today?')
    print(decode_sequence(string_to_matrix(user_input)))
    stop_condition = False
    while stop_condition == False:
        user_input = input()
        if user_input.upper()== 'Q' or user_input.upper() == 'QUIT':
            stop_condition = True
            break
        print(decode_sequence(string_to_matrix(user_input)))
chat()"""
