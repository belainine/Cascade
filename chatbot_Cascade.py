# -*- coding: utf-8 -*-

"""
Chatbot Tutorial
================
**Author:** `Matthew Inkawhich <https://github.com/MatthewInkawhich>`_
"""
######################################################################
#                            tensorboard
#tensorboard --logdir=C:\Users\belainine\myenv\pytorch-chatbot-master\chatbot_tutorial\runs\logs\
######################################################################

######################################################################
# In this tutorial, we explore a fun and interesting use-case of recurrent
# Cascade models. We will train a simple chatbot using movie
# scripts from the `Cornell Movie-Dialogs
#
# Conversational models are a hot topic in artificial intelligence
# research. Chatbots can be found in a variety of settings, including
# customer service applications and online helpdesks. These bots are often
# powered by retrieval-based models, which output predefined responses to
# questions of certain forms. In a highly restricted domain like a
# company’s IT helpdesk, these models may be sufficient, however, they are
# not robust enough for more general use-cases. Teaching a machine to
# carry out a meaningful conversation with a human in multiple domains is
# a research question that is far from solved. Recently, the deep learning
# boom has allowed for powerful generative models like Google’s `Neural
# a large step towards multi-domain generative conversational models. In
# this tutorial, we will implement this kind of model in PyTorch.
#
#
# .. code:: python
#
#   > hello?
#   Bot: hello .
#   > where am I?
#   Bot: you re in a hospital .
#   > who are you?
#   Bot: i m a lawyer .
#   > how are you doing?
#   Bot: i m fine .
#   > are you my friend?
#   Bot: no .
#   > you're under arrest
#   Bot: i m trying to help you !
#   > i'm just kidding
#   Bot: i m sorry .
#   > where are you from?
#   Bot: san francisco .
#   > it's time for me to leave
#   Bot: i know .
#   > goodbye
#   Bot: goodbye .
#
# **Tutorial Highlights**
#
# -  Handle loading and preprocessing of `Cornell Movie-Dialogs
#    Corpus <https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html>`__
#    dataset
# -  Implement a sequence-to-sequence model with `Luong attention
#    mechanism(s) <https://arxiv.org/abs/1508.04025>`__
# -  Jointly train encoder and decoder models using mini-batches
# -  Implement greedy-search decoding module
# -  Interact with trained chatbot
#
# **Acknowledgements**
#
# This tutorial borrows code from the following sources:
#
# 1) FloydHub’s Cornell Movie Corpus preprocessing code:
#    https://github.com/floydhub/textutil-preprocess-cornell-movie-corpus
#


######################################################################
# Preparations
# ------------
#
# To start, Download the data ZIP file
# `here <https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html>`__
# and put in a ``data/`` directory under the current directory.
#
# After that, let’s import some necessities.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
from torch.autograd import Variable
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.tokenize import TweetTokenizer
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from metrics import *
from jiwer import wer
#from tensorboardX import SummaryWriter

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
from loss.loss import Loss
from loss import NLLLoss, Perplexity
import spacy
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
import pickle
from gensim.models import Word2Vec, KeyedVectors
from tqdm import tqdm
spacy.prefer_gpu()
nlp = spacy.load("en")
######################################################################
# Load & Preprocess Data
# ----------------------
#
# The next step is to reformat our data file and load the data into
# structures that we can work with.
#
# The `Cornell Movie-Dialogs
# Corpus <https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html>`__
# is a rich dataset of movie character dialog:
#
# -  220,579 conversational exchanges between 10,292 pairs of movie
#    characters
# -  9,035 characters from 617 movies
# -  304,713 total utterances
#
# This dataset is large and diverse, and there is a great variation of
# language formality, time periods, sentiment, etc. Our hope is that this
# diversity makes our model robust to many forms of inputs and queries.
#
# First, we’ll take a look at some lines of our datafile to see the
# original format.
#

#corpus_name = 'ubuntu'
#corpus_name = 'cornell'
corpus_name = "dailydialogs"
corpus = os.path.join("data", corpus_name)
nb_decoder=3
use_embed=False

######################################################################
# Load and trim data
# ~~~~~~~~~~~~~~~~~~
#
# Our next order of business is to create a vocabulary and load
# query/response sentence pairs into memory.
#
# Note that we are dealing with sequences of **words**, which do not have
# an implicit mapping to a discrete numerical space. Thus, we must create
# one by mapping each unique word that we encounter in our dataset to an
# index value.
#
# For this we define a ``Voc`` class, which keeps a mapping from words to
# indexes, a reverse mapping of indexes to words, a count of each word and
# a total word count. The class provides methods for adding a word to the
# vocabulary (``addWord``), adding all words in a sentence
# (``addSentence``) and trimming infrequently seen words (``trim``). More
# on trimming later.
#

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # UNK-in-sentence token
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = { "<pad>":PAD_token,  "<sos>":SOS_token,  "<eos>":EOS_token, '<unk>':UNK_token}
        self.word2count = {"<pad>":0,  "<sos>":0,  "<eos>":0 , '<unk>':0}
        self.index2word = {PAD_token: "<pad>", SOS_token: "<sos>", EOS_token: "<eos>", UNK_token:'<unk>'}
        self.num_words = 4  # Count SOS, EOS, PAD, UNK
        self.tt = TweetTokenizer(strip_handles=True, reduce_len=True)
    def addSentence(self, sentence):
        
        for word in self.tt.tokenize(sentence):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            if word not in self.word2count:
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
        trim_list = []
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
            else:
                trim_list.append(k)
        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        
        self.word2index = { "<pad>":PAD_token,  "<sos>":SOS_token,  "<eos>":EOS_token, '<unk>':UNK_token}
        #self.word2count = {"<pad>":0,  "<sos>":0,  "<eos>":0 , '<unk>':0,
        #                     '<No>':0}
        self.index2word = {PAD_token: "<pad>", SOS_token: "<sos>", EOS_token: "<eos>", UNK_token:'<unk>'}
        self.num_words = 4 # Count default tokens
        for word in trim_list:
            del self.word2count[word]
        self.word2count["<pad>"] = 5
        self.word2count["<sos>"] = 5
        self.word2count["<eos>"] = 5
        self.word2count['<unk>'] = 5
        for word in keep_words:
            self.addWord(word)


######################################################################
# Now we can assemble our vocabulary and query/response sentence pairs.
# Before we are ready to use this data, we must perform some
# preprocessing.
#
# First, we must convert the Unicode strings to ASCII using
# ``unicodeToAscii``. Next, we should convert all letters to lowercase and
# trim all non-letter characters except for basic punctuation
# (``normalizeString``). Finally, to aid in training convergence, we will
# filter out sentences with length greater than the ``MAX_LENGTH``
# threshold (``filterPairs``).
#

MAX_LENGTH = 16  # Maximum sentence length to consider



# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    
    s = unicodeToAscii(s.lower().strip())
    
    return s

def toNer(text):
    tt = TweetTokenizer(strip_handles=True, reduce_len=True)
    text=' '.join(['<url>' if '/' in w else w for w in tt.tokenize(text)])
    doc = nlp(text)
    text2=''
    i=0
    for ent in doc.ents:
        prefex=re.sub(r'(\w+)-(\w+)',r'\1\2',text[i:ent.start_char].lower())
        prefex=re.sub(r'(\d+)-(\w+)',r'\1 - \2',prefex)
        #print(text[ent.start_char:ent.end_char]+'  '+ent.label_)
        if(text[ent.start_char:ent.end_char].isalpha()==False or ent.label_ in ['PERSON','GPE','ORG']):
            text2+=prefex+' <'+ent.label_+'> '
            i=ent.end_char
    suffix=re.sub(r'(\w+)-(\w+)',r'\1\2',text[i:].lower())
    text2=text2+' '+suffix
    #print(text2)
    return text2.replace('  ',' ').lower().strip() , doc.ents

######################################################################
# Another tactic that is beneficial to achieving faster convergence during
# training is trimming rarely used words out of our vocabulary. Decreasing
# the feature space will also soften the difficulty of the function that
# the model must learn to approximate. We will do this as a two-step
# process:
#
# 1) Trim words used under ``MIN_COUNT`` threshold using the ``voc.trim``
#    function.
#
# 2) Filter out pairs with trimmed words.
#

MIN_COUNT = 4    # Minimum word count threshold for trimming

def trimRareWords(voc, utterances, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    tt = TweetTokenizer(strip_handles=True, reduce_len=True)
    keep_utterances_inv = []
    keep_utterances = []
    print('Trim rare words')
    for utterance in tqdm(utterances):
        keep_input=False
        for k in range(len(utterance)):
            utt=utterance[k]
            input_sentence_list = tt.tokenize(utt)
            
            # Check input sentence
            
            for i in range(len(input_sentence_list)):
                word=input_sentence_list[i]
                if  word not in voc.word2count or voc.word2count[word]<=MIN_COUNT:
                    keep_input = True
                    voc.word2count['<unk>']+=1
                    input_sentence_list[i]='<unk>'
            utt=' '.join(input_sentence_list)
            utterance[k]=utt
            # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        keep_utterances_inv.append(utterance)
        if keep_input :
            keep_utterances.append(utterance)
        
    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(utterances), len(keep_utterances), len(keep_utterances) / len(utterances)))
    return keep_utterances_inv





# Read query/response pairs and return a voc object
def readVocs(corpus, corpus_name):
    print("Reading lines...")

    # combine every two lines into utterances and normalize
    with open(corpus, encoding="utf8") as f:
        content = f.readlines()
    # import gzip
    # content = gzip.open(corpus, 'rt')
    lines = [x.strip() for x in content]
    listUtterances = [x.strip().split('__eou__')[:-1] for x in lines][:600]
    
    print("Counting words...")
    
    voc = Voc(corpus_name)
    for utterances in tqdm(listUtterances):
        for utterance in (utterances):
            voc.addSentence(utterance)    
    result=[]
    listUtterances=trimRareWords(voc, listUtterances, MIN_COUNT)
    for utterances in tqdm(listUtterances):
        utterances=["<pad>",'<pad>']+utterances#[ toNer(utter)[0] for utter in utterances ]
        
        if len(utterances)> nb_decoder:
            for i in range(len(utterances)-nb_decoder):
                liste=list()
                for j in  range(nb_decoder+1):
                    liste.append(utterances[i+j].strip())
                result.append(liste)

    
    return voc, result
# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(utterances):
    tt = TweetTokenizer(strip_handles=True, reduce_len=True)
    # Input sequences need to preserve the last word for EOS token
    for i in range(len(utterances)):
        if len(utterances[i].split()) > MAX_LENGTH:
            return False
    return True

# Filter pairs using filterPair condition
def filterPairs(utterances):
    return [utterance for utterance in utterances if filterPair(utterance)]

# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    voc, utterances = readVocs(datafile, corpus_name)
    print("Read {!s} sentence utterances".format(len(utterances)))
    


    utterances = filterPairs(utterances)
    print("Trimmed to {!s} sentence pairs".format(len(utterances)))
    print("Counted words:", voc.num_words)
    return voc, utterances


# Load/Assemble voc and pairs
save_dir = os.path.join("data", "save")
#datafile= 'Emoubuntu.txt'
#datafile='Emocornell.txtnew'
datafile='Emodailydialogs.txtnew'
#datafile='movie_lines.txtnew'##'movie_subtitles.txt'
voc, utterances = loadPrepareData(corpus, corpus_name, datafile, save_dir)
# Print some pairs to validate
for utterance in utterances[:10]:
    print(utterance)





# Trim voc and pairs
#utterances = trimRareWords(voc, utterances, MIN_COUNT)
print("\npairs:")
for utterance in utterances[:10]:
    print(utterance)

######################################################################
# Prepare Data for Models
# -----------------------
#
# Although we have put a great deal of effort into preparing and massaging our
# data into a nice vocabulary object and list of sentence , our models
# will ultimately expect numerical torch tensors as inputs. One way to
# prepare the processed data for the models can be found in the `Cascade
# In that tutorial, we use a batch size of 1, meaning that all we have to
# do is convert the words in our sentence group to their corresponding
# indexes from the vocabulary and feed this to the models.
#
# However, if you’re interested in speeding up training and/or would like
# to leverage GPU parallelization capabilities, you will need to train
# with mini-batches.
#
# Using mini-batches also means that we must be mindful of the variation
# of sentence length in our batches. To accomodate sentences of different
# sizes in the same batch, we will make our batched input tensor of shape
# *(max_length, batch_size)*, where sentences shorter than the
# *max_length* are zero padded after an *EOS_token*.
#
# If we simply convert our English sentences to tensors by converting
# words to their indexes(\ ``indexesFromSentence``) and zero-pad, our
# tensor would have shape *(batch_size, max_length)* and indexing the
# first dimension would return a full sequence across all time-steps.
# However, we need to be able to index our batch along time, and across
# all sequences in the batch. Therefore, we transpose our input batch
# shape to *(max_length, batch_size)*, so that indexing across the first
# dimension returns a time step across all sentences in the batch. We
# handle this transpose implicitly in the ``zeroPadding`` function.
#
#
# The ``inputVar`` function handles the process of converting sentences to
# tensor, ultimately creating a correctly shaped zero-padded tensor. It
# also returns a tensor of ``lengths`` for each of the sequences in the
# batch which will be passed to our decoders later.
#
# The ``outputVar`` function performs a similar function to ``inputVar``,
# but instead of returning a ``lengths`` tensor, it returns a binary mask
# tensor and a maximum target sentences length. The binary mask tensor has
# the same shape as the output target tensor, but every element that is a
# *PAD_token* is 0 and all others are 1.
#
# ``batch2TrainData`` simply takes a bunch of pairs and returns the input
# and target tensors using the aforementioned functions.
#

def indexesFromSentence(voc, sentence):
    tt = TweetTokenizer(strip_handles=True, reduce_len=True)
    return [voc.word2index[word] for word in tt.tokenize(sentence)] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
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
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs

def batch2TrainData(voc, utterance_batch, reverse=False):
    if reverse:
        utterance_batch = [utterance[::-1] for utterance in utterance_batch]
    tt = TweetTokenizer(strip_handles=True, reduce_len=True)
    utterance_batch.sort(key=lambda x: len(tt.tokenize(x[0])), reverse=True)
    input_batch, output_batch = [], [[] for _ in range(nb_decoder)]
    for utterance in utterance_batch:
        input_batch.append(utterance[0])
        for i in range(1,len(utterance)):
            output_batch[i-1].append(utterance[i])
            
    inp, lengths = inputVar(input_batch, voc)
    output_list, mask_list, max_target_len_list=[],[],[]
    for i in range(len(output_batch)):
        
        output, mask, max_target_len = outputVar(output_batch[i], voc)
        output_list.append(output)
        mask_list.append(mask)
        max_target_len_list.append(max_target_len)
    return inp, lengths, output_list, mask_list, max_target_len_list

# Example for validation
small_batch_size = 5
liste=[random.choice(utterances) for _ in range(small_batch_size)]
print('liste:',liste)
batches = batch2TrainData(voc, liste)
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)


######################################################################
# Define Models
# -------------
#
# Cascade Model
# ~~~~~~~~~~~~~
#
# The brains of our chatbot is (Cascade) model. The
# goal of a Cascade model is to take a variable-length sequence as an
# input, and return a variable-length sequence as an output using a
# fixed-sized model.
#
#
#


######################################################################
# Encoder
# ~~~~~~~
#
# The encoder RNN iterates through the input sentence one token
# (e.g. word) at a time, at each time step outputting an “output” vector
# and a “hidden state” vector. The hidden state vector is then passed to
# the next time step, while the output vector is recorded. The encoder
# transforms the context it saw at each point in the sequence into a set
# of points in a high-dimensional space, which the decoder will use to
# generate a meaningful output for the given task.
#
# Bidirectional RNN:
#
#
#
# Note that an ``embedding`` layer is used to encode our word indices in
# an arbitrarily sized feature space. For our models, this layer will map
# each word to a feature space of size *hidden_size*. When trained, these
# values should encode semantic similarity between similar meaning words.
#
# Finally, if passing a padded batch of sequences to an RNN module, we
# must pack and unpack padding around the RNN pass using
# ``torch.nn.utils.rnn.pack_padded_sequence`` and
# ``torch.nn.utils.rnn.pad_packed_sequence`` respectively.
#
# **Computation Graph:**
#
#    1) Convert word indexes to embeddings.
#    2) Pack padded batch of sequences for RNN module.
#    3) Forward pass through GRU.
#    4) Unpack padding.
#    5) Sum bidirectional GRU outputs.
#    6) Return output and final hidden state.
#
# **Inputs:**
#
# -  ``input_seq``: batch of input sentences; shape=\ *(max_length,
#    batch_size)*
# -  ``input_lengths``: list of sentence lengths corresponding to each
#    sentence in the batch; shape=\ *(batch_size)*
# -  ``hidden``: hidden state; shape=\ *(n_layers x num_directions,
#    batch_size, hidden_size)*
#
# **Outputs:**
#
# -  ``outputs``: output features from the last hidden layer of the GRU
#    (sum of bidirectional outputs); shape=\ *(max_length, batch_size,
#    hidden_size)*
# -  ``hidden``: updated hidden state from GRU; shape=\ *(n_layers x
#    num_directions, batch_size, hidden_size)*
#
#

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


######################################################################
# Decoder
# ~~~~~~~
#
# The decoder RNN generates the response sentence in a token-by-token
# fashion. It uses the encoder’s context vectors, and internal hidden
# states to generate the next word in the sequence. It continues
# generating words until it outputs an *EOS_token*, representing the end
# of the sentence. A common problem with a Cascade decoder is that
# if we rely soley on the context vector to encode the entire input
# sequence’s meaning, it is likely that we will have information loss.
# This is especially the case when dealing with long input sequences,
# greatly limiting the capability of our decoder.
#
#
# `Luong et al. <https://arxiv.org/abs/1508.04025>`__ improved upon
# Bahdanau et al.’s groundwork by creating “Global attention”. The key
# difference is that with “Global attention”, we consider all of the
# encoder’s hidden states, as opposed to Bahdanau et al.’s “Local
# attention”, which only considers the encoder’s hidden state from the
# current time step. Another difference is that with “Global attention”,
# we calculate attention weights, or energies, using the hidden state of
# the decoder from the current time step only. Bahdanau et al.’s attention
# calculation requires knowledge of the decoder’s state from the previous
# time step. Also, Luong et al. provides various methods to calculate the
# attention energies between the encoder output and decoder output which
# are called “score functions”:
#
#
# Overall, the Global attention mechanism can be summarized by the
# following figure. Note that we will implement the “Attention Layer” as a
# separate ``nn.Module`` called ``Attn``. The output of this module is a
# softmax normalized weights tensor of shape *(batch_size, 1,
# max_length)*.
#

# Luong attention layer
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


######################################################################
# Now that we have defined our attention submodule, we can implement the
# actual decoder model. For the decoder, we will manually feed our batch
# one time step at a time. This means that our embedded word tensor and
# GRU output will both have shape *(1, batch_size, hidden_size)*.
#
# **Computation Graph:**
#
#    1) Get embedding of current input word.
#    2) Forward through unidirectional GRU.
#    3) Calculate attention weights from the current GRU output from (2).
#    4) Multiply attention weights to encoder outputs to get new "weighted sum" context vector.
#    5) Concatenate weighted context vector and GRU output using Luong eq. 5.
#    6) Predict next word using Luong eq. 6 (without softmax).
#    7) Return output and final hidden state.
#
# **Inputs:**
#
# -  ``input_step``: one time step (one word) of input sequence batch;
#    shape=\ *(1, batch_size)*
# -  ``last_hidden``: final hidden layer of GRU; shape=\ *(n_layers x
#    num_directions, batch_size, hidden_size)*
# -  ``encoder_outputs``: encoder model’s output; shape=\ *(max_length,
#    batch_size, hidden_size)*
#
# **Outputs:**
#
# -  ``output``: softmax normalized tensor giving probabilities of each
#    word being the correct next word in the decoded sequence;
#    shape=\ *(batch_size, voc.num_words)*
# -  ``hidden``: final hidden state of GRU; shape=\ *(n_layers x
#    num_directions, batch_size, hidden_size)*
#

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded) #[1, 64, 512]
        if(embedded.size(0) != 1):
            raise ValueError('Decoder input sequence length should be 1')
        
        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)
        
        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average

        
        attn_weights = self.attn(rnn_output, encoder_outputs) #[64, 1, 14]
        
        # encoder_outputs [14, 64, 512]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) #[64, 1, 512]
        
        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) #[64, 512]
        context = context.squeeze(1) #[64, 512]
        
        concat_input = torch.cat((rnn_output, context), 1) #[64, 1024]
        concat_output = torch.tanh(self.concat(concat_input)) #[64, 512]
        
        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output) #[64, output_size]
        
        # Return final output, hidden state, and attention weights (for visualization)
        
        return output, hidden, attn_weights,rnn_output,context


######################################################################
# Define Training Procedure
# -------------------------
#
# Masked loss
# ~~~~~~~~~~~
#
# Since we are dealing with batches of padded sequences, we cannot simply
# consider all elements of the tensor when calculating loss. We define
# ``maskNLLLoss`` to calculate our loss based on our decoder’s output
# tensor, the target tensor, and a binary mask tensor describing the
# padding of the target tensor. This loss function calculates the average
# negative log likelihood of the elements that correspond to a *1* in the
# mask tensor.
#

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()

    #crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    NLLLoss(mask)
    loss = F.cross_entropy(inp, target, ignore_index=EOS_token)
    loss = loss.to(device)
    
    return loss, nTotal.item()

def maskNLLLoss1(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()
def custom_loss(logits, labels,mask):

    # create a mask by filtering out all tokens that ARE NOT the padding token
    tag_pad_token = 0
    mask = (labels > tag_pad_token).float()
 
    # count how many tokens we have
    nb_tokens = int(torch.sum(mask).item())
    
    # pick the values for the label and zero out the rest with the mask
    logits = logits[range(logits.shape[0]), labels] * mask.float()
    #ce_loss=F.cross_entropy(logits.shape[0], labels, ignore_index=EOS_token)
    # compute cross entropy loss which ignores all <PAD> tokens
    ce_loss = -torch.sum(logits) / nb_tokens

    return ce_loss,nb_tokens

######################################################################
# Single training iteration
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The ``train`` function contains the algorithm for a single training
# iteration (a single batch of inputs).
#
# We will use a couple of clever tricks to aid in convergence:
#
# -  The first trick is using **teacher forcing**. This means that at some
#    probability, set by ``teacher_forcing_ratio``, we use the current
#    target word as the decoder’s next input rather than using the
#    decoder’s current guess. This technique acts as training wheels for
#    the decoder, aiding in more efficient training. However, teacher
#    forcing can lead to model instability during inference, as the
#    decoder may not have a sufficient chance to truly craft its own
#    output sequences during training. Thus, we must be mindful of how we
#    are setting the ``teacher_forcing_ratio``, and not be fooled by fast
#    convergence.
#
#
#
#    1) Forward pass entire input batch through encoder.
#    2) Initialize decoder inputs as SOS_token, and hidden state as the encoder's final hidden state.
#    3) Forward input batch sequence through decoder one time step at a time.
#    4) If teacher forcing: set next decoder input as the current target; else: set next decoder input as current decoder output.
#    5) Calculate and accumulate loss.
#    6) Perform backpropagation.
#    7) Update encoder and decoder model parameters.
#
#
# .. Note ::
#
#   PyTorch’s RNN modules (``RNN``, ``LSTM``, ``GRU``) can be used like any
#   other non-recurrent layers by simply passing them the entire input
#   sequence (or batch of sequences). We use the ``GRU`` layer like this in
#   the ``encoder``. The reality is that under the hood, there is an
#   iterative process looping over each time step calculating hidden states.
#   Alternatively, you ran run these modules one time-step at a time. In
#   this case, we manually loop over the sequences during the training
#   process like we must do for the ``decoder`` model. As long as you
#   maintain the correct conceptual model of these modules, implementing
#   sequential models can be very straightforward.
#
#


def train(input_variable, lengths, target_variable_list, mask_list, max_target_len_list, encoder, decoder_list, embedding,
          encoder_optimizer, decoder_optimizer_list, batch_size, clip, max_length=MAX_LENGTH):

    # Zero gradients
    encoder_optimizer.zero_grad()
    for decoder_optimizer in decoder_optimizer_list:
        decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    for i in range(len(target_variable_list)):
        target_variable_list[i] = target_variable_list[i].to(device)
        mask_list[i] = mask_list[i].to(device)
    
    # Initialize variables
    loss_list = torch.tensor([0 for _ in decoder_list], dtype=torch.float).to(device)
    print_losses_list = [[0] for decoder in decoder_list]
    n_totals_list = [0 for _ in decoder_list]

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
    encoder_outputs_list=[None for decoder in decoder_list]
    context_list=[[] for _ in decoder_list]
    encoder_outputs_list[0]=encoder_outputs
    # Create initial decoder input (start with SOS tokens for each sentence)
    
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)
    decoder_input_list=[decoder_input for decoder in decoder_list]
    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden_list=[None for decoder in decoder_list]
    decoder_hidden_list[0] = encoder_hidden[:decoder_list[0].n_layers]
    
    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True# if random.random() < teacher_forcing_ratio else False
    
    # Forward batch of sequences through decoder one time step at a time
    for i in range(len(decoder_list)):
        rnn_output_list=list()
        
        if i > 0 :  decoder_hidden_list[i]= decoder_hidden_list[i-1]
        if use_teacher_forcing:
            
    
            for t in range(max_target_len_list[i]):
                if i < len(decoder_list)-1 :
                    # ingnore End-of-sentence for a first and second decoder
                    decoder_input_list[i] = target_variable_list[i][t].view(1, -1)
                    print('{}-{}'.format(i,decoder_input_list[i]))
                    decoder_input_list[i] = decoder_input_list[i].to(device)
                else:
                    print('{}-{}'.format(i,decoder_input_list[i]))
                decoder_output, decoder_hidden_list[i], attn_weights, rnn_output,context= decoder_list[i](
                    decoder_input_list[i], decoder_hidden_list[i], encoder_outputs_list[i]
                )
                
                # Teacher forcing: next input is current target
                if i==len(decoder_list)-1 :
                    # add End-of-sentence for a last decoder
                    decoder_input_list[i] = target_variable_list[i][t].view(1, -1)
                    
                    decoder_input_list[i] = decoder_input_list[i].to(device)
                # Calculate and accumulate loss
                mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable_list[i][t], mask_list[i][t])
                #mask_loss, nTotal = custom_loss(decoder_output, target_variable[t], mask_list[i][t])
                loss_list[i] += mask_loss
                print_losses_list[i].append(mask_loss.item() * nTotal)
                
                n_totals_list[i] += nTotal
                # Next input is current target
                rnn_output_list.append(rnn_output)
                context_list[i].append(context)
                #loss[0] += F.cross_entropy(decoder_output_list[0], target_variable_list[0][t], ignore_index=EOS_token)
            
            if i < len(decoder_list)-1:
                
                rnn_output_list.extend([elt for i in range(len(context_list)) for elt in context_list[len(context_list)-i-1]])
                encoder_outputs_list[i+1]=Variable(torch.stack(rnn_output_list, 0),requires_grad=True).to(device)
        else:
            for t in range(max_target_len_list[i]):
                decoder_output, decoder_hidden_list[i], attn_weights, rnn_output, context= decoder_list[i](
                    decoder_input_list[i], decoder_hidden_list[i], encoder_outputs_list[i]
                )
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input_list[i] = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])

                decoder_input_list[i] = decoder_input_list[i].to(device)
                # Calculate and accumulate loss
                mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable_list[i][t], mask_list[i][t])
                #mask_loss, nTotal = custom_loss(decoder_output, target_variable[t], mask_list[i][t])
                loss_list[i] += mask_loss
                print_losses_list[i].append(mask_loss.item() * nTotal)
                
                n_totals_list[i] += nTotal
                # Next input is current target
                rnn_output_list.append(rnn_output)
                context_list[i].append(context)
                #loss[0] += F.cross_entropy(decoder_output_list[0], target_variable_list[0][t], ignore_index=EOS_token)
            
            if i < len(decoder_list)-1:
                rnn_output_list.extend([elt for i in range(len(context_list)) for elt in context_list[len(context_list)-i-1]])
                encoder_outputs_list[i+1]=Variable(torch.stack(rnn_output_list, 0),requires_grad=True).to(device)
        # Perform backpropatation
        
        
        
        # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        
        # Adjust model weights
    encoder_optimizer.step()
        
    for i,decoder in enumerate(decoder_list):
        loss_list[i].backward(retain_graph=True)
        _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
            
        decoder_optimizer_list[i].step()
    
    return [sum(print_losses_list[i]) / n_totals_list[i] for i in range(len(decoder_list))]


######################################################################
# Training iterations
# ~~~~~~~~~~~~~~~~~~~
#
# It is finally time to tie the full training procedure together with the
# data. The ``trainIters`` function is responsible for running
# ``n_iterations`` of training given the passed models, optimizers, data,
# etc. This function is quite self explanatory, as we have done the heavy
# lifting with the ``train`` function.
#
# One thing to note is that when we save our model, we save a tarball
# containing the encoder and decoder state_dicts (parameters), the
# optimizers’ state_dicts, the loss, the iteration, etc. Saving the model
# in this way will give us the ultimate flexibility with the checkpoint.
# After loading a checkpoint, we will be able to use the model parameters
# to run inference, or we can continue training right where we left off.
#


def trainIters(model_name, voc, utterances, encoder, decoder_list, encoder_optimizer, decoder_optimizer_list, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename):

    # Load batches for each iteration
    utterances_training=utterances[:-200]
    writer = SummaryWriter('runs/logs'+'/'+model_name)
    searcher = GreedySearchDecoder(encoder, decoder_list)
    print(searcher)
    utterances_test=utterances[-200:-100]
    #training_batches = [batch2TrainData(voc, [utterances_training[(k+i*batch_size)%len(utterances_training)] for k in range(batch_size)])
    #                  for i in range(n_iteration)]
    utterances_eval =utterances[-100:] 
    eval_batches = [batch2TrainData(voc, [utterances_eval[i]])
                      for i in range(len(utterances_eval))]
    
    test_batches = [batch2TrainData(voc, [utterances_test[i]])
                      for i in range(len(utterances_test))]
    
    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss_list = [0 for _ in decoder_list]
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1
        
    w2v = KeyedVectors.load_word2vec_format('data/embeddings/GoogleNews-vectors-negative300.bin.gz', binary=True)
    # Training loop
    print("Training...")
    for iteration in tqdm(range(start_iteration, n_iteration + 1)):
        
        training_batch = batch2TrainData(voc, [utterances_training[(k+iteration*batch_size)%len(utterances_training)] for k in range(batch_size)])#training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable_list, mask_list, max_target_len_list = training_batch

        # Run a training iteration with batch
        loss_list = train(input_variable, lengths, target_variable_list, mask_list, max_target_len_list, encoder,
                     decoder_list, embedding, encoder_optimizer, decoder_optimizer_list, batch_size, clip)
       
        print_loss_list = [print_loss_list[i] +loss_list[i]  for i  in range(len(print_loss_list))]

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg =  [print_loss / print_every for print_loss  in print_loss_list]
            writer.add_scalars('run_14h',{'print_loss_avg@'+str(j):print_loss_avg[j] for j in range(len(print_loss_avg))}, iteration)
            
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:s}".format(iteration, iteration / n_iteration * 100, str(['{:.9f}'.format(p) for p in print_loss_avg])))
            print_loss_list = [0 for _ in decoder_list]
        if iteration % (print_every * 10) == 0:
            perplexity , score_bleu=test_perplexity(test_batches,searcher,w2v)
            print('perplexity , score_bleu ',perplexity , score_bleu)
            writer.add_scalar('score_bleu', score_bleu, iteration)
            writer.add_scalar('perplexity', perplexity, iteration)
            eval_loss=calc_valid_loss(eval_batches,searcher)
            print('eval_loss',eval_loss.item())
        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            param={
                'iteration': iteration,
                'en': encoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'loss': loss_list,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }
            for i in range(len(decoder_list)):
                param['de'+str(i)]=decoder_list[i].state_dict()
                param['de_opt'+str(i)]=decoder_optimizer_list[i].state_dict()
            torch.save(param, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

######################################################################
# Define Evaluation
# -----------------
#
# After training a model, we want to be able to talk to the bot ourselves.
# First, we must define how we want the model to decode the encoded input.
#
# Greedy decoding
# ~~~~~~~~~~~~~~~
#
# Greedy decoding is the decoding method that we use during training when
# we are **NOT** using teacher forcing. In other words, for each time
# step, we simply choose the word from ``decoder_output`` with the highest
# softmax value. This decoding method is optimal on a single time-step
# level.
#
# To facilite the greedy decoding operation, we define a
# ``GreedySearchDecoder`` class. When run, an object of this class takes
# an input sequence (``input_seq``) of shape *(input_seq length, 1)*, a
# scalar input length (``input_length``) tensor, and a ``max_length`` to
# bound the response sentence length. The input sentence is evaluated
# using the following computational graph:
#
# **Computation Graph:**
#
#    1) Forward input through encoder model.
#    2) Prepare encoder's final hidden layer to be first hidden input to the decoder's.
#    3) Initialize decoder's first input as SOS_token.
#    4) Initialize tensors to append decoded words to.
#    5) Iteratively decode's one word token at a time:
#        a) Forward pass through decoders.
#        b) Obtain most likely word token and its softmax score.
#        c) Record token and score.
#        d) Prepare current token to be next decoder input.
#    6) Return collections of word tokens and scores.
#
def test_perplexity(test_batches,searcher,w2v):

        
        losses=[]
        all_items=0
        score_bleu=0
        all_words_referances, all_words_candidates=[],[]
        for iterTest in range(1, len(test_batches)+1):
            all_losses=0
            test_batche = test_batches[iterTest - 1]
            # Extract fields from batch
            
            input_variable_test, lengths_test, target_variable_test, mask_list_test, max_target_len_list_test = test_batche  
            #print(input_variable_test, lengths_test, max_target_len_list_test,target_variable_test)
            #lengths_test =[lengths_test]+[ torch.tensor([len(indexes) for indexes in indexes_batch]) for indexes_batch in target_variable_test ]
            input_variable_test=input_variable_test.to(device)
                
            target_variable_test =[ target_variable_test[i].to(device) for i in range(len(target_variable_test))]
            max_target_len_list_test=[torch.tensor([val]) for val in max_target_len_list_test]
            tokens, scores, decoder_output_list , attention_list= searcher(input_variable_test, lengths_test, max_target_len_list_test[:],target_variable_test)
            
            nTotal = mask_list_test[nb_decoder-1].sum()
            
            for i in range(max_target_len_list_test[nb_decoder-1][0]):
                #loss = -torch.log(torch.gather(decoder_output_list[i], 1, target_variable_test[nb_decoder-1][i].view(-1, 1)).squeeze(1))
                mask_loss, nTotal = maskNLLLoss(decoder_output_list[i], target_variable_test[nb_decoder-1][i], mask_list_test[nb_decoder-1][i])
                all_losses += mask_loss.item()
                all_items+=nTotal
            losses.append(all_losses)
            decoded_words_referances = [voc.index2word[token.item()] for token in target_variable_test[nb_decoder-1]]
            decoded_words_candidates = [voc.index2word[token.item()] for token in tokens]
            print('in ',[' '.join(decoded_words_referances[:-1])])
            print('out',[' '.join(decoded_words_candidates)])
            
            score_bleu += corpus_bleu([[decoded_words_referances[:-1]]], [decoded_words_candidates])
            context_list_list=[np.transpose(np.array([context.detach().squeeze(0).cpu().numpy() for context in attention]))for attention in attention_list]
            
            
            inp=' '.join([voc.index2word[token.item()] for token in input_variable_test[:,0]])
            #print('input_variable_test',inp)
            sentences=[inp]+[' '.join([voc.index2word[token.item()] for token in s[:,0]]) for s in target_variable_test[:2]]
            #print('target_variable_test',sentences)
            showAttention(sentences+[ ' '.join(decoded_words_referances)], context_list_list,[lengths_test]+max_target_len_list_test[:2]+[torch.tensor([MAX_LENGTH])],rank=iterTest)
            all_words_referances.append(' '.join(decoded_words_referances))
            all_words_candidates.append(' '.join(decoded_words_candidates))
        # evaluation Metrics Word for all test data
        evaluationMetricsWord(all_words_referances, all_words_candidates,w2v)    
        
        perplexity=np.exp(np.sum(losses)/all_items)#np.mean(losses))

        score_bleu=score_bleu/len(test_batches)
        return perplexity,score_bleu
    
def calc_valid_loss(eval_batches,searcher):#(data_loader, criteria, model):
    
        losses=[]
        all_items=0
        searcher.eval()
        for iterTest in range(1, len(eval_batches)+1):
            all_losses=0
            test_batche = eval_batches[iterTest - 1]
            # Extract fields from batch
            
            input_variable_test, lengths_test, target_variable_test, mask_list_test, max_target_len_list_test = test_batche  
            #print(input_variable_test, lengths_test, max_target_len_list_test,target_variable_test)
            #lengths_test =[lengths_test]+[ torch.tensor([len(indexes) for indexes in indexes_batch]) for indexes_batch in target_variable_test ]
            input_variable_test=input_variable_test.to(device)
                
            target_variable_test =[ target_variable_test[i].to(device) for i in range(len(target_variable_test))]
            max_target_len_list_test=[torch.tensor([val]) for val in max_target_len_list_test]
            tokens, scores, decoder_output_list , attention_list= searcher(input_variable_test, lengths_test, max_target_len_list_test[:],target_variable_test)
            
            nTotal = mask_list_test[nb_decoder-1].sum()
            
            for i in range(max_target_len_list_test[nb_decoder-1][0]):
                mask_loss, nTotal = maskNLLLoss(decoder_output_list[i], target_variable_test[nb_decoder-1][i], mask_list_test[nb_decoder-1][i])
                #loss = -torch.log(torch.gather(decoder_output_list[i], 1, target_variable_test[nb_decoder-1][i].view(-1, 1)).squeeze(1))
                all_losses += mask_loss.item()
                all_items+=nTotal
            losses.append(all_losses)
            
            
        valid_loss=np.exp(np.sum(losses)/all_items)#np.exp(np.mean(losses))
        searcher.train()
        return valid_loss 

    
class Sentence:
    def __init__(self, decoder_hidden, last_idx=SOS_token, sentence_idxes=[], sentence_scores=[]):
        if(len(sentence_idxes) != len(sentence_scores)):
            raise ValueError("length of indexes and scores should be the same")
        self.decoder_hidden = decoder_hidden
        self.last_idx = last_idx
        self.sentence_idxes =  sentence_idxes
        self.sentence_scores = sentence_scores

    def avgScore(self):
        if len(self.sentence_scores) == 0:
            raise ValueError("Calculate average score of sentence, but got no word")
        # return mean of sentence_score
        return sum(self.sentence_scores) / len(self.sentence_scores)

    def addTopk(self, topi, topv, decoder_hidden, beam_size, voc):
        topv = torch.log(topv)
        terminates, sentences = [], []
        for i in range(beam_size):
            if topi[0][i] == EOS_token:
                terminates.append(([voc.index2word[idx.item()] for idx in self.sentence_idxes] + ['<eos>'],
                                   self.avgScore())) # tuple(word_list, score_float
                continue
            idxes = self.sentence_idxes[:] # pass by value
            scores = self.sentence_scores[:] # pass by value
            idxes.append(topi[0][i])
            scores.append(topv[0][i])
            sentences.append(Sentence(decoder_hidden, topi[0][i], idxes, scores))
        return terminates, sentences

    def toWordScore(self, voc):
        words = []
        for i in range(len(self.sentence_idxes)):
            if self.sentence_idxes[i] == EOS_token:
                words.append('<eos>')
            else:
                words.append(voc.index2word[self.sentence_idxes[i].item()])
        if self.sentence_idxes[-1] != EOS_token:
            words.append('<eos>')
        return (words, self.avgScore())
    
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder_list):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder_list = decoder_list
    def __str__(self):
        return 'encoder (%s'% encoder+'-->decoder_list -->'+'\n %s'%([decoder_list[i] for i in range(len(decoder_list))])
        
    def forward(self, input_seq, input_length, max_length,indexes_batchs):
        # Forward input through encoder model
        
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        #print('input_seq',input_seq)
        #print('encoder_outputs',encoder_outputs.shape)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder_list[0].n_layers]
        decoder_hidden_list=[None for decoder in decoder_list]
        decoder_hidden_list[0]=decoder_hidden
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones( 1, device=device, dtype=torch.long) * SOS_token
        decoder_input_list=[decoder_input for i in range(nb_decoder)]
        # Initialize tensors to append decoded words to
        all_tokens = [torch.zeros([0], device=device, dtype=torch.long) for _ in range(nb_decoder)]
        all_scores = [torch.zeros([0], device=device) for _ in range(nb_decoder)]
        encoder_outputs_list=[ None for i in range(nb_decoder+1)]
        encoder_outputs_list[0]=encoder_outputs
        #print('encoder_outputs',encoder_outputs.shape)
        context_list_list=list()
        attention_list=list()
        last_decoder_output_list=list()
        # Iteratively decode one word token at a time
       
        for i in range(nb_decoder):
            rnn_output_list=list()
            context_list=list()
            attn_weights_list=list()
            if i>0 : decoder_hidden_list[i]=decoder_hidden_list[i-1]
            for t in range(max_length[i][0]):
                
                
                
                # Prepare current token to be next decoder input (add a dimension)
                decoder_input_list[i] = torch.unsqueeze(decoder_input_list[i], 1)
                # Forward pass through decoder
                decoder_output, decoder_hidden_list[i], attn_weights, rnn_output ,context = self.decoder_list[i](decoder_input_list[i], decoder_hidden_list[i], encoder_outputs_list[i])
                attn_weights_list.append(attn_weights.squeeze(0))
                # Obtain most likely word token and its softmax score
                decoder_scores, decoder_input_list[i] = torch.max(decoder_output, dim=1)#decoder_output.topk(1)
                # Record token and score
                all_tokens[i] = torch.cat((all_tokens[i], decoder_input_list[i]), dim=0)
                all_scores[i] = torch.cat((all_scores[i], decoder_scores), dim=0)
                rnn_output_list.append(rnn_output)
                context_list.append(context)
                if i < nb_decoder-1:
                    if t >= len(indexes_batchs[i][0]):
                        val=2  
                    else:
                        val=indexes_batchs[i][0][t]
                    decoder_input_list[i]=torch.LongTensor([val])
                else:
                    last_decoder_output_list.append(decoder_output)
                    
                decoder_input_list[i]=decoder_input_list[i].to(device)
                
                        #
                
                #print('decoder_input_list[i]',decoder_input_list[i])
            context_list_list=context_list+context_list_list     
            attention_list.append(attn_weights_list)
            if i < nb_decoder-1:
                #print('zcontext',torch.stack(rnn_output_list+context_list, 0).shape)
                encoder_outputs_list[i+1]=torch.stack(rnn_output_list+context_list_list, 0)
                #print(context_list)
                    

                
            #print('all_tokens',[voc.index2word[token.item()] for token in all_tokens])

        # Return collections of word tokens and scores
        return all_tokens[nb_decoder-1], all_scores[nb_decoder-1],last_decoder_output_list,attention_list
    def beam_decode(self, input_seq, input_length, max_length,indexes_batchs,  voc, beam_size):
                # Forward input through encoder model
        
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        #print('input_seq',input_seq)
        #print('encoder_outputs',encoder_outputs.shape)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder_list[0].n_layers]
        decoder_hidden_list=[None for decoder in decoder_list]
        decoder_hidden_list[0]=decoder_hidden
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones( 1, device=device, dtype=torch.long) * SOS_token
        decoder_input_list=[decoder_input for i in range(nb_decoder)]
        # Initialize tensors to append decoded words to
        all_tokens = [torch.zeros([0], device=device, dtype=torch.long) for _ in range(nb_decoder)]
        all_scores = [torch.zeros([0], device=device) for _ in range(nb_decoder)]
        encoder_outputs_list=[ None for i in range(nb_decoder+1)]
        encoder_outputs_list[0]=encoder_outputs
        #print('encoder_outputs',encoder_outputs.shape)
        context_list_list=list()
        attention_list=list()
        last_decoder_output_list=list()
        # Iteratively decode one word token at a time
        
        terminal_sentences, prev_top_sentences, next_top_sentences = [], [], []
        prev_top_sentences.append(Sentence(decoder_hidden))
        
        for i in range(nb_decoder):
            rnn_output_list=list()
            context_list=list()
            attn_weights_list=list()
            if i>0 : decoder_hidden_list[i]=decoder_hidden_list[i-1]
            for t in range(max_length[i][0]):
                # Prepare current token to be next decoder input (add a dimension)
                decoder_input_list[i] = torch.unsqueeze(decoder_input_list[i], 1)
                # Forward pass through decoder
                if i < nb_decoder-1:
                    decoder_output, decoder_hidden_list[i], attn_weights, rnn_output ,context = self.decoder_list[i](decoder_input_list[i], decoder_hidden_list[i], encoder_outputs_list[i])
                    
                    # Obtain most likely word token and its softmax score
                    decoder_scores, decoder_input_list[i] = decoder_output.topk(1) #torch.max(decoder_output, dim=1)#
                    if t >= len(indexes_batchs[i][0]):
                        val=2
                    else:
                        val=indexes_batchs[i][0][t]
                    decoder_input_list[i]=torch.LongTensor([val])
                    decoder_input_list[i]=decoder_input_list[i].to(device)
                    last_decoder_output_list.append(decoder_output)
                else:
                    
                    for sentence in prev_top_sentences:
                        decoder_input_list[i] = torch.LongTensor([[sentence.last_idx]])
                        decoder_input_list[i] = decoder_input_list[i].to(device)
            
                        decoder_hidden_list[i] = sentence.decoder_hidden
                        
                        decoder_output, decoder_hidden_list[i], attn_weights, rnn_output ,context = self.decoder_list[i](decoder_input_list[i], decoder_hidden_list[i], encoder_outputs_list[i])
                        decoder_scores, decoder_input_list[i] = decoder_output.topk(beam_size)
                        term, top = sentence.addTopk(decoder_input_list[i], decoder_scores, decoder_hidden_list[i], beam_size, voc)
                        terminal_sentences.extend(term)
                        next_top_sentences.extend(top)
    
                    next_top_sentences.sort(key=lambda s: s.avgScore(), reverse=True)
                    prev_top_sentences = next_top_sentences[:beam_size]
                    #print('prev_top_sentences : {}'.format([ ' '.join( s.toWordScore( voc)[0]) for s in prev_top_sentences]))
                    next_top_sentences = []
                # Record token and score
                #all_tokens[i] = torch.cat((all_tokens[i], decoder_input_list[i]), dim=0)
                #all_scores[i] = torch.cat((all_scores[i], decoder_scores), dim=0)
                
                    
                attn_weights_list.append(attn_weights.squeeze(0))
                rnn_output_list.append(rnn_output)
                context_list.append(context)    
                decoder_input_list[i]=decoder_input_list[i].to(device)
 
            context_list_list=context_list+context_list_list     
            attention_list.append(attn_weights_list)
            if i < nb_decoder-1:
                #print('zcontext',torch.stack(rnn_output_list+context_list, 0).shape)
                encoder_outputs_list[i+1]=torch.stack(rnn_output_list+context_list_list, 0)
                
    
        terminal_sentences += [sentence.toWordScore(voc) for sentence in prev_top_sentences]
        terminal_sentences.sort(key=lambda x: x[1], reverse=True)
    
        #n = min(len(terminal_sentences), 15)
        return prev_top_sentences[0].sentence_idxes,all_scores[nb_decoder-1],last_decoder_output_list,attention_list


import matplotlib.pyplot as plt
#plt.switch_backend('Qt5Agg')
import matplotlib.ticker as ticker
import numpy as np


def  showAttention(input_sentences,  attentions,lengths,rank=1):
    h, w = 20, 20        # for raster image
    tt = TweetTokenizer(strip_handles=True, reduce_len=True)
    nrows, ncols = 1, len(attentions)  # array of sub-plots
    figsize = [15, 15]     # figure size, inches
    
    # prep (x,y) for extra plotting on selected sub-plots

    #input_sentence='1 2 3 4'
    #output_words='1 2 3'
    
    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    # plot simple raster image on each sub-plot
    input_sentences=[ s.replace('<pad>','<pad>') for s in input_sentences]
    arr= ax.flat if nb_decoder!=1 else [ax]
    for i, axi in enumerate(arr):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]

        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        # write row/col indices as axes' title for identification
        # axi.set_title("Row:"+str(rowid)+", Col:"+str(colid))
        max_len=lengths[i+1][0]
        encodes=tt.tokenize(input_sentences[i+1].strip())[:max_len] +[ '<pad>' for _ in range(max_len-len(tt.tokenize(input_sentences[i+1].strip())))]
        axi.set_xticklabels(['']+ encodes , rotation=90,fontsize=15)
        
        decodes=[tt.tokenize(input_sentences[j].strip())[:lengths[j][0]-1] +['<eos>'] for j in range(i,-1,-1)]

        decodes= list(itertools.chain(*decodes))
        axi.set_yticklabels( ['']+  decodes,fontsize=15)
        
        axi.xaxis.set_major_locator(ticker.MultipleLocator(1))
        axi.yaxis.set_major_locator(ticker.MultipleLocator(1))
        img = attentions[i][:len(decodes)]
        axi.imshow(img, alpha=0.9)
    # one can access the axes by ax[row_id][col_id]
    # do additional plotting on ax[row_id][col_id] of your choice
        #axi.plot(xs, 3*ys, color='red', linewidth=3)
        #axi.plot(ys**2, xs, color='green', linewidth=3)
    
    plt.tight_layout(True)
    #plt.show()
    plt.savefig('data/image/context_list{}.png'.format(rank))
    plt.close('all')
    
######################################################################
# Evaluate my text
# ~~~~~~~~~~~~~~~~
#
# Now that we have our decoding method defined, we can write functions for
# evaluating a string input sentence. The ``evaluate`` function manages
# the low-level process of handling the input sentence. We first format
# the sentence as an input batch of word indexes with *batch_size==1*. We
# do this by converting the words of the sentence to their corresponding
# indexes, and transposing the dimensions to prepare the tensor for our
# models. We also create a ``lengths`` tensor which contains the length of
# our input sentence. In this case, ``lengths`` is scalar because we are
# only evaluating one sentence at a time (batch_size==1). Next, we obtain
# the decoded response sentence tensor using our ``GreedySearchDecoder``
# object (``searcher``). Finally, we convert the response’s indexes to
# words and return the list of decoded words.
#
# ``evaluateInput`` acts as the user interface for our chatbot. When
# called, an input text field will spawn in which we can enter our query
# sentence. After typing our input sentence and pressing *Enter*, our text
# is normalized in the same way as our training data, and is ultimately
# fed to the ``evaluate`` function to obtain a decoded output sentence. We
# loop this process, so we can keep chatting with our bot until we enter
# either “q” or “quit”.
#
# Finally, if a sentence is entered that contains a word that is not in
# the vocabulary, we handle this gracefully by printing an error message
# and prompting the user to enter another sentence.
#    


def evaluate(encoder, decoder_list, searcher, voc, sentences, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    beam=1
    indexes_batchs = [[indexesFromSentence(voc, sentences[i])] for i in range(nb_decoder)]
    print('sentences--',sentences)
    
    # Create lengths tensor
    
    lengths = [ torch.tensor([len(indexes) for indexes in indexes_batch]) for indexes_batch in indexes_batchs ]+[torch.tensor([max_length])]
    
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batchs[0]).transpose(0, 1)
    
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = [length.to(device) for length in lengths]
    # Decode sentence with searcher
    try:
        os.makedirs('data/image/'+corpus_name)
    except OSError:
        pass
    if beam==1:
        tokens, scores, _,attention_list = searcher(input_batch, lengths[0], lengths[1:],indexes_batchs[1:])
    else:
        tokens, scores, _,attention_list = searcher.beam_decode(input_batch, lengths[0], lengths[1:],indexes_batchs[1:],voc,beam)
    #print([np.array([context.detach().squeeze(0).cpu().numpy() for context in attention]).shape for attention in attention_list])
    # indexes -> words
    context_list_list=[np.transpose(np.array([context.detach().squeeze(0).cpu().numpy() for context in attention]))for attention in attention_list]
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    showAttention(sentences+[ ' '.join(decoded_words)], context_list_list,lengths)
    plt.savefig('data/image/{}/context_list{}.png'.format(corpus_name,i))
    plt.close('all')
    return decoded_words


def evaluateInput(encoder, decoder_list, searcher, voc):
    input_sentence = ''
    input_sentences_temp=['<pad>','<pad>','<pad>']
    
    while(1):
        #try:
            
            # Get input sentence
            input_sentence  = input('> ')
            input_sentences_temp=input_sentences_temp[:2]+[input_sentence]
            #input_sentences[1]=input_sentences[2]
            input_sentences=[toNer(s)[0] for s in input_sentences_temp]

            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentences = trimRareWords(voc, [input_sentences], 0)
            input_sentences=input_sentences[0]
            print('input_sentences : ',input_sentences)
            input_sentences =[ normalizeString(input_sentence) for input_sentence in input_sentences]
            #print(input_sentence)
            # Evaluate sentence
            print('input_sentences',input_sentences)
            output_words = evaluate(encoder, decoder_list, searcher, voc, input_sentences)
            
            # Format and print response sentence
            output_words1=[]
            for i in range(len(output_words)):
                if  (output_words[i] != '<eos>' and output_words[i] != '<pad>'):
                    output_words1.append(output_words[i])
                else:
                    break
            print('Bot:', ' '.join(output_words1))
            input_sentences_temp=input_sentences_temp[2:]+[' '.join(output_words1)]
            # except KeyError:
            #print("Error: Encountered unknown word.")






def load_embeddings( voc, emb_size, w2v):
        print('load embeddings glove.840B.300d ...')
        
        vocab=voc.word2index
        embedding_matrix = np.zeros((voc.num_words, emb_size))
        
        for word in  tqdm(vocab):

            if  word in w2v.vocab:
                i=vocab[word]
                embedding_matrix[i] = w2v[word]
            else:
                i=vocab[word]
                embedding_matrix[i] = np.random.normal(0,0.1,300)
        print(embedding_matrix.shape)
        return embedding_matrix
######################################################################
# Run Model
# ---------
#
# Finally, it is time to run our model!
#
# Regardless of whether we want to train or test the chatbot model, we
# must initialize the individual encoder and decoder models. In the
# following block, we set our desired configurations, choose to start from
# scratch or set a checkpoint to load from, and build and initialize the
# models. Feel free to play with different model configurations to
# optimize performance.
#

# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 6
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.2
batch_size = 5

# Set checkpoint to load from; set to None if starting from scratch
#loadFilename = './data/save/cb_model/dailydialogs/2-2_300/240000_checkpoint.tar'

loadFilename = None
checkpoint_iter = 5000
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename, map_location={'cuda:0': 'cpu'})
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd_list=[ None for i in range(nb_decoder)]
    decoder_optimizer_sd_list=[ None for i in range(nb_decoder)]
    for i in range(nb_decoder):
        #checkpoint['de'+str(i)]=checkpoint['de'+str(i)]
        #checkpoint['de_opt'+str(i)]=checkpoint['de_opt'+str(i)]
        decoder_sd_list[i]= checkpoint['de'+str(i)]
        decoder_optimizer_sd_list[i]= checkpoint['de_opt'+str(i)]
    encoder_optimizer_sd = checkpoint['en_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
if use_embed:
    w2v = KeyedVectors.load_word2vec_format('data/embeddings/GoogleNews-vectors-negative300.bin.gz', binary=True)
    pretrained_weight = load_embeddings(voc, hidden_size,w2v)
    embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
    
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder_list=[ None for i in range(nb_decoder)]
for i in range(nb_decoder):
    decoder_list[i]= LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    
    for i in range(nb_decoder):
        decoder_list[i].load_state_dict(decoder_sd_list[i])
# Use appropriate device
encoder = encoder.to(device)
for i in range(nb_decoder):
    decoder_list[i] = decoder_list[i].to(device)
print('Models built and ready to go!')


######################################################################
# Run Training
# ~~~~~~~~~~~~
#
# Run the following block if you want to train the model.
#
# First we set training parameters, then we initialize our optimizers, and
# finally we call the ``trainIters`` function to run our training
# iterations.
#

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.00
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 25
print_every = 100
save_every = 50000



# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer_list=[None for i in range(nb_decoder)]
for i in range(nb_decoder):
    decoder_optimizer_list[i] = optim.Adam(decoder_list[i].parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    for i in range(nb_decoder):
        decoder_optimizer_list[i].load_state_dict(decoder_optimizer_sd_list[i])



# Run training iterations
print("Starting Training!")
modeTrain=True
if modeTrain:
    # Ensure dropout layers are in train mode
    encoder.train()
    for i in range(nb_decoder):
        decoder_list[i].train()
    trainIters(model_name, voc, utterances, encoder, decoder_list, encoder_optimizer, decoder_optimizer_list,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename)


######################################################################
# Run Evaluation
# ~~~~~~~~~~~~~~
#
# To chat with your model, run the following block.
#

# Set dropout layers to eval mode
encoder.eval()
for decoder in decoder_list:
    decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder_list)

# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, decoder_list, searcher, voc)