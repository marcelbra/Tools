
import sys
from collections import Counter
import pickle
import numpy

from Node import Node

pad_string = '<pad>'
unk_string = '<unk>'

NoConstLabelID = 0

   
def _read_parses(path, max_sent_len):
   ''' read all parses from a file and return the set of constituents for each parse '''
   print("reading parses from", path, "...", file=sys.stderr)
   parses = []
   with open(path, "r") as file:
      for line in file:
         words, constituents = Node(line).words_and_constituents()
         if max_sent_len <= 0 or len(words) <= max_sent_len:
            parses.append((words, constituents))
   print("done", file=sys.stderr)
   return parses
            

def _build_dict(counter, min_freq=0, add_pad_symbol=False):
   ''' 
   Create a dictionary which maps strings with some minimal frequency to numbers.
   We don't use pack_padded sequence, so it is OK to assign ID 1 to the
   padding symbol.
   '''

   symlist = [unk_string] + ([pad_string] if add_pad_symbol else []) + \
             [sym for sym, freq in counter.most_common() if freq >= min_freq]
   string2ID = {sym:i for i,sym in enumerate(symlist)}

   return string2ID, symlist


class Data(object):
   
   def __init__(self, *args):
      # During testing, the constructor is called with just one argument,
      # which is the filename from which to read the parameters
      if len(args) == 1:
         self._init_test(*args)
      # During training, the constructor is called with 2 obligatory
      # and a couple of optionale arguments (see _init_train)
      else:
         self._init_train(*args)

   ### functions needed during training ###############################################

   def _init_train(self, path_train, path_dev, word_trunc_len=10, 
                   min_char_freq=2, max_sent_len=50):
      '''
      path_train: path to the file with the training data
      path_train: path to the file with the dev data
      word_trunc_len: length of the word prefixes/suffix
      min_char_freq: minimal frequency of characters which are not mapped to unknown
      max_sent_len: training sentences longer than this are ignored to speed up training
      '''
      self.word_trunc_len = word_trunc_len
      self.train_parses = _read_parses(path_train, max_sent_len)
      self.dev_parses   = _read_parses(path_dev, max_sent_len)
      
      print("building mapping tables ...", file=sys.stderr)
      letter_count = Counter()
      label_count = Counter()
      for words, constituents in self.train_parses:
         letter_count.update(''.join(words))
         labels, *_ = zip(*constituents)
         label_count.update(labels)
      
      self.char2ID, _ = _build_dict(letter_count, min_char_freq, add_pad_symbol=True)
      self.label2ID, self.ID2label = _build_dict(label_count)
      
      print("done", file=sys.stderr, flush=True)


   def num_char_types(self):
      ''' returns the number of characters (incl. unknown and padding symbol) '''
      return len(self.char2ID)


   def num_label_types(self):
      ''' returns the number of grammar labels incl. the "no constituent" label
          which is mapped to the index 0 '''
      return len(self.ID2label)
   
      
   def _get_charIDs(self, word):
      ''' maps a word to a sequence of character IDs '''

      padID = self.char2ID[pad_string]
      unkID = self.char2ID[unk_string]

      charIDs = [ self.char2ID.get(c, unkID) for c in word ]

      # add enough padding symbols
      fwd_charIDs = [padID] * self.word_trunc_len + charIDs
      bwd_charIDs = [padID] * self.word_trunc_len + charIDs[::-1]

      # truncate
      fwd_charIDs = fwd_charIDs[-self.word_trunc_len:]
      bwd_charIDs = bwd_charIDs[-self.word_trunc_len:]

      return fwd_charIDs, bwd_charIDs


   def words2charIDvec(self, words):
      """ converts a sequence of words to a suffix letter matrix
          and a prefix letter matrix """
      fwd_charID_seqs = []
      bwd_charID_seqs = []
      for word in words:
         fwd_charIDs, bwd_charIDs = self._get_charIDs(word)
         fwd_charID_seqs.append(fwd_charIDs)
         bwd_charID_seqs.append(bwd_charIDs)

      fwd_charID_seqs = numpy.asarray(fwd_charID_seqs, dtype='int32')
      bwd_charID_seqs = numpy.asarray(bwd_charID_seqs, dtype='int32')

      return fwd_charID_seqs, bwd_charID_seqs


   def labelID(self, label):
      """ returns the index of a given label """
      return self.label2ID.get(label, self.label2ID[unk_string])


   def store_parameters(self, filename):
      """ stores the parameters required for data preprocessing to a file """
      
      all_params = (self.word_trunc_len, self.char2ID, self.ID2label)
      with open(filename, "wb") as file:
         pickle.dump(all_params, file)

         
   ### functions needed during parsing ###############################################

   def _init_test(self, filename):
      """ load parameters from a file """
      
      with open(filename, "rb") as file:
         self.word_trunc_len, self.char2ID, self.ID2label = pickle.load(file)
      self.label2ID = {l:i for i,l in enumerate(self.ID2label)}

      
   def sentences(self, filename):
      """ reads the data to be parsed. Each line contains one tokenized sentence. """
      with open(filename, "r") as f:
         for line in f:
            words = line.split()
            yield words

