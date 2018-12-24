import os
import collections
import pickle
import re
import logging


class Model (object):

    def __init__ (self):
        self.words = list ()
        self.word_tuples = list ()
        self.words_model = dict ()
        self.word_tuples_model = dict ()

    def train_models (self, corpus, model_name = 'models_compressed.pkl'):
        """Takes in a preferably long string (corpus/training data),
        split that string into a list, we \"chunkify\" resulting in
        a list of 2-elem list. Finally we create a dictionary,
        where each key = first elem and each value = Counter([second elems])
        """
        self.words = re.findall ('[a-z]+', corpus.lower ())
        self.words_model = collections.Counter (self.words)

        def chunks (l, n):
            for i in range (0, len (l) - n + 1):
                yield l[i:i+n]

        self.word_tuples = list (chunks(self.words, 2))

        self.word_tuples_model = {first:collections.Counter () for first, second in self.word_tuples}

        for tup in self.word_tuples:
            try:
                self.word_tuples_model[tup[0]].update ([tup[1]])
            except:
                # hack-y fix for uneven # of elements in WORD_TUPLES
                pass

        if model_name:
            self.save_models (model_name)


    def train_bigtxt (self):
        """unnecessary helper function for training against
        default corpus data (big.txt)"""

        bigtxtpath = os.path.join (os.path.dirname (__file__), 'big.txt')
        with open (bigtxtpath, 'rt') as bigtxtfile:
            self.train_models (str (bigtxtfile.read ()))


    def save_models (self, path = None):
        """Save models to 'path'. If 'path' not specified,
        save to module's folder under name 'models_compressed.pkl'"""
        if path == None:
            path = os.path.join(os.path.dirname(__file__), 'models_compressed.pkl')

        pickle.dump({'words_model': self.words_model,
                     'word_tuples_model': self.word_tuples_model},
                    open (path, 'wb'),
                    protocol = 2)


    def load_models (self, load_path = None):
        """Load autocomplete's built-in model (uses Norvig's big.txt). Optionally
        provide the path to Python pickle object."""
        if load_path is None:
            load_path = os.path.join(os.path.dirname(__file__), 'models_compressed.pkl')
        try:
            models = pickle.load (open(load_path,'rb'))
            self.words_model = models['words_model']
            self.word_tuples_model = models['word_tuples_model']
        except (IOError, KeyError, ValueError) as e:
            logging.info ('train language model')
            self.train_bigtxt ()
