import copy
import logging
from collections import Counter

import models


NEARBY_KEYS = {
    'a': 'bg',
    'b': 'aghic',
    'c': 'bhidj',
    'd': 'cijke',
    'e': 'djklf',
    'f': 'ekl',
    'g': 'abhnm',
    'h': 'abcionmg',
    'i': 'bcdjponh',
    'j': 'cdekqpoi',
    'k': 'deflrqpj',
    'l': 'fekqkr',
    'm': 'ghntst',
    'n': 'ghioutsm',
    'o': 'hijpvutn',
    'p': 'ijkqwvuo',
    'q': 'jklrxwvp',
    'r': 'lkqwx',
    's': 'mntzy',
    't': 'mnouzys',
    'u': 'nopvzt',
    'v': 'opqwu',
    'w': 'pqrxv',
    'x': 'qrw',
    'y': 'stz',
    'z': 'ystu',
    '0': 'ztuv',
    '1': 'uvw',
    '2': 'vwx',
    '3': 'wx',
    '4': 'yz',
    '5': 'yz',
    '6': '',
    '7': '',
    '8': '',
    '9': ''
}


class Autocomplete (object):

    def __init__ (self, load_path = None):
        self.model = models.Model ()
        self.model.load_models (load_path)

    def this_word (self, word, top_n = 4):
        """given an incomplete word, return top n suggestions"""
        try:
            words = [(k, v) for k, v in self.model.words_model.most_common ()
                    if k.startswith (word)][:top_n]
            if not words:
                possible_words = list ()
                list_word = list (word)
                for index, letter in enumerate (list_word):
                    for possible_letter in NEARBY_KEYS[letter]:
                        new_word = copy.deepcopy (list_word)
                        new_word[index] = possible_letter
                        possible_words.append (''.join (new_word))
                for new_word in possible_words:
                    words += [(k, v) for k, v in self.model.words_model.most_common () if k.startswith (new_word)][:top_n]
                if len (words) > top_n:
                    words = sorted (words, reverse = True, key = lambda x: x[1])[:top_n]
            return words
        except KeyError as e:
            logging.error (e, exc_info = True)

    def this_word_given_last (self, first_word, second_word, top_n = 4):
        """given a word, return top n suggestions determined by the frequency of
        words prefixed by the input GIVEN the occurence of the last word"""
        try:
            possible_second_words = [second_word[:-1] + char for char in NEARBY_KEYS[second_word[-1]] if len (second_word) > 2]
            possible_second_words.append (second_word)
            probable_words = {w:c for w, c in
                              self.model.word_tuples_model[first_word.lower ()].items ()
                              for sec_word in possible_second_words
                              if w.startswith (sec_word)}

            return Counter (probable_words).most_common (top_n)
        except KeyError as e:
            logging.error (e, exc_info = True)

    def predict (self, first_word, second_word, top_n = 4):
        """given the last word and the current word to complete, we call
        predict_currword or predict_currword_given_lastword to retrive most n
        probable suggestions"""
        if first_word and second_word:
            return self.this_word_given_last (first_word, second_word, top_n = top_n)
        else:
            return self.this_word (second_word, top_n)

    def split_predict (self, text, top_n = 4):
        """takes in string and will right split accordingly.
        Optionally, you can provide keyword argument "top_n" for
        choosing the number of suggestions to return (default is 10)"""
        splitted = text.lower ().rsplit (' ')
        if len (splitted) >= 2:
            prev = splitted[-2]
            current = splitted[-1]
        else:
            current = splitted[-1]
            prev = None
        return self.predict (prev, current, top_n = top_n)

def main ():
    autocomplete = Autocomplete ()
    print autocomplete.split_predict ('it wor')

if __name__ == '__main__':
    main ()
