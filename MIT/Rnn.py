# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 12:49:42 2021

@author: Ismail
"""


my_rnn - RNN()

hidden_state = [0, 0, 0, 0]

sentence = ["I", "love", "recurrent", "neural"]


for word in sentence:
    prediction, hidden_state = my_rnn(word, hidden_state)
    
next_word_prediction = prediction

