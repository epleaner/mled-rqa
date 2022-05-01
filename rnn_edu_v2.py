import pandas as pd
import numpy as np
import json

import keras
from keras.layers import Input, Dense, LSTM, TimeDistributed, Lambda, multiply
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

def one_hot(skill_matrix, vocab_size):
    '''
    params:
        skill_matrix: 2-D matrix (student, skills)
        vocal_size: size of the vocabulary
    returns:
        a ndarray with a shape like (student, sequence_len, vocab_size)
    '''
    seq_len = skill_matrix.shape[1]
    result = np.zeros((skill_matrix.shape[0], seq_len, vocab_size))
    for i in range(skill_matrix.shape[0]):
        result[i, np.arange(seq_len), skill_matrix[i]] = 1.
    return result

def dkt_one_hot(skill_matrix, response_matrix, vocab_size):
    seq_len = skill_matrix.shape[1]
    skill_response_array = np.zeros((skill_matrix.shape[0], seq_len, 2 * vocab_size))
    for i in range(skill_matrix.shape[0]):
        skill_response_array[i, np.arange(seq_len), 2 * skill_matrix[i] + response_matrix[i]] = 1.
    return skill_response_array

def preprocess(skill_df, response_df, skill_num):
    skill_matrix = skill_df.iloc[:, 1:].values
    response_array = response_df.iloc[:, 1:].values
    skill_array = one_hot(skill_matrix, skill_num)
    skill_response_array = dkt_one_hot(skill_matrix, response_array, skill_num)
    return skill_array, response_array, skill_response_array

def build_skill2skill_model(input_shape, lstm_dim=32, dropout=0.0):
    input = Input(shape=input_shape, name='input skills')
    lstm = LSTM(lstm_dim, 
                return_sequences=True, 
                dropout=dropout,
                name='lstm layer')(input)
    output = TimeDistributed(Dense(input_shape[-1], activation='softmax'), name='probability')(lstm)
    model = Model(inputs=[input], outputs=[output])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def reduce_dim(x):
    x = K.max(x, axis=-1, keepdims=True)
    return x

def build_dkt_model(input_shape, lstm_dim=32, dropout=0.0):
    input_skills = Input(shape=input_shape, name='input skills')
    lstm = LSTM(lstm_dim, 
                return_sequences=True, 
                dropout=dropout,
                name='lstm layer')(input_skills)
    dense = TimeDistributed(Dense(int(input_shape[-1]/2), activation='sigmoid'), name='probability for each')(lstm)
    
    skill_next = Input(shape=(input_shape[0], int(input_shape[1]/2)), name='next_skill_tested')
    merged = multiply([dense, skill_next], name='multiply')
    reduced = Lambda(reduce_dim, output_shape=(input_shape[0], 1), name='reduce dim')(merged)
    
    model = Model(inputs=[input_skills, skill_next], outputs=[reduced])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

if __name__ == '__main__':

    response_df = pd.read_csv('correct.tsv', sep='\t').drop('Unnamed: 0', axis=1)
    skill_df = pd.read_csv('skill.tsv', sep='\t').drop('Unnamed: 0', axis=1)
    assistment_df = pd.read_csv('assistment_id.tsv', sep='\t').drop('Unnamed: 0', axis=1)
    skill_dict = {}
    with open('skill_dict.json', 'r', encoding='utf-8') as f:
        loaded = json.load(f)
        for k, v in loaded.items():
            skill_dict[k] = int(v)

    skill_num = len(skill_dict) + 1 # including 0

    skill_array, response_array, skill_response_array = preprocess(skill_df, response_df, skill_num)

    print('skill2skill')
    skill2skill_model = build_skill2skill_model((99, skill_num), lstm_dim=64)

    print('dkt')
    dkt_model = build_dkt_model((99, 2 * skill_num), lstm_dim=64)

    skill2skill_model.fit(skill_array[:, 0:-1], 
                          skill_array[:, 1:],
                          epochs=20, 
                          batch_size=32, 
                          shuffle=True,
                          validation_split=0.2)

    dkt_model.fit([skill_response_array[:, 0:-1], skill_array[:, 1:]],
                  response_array[:, 1:, np.newaxis],
                  epochs=20, 
                  batch_size=32, 
                  shuffle=True,
                  validation_split=0.2)