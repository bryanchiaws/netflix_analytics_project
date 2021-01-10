#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 23:09:58 2020

@author: bryanchia
"""

import pandas as pd
from datetime import datetime as dt

directory = '/Users/bryanchia/Desktop/Netflix Project'

file = '/bryan_viewing.csv'

df = pd.read_csv(directory + file)

#clean data

df['Show'] = df['Title'].str.split(':', -1, True)[0]
df['Season'] = df['Title'].str.split(':', -1, True)[1]
df['Episode'] = df['Title'].str.split(':', -1, True)[2]

df['Date'] = [ dt.strptime(x, '%d/%m/%Y') if y == 'raena' else dt.strptime(x, '%m/%d/%Y') for x, y in \
    zip(df['Date'], df['Dataset'])]

df = df.rename({'Title': 'Full Title'}, axis=1)

df_show_chars = pd.read_pickle(directory + '/show_chars_full.pkl')

df_show_actors = pd.read_pickle(directory + '/show_actors_full.pkl')

df_complete = df.merge(df_show_chars, left_on = 'Show', right_on = 'Title', how = 'left')

df_complete = df_complete[df_complete['Full Title'] != '']

df_complete.to_pickle(directory + '/show_data_full.pkl')

s = pd.DataFrame([pd.Series(x)for x in df_show_actors['Actors']]).stack().reset_index(level = 1, drop = True) 

s.name = 'Actor'

df_actors_complete = df_show_actors.join(s, how =  'right' ).drop('Actors', axis = 1).reset_index(drop = True)

df_actors_complete.to_pickle(directory + '/show_actors_stack_full.pkl')

