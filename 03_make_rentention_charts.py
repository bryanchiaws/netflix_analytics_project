#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 00:45:40 2020

@author: bryanchia
"""

import pandas as pd
from datetime import datetime as dt
import numpy as np
import string 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import PercentFormatter

directory = '/Users/bryanchia/Desktop/Netflix Project'

df_chars = pd.read_pickle(directory + '/show_data_full.pkl')

#df_chars['Length'] = df_chars['Length'].to_string()

def convert_minutes(x):
    
    x = str(x)
    
    if x == '':
        return np.NaN
    elif ('h' in x) & ('min' not in x):
        return pd.to_numeric(x.split('h', -1)[0])*60.0
    elif ('h' not in x) & ('min' in x):
        return pd.to_numeric(x.split('min', -1)[0])*1.0
    elif ('h' in x) & ('min' in x):
        return pd.to_numeric(x.split('h', -1)[0])*60.0 + pd.to_numeric(x.split('h', -1)[1].split('min', -1)[0])

#Remove some shows I didn't watch
df_chars = df_chars[df_chars['Show'] != 'I Am Not Okay With This']
df_chars = df_chars[df_chars['Show'] != 'Dead to Me']
#df_chars = df_chars[df_chars['Show'] != 'The Inbetweeners']


df_chars['Length_Minutes'] = df_chars['Length'].apply(convert_minutes)
    
#Account for mistake for Kingdom
df_chars['Episodes'] = [*map(lambda x, y: '12 episodes' if ('Kingdom' in x)\
                                 else y, df_chars['Show'], df_chars['Episodes'])]

df_chars['Num_Episodes'] = pd.to_numeric([*map(lambda x: str(x).split('episodes', -1)[0] if str(x).find('episodes') != -1\
                                 else 1, df_chars['Episodes'])])
    
df_chars.loc[df_chars['Season'].isnull(), 'Season'] = 'Season 1'
    
#Account for mistake for s0me limited series
df_chars['Length_Minutes'] = [*map(lambda x, y, z, a: y/z if ('Limited' in str(x)) & ('Unbelievable' not in str(a))\
                                 else y, df_chars['Full Title'], df_chars['Length_Minutes'], df_chars['Num_Episodes'],df_chars['Title'])]

    
df_chars['Season_Num'] = pd.to_numeric([*map(lambda x: x.split('Season', -1)[1] if ('Season' in x)\
                                               else 1, df_chars['Season'])])

df_chars['Total Time'] = [*map(lambda x, y: x * y, df_chars['Num_Episodes'], df_chars['Length_Minutes'])]

df_chars['Total_Seasons'] = pd.to_numeric([*map(lambda x: str(x).split('.E', -1)[0].split('S', -1)[1] if ('S' in str(x))\
                                               else x, df_chars['Total Seasons'])])

#Retention Analysis

df_sum_retention = df_chars.groupby(['Title']).agg(
     time_spent = ('Length_Minutes', sum),
     episode_length = ('Length_Minutes', max),
     time_spent_h = ('Length_Minutes', lambda x: sum(x)/60.0),
     total_time = ('Total Time', min),
     total_time_h = ('Total Time', lambda x: min(x)/60.0),
     date_range = ('Date', lambda x: (max(x) - min(x)).days),
     total_seasons= ('Total_Seasons', max),
     seasons_watched = ('Season_Num', 'nunique'),
     episodes_watched = ('Episode', 'nunique'),
     num_episodes = ('Num_Episodes', max)
     ).reset_index()

df_sum_retention['perc_watched'] = [*map(lambda x, y: x/y, df_sum_retention['time_spent'], df_sum_retention['total_time'] )]

df_sum_soaps = df_sum_retention[(df_sum_retention['total_seasons'] > 1) & (df_sum_retention['total_time'] > 1000)& (df_sum_retention['episode_length'] > 30)]

df_sum_sitcoms = df_sum_retention[(df_sum_retention['total_seasons'] > 1) & (df_sum_retention['episode_length'] <= 30)]

df_sum_series = df_sum_retention[df_sum_retention['total_seasons'] > 1]

df_sum_ls = df_sum_retention[(df_sum_retention['total_seasons'] == 1) & (df_sum_retention['num_episodes'] > 1)]

df_sum_tv = df_sum_retention[(df_sum_retention['num_episodes'] > 1)]

df_sum_failed = df_sum_retention[(df_sum_retention['episodes_watched'] == 1) & (df_sum_retention['num_episodes'] > 1)]

#Plot Charts Function

def create_charts(df, title, list_an = None, txt = ''):
    
    df['Title'] = [ '\n'.join(wrap(l, 15)) for l in df['Title'] ]
    
    cp = cm.get_cmap('viridis', len(df['perc_watched'])).colors

    fig, ax = plt.subplots(figsize=(32,28))

    ax.barh(
        df['Title'],
        df['perc_watched'],
        color = cp
    )
    
    for i, v in list(enumerate(df['perc_watched'])):
        ax.text(v + 0.01, i + 0.25, str(round(df['time_spent_h'][i], 1)) + '/' + str(round(df['total_time_h'][i], 1)) + ' h', 
                fontsize = 28)
    
    ax.set_xlabel('Retention Rate', labelpad = 10, fontsize = 28)
    ax.set_title(title, pad=5,
                 weight='bold', fontsize = 50,
                 loc ='left')
    
    if list_an is None:
        pass
    else:
        for a, b, c in list_an:
        
            ax.annotate(c, xy=(b + 0.14, a + 0.15), xytext=(b + 0.19, a + 0.15),
                        fontsize = 28, 
                        arrowprops=dict(arrowstyle="<-", lw = 2, shrinkA = 5),
                        va='center', 
                        bbox = dict(boxstyle='round, pad=1', fc = 'w', lw = 2))
    
    ax.invert_yaxis()
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color='#EEEEEE')
    ax.yaxis.grid(False)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    
    plt.xticks(size= 28)
    plt.yticks(size= 28)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Cambria'
    
    plt.figtext(0.12, 0.05, txt, wrap=True, horizontalalignment='left', verticalalignment = 'top', fontsize=28)
    
    return(fig)

#Chart 1: TV Shows
    
df_chart1 = df_sum_tv.sort_values(by=['perc_watched', 'time_spent_h'], ascending = False)[0:20].reset_index(drop = True)

list_an1 = [list(enumerate(df_chart1['perc_watched']))[12] + ('I\'ve finished or even watched\neverything above this more\nthan once. Not. Proud.',),
            list(enumerate(df_chart1['perc_watched']))[15] + ('My wife was definitely\nresponsible for that one.',)]

create_charts(df_chart1,
              'Retention Rate For Top 20 Netflix TV Shows Watched',
              list_an1, 
              '[1] Retention Rate is calculated as the Total Time Spent on a TV Show divided by the Total Running Time of all seasons and episodes of the TV Show.\n Shows are sorted by retention rate and total hours spent.')\
    .savefig(directory + '/figures/01_all_tv_viewing.png', bbox_inches='tight')

#Chart 2: Soaps
    
df_chart2= df_sum_soaps.sort_values(by=['perc_watched', 'time_spent_h'], ascending = False)[0:15].reset_index(drop = True)

list_an2 = [list(enumerate(df_chart2['perc_watched']))[12] + ('Quite sure I watched the first 8 seasons of \nthe Walking Dead from Thanksgiving till\nFinals Week 2017 so it probably ranks higher...',), 
            list(enumerate(df_chart2['perc_watched']))[9] + ('Shonda Rhimes coming\nin hot at No. 5 and No 8!',),
            list(enumerate(df_chart2['perc_watched']))[5] + ('I am not proud of this... but neither am I ashamed.\n This was a cultural institution mind you.\nI got the Blackberry back in 2011 because of it.',)]

create_charts(df_chart2,
              'Retention Rate For Top 15 Netflix Long-Running TV Soaps Watched', 
              list_an2,
              txt = '[1] Retention Rate is calculated as the Total Time Spent on a TV Show divided by the Total Running Time of all seasons and episodes of the TV Show.\n Shows are sorted by retention rate and total hours spent.\
                      \n[2] Long-Running TV Soaps are defined as TV shows with more than one season, 1000 minutes of content, and where episodes last longer than half an hour.')\
    .savefig(directory + '/figures/02_soaps_tv_viewing.png', bbox_inches='tight')

#Chart 3: Sitcoms
    
df_chart3= df_sum_sitcoms.sort_values(by=['perc_watched', 'time_spent_h'], ascending = False)[0:10].reset_index(drop = True)

list_an3 = [list(enumerate(df_chart3['perc_watched']))[9] + ('Not worth watching after the first season.\nUnpopular opinion but FIGHT ME.',),
            list(enumerate(df_chart3['perc_watched']))[1] + ('Not a sitcom\n but wayy addictive.',)]

create_charts(df_chart3,
              'Retention Rate For Top 10 Netflix TV Sitcoms/Reality Shows Watched', 
              list_an3,
              txt = '[1] Retention Rate is calculated as the Total Time Spent on a TV Show divided by the Total Running Time of all seasons and episodes of the TV Show.\n Shows are sorted by retention rate and total hours spent.\
                      \n[2] TV Sitcoms/Reality Shows are defined as TV shows with more than one season and where episodes last less than half an hour.')\
    .savefig(directory + '/figures/03_sticoms_tv_viewing.png', bbox_inches='tight')
    
#Chart 4: Limited Series
    
df_chart4= df_sum_ls.sort_values(by=['perc_watched', 'time_spent_h'], ascending = False)[0:10].reset_index(drop = True)

list_an4 = [list(enumerate(df_chart4['perc_watched']))[4] + ('And Jerry with\nthe mat talk!!',), 
            list(enumerate(df_chart4['perc_watched']))[1] + ('Food shows\ndoing well...',),
            list(enumerate(df_chart4['perc_watched']))[3] + ('True crime\ndoes well too...',),
            list(enumerate(df_chart4['perc_watched']))[2] + ('Probs not a limited series...\non that cliff-hanger\nit ended on it better not be!',)]

create_charts(df_chart4,
              'Retention Rate For Top 10 Netflix Limited Series Watched', 
              list_an4,
              txt = '[1] Retention Rate is calculated as the Total Time Spent on a TV Show divided by the Total Running Time of all seasons and episodes of the TV Show.\n Shows are sorted by retention rate and total hours spent.\
                      \n[2] Limited Series are defined as TV shows with just one season but more than 1 episode.')\
    .savefig(directory + '/figures/04_ls_tv_viewing.png', bbox_inches='tight')
    
#Chart 5: What not to watch:
    
df_chart5= df_sum_failed[df_sum_failed['Title'] != 'Street Food'].sort_values(by=['episode_length'], ascending = True)[0:15].reset_index(drop = True)

def create_el_charts(df, title, list_an = None, txt = ''):
    
    df['Title'] = [ '\n'.join(wrap(l, 15)) for l in df['Title'] ]
    
    cp = cm.get_cmap('viridis', len(df['perc_watched'])).colors

    fig, ax = plt.subplots(figsize=(32,28))

    ax.barh(
        df['Title'],
        df['episode_length'],
        color = cp
    )
    
    for i, v in list(enumerate(df['episode_length'])):
        ax.text(v + 0.1, i + 0.25, str(df['episode_length'][i]).split('.')[0] + 'm', 
                fontsize = 28)
    
    ax.set_xlabel('Episode Length (Minutes)', labelpad = 10, fontsize = 28)
    ax.set_title(title, pad=5,
                 weight='bold', fontsize = 50,
                 loc ='left')
    
    if list_an is None:
        pass
    else:
        for a, b, c in list_an:
        
            ax.annotate(c, xy=(b + 0.14, a + 0.2), xytext=(b + 0.19, a + 0.2),
                        fontsize = 28, 
                        arrowprops=dict(arrowstyle="<-", lw = 2, shrinkA = 5),
                        va='center', 
                        bbox = dict(boxstyle='round, pad=1', fc = 'w', lw = 2))
    
    ax.invert_yaxis()
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color='#EEEEEE')
    ax.yaxis.grid(False)
    
    plt.xticks(size= 28)
    plt.yticks(size= 28)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Cambria'
    
    plt.figtext(0.12, 0.05, txt, wrap=True, horizontalalignment='left', verticalalignment = 'top', fontsize=28)
    
    return(fig)

create_el_charts(df_chart5,
              '15 Netflix TV Shows With The Lowest Retention', 
              txt = '[1] Listed are shows where only 1 episode was viewed, ranked by episode length.')\
    .savefig(directory + '/figures/05_failed_tv_viewing.png', bbox_inches='tight')
    