#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 00:45:40 2020

@author: bryanchia
"""

import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as td
import numpy as np
import string 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import PercentFormatter
from sklearn import linear_model as lm
import statsmodels.api as sm

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

def convert_day(x):
 
    if x == 2:
        return 'Monday'
    elif x == 3:
        return 'Tuesday'
    elif x == 4:
        return 'Wednesday'
    elif x == 5:
        return 'Thursday'
    elif x == 6:
        return 'Friday'
    elif x == 7:
        return 'Saturday'
    elif x == 1:
        return 'Sunday'

#Remove some shows I didn't watch
df_chars = df_chars[df_chars['Show'] != 'I Am Not Okay With This']
df_chars = df_chars[df_chars['Show'] != 'Dead to Me']
df_chars = df_chars[df_chars['Show'] != 'The Crown']
df_chars = df_chars[df_chars['Show'] != 'Insatiable']
#df_chars = df_chars[df_chars['Show'] != 'The Inbetweeners']

df_chars['Length_Minutes'] = [convert_minutes(x) for x in df_chars['Length']]


df_chars['Episodes'] = ['12 episodes' if ('Kingdom' in x) else y for x, y in zip(df_chars['Show'], df_chars['Episodes'])]
    
#Account for differences in Date
#df_chars['Date'] = [*map(lambda x, y: y if ('zoe' in x)\
#                                 else y - td(1), df_chars['Dataset'], df_chars['Date'])]

df_chars['Num_Episodes'] = pd.to_numeric([str(x).split('episodes', -1)[0] if str(x).find('episodes') != -1\
                                          else 1 for x in df_chars['Episodes']])

df_chars.loc[df_chars['Season'].isnull(), 'Season'] = 'Season 1'

#df_chars['Season'] = [ 'Season 1' if x == '' else x for x in df_chars['Season']]

 #Account for mistake for American Crime Story
df_chars['Season'] = [*map(lambda x, y: 'Season 2' if ('Gianni' in x)\
                                 else y, df_chars['Original Title'], df_chars['Season'])]
    
#Account for mistake for some limited series
df_chars['Length_Minutes'] = [*map(lambda x, y, z, a: y/z if ('Limited' in str(x)) & ('Unbelievable' not in str(a))\
                                 else y, df_chars['Full Title'], df_chars['Length_Minutes'], df_chars['Num_Episodes'],df_chars['Title'])]

    
df_chars['Season_Num'] = pd.to_numeric([*map(lambda x: x.split('Season', -1)[1] if ('Season' in x)\
                                               else 1, df_chars['Season'])])

df_chars['Total Time'] = [*map(lambda x, y: x * y, df_chars['Num_Episodes'], df_chars['Length_Minutes'])]

df_chars['Total_Seasons'] = pd.to_numeric([*map(lambda x: str(x).split('.E', -1)[0].split('S', -1)[1] if ('S' in str(x))\
                                               else x, df_chars['Total Seasons'])])
    
#Account for mistake for American Crime Story
df_chars['Total_Seasons'] = [*map(lambda x, y: 3 if ('American Crime Story' in x)\
                                 else y, df_chars['Show'], df_chars['Total_Seasons'])]
    
#Account for mistake for Derry Girls
df_chars['Total_Seasons'] = [*map(lambda x, y: 2 if ('Derry' in x)\
                                 else y, df_chars['Show'], df_chars['Total_Seasons'])]

    
df_chars['Eps_Per_Season'] = pd.to_numeric([*map(lambda x,y: round(x/y, 0), df_chars['Num_Episodes'], df_chars['Total_Seasons'])])

#Filter out Second Time Watching

df_chars['Dummy'] = 1
    
df_chars['Times Watched'] = df_chars.sort_values(['Date'],ascending=True).groupby(['Full Title'])['Dummy'].transform(pd.Series.cumsum)

#Filter to TV Shows

df_chars = df_chars[df_chars['Num_Episodes'] > 1]
    
#Binge Analysis: Time Spent/Days Watched

df_sum_binge = df_chars.groupby(['Title', 'Season', 'Dataset']).agg(
     time_spent = ('Length_Minutes', sum),
     episode_length = ('Length_Minutes', max),
     total_episodes = ('Eps_Per_Season', max),
     date_range = ('Date', lambda x: (max(x) - min(x)).days + 1),
     total_days = ('Date', 'nunique')
     ).reset_index()

df_sum_binge['binge_rate_days'] = df_sum_binge['time_spent'] / df_sum_binge['total_days']

df_sum_binge['binge_rate_days'] = [*map(lambda x, y: (x/y)/60.0, df_sum_binge['time_spent'], df_sum_binge['total_days'] )]

df_sum_binge['binge_rate_days_consec'] = [*map(lambda x, y: (x/y)/60.0, df_sum_binge['time_spent'], df_sum_binge['date_range'] )]

#Account For Day of Week

df_chars['Day of Week'] = [*map(lambda x: x.isoweekday() ,df_chars['Date'])]

df_chars['Day of Week'] = df_chars['Day of Week'].map({2: 'Monday', 3: 'Tuesday', 4: 'Wednesday', 5: 'Thursday', 6: 'Friday',\
                                                7: 'Saturday', 1: 'Sunday'})

df_reg = df_chars.groupby(['Title', 'Season', 'Dataset', 'Date', 'Day of Week']).agg(
     time_spent = ('Length_Minutes', sum)
     ).reset_index()

df_reg['time_spent_h'] = [*map(lambda x: x/60.0, df_reg['time_spent'])]

def create_dummy(wd):
    df_reg[wd] = [*map ( lambda x: 1 if x == wd else 0, df_reg['Day of Week'])]
    
for x in df_reg['Day of Week'].unique():
    create_dummy(x)

X = sm.add_confnt(df_reg.iloc[:,7:13])

results = sm.OLS(df_reg['time_spent_h'], X ).fit(cov_type='HC1')

results.summary()

df_reg_output = pd.merge(pd.merge(df_reg, pd.DataFrame(results.predict(), columns = ['yhat']), \
                         how = 'left', left_index = True, right_index = True),\
                         pd.DataFrame(results.resid, columns = ['resid']), how = 'left', left_index = True, right_index = True)

check = df_reg.groupby(['Day of Week']).agg(
     time_spent = ('time_spent_h', 'mean')
     ).reset_index()

df_sum_binge_norm = df_reg_output.groupby(['Title', 'Season', 'Dataset']).agg(
     time_spent = ('time_spent_h', sum),
     time_spent_norm = ('resid', sum),
     date_range = ('Date', lambda x: (max(x) - min(x)).days + 1),
     total_days = ('Date', 'nunique')
     ).reset_index()

df_sum_binge_norm['binge_rate_days'] = [*map(lambda x, y: (x/y), df_sum_binge_norm['time_spent'], df_sum_binge_norm['total_days'] )]
df_sum_binge_norm['binge_rate_days_norm'] = [*map(lambda x, y: (x/y), df_sum_binge_norm['time_spent_norm'], df_sum_binge_norm['total_days'] )]

df_sum_binge_norm['binge_rate_days_consec'] = [*map(lambda x, y: (x/y), df_sum_binge_norm['time_spent'], df_sum_binge_norm['date_range'] )]
df_sum_binge_norm['binge_rate_days_consec_norm'] = [*map(lambda x, y: (x/y), df_sum_binge_norm['time_spent_norm'], df_sum_binge_norm['date_range'] )]

df_final_sum = df_sum_binge_norm.groupby(['Title']).agg(
     full_binge_rate_days = ('binge_rate_days', 'mean'),
     full_binge_rate_days_norm = ('binge_rate_days_norm', 'mean'),
     full_binge_rate_days_consec = ('binge_rate_days_consec', 'mean'),
     full_binge_rate_days_consec_norm = ('binge_rate_days_consec_norm', 'mean')).reset_index()

#Binge Analysis: Percent Watched/Days Watched

df_sum_binge['p_binge_rate_days'] = [*map(lambda x, y, a, b: 1/y if (x > (a*b)) else (x/(a*b))/y,\
                                          df_sum_binge['time_spent'], \
                                              df_sum_binge['total_days'], \
                                                  df_sum_binge['episode_length'],\
                                              df_sum_binge['total_episodes'] , )]
    
df_sum_binge['p_binge_rate_days_consec'] = [*map(lambda x, y, a, b: 1/y if (x > (a*b)) else (x/(a*b))/y,\
                                          df_sum_binge['time_spent'], \
                                              df_sum_binge['date_range'], \
                                                  df_sum_binge['episode_length'],\
                                              df_sum_binge['total_episodes'] , )]
    
df_sum_binge['correct_episodes'] = [*map(lambda x, y, a: x/y if (x > (a*y)) else a,\
                                          df_sum_binge['time_spent'], \
                                                  df_sum_binge['episode_length'],\
                                              df_sum_binge['total_episodes'])]    
    
eps_map = df_sum_binge[['Title', 'Season', 'correct_episodes']].drop_duplicates()

#eps_map['Dummy'] = 1   
#eps_map['Times Watched'] = eps_map.groupby(['Title', 'Season'])['Dummy'].transform(pd.Series.cumsum)
    
#Account For Day of Week

df_reg_p = df_chars.groupby(['Title', 'Season', 'Dataset', 'Date', 'Day of Week']).agg(
     time_spent = ('Length_Minutes', sum),
     episode_length = ('Length_Minutes', max)
     ).reset_index()

df_reg_p = pd.merge(df_reg_p, eps_map, \
                                           how = 'left', on = ['Title', 'Season'])

df_reg_p['percent_watched'] = [*map(lambda x, a, b: x/(a*b), df_reg_p['time_spent'],\
                                   df_reg_p['episode_length'],\
                                       df_reg_p['correct_episodes'])]

df_reg_p = df_reg_p[df_reg_p['percent_watched'].notna()]

def create_dummy_p(wd):
    df_reg_p[wd] = [*map ( lambda x: 1 if x == wd else 0, df_reg_p['Day of Week'])]
    
for x in df_reg_p['Day of Week'].unique():
    create_dummy_p(x)

X = sm.add_constant(df_reg_p.iloc[:,10:16])

results_p = sm.OLS(df_reg_p['percent_watched'], X ).fit(cov_type='HC1')

results_p.summary()

df_reg_output_p = pd.merge(pd.merge(df_reg_p, pd.DataFrame(results_p.predict(), columns = ['yhat']), \
                         how = 'left', left_index = True, right_index = True),\
                         pd.DataFrame(results_p.resid, columns = ['resid']), how = 'left', left_index = True, right_index = True)


df_sum_binge_norm_p = df_reg_output_p.groupby(['Title', 'Season', 'Dataset']).agg(
     percent_watched = ('percent_watched', sum),
     percent_watched_norm = ('resid', sum),
     date_range = ('Date', lambda x: (max(x) - min(x)).days + 1),
     total_days = ('Date', 'nunique')
     ).reset_index()

df_sum_binge_norm_p['binge_rate_days'] = [*map(lambda x, y: (x/y), df_sum_binge_norm_p['percent_watched'], df_sum_binge_norm_p['total_days'] )]
df_sum_binge_norm_p['binge_rate_days_norm'] = [*map(lambda x, y: (x/y), df_sum_binge_norm_p['percent_watched_norm'], df_sum_binge_norm_p['total_days'] )]

df_sum_binge_norm_p['binge_rate_days_consec'] = [*map(lambda x, y: (x/y), df_sum_binge_norm_p['percent_watched'], df_sum_binge_norm_p['date_range'] )]
df_sum_binge_norm_p['binge_rate_days_consec_norm'] = [*map(lambda x, y: (x/y), df_sum_binge_norm_p['percent_watched_norm'], df_sum_binge_norm_p['date_range'] )]

#Final Statistics by Show

df_final_sum_p = df_sum_binge_norm_p.groupby(['Title']).agg(
     full_percent_binge_rate_days = ('binge_rate_days', 'mean'),
     full_percent_binge_rate_days_norm = ('binge_rate_days_norm', 'mean'),
     full_percent_binge_rate_days_consec = ('binge_rate_days_consec', 'mean'),
     full_percent_binge_rate_days_consec_norm = ('binge_rate_days_consec_norm', 'mean')).reset_index()

######################
#CREATE CHARTS
######################

from textwrap import wrap

#Plot Charts Function

def create_charts(df, var, num, denom, title, list_an = None, txt = ''):
    
    #df['Label'] = [ '\n'.join(wrap(l, 24)) for l in df['Label'] ]
    
    cp = cm.get_cmap('viridis', len(df[var])).colors

    fig, ax = plt.subplots(figsize=(40,50))

    ax.barh(
        df['Label'],
        df[var],
        color = cp
    )
    
    if ('percent' in num) & ('full' not in num):
        for i, v in list(enumerate(df[var])):
            ax.text(v + 0.01, i + 0.25, str(round(df[num][i]*100, 0)) + '% /' + str(round(df[denom][i], 1)) + ' days', 
                    fontsize = 46)   
    elif ('full' in num) & ('percent' not in num) :
        for i, v in list(enumerate(df[var])):
            ax.text(v + 0.01, i + 0.25, str(round(df[num][i], 1)) + ' h/day', 
                    fontsize = 46)
    elif ('full' in num) & ('percent' in num) :
        for i, v in list(enumerate(df[var])):
            ax.text(v + 0.01, i + 0.25, str(round(df[num][i]*100, 0))+ ' %/day', 
                    fontsize = 46)
    else:
        for i, v in list(enumerate(df[var])):
            ax.text(v + 0.01, i + 0.25, str(round(df[num][i], 1)) + 'h /' + str(round(df[denom][i], 1)) + ' days', 
                    fontsize = 46)
    
    ax.set_xlabel('Binge Rate', labelpad = 10, fontsize = 46)
    ax.set_title(title, pad=5,
                 weight='bold', fontsize = 60,
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
    #ax.xaxis.set_major_formatter(PercentFormatter(1))
    
    plt.xticks(size= 46)
    plt.yticks(size= 46)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Cambria'
    
    plt.figtext(0.12, 0.08, txt, wrap=True, horizontalalignment='left', verticalalignment = 'top', fontsize=46)
    
    return(fig)

#Chart 1: Binge Rate: Time Spent / Days
    
df_sum_binge_norm['Label'] = df_sum_binge_norm['Title'] + '\n' + df_sum_binge_norm['Season']

#Discrete Binge Rate

df_chart1 = df_sum_binge_norm.sort_values(by=['binge_rate_days'], ascending = False)[0:20].reset_index(drop = True)

create_charts(df_chart1,
              'binge_rate_days',
              'time_spent',
              'total_days',
              'Discrete Binge Rate For Top 20 Netflix TV Show Seasons Watched', 
              txt = '[1] Discrete Binge Rate is calculated as the total time spent on a season divided by the number of\ndays over which the season was viewed.')\
    .savefig(directory + '/04_binge_analysis_figures/01_discrete_binge_rate.png', bbox_inches='tight')

#Consecutive Binge Rate

df_chart2 = df_sum_binge_norm.sort_values(by=['binge_rate_days_consec'], ascending = False)[0:20].reset_index(drop = True)

create_charts(df_chart2,
              'binge_rate_days_consec',
              'time_spent',
              'date_range',
              'Consecutive Binge Rate For Top 20 Netflix TV Show Seasons Watched', 
              txt = '[1] Consecutive Binge Rate is calculated as the total time spent on a season divided by the range\nof dates over which the season was viewed.')\
    .savefig(directory + '/04_binge_analysis_figures/02_consec_binge_rate.png', bbox_inches='tight')

#Normalized Consecutive Binge Rate

df_chart3 = df_sum_binge_norm.sort_values(by=['binge_rate_days_consec_norm'], ascending = False)[0:20].reset_index(drop = True)

create_charts(df_chart3,
              'binge_rate_days_consec_norm',
              'time_spent_norm',
              'date_range',
              'Normalized Consecutive Binge Rate For Top 20 Netflix TV Show Seasons Watched', 
              txt = '[1] Normalized Consecutive Binge Rate is calculated as the total time spent on a season divided by\nthe range of dates over which the season was viewed, normalized for the day of the week.')\
    .savefig(directory + '/04_binge_analysis_figures/03_norm_consec_binge_rate.png', bbox_inches='tight')

#Normalized Consecutive Percentage Binge Rate

df_sum_binge_norm_p['Label'] = df_sum_binge_norm_p['Title'] + '\n' + df_sum_binge_norm_p['Season']

df_chart4 = df_sum_binge_norm_p.sort_values(by=['binge_rate_days_consec_norm'], ascending = False)[0:20].reset_index(drop = True)

create_charts(df_chart4,
              'binge_rate_days_consec_norm',
              'percent_watched_norm',
              'date_range',
              'Normalized Consecutive Percentage Binge Rate For Top 20 Netflix TV Show Seasons Watched', 
              txt = '[1] Normalized Consecutive Percentage Binge Rate is calculated as the percentage of a season finished (total time\nspent divided by total watching hours of a season) divided by the range of dates over which the season was viewed,\nnormalized for the day of the week.')\
    .savefig(directory + '/04_binge_analysis_figures/04_norm_consec_p_binge_rate.png', bbox_inches='tight')

#Consecutive Percentage Binge Rate

df_sum_binge_norm_p['Label'] = df_sum_binge_norm_p['Title'] + '\n' + df_sum_binge_norm_p['Season']

df_chart5 = df_sum_binge_norm_p.sort_values(by=['binge_rate_days_consec'], ascending = False)[0:20].reset_index(drop = True)

create_charts(df_chart5,
              'binge_rate_days_consec',
              'percent_watched',
              'date_range',
              'Consecutive Percentage Binge Rate For Top 20 Netflix TV Show Seasons Watched', 
              txt = '[1] Consecutive Percentage Binge Rate is calculated as the percentage of a season finished (total time spent\ndivided by total watching hours of a season) divided by the range of dates over which the season was viewed.')\
    .savefig(directory + '/04_binge_analysis_figures/05_consec_p_binge_rate.png', bbox_inches='tight')

#Show Summaries (Percentage)
    
df_chart6 = df_final_sum_p.sort_values(by=['full_percent_binge_rate_days_consec_norm'], ascending = False)[0:10].reset_index(drop = True)

df_chart6['Label'] = df_chart6['Title']

create_charts(df_chart6,
              'full_percent_binge_rate_days_consec_norm',
              'full_percent_binge_rate_days_consec_norm',
              'date_range',
              'Consecutive Percentage Binge Rate For Top 10 Netflix TV Shows Watched', 
              txt = '[1] Consecutive Percentage Binge Rate is calculated as the percentage of a show finished (total time spent\ndivided by total watching hours of a show) divided by the range of dates over which the show was viewed.\nThe average of all seasons are then taken.')\
    .savefig(directory + '/04_binge_analysis_figures/06_summary_percent_binge_rate.png', bbox_inches='tight')

#Show Summaries (Percentage)
    
df_chart7 = df_final_sum.sort_values(by=['full_binge_rate_days_consec_norm'], ascending = False)[0:10].reset_index(drop = True)

df_chart7['Label'] = df_chart7['Title']

create_charts(df_chart7,
              'full_binge_rate_days_consec_norm',
              'full_binge_rate_days_consec_norm',
              'date_range',
              'Consecutive Binge Rate For Top 10 Netflix TV Shows Watched', 
              txt = '[1] Consecutive Binge Rate is calculated as the total time spent on a show divided by the range of dates\nover which the show was viewed. The average of all seasons are then taken.')\
    .savefig(directory + '/04_binge_analysis_figures/07_summary_binge_rate.png', bbox_inches='tight')
    
df_chars['Test'] = ['{}+{}'.format(x,y) for (x,y) in zip(df_chars['Title'], df_chars['Episode'])]

df_duplicates = df_chars[df_chars.duplicated()]
s

