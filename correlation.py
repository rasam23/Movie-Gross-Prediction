# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 00:40:38 2017

"""

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(context="paper", font="monospace")
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
import numpy as np


data=pd.read_csv('movie_project.csv')
data.columns
df = pd.DataFrame()
df['budget'] = data.budget
df['duration']=data.duration
df['director_facebook_likes']=data.director_facebook_likes
df['actor_facebook_likes']=data['actor_3_facebook_likes']+data['actor_1_facebook_likes']+data['actor_2_facebook_likes']
df['cast_facebook_likes']=data['cast_total_facebook_likes']
df['movie_facebook_likes']=data['movie_facebook_likes']
df['Drama']=data.Drama
df['Comedy']=data.Comedy
df['Mystery']=data.Mystery
df['News']=data.News
df['Sports']=data.Sports
df['Sci-Fi']=data['Sci-Fi']
df['Romance']=data.Romance
df['Family']=data['family']
df['Biography']=data.Biography
df['Musical']=data.Musical
df['Adventure']=data.Adventure
df['Crime']=data.Crime
df['War']=data.War
df['Fantasy']=data.Fantasy
df['Thriller']=data.Thriller
df['Horror']=data.Horror
df['Animation']=data.Animation
df['Action']=data.Action
df['History']=data.History
df['Western']=data.Western
df['0']=data['0']
df['imdb_score']=data.imdb_score
df['imdb_votes']=data.imdbVotes
df['Gross']=data.gross
df['Award']=data.AwardsOther
df['Win']=data.Wins
df['Nominations']=data.Nominations
df['num_critic_for_reviews']=data.num_critic_for_reviews
df['num_voted_users']=data.num_voted_users
df['num_users_for_review']=data.num_user_for_reviews
df['DVD']=data['DVD']

corrmat = df.corr()
corrmat
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)

df1 = pd.DataFrame()
df1=df[(df['Drama'] ==1)]
df1=df1.drop(df1.columns[[1, 3,  4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]], axis=1)

df2=pd.DataFrame()
df2['Drama_Gross']= df['Gross'].where(df['Drama'] == 1)
df2['Comedy_Gross']= df['Gross'].where(df['Comedy'] == 1)
df2['Mystery_Gross']= df['Gross'].where(df['Mystery'] == 1)
df2['News_Gross']= df['Gross'].where(df['News'] == 1)
df2['Sports_Gross']= df['Gross'].where(df['Sports'] == 1)
df2['Sci-Fi_Gross']= df['Gross'].where(df['Sci-Fi'] == 1)
df2['Romance_Gross']= df['Gross'].where(df['Romance'] == 1)
df2['Family_Gross']= df['Gross'].where(df['Family'] == 1)
df2['Biography_Gross']= df['Gross'].where(df['Biography'] == 1)
df2['Musical_Gross']= df['Gross'].where(df['Musical'] == 1)
df2['Adventure_Gross']= df['Gross'].where(df['Adventure'] == 1)
df2['Crime_Gross']= df['Gross'].where(df['Crime'] == 1)
df2['War_Gross']= df['Gross'].where(df['War'] == 1)
df2['Fantasy_Gross']= df['Gross'].where(df['Fantasy'] == 1)
df2['Thriller_Gross']= df['Gross'].where(df['Thriller'] == 1)
df2['Horror_Gross']= df['Gross'].where(df['Horror'] == 1)
df2['Animation_Gross']= df['Gross'].where(df['Animation'] == 1)
df2['Action_Gross']= df['Gross'].where(df['Action'] == 1)
df2['History_Gross']= df['Gross'].where(df['History'] == 1)
df2['Western_Gross']= df['Gross'].where(df['Western'] == 1)
df2=df2.fillna(0)
df3=pd.DataFrame()
df3['avg']=df2.mean()
df3['avg'] = df3['avg'].astype(int)
df3['total_gross']=df2.sum()

df3['avg'].plot(kind='bar',color='red')
df3['total_gross'].plot(kind='bar',color='blue')

#Checking correlation of Budget with Gross of movie
y=df.Gross
y=y.fillna(0)
y=y.astype(int)

X=df.budget
X=X.fillna(0)
X=X.astype(int)
X=sm.add_constant(X)

lr_model=sm.OLS(y,X).fit()
print(lr_model.summary())
print(lr_model.params)

print("Correlation coefficient is=",np.corrcoef(df.Gross,df.budget) [0,1])
#Gross=0.2552*Budget+3.223e+07
plt.figure()
plt.scatter(df.scaled_budget,df.scaled_gross)
plt.xlabel('budget')
plt.ylabel('Gross')
#Converting budget and gross to logarithmic scale
from sklearn import preprocessing
df['scaled_budget'] = np.log(df.budget)
df['scaled_gross'] = np.log(df.Gross)

