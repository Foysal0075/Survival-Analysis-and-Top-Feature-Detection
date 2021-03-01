# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 05:53:55 2020

@author: Foysal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator


data = pd.read_csv('processed_covid_data.csv')
df = pd.DataFrame(data)
df = df.drop('Unnamed: 0', axis=1)

df['gender'] = df['gender'].replace(1,'male')
df['gender'] = df['gender'].replace(2,'female')

gender = df['gender']
gender = gender.astype('category')

fig = plt.figure(figsize=(7,5))
for value in gender.unique():
    mask = gender == value
    if value == 'male':
        time_cell, survival_prob_cell = kaplan_meier_estimator(df['survival'][mask].astype(bool), df['hospital duration'][mask])
        plt.step(time_cell, 1 - survival_prob_cell, where="post", color = '#FB014B',label="Male")
    elif value=='female':
        time_cell, survival_prob_cell = kaplan_meier_estimator(df['survival'][mask].astype(bool), df['hospital duration'][mask])
        plt.step(time_cell, 1 - survival_prob_cell, where="post",color = '#0163FB',label="Female")


plt.ylabel("est. Probability of Survival")
plt.xlabel("Hospital Duration")
plt.legend(loc="lower right")
limit = plt.gca()
limit.set_xlim([0, 48])
limit.set_ylim([0, 1])
plt.show()
plt.savefig('corona_gender_onset.pdf')



group1 = np.array(df.query('Age<=34').index)
group2 =np.array(df.query('Age>34 and Age<=47').index)
group3 =np.array(df.query('Age>47 and Age<=64').index)
group4 =np.array(df.query('Age>64').index)

groups = [ 'Age: 0-34', 'Age: 35-47', 'Age: 48-64', 'Age: 65+']

fig = plt.figure(figsize=(6, 4))
for value in groups:

    if value == 'Age: 65+':
        time_cell, survival_prob_cell = kaplan_meier_estimator(df.loc[group4]['survival'].astype(bool), df.loc[group4]['hospital duration'])
        plt.step(time_cell, 1 - survival_prob_cell, where="post", color = '#FB0101',label="%s" % (value))

    elif value == 'Age: 0-34':
        time_cell, survival_prob_cell = kaplan_meier_estimator(df.loc[group1]['survival'].astype(bool), df.loc[group1]['hospital duration'])
        plt.step(time_cell, 1 - survival_prob_cell, where="post", color = '#18C239',label="%s" % (value))

    elif value == 'Age: 35-47':
        time_cell, survival_prob_cell = kaplan_meier_estimator(df.loc[group2]['survival'].astype(bool), df.loc[group2]['hospital duration'])
        plt.step(time_cell, 1 - survival_prob_cell, where="post", color = '#5101FB',label="%s" % (value))

    elif value == 'Age: 48-64':
        time_cell, survival_prob_cell = kaplan_meier_estimator(df.loc[group3]['survival'].astype(bool), df.loc[group3]['hospital duration'])
        plt.step(time_cell, 1 - survival_prob_cell, where="post", color = '#FB6401',label="%s" % (value))


plt.ylabel("est. Probability of Survival")
plt.xlabel("Hospital Duration")
plt.legend(loc="lower right")
limit = plt.gca()
limit.set_xlim([0, 44])
limit.set_ylim([0, 1])
#plt.savefig('corona_age_group_onset.pdf')
plt.show()


fig = plt.figure(figsize=(6,4))
time_cell, survival_prob_cell = kaplan_meier_estimator(df['survival'].astype(bool), df['hospital duration'])
plt.step(time_cell, 1 - survival_prob_cell, where="post", color = '#FF7A00',label="Age: 0-over")
plt.ylabel("est. Probability of Survival")
plt.xlabel("Hospital Duration")
plt.legend(loc="lower right")
limit = plt.gca()
limit.set_xlim([0, 48])
limit.set_ylim([0, 1])
plt.show()


age_groups = [len(group1), len(group2), len(group3), len(group4)]

fig = plt.figure(figsize=(6,4))
ax = fig.add_axes([0,0,1,1])
ax.bar(groups, age_groups)
ax.set_ylabel('Number of Patients')
ax.set_title('Patients by age group')
plt.show()



male=[]
female= []

male.append((df.loc[group1]['gender']=='male').value_counts()[True])
male.append((df.loc[group2]['gender']=='male').value_counts()[True])
male.append((df.loc[group3]['gender']=='male').value_counts()[True])
male.append((df.loc[group4]['gender']=='male').value_counts()[True])

female.append((df.loc[group1]['gender']=='male').value_counts()[False])
female.append((df.loc[group2]['gender']=='male').value_counts()[False])
female.append((df.loc[group3]['gender']=='male').value_counts()[False])
female.append((df.loc[group4]['gender']=='male').value_counts()[False])


# male   = [92, 137 , 160, 161 ]
# female = [146, 161, 136, 143]

x = np.arange(len(groups))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(8,5))
rects1 = ax.bar(x - width/2, male, width, label='Male')
rects2 = ax.bar(x + width/2, female, width, label='Female')

ax.set_ylabel('Number of Patients')
ax.set_title('Patients by age group')
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.legend(loc='best')


def autolabel(rects):

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.show()




























