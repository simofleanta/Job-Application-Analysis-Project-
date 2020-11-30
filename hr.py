import pandas as pd
import seaborn as sns 
from pandas import DataFrame
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import numpy as np
import plotly
import statistics
import plotly.express as px
import stats
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import plotly.express as px


#sometimes it is hard to choose which jobs are more relevant, so the person hunting for jobs needs to focus on most relevant ones. 
#most relevant ones are those where interview offer is responsive. 
#q to answer: to which jobs should apply more based on where the interview call comes positive. 
# out of couple of countries, which are the countries most responsive?
# on what seniority level should the person apply? 

#to answer the questions, I decided to follow a set of exploratory data analysis process:
# #groupings, 
#filters
#pivots
#counts
#graphs 

hr=pd.read_csv('hr.csv')
print(hr.columns)
df=DataFrame(hr.head(113))
print(df.head(113))


# more jobs applied for seem graduate 
sns.violinplot(x=df["Job_seniority"], y=df["interview_call"], palette="Blues")

sns.violinplot(x=df["region"], y=df["interview_call"], palette="Blues")


#aggregation 
operations=['mean','sum','min','max']
a=df.groupby(['Job_seniority','Domain'], as_index=False)[['interview_call']].agg(operations)
print(a.reset_index())

x=df.groupby(['Domain'])[['interview_call']]
print(x.count())

y=df.groupby(['interview_call'])
print(y.count())

#filter interview call 1 
interview_call=df[df.interview_call==1]
No_interview_call=df[df.interview_call==2]
No_answer_interview_call=df[df.interview_call==3]
#print(interview_call)

#I will focus on one and use the above data instead of the df 

""" count how many 1=yes interviews by  domain"""
#am having 
yes=interview_call.groupby(['Domain'])
#print(yes.count())

#am not having interview
no=No_interview_call.groupby(['Domain'])
#print(no.count())

#am not having an answer
no_answer=No_answer_interview_call.groupby(['Domain'])
#print(no_answer.count())

""" count how many interviews based on job seniority"""

#am having 
seniority=interview_call.groupby(['Job_seniority'])
print(seniority.count())

#filter seniority with answer 1(it seems that you get nothing on Graduate level)
#so to filter based on an answer use filtered data on the answer desired
Junior=interview_call[interview_call.Job_seniority=='Junior']
#print(Junior)

#normal jobs are mostly 1 (wonder why?)
Normal=interview_call[interview_call.Job_seniority=='Normal']
print(Normal)

#it seems that bi is more requested. Now I'll check country that most requires 
#image bi 
domain=interview_call[interview_call.Domain=='data intelligence']
print(domain.count())

#image data
domain=interview_call[interview_call.Domain=='data']
print(domain)
domain=interview_call[interview_call.Domain=='data']
print(domain.count())

#imgage on region

region_br=interview_call[interview_call.region=='Brussels']
print(region_br)

region_b=interview_call[interview_call.region=='Berlin']
print(region_b)


# pivot on yes

pivot1=interview_call.pivot_table(index='interview_call',columns='Domain', aggfunc={'interview_call':'count'}).fillna(0)
pivot1['Max']=pivot1.idxmax(axis=1)
print(pivot1)

#pivot on yes, on  bi on seniority

pivot1=domain.pivot_table(index='interview_call',columns='Job_seniority', aggfunc={'interview_call':'count'}).fillna(0)
pivot1['Max']=pivot1.idxmax(axis=1)
print(pivot1)

#pivot on yes, on bi, on region 
pivot1=domain.pivot_table(index='interview_call',columns='region', aggfunc={'interview_call':'count'}).fillna(0)
pivot1['Max']=pivot1.idxmax(axis=1)
print(pivot1)

#graphs to resue the findings 


fig, ax=plt.subplots(figsize=(6,4))
sns.set_style('darkgrid')
interview_call.groupby('Domain')['interview_call'].count().sort_values().plot(kind='bar')
plt.ylabel('interview_call')
ax.get_yaxis().get_major_formatter().set_scientific(False)
plt.title('Most intervwiew call come from the jobs')
plt.show()

fig, ax=plt.subplots(figsize=(6,4))
sns.set_style('darkgrid')
interview_call.groupby('Job_seniority')['interview_call'].count().sort_values().plot(kind='bar')
plt.ylabel('interview_call')
ax.get_yaxis().get_major_formatter().set_scientific(False)
plt.title('Most intervwiew call come from the jobs seniority levels')
plt.show() 

fig, ax=plt.subplots(figsize=(6,4))
sns.set_style('darkgrid')
interview_call.groupby('region')['interview_call'].count().sort_values().plot(kind='bar')
plt.ylabel('interview_call')
ax.get_yaxis().get_major_formatter().set_scientific(False)
plt.title('Most intervwiew call come from the regions')
plt.show()

fig = px.density_heatmap(interview_call, x="Job", y="interview_call", nbinsx=20, nbinsy=20, color_continuous_scale="Blues_r",title='Job distribution on interview call')
plotly.offline.plot(fig, filename='bike')

"""Conclusions:"""

#Most responsive are jobs with BI followed by DA and then BA
#This sequence shows BI as a balance between business analysis and data analysis.#tech & Business
#seniority level not very relevant, mostly jobs with normal seniority level were more responsive

#Suggestion:
#apply to BI Jobs

#requirements of bi:
#-bi tools
#-sql+py which are basis applying data analysis into a business context as mentioned above AS balance between tech& business











 






























