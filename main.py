# """
# This is the main script for the streamlit app to visualize the daily standup update records.
# The data is stored in a csv file, and the script will read the data and visualize it.
# The script is written in Python 3.8.5. 
# author: Yannan Su
# date: 2023-03-02

# streamlit run /Users/su/DS-projects/standup-updates-dashboard/main.py
# """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
# import calmap
import streamlit as st


sns.set_style('white')
sns.set_palette('Set2')
st.set_page_config(layout="wide")

st.markdown("# PhD's Daily Progress Records")

st.markdown("## Introduction") 
intro = "Doing PhD is a challenging. It becomes even more challenging when you approach the end of it. \
            You have to wrap up your research, write many papers and prepare your thesis. \
            \nTo make sure I am on track, I set up a daily standup meeting with my colleague and recorded my daily progress in a spreadsheet. \
            \nThis app is to visualize the data and help me to track my progress. "
st.markdown(intro)

st.markdown("## More Motivations")
motiv = "Besides tracking my progress, I also regard this project as: \
     \n- a teaser for project management and agile methodologies\
     \n- a practice for developing a data science project in a full cycle, from data definition to project deployment"
st.markdown(motiv)

st.markdown("## Data Definition")
datinfo = "The data is stored in a csv file, each row is a daily record. \
           \nThere can be two tasks in one day, and each task has a progress(%) and a spent time (h). \
           Tasks are categorized by the related paper (manuscript 1/2/3 (M1/M2/M3), thesis) \
            and the type of work (Draft, Revision, Analysis, Literature, Experiments, etc.). \
           \nThere is also a productivity rating (1-5) and a word count for each day. \
           Extra tasks (meetings, teaching, talks, etc.) are recorded in the extra event and extra time column."
st.markdown(datinfo)

# add sidebars for selecting person and month
sidebar_person = st.sidebar
people = ['Yannan']
person_selector = sidebar_person.selectbox("Select a Person", people)

sidebar_month = st.sidebar
months = {'All months': '2023-',
          'Jan 2023': '2023-1-', 
          'Feb 2023': '2023-2-', 
          'Mar 2023': '2023-3-', 
          'Apr 2023': '2023-4-', 
          'May 2023': '2023-5-', 
          'Jun 2023': '2023-6-', 
          'Jul 2023': '2023-7-', 
          'Aug 2023': '2023-8-', 
          'Sep 2023': '2023-9-', 
          'Oct 2023': '2023-10-', 
          'Nov 2023': '2023-11-', 
          'Dec 2023': '2023-12-'}
month_selector = sidebar_month.selectbox("Select a Month", months.keys())

st.markdown(f"## Selected data from: {month_selector}")

# add checkboxes for selecting figures
show_fig_all = st.sidebar.checkbox("Show all figures")
show_fig1 = st.sidebar.checkbox("All in one plot over time")
show_fig2 = st.sidebar.checkbox("Relation between productivity and task time")
show_fig3 = st.sidebar.checkbox("Calendar heatmap")
show_fig4 = st.sidebar.checkbox("Working hours by date")
show_fig5 = st.sidebar.checkbox("Working hours by weekday")
show_fig6 = st.sidebar.checkbox("Working hours for each task")
show_fig7 = st.sidebar.checkbox("Components of tasks")
show_fig8 = st.sidebar.checkbox("Progress by date")
show_fig9 = st.sidebar.checkbox("Progress by weekday")
show_fig10 = st.sidebar.checkbox("Lingering tasks")
show_fig11 = st.sidebar.checkbox("Word counts by date")
show_fig12 = st.sidebar.checkbox("Word counts by weekday")

if show_fig_all:
    show_fig1 = True
    show_fig2 = True
    show_fig3 = True
    show_fig4 = True
    show_fig5 = True
    show_fig6 = True
    show_fig7 = True
    show_fig8 = True
    show_fig9 = True
    show_fig10 = True
    show_fig11 = True
    show_fig12 = True

# """
# Prepare data
# """

# load data
dat = pd.read_csv(f'Daily-Standup-update-{person_selector}_2023.csv')

# select the month
dat = dat[dat['Date'].str.contains(months[month_selector])]
dat["Date"] = pd.to_datetime(dat["Date"], format="%Y-%m-%d").dt.date

# fill the missing values with 0 for all numerical columns
dat = dat.fillna(0)

dat['TotalProgress'] = dat["Progress1/%"] + dat["Progress2/%"]
dat['TotalTaskTime'] = dat["SpentTime1/h"] + dat["SpentTime2/h"]
dat['TotalTime'] = dat["TotalTaskTime"] + dat["ExtraTime/h"]

# split the df into twp df: date, task1, task2
dat_task_1 = dat[['WeekID', 'Date', 'Weekday', 
                        'Task1','Manuscript1', 'Type1', 'Progress1/%', 'SpentTime1/h']]

dat_task_2 = dat[['WeekID', 'Date', 'Weekday', 
                        'Task2', 'Manuscript2', 'Type2', 'Progress2/%', 'SpentTime2/h']]
# remove the numbers in the column names of dat_task_1
dat_task_1.columns = ['WeekID', 'Date', 'Weekday', 'Task','Manuscript', 'Type', 'Progress/%', 'SpentTime/h']
dat_task_1['TaskIndex'] = 'Task1'

dat_task_2.columns = ['WeekID', 'Date', 'Weekday', 'Task', 'Manuscript', 'Type', 'Progress/%', 'SpentTime/h']
dat_task_2['TaskIndex'] = 'Task2'
# concat the two df vertically
dat_task = pd.concat([dat_task_1, dat_task_2], axis=0)

dat_task.sort_values(by=['WeekID', 'Date', 'TaskIndex'], inplace=True)

# """
# All in one plot over time
# """
# plot the total progress of each day
# overlay the total task time spent on each task for each day
# overlay the task time + extra time spent on each task for each day
# overlay the productivity for each day

# normalize the values by dividing (max - min)
TotalProgress= (dat['TotalProgress'] - dat['TotalProgress'].min()) / (dat['TotalProgress'].max() - dat['TotalProgress'].min())
TotalTaskTime = (dat['TotalTaskTime'] - dat['TotalTaskTime'].min()) / (dat['TotalTaskTime'].max() - dat['TotalTaskTime'].min())
TotalTime = (dat['TotalTime'] - dat['TotalTime'].min()) / (dat['TotalTime'].max() - dat['TotalTime'].min())
ProductivityRating = (dat['ProductivityRating (1-5)'] - dat['ProductivityRating (1-5)'].min()) / (dat['ProductivityRating (1-5)'].max() - dat['ProductivityRating (1-5)'].min())
WordCount = (dat['WordCount'] - dat['WordCount'].min()) / (dat['WordCount'].max() - dat['WordCount'].min())

fig1, ax = plt.subplots(figsize=(15, 5))
x_date = np.arange(len(dat['Date']))
ax.plot(x_date, TotalProgress, label='TotalProgress', marker='o')
ax.plot(x_date, TotalTaskTime, label='TotalTaskTime', marker='o')
ax.plot(x_date, TotalTime, label='TotalTime', marker='o')
ax.plot(x_date, ProductivityRating, label='Productivity', marker='o')
ax.plot(x_date, WordCount, label='WordCount', marker='o')

# ax.plot(dat['Date'], dat['TotalTime'], label='TotalTime')
# ax.plot(dat['Date'], dat['ProductivityRating (1-5)'], label='Productivity')
# ax.plot(dat['Date'], dat['WordCount'], label='WordCount')
plt.legend()
ax.set_xticks(x_date)
ax.set_xticklabels(dat['Date'], rotation=90)
# plt.title('All in one plot over time (normalized)')
if show_fig1:
    st.markdown("### All in one plot over time (normalized by (max - min))")
    st.pyplot(fig1)

# """
# Regression plot between productivity and total task time
# """
# regression plot between productivity and total task time
fig2, ax = plt.subplots(figsize=(15, 5))
sns.regplot(x=dat['TotalTaskTime'], y=dat['ProductivityRating (1-5)'], data=dat, ax=ax)
# plt.title('Regression plot between productivity and total task time')
if show_fig2:
    st.markdown("### Regression plot between productivity and total task time")
    st.pyplot(fig2)


# """
# Calendar heatmap with calmap
# """
# hrs_by_date = dat[['Date', 'TotalTime']]
# # set date as index
# hrs_by_date.set_index('Date', inplace=True)
# hrs_by_date.index = pd.to_datetime(hrs_by_date.index)

# fig3_1, axes = calmap.calendarplot(hrs_by_date.groupby('Date')['TotalTime'].sum(), fig_kws={'figsize': (12, 6)})
# plt.title('Total Time Spent on Tasks')

# productivity_by_date = dat[['Date', 'ProductivityRating (1-5)']]
# # set date as index
# productivity_by_date.set_index('Date', inplace=True)
# productivity_by_date.index = pd.to_datetime(productivity_by_date.index)
# fig3_2, axes = calmap.calendarplot(productivity_by_date.groupby('Date')['ProductivityRating (1-5)'].sum(), fig_kws={'figsize': (12, 6)})
# plt.title('Productivity Rating (1-5)')

# """
# Calendar heatmap with seaborn
# """

# Seaborn version

def create_heatmap(val):
# Format the data for the heatmap
    val_by_date = dat[['Date', val]]

    # init_val[val] = init_val['Date'].map(val_by_date.set_index('Date')[val])
    # concat the two df vertically
    # val_by_date = pd.concat([val_by_date, init_val], axis=0)

    # the day before the first date in the df
    if val_by_date['Date'].min() == '2023-1-1':
        prev_day = '2023-1-1'
    else:
        prev_day = val_by_date['Date'].min() - datetime.timedelta(days=1)
    
    # the next day of the date in the df
    if val_by_date['Date'].max() == '2023-6-30':
        next_day = '2023-6-30'
    else:
        next_day = val_by_date['Date'].max() + datetime.timedelta(days=1)

    val_by_date = pd.concat([pd.DataFrame({'Date': pd.date_range(start='2023-1-1', end=prev_day), val: None}),
                             val_by_date, 
                             pd.DataFrame({'Date': pd.date_range(start=next_day, end='2023-6-30'), val: None})], axis=0)

    val_by_date["Date"] = pd.to_datetime(val_by_date["Date"])
    val_by_date['weekday'] = val_by_date['Date'].dt.dayofweek
    val_by_date['month'] = val_by_date['Date'].dt.month
    val_by_date['week'] = val_by_date['Date'].dt.isocalendar().week
    val_by_date_calmap = val_by_date.pivot_table(index='weekday', columns=['month', 'week'], values=val, aggfunc='sum')
    # fill the nan with 0
    val_by_date_calmap = val_by_date_calmap.fillna(1e-2)
    return val_by_date_calmap

# Create the heatmap
fig3_1, ax3_1 = plt.subplots(figsize=(12, 2.2))
sns.heatmap(create_heatmap('TotalTime'), cmap='YlGnBu', linewidths=0.5, ax=ax3_1)
# change the number of xticks as the number of months
ax3_1.set_xticks(np.arange(0, 6*5, 5)+2)
ax3_1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], rotation=0)
ax3_1.set_xlabel('Month')
# change the yticks and labels to only show the weekday
ax3_1.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
ax3_1.set_ylabel('Weekday')
ax3_1.set_title('Total Working Time')

fig3_2, ax3_2 = plt.subplots(figsize=(12, 2.2))
sns.heatmap(create_heatmap('ProductivityRating (1-5)'), cmap='YlGnBu', linewidths=0.5, ax=ax3_2)
# change the number of xticks as the number of months
ax3_2.set_xticks(np.arange(0, 6*5, 5)+2)
ax3_2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], rotation=0)
ax3_2.set_xlabel('Month')
# change the yticks and labels to only show the weekday
ax3_2.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
ax3_2.set_ylabel('Weekday')
ax3_2.set_title('Productivity Rating') 

if show_fig3:
    st.markdown("### Calendar heatmap")
    st.markdown("Note that the heatmap shows gaps between months. \
                This is the issue due to Seaborn heatmap. It can be fixed by using `calmap` package to create the heatmap. \
                I did not use calmap because it is not easily compatible with Streamlit. \ ")
    st.pyplot(fig3_1)
    st.pyplot(fig3_2)


# """
# Working hours by date
# """
dat_extra = dat[['WeekID', 'Date', 'Weekday', "ExtraEvent", "ExtraTime/h"]]

dat_extra.rename(columns={'ExtraTime/h': 'SpentTime/h'}, inplace=True)
dat_extra['TaskIndex'] = 'Extra'
dat_extra['Type'] = 'Extra'
# concat it with the dat
dat_st = pd.concat([dat_task, dat_extra], axis=0)


# plot stacked bar chart of Time of each day, colored by TaskIndex
fig4, ax = plt.subplots(figsize=(15, 5))
dat_time = dat_st[["Date", "TaskIndex", "SpentTime/h"]].groupby(["Date", "TaskIndex"]).sum().reset_index()
dat_time.pivot(index="Date", columns= "TaskIndex", values="SpentTime/h").plot(kind="bar", rot=90, ax=ax, stacked=True)
plt.ylabel('Working Time/h')
plt.axhline(y=dat_time.groupby("Date")["SpentTime/h"].sum().mean(), color='black', linestyle='--', label=' mean')
# plt.title('Working hours by date')
if show_fig4:
    st.markdown("### Working hours by date")
    st.pyplot(fig4)

# """
# Working hours by weekday
# """
# plot stacked bar chart of Time of each day, colored by TaskIndex
fig5, ax = plt.subplots(figsize=(15, 5))
dat_time = dat_st[["Weekday", "TaskIndex", "SpentTime/h"]].groupby(["Weekday", "TaskIndex"]).mean().reset_index()
# dat_time = dat_st[["Weekday", "TaskIndex", "SpentTime/h"]].groupby(["Weekday", "TaskIndex"]).sum().reset_index()
dat_time.pivot(index="Weekday", columns= "TaskIndex", values="SpentTime/h").plot(kind="bar", rot=90, ax=ax, stacked=True)
plt.axhline(y=dat_time.groupby("Weekday")["SpentTime/h"].sum().mean(), color='black', linestyle='--', label=' mean')
plt.ylabel('Working Time/h')
# plt.title('Working hours by weekday')
if show_fig5:
    st.markdown("### Working hours by weekday")
    st.pyplot(fig5)

# """
# Working hours by task
# """
# plot pie chart - time spent on each task type
fig6, axes = plt.subplots(figsize=(15, 5), ncols=2)
dat_st.groupby('Type')['SpentTime/h'].sum().plot.pie(autopct='%1.1f%%', ax=axes[0])
dat_st.groupby('Manuscript')['SpentTime/h'].sum().plot.pie(autopct='%1.1f%%', ax=axes[1])
# plt.suptitle('Working hours by task')
if show_fig6:
    st.markdown("### Working hours by task")
    st.pyplot(fig6)

# """
# Components of tasks
# """
fig7, axes = plt.subplots(figsize=(15, 5), ncols=2)
# pie chart of the Task Type 
dat_task['Type'].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[0])
# pie chart of manuscripts
dat_task['Manuscript'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[1])
plt.tight_layout()
# plt.suptitle('Task Type and Manuscript')
if show_fig7:
    st.markdown("### Task Type and Manuscript")
    st.pyplot(fig7)

# """
# Progress by date
# """
dat_state = dat_task[["Date", "Weekday", "TaskIndex", "Progress/%"]]

# replace nan with 0
dat_state = dat_state.replace(np.nan, 0)

# remove % sign in the column 'Progress/%'
# dat_state['Progress/%'] = dat_state['Progress/%'].str.replace('%', '')

# convert the column 'Progress/%' to float
# dat_state['Progress/%'] = dat_state['Progress/%'].astype(float)

# plot the accomplish state of each day as bars
fig8, ax = plt.subplots(figsize=(15, 5))
dat_state_by_date = dat_state[["Date", "Progress/%"]].groupby("Date").sum().reset_index()
dat_state_by_date[["Date", "Progress/%"]].plot(x="Date", y=["Progress/%"], kind="bar", rot=90, ax=ax, color='olive')
plt.ylabel('Progress/% (%)')
# plot two horizontal lines indicate 100% and 200%
plt.axhline(y=100, color='r', linestyle='-')
plt.axhline(y=200, color='r', linestyle='-')
# plt.title('Progress by date')
if show_fig8:
    st.markdown("### Progress by date")
    st.pyplot(fig8)


# """
# Progress by weekday
# """
# fig5, ax = plt.subplots(figsize=(15, 5))
fig9 = sns.catplot(x="Weekday", y="Progress/%", hue="TaskIndex", kind="bar", data=dat_state, ci=68, height=6, aspect=1.5)
# plt.title('Progress by weekday')
if show_fig9:
    st.markdown("### Progress by weekday")
    st.pyplot(fig9)

# """
# Lingering tasks
# """
if show_fig10:
    st.markdown("### 'Lingering' tasks")
    st.dataframe(dat_task.groupby(['Manuscript', 'Type'])['Task'].value_counts().reset_index(name='count').sort_values(by='count', ascending=False))

# """
# Word counts by date
# """
# bar plot wordcount of each day
fig11, ax = plt.subplots(figsize=(15, 5))
dat_raw_wc = dat[["Date", "WordCount"]].groupby("Date").sum().reset_index().sort_values(by='Date')
dat_raw_wc[["Date", "WordCount"]].plot(x="Date", y=["WordCount"], kind="bar", rot=90, ax=ax)
plt.axhline(y=dat_raw_wc['WordCount'].mean(), color='gray', linestyle='--', label='monthly mean')
plt.axhline(y=dat_raw_wc.query("WordCount!=0")['WordCount'].mean(), color='black', linestyle='--', label='writing days mean')
plt.ylabel('Word Count')
plt.legend()
# plt.title('Word counts by date')
if show_fig11:
    st.markdown("### Word counts by date")
    st.pyplot(fig11)

# """
# Word counts by weekday
# """
# fig8, ax = plt.subplots(figsize=(15, 5))
fig12 = sns.catplot(x="Weekday", y="WordCount", kind="bar", data=dat, ci=95, height=5, aspect=3)
# plt.title('Word counts by weekday')
if show_fig12:
    st.markdown("### Word counts by weekday")
    st.pyplot(fig12)




