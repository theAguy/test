import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#######################################################################################################################

@st.cache
def load_df():
    # Make sure to place the right path to you csv
    df = pd.read_csv('C:/Users/DELL/crime.csv', encoding='unicode_escape')
    df = df.rename(columns={
        'Lat': 'lat',
        'Long': 'lon'
    })
    df = df.dropna(subset=['lat', 'lon'])

    np.random.seed(40001)

    coords_mask = df['lat'] > 40
    return df[coords_mask].sample(n=20000)

df = load_df()

#######################################################################################################################
# Q1
#######################################################################################################################
st.title('Welcome to CSI Boston site')

st.subheader('Map of all crimes')
hour_to_filter = st.slider('Hour of crime', 0, 23)
filtered_data = df[df['HOUR']==hour_to_filter]
st.subheader(f'Map of all crimes at {hour_to_filter}:00')
st.map(filtered_data)
#######################################################################################################################
# Q2
#######################################################################################################################
st.subheader('Top 10 offenses')
st.set_option('deprecation.showPyplotGlobalUse', False)
## finding the 10th largest offenses
df_offense = df.groupby(['OFFENSE_CODE_GROUP']).size().sort_values(ascending=False)
offense_list = df_offense.keys()
filtered_offense_list = offense_list[:10]

df_offense = df.groupby(['OFFENSE_CODE_GROUP']).size().sort_values(ascending=False)
offense_list = df_offense.keys()
filtered_offense_list = offense_list[:10]

df_filtered_offense = df.loc[df['OFFENSE_CODE_GROUP'].isin(filtered_offense_list)]
df_filtered_offense_to_plot = df_filtered_offense.groupby(['OFFENSE_CODE_GROUP','DAY_OF_WEEK']).size()

dd = df_filtered_offense_to_plot.to_frame()
dd = dd.reset_index()

cols = st.selectbox('Choose offense', filtered_offense_list)
filtered_data_offense = dd[dd['OFFENSE_CODE_GROUP']==cols]
ax = sns.barplot(x="DAY_OF_WEEK", y=0, data=filtered_data_offense)
ax.set(xlabel='Day in the week', ylabel='Number of offenses')
st.pyplot()

#######################################################################################################################
# Q3
#######################################################################################################################

st.subheader('Serious crimes by district by year')

serious_crimes = ['Larceny', 'Robbery', 'Drug Violation', 'Auto Theft']
df_serious = df.loc[df['OFFENSE_CODE_GROUP'].isin(serious_crimes)]
district_list = df_serious['DISTRICT'].unique()
district_list = district_list.tolist()
district_list = [district for district in district_list if str(district) != 'nan']
select = st.selectbox('Choose district', district_list)
filtered_data_district = df_serious[df_serious['DISTRICT'].isin(district_list)]



#handling 2016
df_serious_2016 = filtered_data_district.loc[filtered_data_district['YEAR']==2016]
df_serious_2016_grouped = df_serious_2016.groupby(['DISTRICT','MONTH','OFFENSE_CODE_GROUP']).count()
df_serious_2016_grouped = df_serious_2016_grouped.groupby(['DISTRICT','MONTH']).INCIDENT_NUMBER.mean()
df_serious_2016_grouped = df_serious_2016_grouped.to_frame()
df_serious_2016_grouped = df_serious_2016_grouped.reset_index()
df_serious_2016_grouped["MONTH"] = pd.to_numeric(df_serious_2016_grouped["MONTH"], errors='coerce')
df_serious_2016_grouped_for_select = df_serious_2016_grouped[df_serious_2016_grouped['DISTRICT']==select]


#handling 2017
df_serious_2017 = filtered_data_district.loc[filtered_data_district['YEAR']==2017]
df_serious_2017_grouped = df_serious_2017.groupby(['DISTRICT','MONTH','OFFENSE_CODE_GROUP']).count()
df_serious_2017_grouped = df_serious_2017_grouped.groupby(['DISTRICT','MONTH']).INCIDENT_NUMBER.mean()
df_serious_2017_grouped = df_serious_2017_grouped.to_frame()
df_serious_2017_grouped = df_serious_2017_grouped.reset_index()
df_serious_2017_grouped["MONTH"] = pd.to_numeric(df_serious_2017_grouped["MONTH"], errors='coerce')
df_serious_2017_grouped_for_select = df_serious_2017_grouped[df_serious_2017_grouped['DISTRICT']==select]



#plotting
fig, axes = plt.subplots(1, 2, figsize=(10, 2))
ax1 = sns.barplot(x=df_serious_2016_grouped["MONTH"], y='INCIDENT_NUMBER', data=df_serious_2016_grouped_for_select, ax = axes[0])
ax1.set(xlabel='Month in the year', ylabel='Avg serious offense in 2016')

ax2 = sns.barplot(x=df_serious_2017_grouped["MONTH"], y='INCIDENT_NUMBER', data=df_serious_2017_grouped_for_select, ax = axes[1])
ax2.set(xlabel='Month in the year', ylabel='Avg serious offense in 2017')

st.pyplot()
