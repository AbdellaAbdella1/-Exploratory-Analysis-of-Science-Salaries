#!/usr/bin/env python
# coding: utf-8

# # Abdella Abdella Cambridge Spark Bootcamp Project Description 
# 
# Project Title: Exploratory Analysis of Science Salaries
# 
# Description:
# This project aims to perform an exploratory analysis of science salaries using data obtained from Kaggle. The dataset was collected on 02/03/2024 and provides insights into the salaries of professionals in the field of data science. The dataset is sourced from Kaggle and can be accessed through the following link: [Science Salaries Dataset](https://www.kaggle.com/datasets/zain280/data-science-salaries).
# 
# Cambridge Spark, a prominent education institution specializing in data science and machine learning bootcamps, undertook this project to understand the trends and factors influencing salaries in the science domain. The dataset offers a variety of features such as job titles, years of experience, educational qualifications, and salary information. Through this analysis, we aim to uncover patterns, correlations, and insights that can provide valuable information to professionals, job seekers, and employers in the science and data science sectors.
# 
# The analysis will involve data cleaning, exploratory data analysis (EDA), visualization, and possibly machine learning modeling to predict salary trends based on various factors. By leveraging Python programming and data analysis libraries such as Pandas, NumPy, Matplotlib, and Seaborn, we will delve into the dataset to extract meaningful insights and present them in a clear and interpretable manner.
# 
# This project not only serves as a learning experience for participants of the Cambridge Spark bootcamp but also contributes to the broader understanding of salary dynamics within the science and data science domains. The findings and conclusions drawn from this analysis can potentially inform hiring practices, salary negotiations, and career decisions within the industry.

# # Data-Science-Salaries-2023
# Salaries of Different Data Science Fields in the Data Science Domain

# In[43]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.express as px
import seaborn as sns


# In[44]:


# Load the dataset.
df = pd.read_csv("ds_salaries.csv")


# In[46]:


df


# In[7]:


# Display the statistical summary of the DataFrame.
df.describe()


# In[8]:


# Display information about the DataFrame.
df.info()


# In[9]:


# Display the number of unique values for each column.
df.nunique()


# In[10]:


# Transformation of the codes of the categorical variables

df['experience_level'] = data['experience_level'].replace({'SE': 'Expert', 'MI': 'Intermediate', 'EN': 'Junior', 'EX': 'Director'})

df['employment_type'] = data['employment_type'].replace({'FT': 'Full-time', 'CT': 'Contract', 'FL': 'Freelance', 'PT': 'Part-time'})

def country_name(country_code):
    try:
        return pycountry.countries.get(alpha_2=country_code).name
    except:
        return 'other'
    
df['company_location'] = data['company_location'].apply(country_name)
df['employee_residence'] = data['employee_residence'].apply(country_name)


# In[14]:


# Categorical variables

for column in ['work_year','experience_level','employment_type','company_size','remote_ratio','job_title','company_location']:
    print(df[column].unique())


# In[28]:


# Extract the "job title" column
job_titles = df['job_title']

# Calculate the frequency of each job title
title_counts = job_titles.value_counts()

# Extract the top 20 most frequent job titles
top_20_titles = title_counts.head(20)

# Create a DataFrame for the top 20 titles
top_20_df = pd.DataFrame({'Job Title': top_20_titles.index, 'Count': top_20_titles.values})

# Plotting the count plot
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")
ax = sns.barplot(data=top_20_df, x='Count', y='Job Title', palette='seismic')
plt.xlabel('Count')
plt.ylabel('Job Titles')
plt.title('Top 20 Most Frequent Job Titles')

# Add count labels to the bars
for i, v in enumerate(top_20_df['Count']):
    ax.text(v + 0.2, i, str(v), color='blue', va='center')

plt.tight_layout()
plt.show()


# In[26]:


# Calculate the number of individuals in each experience level
level_counts = df['experience_level'].value_counts()

# Create a pie chart
plt.figure(figsize=(7,12),dpi=80)
plt.pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%')
plt.title('Experience Level Distribution')

plt.show()


# In[27]:


# Create a cross-tabulation of the two columns
cross_tab = pd.crosstab(data['experience_level'], data['company_size'])

# Create a heatmap using the cross-tabulation data
plt.figure(figsize=(10, 8))
sns.heatmap(cross_tab, annot=True, fmt="d", cmap='seismic')

plt.xlabel('Company Size')
plt.ylabel('Experience Level')
plt.title('Relationship between Experience Level and Company Size')

plt.show()


# In[33]:


# Create bar chart
average_salary = data.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False)
top_ten_salaries = average_salary.head(10)

plt.figure(figsize=(15,10),dpi=80)
plt.bar(top_ten_salaries.index, top_ten_salaries)

# Add labels to the chart
plt.xlabel('Job')
plt.ylabel('Salary $')
plt.title('Average of the ten highest salaries by Job Titles')
plt.xticks(rotation=35, ha='right')
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

plt.show()


# In[39]:


# Create bar chart
average_salary = data.groupby('company_location')['salary_in_usd'].mean().sort_values(ascending=False)
top_ten_countries = average_salary.head(10)

plt.figure(figsize=(15,10),dpi=80)
plt.bar(top_ten_countries.index, top_ten_countries)

# Add labels to the chart
plt.xlabel('Country')
plt.ylabel('Salary $')
plt.title('Average of the ten highest salaries by country')
plt.xticks(rotation=20, ha='right')
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

plt.show()


# In[36]:


common_jobs = ['Data Engineer', 'Data Scientist', 'Data Analyst', 'Machine Learning Engineer', 'Analytics Engineer','Research Scientist', 'Data Science Manager', 'Applied Scientist']
common_jobs = data[data['job_title'].isin(common_jobs)]


# In[37]:


salary_common_jobs = common_jobs.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False)
remote_common_jobs = common_jobs.groupby('job_title')['remote_ratio'].mean().sort_values(ascending=False)
salary_common_country = common_jobs.groupby('company_location')['salary_in_usd'].mean().sort_values(ascending=False)


# In[38]:


plt.figure(figsize=(15,10),dpi=80)
plt.bar(salary_common_jobs.index, salary_common_jobs)

# Add labels to the chart
plt.xlabel('Job')
plt.ylabel('Salary $')
plt.title('Average salary for common Job Titles')
plt.xticks(rotation=20, ha='right')
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

plt.show()


# In[40]:


# Create bar chart
remote_common_jobs = common_jobs.groupby('job_title')['remote_ratio'].mean().sort_values(ascending=False)

plt.figure(figsize=(15,10),dpi=80)
plt.bar(remote_common_jobs.index, remote_common_jobs)

# Add labels to the chart
plt.xlabel('Job')
plt.ylabel('% remote')
plt.title('Remote rate by Job Titles')
plt.xticks(rotation=20, ha='right')
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

plt.show()


# In[41]:


# Create bar chart
salary_common_country = common_jobs.groupby('company_location')['salary_in_usd'].mean().sort_values(ascending=False)

plt.figure(figsize=(15,10),dpi=80)
plt.bar(salary_common_country.head(10).index, salary_common_country.head(10))

# Add labels to the chart
plt.xlabel('Country')
plt.ylabel('Salary $')
plt.title('Average of the 10 highest salaries of common jobs by country')
plt.xticks(rotation=20, ha='right')
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

plt.show()


# In[34]:


salary_by_country = data.groupby('company_location', as_index=False)['salary_in_usd'].mean()

fig = px.choropleth(salary_by_country,locations='company_location',locationmode='country names',color='salary_in_usd',
                    projection='equirectangular',hover_name='company_location',
                    labels={'salary_in_usd':'Average Salary in USD'},title='Distribution of average salary by company location')


fig.show("notebook")


# In[ ]:




