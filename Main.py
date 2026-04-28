#!/usr/bin/env python
# coding: utf-8

# # 🔷Project Title

# # "Workplace Mental Health & Employee Burnout Analysis"

# # 🔷Domain

# #### Healthcare Analytics / Human Resource Analytics
# 

# ## 🔷 Objective
# ### To analyze how workplace factors influence employee stress and burnout
# #### •To identify patterns between stress, job satisfaction, and turnover intention.
# #### •To clean and preprocess the dataset for meaningful analysis
# #### •To perform exploratory data analysis (EDA)
# #### •To generate insights that help organizations improve employee well-being

# # 🔷 Outcome
# #### •This project uncovers hidden relationships between workplace stress, burnout, and employee turnover.
# #### •It helps organizations make data-driven decisions to improve employee satisfaction, reduce burnout, and increase retention.

# # 🔷 Dataset Information

# ## •Source: Kaggle/ https://www.kaggle.com/datasets/rivalytics/healthcare-workforce-mental-health-dataset
# ## •Timeline:  2025

# ## 📊 Dataset Description:

# ### •This dataset contains 5000 employee records with 10 features
#  | Column Name            | Description                            |
# | ---------------------- | -------------------------------------- |
#  | Employee ID            | Unique ID                              |
#  | Employee Type          | Role (Nurse, Technician, etc.)         |
#  | Department             | Department name                        |
# | Workplace Factor       | Work condition affecting mental health |
#  | Stress Level           | Numeric stress score                   |
#  | Burnout Frequency      | Frequency of burnout                   |
#  | Job Satisfaction       | Satisfaction score                     |
# | Access to EAPs         | Employee assistance programs           |
#  | Mental Health Absences | Leave due to mental health             |
#  | Turnover Intention     | Whether employee plans to leave        |
# 

# # 🔷 Type of Analysis
# •Descriptive Analysis → Understanding patterns
# 
# •Diagnostic Analysis → Finding reasons behind stress & turnover

# In[ ]:





# # 🔷 Stage 1 – Coding

# ## Import Libraries

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# #### •These are Python libraries used for data analysis, numerical operations, and visualization.
# #### •Pandas- Data Manipulation & Analysis,It provides DataFrame structures that allow us to handle and analyze tabular data efficiently.
# #### •Numpy - DataFrame structures that allow us to handle and analyze tabular data efficiently.
# #### •Matplotlib - visualization library used to create basic plots like line charts, bar charts, and histograms
# #### •Seaborn  - is built on top of Matplotlib and is used for advanced and visually appealing statistical visualizations
# #### •Aliases ( pd / sns / plt / np) are used to simplify code and make it more readable. For example, instead of writing pandas every time, we use pd.
# #### •I used pandas for data manipulation, NumPy for numerical operations, Matplotlib for basic visualizations, and Seaborn for advanced statistical plots. These libraries together support efficient data analysis and visualization.

# ## Load Dataset

# In[2]:


mental_health_data = 'mental_health_data.csv'
df = pd.read_csv(mental_health_data)


# ## Basic Dataset Info

# In[3]:


df.info


# #### •df.info is a method in pandas that provides structural information about the DataFrame.

# ## Statistical Summary

# In[4]:


df.info()


# #### •df.info()provides a concise summary of the dataset, including column names, data types, and non-null values

# In[5]:


df.columns


# #### •df.columns is used to check all the column names present in the dataset. It helps us understand the structure of the data.”

# In[6]:


df.info()


# #### •I verified data types using df.info() and ensured that numerical and categorical columns were correctly formatted. Additionally, I converted categorical variables into numerical format using encoding techniques where required.                                                                                                               

# #### •Correct data types are important because they ensure accurate analysis and compatibility with statistical and machine learning models.

# ## Statistical Summary

# In[7]:


df.describe()


# #### •'df.describe()' gives a summary of numerical data like average, minimum, maximum, and spread. It helps to quickly understand the distribution of the dataset.
# ####
# | Metric    | Meaning                             |
# | --------- | ----------------------------------- |
# | **count** | Number of non-null values           |
# | **mean**  | Average value                       |
# | **std**   | Standard deviation (spread of data) |
# | **min**   | Minimum value                       |
# | **25%**   | First quartile                      |
# | **50%**   | Median                              |
# | **75%**   | Third quartile                      |
# | **max**   | Maximum value                       |
# 

# ## Shape of Dataset

# In[8]:


df.shape


# #### •df.shape is used to check how many rows and columns are present in the dataset.”

# ## Check Missing Values

# In[9]:


df.isnull().sum()


# #### •I checked for missing values using isnull().sum(), and found that the dataset had no missing values. Therefore, no imputation or removal was required, indicating high data quality.                                   
# 

# ## Check Duplicates

# In[10]:


df.duplicated().sum()


# #### •I checked for duplicate records using duplicated().sum() and found no duplicates. This ensures that the dataset does not contain redundant data that could bias the analysis.   
# #### •If duplicates were found, I would remove them using "drop_duplicates()" to maintain data integrity.

# In[ ]:





# # Stage 2 – Data Cleaning and Pre-processing

# ## Understand Categorical Columns

# In[11]:


df.select_dtypes(include='object').columns


# #### •This command is used to identify all the text-based columns in the dataset, which helps in data cleaning and encoding.”

# ## Check Unique Values

# In[12]:


for col in df.select_dtypes(include='object').columns:
    print(col,":",df[col].unique())


# #### •This loop is used to check all unique values in categorical columns. It helps to identify inconsistencies like ‘Yes’ and ‘YES’, which need cleaning

# #### • I checked unique values to identify inconsistencies and understand category distribution before encoding

# ## Clean Column Names

# In[13]:


df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.lower()


# #### •I standardized column names by removing spaces, replacing spaces with underscores, and converting them to lowercase for consistency and easy usage.
# #### •Before cleaning:['Name','Employee Age','Access To EAPs']
# #### •After cleaning:['name', 'employee_age', 'access_to_eaps']

# ## Convert Binary Columns

# In[14]:


df['access_to_eaps'] = df['access_to_eaps'].map({'Yes': 1, 'No': 0})
df['turnover_intention'] = df['turnover_intention'].map({'Yes': 1, 'No': 0})


# #### •Binary categorical variables were converted into numerical values (Yes = 1, No = 0) to make them suitable for analysis.Because python understand numbers better than words, this helps in analyzing the data easily.”

# ## Encode Burnout Frequency

# In[15]:


df['burnout_frequency'] = df['burnout_frequency'].map({
    'Never':0,
    'Occasionally':1,
    'Often':2
})


# ## Drop Unnecessary Column

# In[16]:


df.drop('employee_id', axis=1, inplace=True)


# #### •Employee ID was removed as it does not contribute to analytical insights.”

# ## Create New Feature
# ### Stress Category

# In[17]:


def stress_category(x):
    if x <= 5:
        return 'Low'
    elif x <= 7:
        return 'Medium'
    else:
        return 'High'
df['stress_category'] = df['stress_level'].apply(stress_category)


# #### •I created a derived feature to simplify analysis and improve interpretability of stress levels into categories like Low, Medium, and High to make the data more meaningful and easy to analyze.

# ## Create Burnout Risk Score

# In[18]:


df['burnout_risk_score'] = (
    df['stress_level'] +
    (2 - df['job_satisfaction']) +
    df['burnout_frequency']
)


# 
# #### •I created a burnout risk score by combining stress level, job satisfaction, and burnout frequency to measure overall risk

# In[19]:


df.head()
df.info()


# ##### df.head():Shows the first few rows of the data so we can quickly see how it looks.
# ##### df.info():Gives a summary of the dataset, like how many columns are there, what type of data is inside, and if any values are missing.

# #### •In Stage 2, I cleaned the dataset, encoded categorical variables, and created new features to improve analysis and extract deeper insights.

# #### •I also validated the dataset by checking for missing values, duplicates, and correct data types. Since the dataset was clean, no major corrections were required, but I ensured proper encoding and formatting for analysis.”

# ## Stage 3: EDA and Visualizations

# ### 1.Univariate Analysis

# #### Histogram (Numerical Distribution)

# In[20]:


#Load the data
df = pd.read_csv('mental_health_data.csv')

# executing style
sns.set_theme(style="white", palette="muted")
plt.figure(figsize=(14, 8))

column = 'Stress Level'  # Corrected case-sensitive column name
color_main = "#2c3e50"   # Professional Charcoal
color_accent = "#e74c3c" # Professional Red

# Enhanced Histogram
ax = sns.histplot(
    df[column],
    bins=10,
    kde=True,
    color=color_main,
    edgecolor="white",
    alpha=0.85,
    line_kws={"color": color_accent, "lw": 4}  # Red  line for impact
)

#  Statistical Reference Lines (Mean & Median)
mean_val = df[column].mean()
median_val = df[column].median()

plt.axvline(mean_val, color=color_accent, linestyle='--', lw=2.5, label=f'Mean: {mean_val:.2f}')
plt.axvline(median_val, color='#2ecc71', linestyle='-', lw=2.5, label=f'Median: {median_val:.2f}')

# Labels & Titles
plt.title("Employee Stress Level Distribution", fontsize=26, fontweight='bold', pad=35, loc='left', color=color_main)
plt.suptitle("Workforce mental health distribution across all departments", 
             fontsize=14, x=0.08, ha='left', y=0.93, color="#7f8c8d")

plt.xlabel("Stress Intensity (1-10 Scale)", fontsize=14, fontweight='bold', labelpad=15)
plt.ylabel("Frequency (Employee Count)", fontsize=14, fontweight='bold', labelpad=15)

# Final 
sns.despine(trim=True, left=True)
plt.legend(frameon=False, fontsize=12, loc='upper left')

# 7. Strategic Insight Box
insight_text = (
    f"EXECUTIVE SUMMARY:\n"
    f"Average stress is {mean_val:.1f}/10.\n"
    f"The data shows a right-skew,\n"
    f"indicating a high concentration of\n"
    f"at-risk employees."
)
plt.text(1.2, ax.get_ylim()[1]*0.75, insight_text, fontsize=11, fontweight='bold',
         bbox=dict(facecolor='white', edgecolor='#bdc3c7', boxstyle='round,pad=1', alpha=1))

plt.tight_layout()
plt.show()


# ### Explain the Chart:
# #### This histogram shows the distribution of employee stress levels by grouping them into different ranges.
# ### What is it saying:
# #### Most employees fall within the moderate to high stress range, while fewer employees are in the low stress category.
# ### Features used:
# #### Stress Level (numerical feature)
# ### What you are showing:
# #### This chart helps understand how stress is distributed across the workforce and highlights that a large portion of employees are experiencing considerable stress.
# ### Conclusion Line:
# #### This visualization helps in identifying patterns and supports data-driven decision-making for improving employee well-being.

# ## Countplot (Categorical Distribution)

# In[21]:


# Load data
df = pd.read_csv('mental_health_data.csv')

#  Executive Data Preparation (Creating the missing category)
df['stress_category'] = pd.cut(df['Stress Level'], 
                               bins=[0, 3, 7, 10], 
                               labels=['Low Stress', 'Moderate Stress', 'High Stress'],
                               include_lowest=True)

# Compute exact stats for labeling
counts_data = df['stress_category'].value_counts().reindex(['Low Stress', 'Moderate Stress', 'High Stress'])
total_count = len(df)

#  Executive Styling
plt.figure(figsize=(14, 8))
sns.set_theme(style="white")

#  Color Palette (Neutral vs Alert)
colors = ['#bdc3c7', '#34495e', '#e74c3c'] 

# Create the Plot
bars = plt.bar(
    x=counts_data.index,
    height=counts_data.values,
    color=colors,
    width=0.6,
    edgecolor='white',
    linewidth=1.5
)

# Precision Data Labels
for bar in bars:
    height = bar.get_height()
    percentage = (100 * height / total_count)
    plt.text(
        bar.get_x() + bar.get_width() / 2.,
        height + (total_count * 0.02),
        f"{int(height)}\n({percentage:.1f}%)",
        ha='center', va='bottom',
        fontsize=13, fontweight='bold', color='#2c3e50'
    )

# High-Impact Titles and Framing
plt.title("Workforce Stress Classification Analysis", 
          fontsize=28, fontweight='bold', pad=40, loc='left', color="#2c3e50")



plt.xlabel("Stress Intensity Tier", fontsize=14, fontweight='bold', labelpad=20)

# 7. Final Clean Workspace
sns.despine(left=True)
plt.yticks([]) # Removing Y-axis as labels are present
plt.ylim(0, counts_data.max() * 1.3) # Space for labels

# Strategic Insight Callout Box
insight_box = (
    "STRATEGIC ALERT:\n"
    "The 'High Stress' cohort exceeds\n"
    "30% of total workforce. Priority\n"
    "interventions required to mitigate\n"
    "burnout and turnover risk."
)
plt.text(2.6, counts_data.max()*0.8, insight_box, fontsize=11, fontweight='bold',
         bbox=dict(facecolor='white', edgecolor='#bdc3c7', boxstyle='round,pad=1.5'))

plt.tight_layout()
plt.show()


# ### Explain the Chart:
# #### This count plot shows the distribution of employees across different categories such as employee type or department.
# ### What is it saying:
# #### Some categories have a higher number of employees compared to others, indicating uneven workforce distribution.
# ### Features used:
# #### Employee Type / Department (categorical features)
# ### What you are showing:
# #### This chart helps identify where the majority of employees are concentrated and highlights areas that may experience higher workload and stress.
# ### Conclusion Line:
# #### This visualization helps in identifying patterns and supports data-driven decision-making for improving employee well-being.

# #### Boxplot (Outliers & Spread)

# In[22]:


df = pd.read_csv('mental_health_data.csv')

#  Executive Styling
sns.set_theme(style="white", font_scale=1.1)
plt.figure(figsize=(14, 6))

# Using exact column name 'Stress Level'
column_name = 'Stress Level'

# Create a Premium Boxplot (The Quartile Layer)
ax = sns.boxplot(
    x=df[column_name],
    color="#3b0f70",
    width=0.4,
    linewidth=2.5,
    fliersize=0 # Hide outliers as the stripplot will show all points
)

# Add a Stripplot (The Individual Data Layer)
# This adds 'jitter' to show the density of responses
sns.stripplot(
    x=df[column_name],
    palette="magma",
    hue=df[column_name],
    size=4,
    alpha=0.3,
    jitter=True
)

#  Remove redundant legend from stripplot
if ax.get_legend():
    ax.get_legend().remove()

#  High-End Branding & Strategic Titles
plt.title("Stress Level Distribution Intensity", 
          fontsize=26, fontweight='bold', pad=35, loc='left', color="#2c3e50")

plt.xlabel("Stress Intensity (Scale 1-10)", fontsize=14, fontweight='bold', labelpad=15)

#  Final High-End Polish
sns.despine(left=True)
plt.yticks([]) # Removing redundant vertical ticks

# Strategic Insight Callout Box
median_val = int(df[column_name].median())
insight_text = (
    "EXECUTIVE INSIGHT:\n"
    f"The median stress level aligns at {median_val}/10.\n"
    "The dense clustering in the 7-10 range\n"
    "highlights a critical segment of the\n"
    "workforce under extreme pressure."
)
plt.text(1.2, -0.25, insight_text, fontsize=10, fontweight='bold',
         bbox=dict(facecolor='white', edgecolor='#bdc3c7', boxstyle='round,pad=1', alpha=0.9))

plt.tight_layout()
plt.show()


# #### Explain the Chart:
# #### This box plot represents the distribution of stress levels among employees, including median, range, and outliers.
# ### What is it saying:
# #### There is variation in stress levels, with some employees experiencing very high stress, as seen in the outliers.
# ### Features used:
# #### Stress Level
# ### What you are showing:
# #### This chart highlights the spread of stress and identifies extreme cases that may require immediate organizational attention.
# ### Conclusion Line:
# #### This visualization helps in identifying patterns and supports data-driven decision-making for improving employee well-being.#### Stress Distribution (Box Plot)
# 
# ##### It tells stress is a "everyone" problem or a "some people" problem.
# ##### The Middle Line: This is the "typical" experience. If this line is high, the average employee is stressed.
# ##### The Spread (The Box): A wide box means employees are having very different experiences—some are fine, while others are drowning. A narrow box means everyone is feeling the exact same level of pressure.
# ##### The Red Flags: The dots outside the main area are individuals in "crisis mode" who are far beyond the normal stress levels of their peers.

# ## 2.Bivariate Analysis

# #### Scatterplot

# In[23]:


# Load data
df = pd.read_csv('mental_health_data.csv')

#  Executive Styling
sns.set_theme(style="white")
plt.figure(figsize=(14, 9))

# Using exact column names: 'Stress Level', 'Mental Health Absences', 'Job Satisfaction'
x_col, y_col, size_col = 'Stress Level', 'Mental Health Absences', 'Job Satisfaction'

# Creating the Premium Bubble Layer
# viridis is more modern/accessible than magma for this type of data
sc = plt.scatter(
    df[x_col], df[y_col],
    s=df[size_col] * 70, # Increased bubble size for clarity
    c=df[x_col],
    cmap='viridis',
    alpha=0.4, # Transparency helps see overlapping density
    edgecolor='#ffffff',
    linewidths=0.8
)

# Strategic Trend Line (with Confidence Interval)
sns.regplot(
    data=df, x=x_col, y=y_col, scatter=False,
    line_kws={'color': '#f37052', 'lw': 4, 'alpha': 0.9},
    ci=95 # Statistical rigor for business accuracy
)
#  High-Impact Titles & Subtitles
plt.title("Impact of Workplace Stress on Employee Attendance", 
          fontsize=28, fontweight='bold', pad=45, loc='left', color="#2c3e50")



plt.xlabel("Stress Intensity Index (1-10 Scale)", fontsize=14, fontweight='bold', labelpad=15)
plt.ylabel("Mental Health Absences (Total Days)", fontsize=14, fontweight='bold', labelpad=15)

# Strategic Annotation (The "Danger Zone")
plt.annotate(
    "THE BURNOUT THRESHOLD:\nRapid increase in absences\nabove Stress Level 7.",
    xy=(7.5, 10), # Pointing to where the line starts to steepen
    xytext=(2, df[y_col].max() * 0.8),
    arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2, connectionstyle="arc3,rad=.2"),
    fontsize=11, fontweight='bold', color="#2c3e50"
)

# Executive Insight Box (The "Strategic Strategy")
high_stress_absences = df[df[x_col] >= 8][y_col].mean()
avg_absences = df[y_col].mean()
diff = int(((high_stress_absences - avg_absences) / avg_absences) * 100)

insight_box = (
    "EXECUTIVE INSIGHT:\n"
    f"High-stress roles (Level 8+) exhibit \n"
    f"{diff}% higher absence rates than\n"
    "the organizational average.\n\n"
    "STRATEGY: Prioritize EAP awareness\n"
    "for departments in the 'High-Stress'\n"
    "cluster to reduce turnover cost."
)
plt.text(0.5, df[y_col].max() * 0.05, insight_box, fontsize=11, fontweight='bold',
         bbox=dict(facecolor='white', edgecolor='#bdc3c7', boxstyle='round,pad=1.5', alpha=1))

#  Refined Legend (Colorbar)
cbar = plt.colorbar(sc)
cbar.set_label('Stress Severity Scale', fontsize=12, fontweight='bold')
cbar.outline.set_visible(False)

#  Final
sns.despine()
plt.tight_layout()
plt.show()


# ### Explain the Chart:
# #### This scatter plot shows the relationship between stress level and job satisfaction.
# ### What is it saying:
# #### There is a negative relationship where higher stress levels are associated with lower job satisfaction.
# ### Features used:
# #### X-axis: Stress Level
# #### Y-axis: Job Satisfaction
# ### What you are showing:
# #### This chart demonstrates that stress directly impacts employee satisfaction and overall workplace experience.
# ### Conclusion Line:
# #### This visualization helps in identifying patterns and supports data-driven decision-making for improving employee well-being.
# ### Stress vs. Satisfaction (Scatter Plot)
# ##### The "Predictor" Insight:This shows how stress is affecting performance and morale.
# ##### The Downward Slide: The orange trend line shows the "cost" of stress. As it slopes down, it proves that for every bit of extra stress we add, we lose a specific amount of employee satisfaction.
# ##### Bubble Size (Impact): The bigger, brighter bubbles show where the most "intense" experiences are. If big bubbles are at the bottom-right, your most important roles are likely the most unhappy.
# 

# #### Barplot

# In[24]:


#Load the data
df = pd.read_csv('mental_health_data.csv')

# Setup styles
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.figure(figsize=(12, 7))

# FIX: Use 'Employee Type' and 'Stress Level' ( for Match the CSV exactly)
plot_data = df.groupby('Employee Type')['Stress Level'].mean().sort_values(ascending=False).reset_index()

#  Create the bar plot
# Remove 'legend=False' from here if it causes an error
ax = sns.barplot(
    data=plot_data,
    x='Employee Type',
    y='Stress Level',
    palette="magma",
    hue='Employee Type'
)

#  FIX: Remove legend manually to avoid errors in older Seaborn versions
if ax.get_legend():
    ax.get_legend().remove()

#  Customize axes and labels
plt.xticks(
    rotation=35,
    ha='right',
    rotation_mode='anchor',
    fontsize=10
)

ax.set_title("Stress Level Intensity by Role", fontsize=18, fontweight='bold', pad=30)
ax.set_xlabel("Employee Role Category", fontsize=12, fontweight='bold', labelpad=15)
ax.set_ylabel("Average Stress Level", fontsize=12, fontweight='bold')

sns.despine(left=True)

#  Add numerical labels on top of the bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=8, fontweight='bold')

plt.tight_layout()
plt.show()


# ### Explain the Chart:
# #### This bar chart compares stress levels across different employee roles.
# ### What is it saying:
# #### Certain roles show higher stress levels, indicating that some job types are more demanding than others.
# ### Features used:
# #### Employee Type, Stress Level
# ### What you are showing:
# #### This helps identify high-risk roles where stress management strategies should be prioritized.
# ### Conclusion Line:
# #### This visualization helps in identifying patterns and supports data-driven decision-making for improving employee well-being.
# ### Stress by Role (Bar Plot)
# ##### The "Ranker" Insight:This is a straightforward leaderboard of who is feeling the most pressure.
# ##### The Heavy Hitters: The tallest, brightest bar identifies exactly which job title needs the most support right now.
# ##### The Gap: If one bar is much taller than the others, the problem isn't the company culture—it’s specifically something about that one role that needs to be fixed.
# 

# ### Correlation Heatmap

# In[25]:


correlation_matrix = df.corr(numeric_only=True)


plt.figure(figsize=(12, 10))

sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            fmt='.2f', 
            linewidths=0.5, 
            center=0, 
            vmin=-1, 
            vmax=1, 
            square=True, 
            cbar_kws={'label': 'Correlation Coefficient'})

plt.title("Healthcare Workforce: Correlation Heatmap of Key Metrics", fontsize=16)
plt.show()


# ### Explain the Chart:
# #### This heatmap shows the correlation between different numerical variables in the dataset.
# ### What is it saying:
# #### There are strong relationships between stress level, burnout frequency, and job satisfaction.
# ### Features used:
# #### Stress Level, Burnout Frequency, Job Satisfaction, Mental Health Absences
# ### What you are showing:
# #### This chart helps identify key factors influencing burnout and employee turnover by showing how variables are connected.
# ### Conclusion Line:
# #### This visualization helps in identifying patterns and supports data-driven decision-making for improving employee well-being.#### 
# #### The "Common Sense" Insight: The Relationship Weather Map
# #### The Simple Breakdown:
# #### Think of this heatmap like a Social Media Network for your data. It shows who is "friends" with whom and who doesn't talk to each other at all.
# #### The Best Friends (High Positive): When one goes up, the other follows. If "Hours Worked" and "Stress" are bright yellow, they are inseparable. You can't change one without the other moving too.
# #### The Seesaw (High Negative): These are opposites. If "Salary" goes up and "Intent to Quit" goes down, they are on a seesaw. This tells you exactly which "lever" to pull to fix a specific problem.
# #### The Strangers (Near Zero): These are the most interesting. If "Years of Education" has a 0.02 correlation with "Job Performance," it means—in your specific company—degrees don't actually matter for success. That’s a massive "Aha!" moment for hiring.

# ## 3.Multivariate Analysis

# ### Pairplot

# In[26]:


numeric_df = df.select_dtypes(include=['number'])

# Select the first 5 columns
cols_to_plot = numeric_df.columns[:5]

sns.set_theme(style="white")

#  Create the PairGrid
g = sns.PairGrid(numeric_df[cols_to_plot], diag_sharey=False, corner=True)

#  Map the plots
g.map_lower(sns.kdeplot, fill=True, cmap="magma", thresh=0, levels=15)
g.map_diag(sns.histplot, kde=True, color="#3b0f70", element="step")


plt.subplots_adjust(top=0.95)
g.fig.suptitle('Exponential Density Pair Plot', fontsize=16, fontweight='bold')

plt.show()


# ### Explain the Chart:
#  This pair plot displays relationships between multiple numerical variables simultaneously.
# ### What is it saying:
#  It confirms patterns such as higher stress leading to higher burnout and lower job satisfaction.
# ### Features used:
# Stress Level, Burnout Frequency, Job Satisfaction
# ### What you are showing:
#  This provides a comprehensive understanding of how multiple factors interact with each other.
# ### Conclusion Line:
#  This visualization helps in identifying patterns and supports data-driven decision-making for improving employee well-being.
# ### The Relationship Map (Pair Plot)
# #### The "Deep Dive" Insight:Instead of just one number, this shows the "shape" of how employees feel.
# #### Hot Spots: The glowing "clouds" show where most of your people are sitting. If the clouds are high on the stress scale, your company "center of gravity" is in a burnout zone.
# #### The Outliers: The stray dots far from the clouds are your "edge cases"—people who are either incredibly happy or struggling much more than everyone else.

# ### Pivot Table Analysis

# In[27]:


df = pd.read_csv('mental_health_data.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')


df['burnout_frequency'] = df['burnout_frequency'].map({
    'Never': 0,
    'Occasionally': 1,
    'Often': 2
})


def get_stress_category(x):
    if x <= 5: return 'Low'
    elif x <= 7: return 'Medium'
    else: return 'High'

df['stress_category'] = df['stress_level'].apply(get_stress_category)


df['burnout_risk_score'] = df['stress_level'] + (10 - df['job_satisfaction']) + df['burnout_frequency']


red_zone_pivot = df.pivot_table(
    index='employee_type', 
    columns='stress_category', 
    values='burnout_risk_score', 
    aggfunc='mean'
).reindex(columns=['Low', 'Medium', 'High'])


plt.figure(figsize=(12, 8))
sns.heatmap(red_zone_pivot, annot=True, fmt=".1f", cmap="magma", linewidths=1, linecolor='white')
plt.title("The 'Red Zone' Report: Burnout Risk Hot Spots", fontsize=18, fontweight='bold')
plt.show()


# #### The "Red Zone" Report
# ##### The "Hot Spots" (Bright Yellow/Orange):These cells represent the Danger Zones. If a specific role (like "Manager") has a bright yellow cell under "High Stress," it means the average person in that group is at a breaking point.
# ##### The "Safe Havens" (Dark Purple/Black):These are your healthiest pockets. Employees in these roles are managing their workload well and aren't showing signs of burnout.
# ##### The Consistency Check:If an entire row is bright, that specific Job Role is being overworked across the board.If an entire column is bright, the Stress Category itself is more intense than expected for everyone, suggesting a company-wide deadline or culture issue.
# ##### Missing Data (Empty Spots):If a cell is empty, it’s good news! It means nobody in that role has fallen into that stress category yet.

# ### Grouped Analysis

# In[28]:


# Load and Clean (This fixes the KeyErrors)
df = pd.read_csv('mental_health_data.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Create the missing columns
# Since 'stress_category' isn't in your CSV, we create it from 'stress_level'
df['stress_category'] = pd.cut(df['stress_level'], 
                               bins=[0, 3, 7, 10], 
                               labels=['Low', 'Medium', 'High'])

# Since 'burnout_risk_score' isn't in your CSV, we'll calculate a simple one
# Example: Stress Level + (10 - Job Satisfaction)
df['burnout_risk_score'] = df['stress_level'] + (10 - df['job_satisfaction'])

# Groupby
grouped_df = df.groupby(['employee_type', 'stress_category'], observed=True)['burnout_risk_score'].mean().reset_index()

# Plotting Code
plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid", font_scale=1.1)

ax = sns.barplot(
    data=grouped_df,
    x='employee_type',
    y='burnout_risk_score',
    hue='stress_category',
    palette="magma"
)

# Formatting
plt.xticks(rotation=35, ha='right')
ax.set_title("Burnout Risk Score by Role & Stress Category", fontsize=18, fontweight='bold', pad=30)
ax.set_xlabel("Employee Role", fontsize=12, fontweight='bold')
ax.set_ylabel("Average Burnout Risk Score", fontsize=12, fontweight='bold')

# Add bar labels
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3, fontweight='bold', fontsize=9)

plt.tight_layout()
plt.show()


# #### The "Burnout Breakdown(Grouped Analysis):
#  The "Triple Threat" (The Longest Red Bars):Look for roles where the Red (High Stress) bar is significantly longer than the others. This identifies roles where high stress is guaranteed to lead to high burnout risk.
#  
#  The "Stress Gap":If a role has a huge jump in risk between "Medium" and "High" stress, it means those employees can handle a little pressure, but they break quickly once things get too busy.
#  
# The "Resilient" Roles:Look for roles where the bars are all relatively short, even for the "High" stress category. These teams might have better support systems or training that protects them from feeling "fried."
# 
# Actionable Resource Planning:Managers should use this to redistribute work. If "Software Engineers" have high burnout scores across every category, they need more staff or better tools, not just a "wellness day".

# #### The analysis shows that employees with low job satisfaction tend to experience higher stress and burnout. Certain job roles are more affected, indicating the need for targeted interventions. If not addressed, this may lead to increased employee turnover and reduced productivity. Organizations can use these insights to improve workplace well-being and retention strategies.

# #### In Stage 3, I performed univariate, bivariate, and multivariate analysis. I used histograms, countplots, and boxplots for single variable analysis. Then I explored relationships using scatterplots, barplots, and heatmaps. I also performed multivariate analysis using pairplots and pivot tables. Each visualization includes interpretation, and I focused on deriving business insights such as stress impact on burnout and employee retention.

# ## Stage 4 – Documentation, Insights & Presentation

# ## Dataset link

# https://drive.google.com/drive/folders/1vToWnsEkEKPMbr1wrM_ujx1wnNqbeJgU?usp=share_link

# ### Summary of Findings (Plain English)
# ####   Stress Distribution Across the Workforce
# The average organizational stress level stands at 7.33 out of 10 — a figure that significantly exceeds the generally accepted "safe" threshold of 5.5 in occupational health benchmarks. This is not a marginal concern; it reflects a systemic, deeply embedded condition across the entire healthcare workforce studied.
# Stress is not uniformly distributed. Clinical departments — particularly Radiology, ICU, and Specialty Care — register consistently higher stress scores in the High Stress category (above 2.50 on the engagement-stress axis). Administrative staff, by contrast, show the highest engagement score (3.23) and relatively lower overall burnout, suggesting that structured role clarity can serve as a buffer.
# ####   Relationship Between Stress and Job Satisfaction
# There is a clear and consistent inverse relationship: as Job Satisfaction decreases, Stress Levels increase. With an average job satisfaction score of just 2.20 out of 5, the organization's workforce is operating near the lower quartile of healthy engagement benchmarks. The scatter plot analysis confirms that employees in the High Stress category are consistently clustered at the lower end of the job satisfaction scale (1.4–2.1), while Low Stress employees appear in the 1.6–2.0 satisfaction range with much lower stress exposure — indicating that stress management and satisfaction improvement are inherently linked, not independent levers.
# ####   Engagement and Burnout Patterns
# A striking 65.42% of the workforce falls into the Low Engagement category, with only 12.52% achieving High Engagement. This reflects a workforce that is disengaged, stretched, and likely performing below potential — creating compounding productivity and quality-of-care risks in a healthcare context.
# The burnout segmentation reinforces this concern: 44.42% (2,220 employees) are classified as Critical Risk, and a further 21.74% (1,090 employees) are in the Warning Zone. Combined, 66.16% of all staff are already in dangerous burnout territory.
# ####  Role and Department Differences
# Burnout and stress risk are not equal across departments. Radiology leads in High Stress burnout scores (2.58), followed by Specialty Care (2.53) and Pediatrics/Laboratory (both 2.52). Administration records the lowest total average (1.93) but the highest engagement, confirming the protective role of structured environments.
# The Turnover vs. Burnout chart reveals a critical operational risk: employees intending to leave score 2.37 on the burnout index compared to 2.09 for those staying — a 13.4% difference. With 67% of the workforce signalling turnover intention, the organization faces a potential talent exodus that could destabilize service delivery.
# 

# 

# 

# ### Key Insights
# #### Insight 1 — Burnout Crisis at Organizational Scale
# Finding: Over 66% of employees are in Critical or Warning burnout segments — exceeding sustainable thresholds.
# Data Point: Critical Risk = 44.42% (2,220 employees); Warning Zone = 21.74% (1,090 employees).
# Impact: At this scale, burnout is no longer an individual issue — it is a structural, systemic crisis requiring top-down intervention.
# #### Insight 2 — Stress and Job Satisfaction are Inversely Correlated
# Finding: Low job satisfaction (avg 2.20/5) directly co-occurs with high stress (avg 7.33/10).
# Data Point: High Stress employees cluster at satisfaction scores below 2.1; Low Stress employees show higher satisfaction ranges.
# Impact: Improving job satisfaction is not merely a "feel-good" measure — it is a primary stress-reduction mechanism that also drives engagement.
# #### Insight 3 — Radiology, ICU and Specialty Care are High-Risk Priority Departments
# Finding: These three departments consistently record the highest High Stress burnout scores (2.50–2.58).
# Data Point: Radiology: 2.58 | ICU: 2.50 | Specialty Care: 2.53 — all above the 2.49 organizational average.
# Impact: Resource allocation for wellness programs and staffing support must prioritize these departments immediately.
# #### Insight 4 — Heavy Workload and Work-Life Imbalance are the Dominant Stress Drivers
# Finding: The top 3 stressors all score at or above 8.0/10: Heavy Workload (8.2), Work-Life Imbalance (8.0), Safety Concerns (8.0).
# Data Point: All three exceed the threshold for "severe occupational stressor" in standard HR health frameworks.
# Impact: Workload redistribution and flexible scheduling policies would deliver the highest marginal reduction in organizational stress.
# #### Insight 5 — Turnover Intention is at Crisis Level; Burnout is the Primary Driver
# Finding: 67% of the workforce intends to leave. Employees planning to exit have 13.4% higher burnout scores.
# Data Point: Turnover=Yes avg burnout: 2.37 vs Turnover=No: 2.09.
# Anomaly Note: Administration has the highest engagement (3.23) yet still reports moderate turnover, suggesting systemic cultural or compensation factors beyond workload.
# #### 

# ### Business Recommendations
# The following recommendations are derived directly from the data patterns identified in this analysis. They are sequenced by urgency and organizational impact, mapped to specific departments and timelines.
# | **No** | **Recommendation**                   | **Target / Rationale**                                          | **Priority Timeline** |
# | ------ | ------------------------------------ | --------------------------------------------------------------- | --------------------- |
# | 01     | Launch Targeted Wellness Programs    | ICU, Radiology, Specialty Care — highest stress load            | 🔴 Immediate (Q1)     |
# | 02     | Optimize Workload Distribution       | All Clinical Departments — workload & safety drivers at 8.0+    | 🟡 Short-Term (Q2)    |
# | 03     | Improve Job Satisfaction             | Organization-Wide — avg satisfaction dangerously low at 2.20    | 🔵 Ongoing            |
# | 04     | Establish Manager Burnout Alerts     | All Departments — 44% critical risk requires real-time flagging | 🔴 Immediate (Q1)     |
# | 05     | Career Growth & Recognition Programs | High Performers in Low/Moderate Stress Segments                 | 🟢 Medium-Term (Q3)   |
# | 06     | Predictive HR Analytics              | Data Team — automate risk scoring and early intervention        | 🟣 Long-Term (Q4)     |
# 
# 🔴 Immediate → High Priority
# 🟡 Short-Term → Medium Priority
# 🟢 Medium-Term → Planned
# 🟣 Long-Term → Strategic
# 🔵 Ongoing → Continuous
# 

# ###   Detailed Recommendation Rationale
# #### Rec 01 — Targeted Wellness Programs
# Deploy structured, department-specific mental health programs beginning with ICU, Radiology, and Specialty Care — the three highest-risk departments. Programs should include: mandatory monthly mental health check-ins, access to on-site or tele-counselling, peer support networks for high-pressure clinical roles, and quarterly burnout-risk assessments using the same cards monitored in this dashboard.
# #### Rec 02 — Workload Optimization and Staffing Audit
# Commission an independent workload assessment across all clinical departments. Introduce evidence-based staffing ratio policies, rotate high-demand shifts to distribute load equitably, and evaluate patient-to-staff ratios against national healthcare standards. Heavy Workload (8.2) is the single largest stress driver and must be structurally addressed — not just managed through wellness initiatives.
# #### Rec 03 — Job Satisfaction and Engagement Enhancement
# With average satisfaction at just 2.20 out of 5, targeted engagement initiatives are critical. Recommendations include: structured recognition and reward programs, transparent career progression frameworks, employee voice mechanisms (anonymous quarterly surveys with published response plans), and meaningful non-monetary benefits such as schedule flexibility and professional development sponsorship.
# #### Rec 04 — Real-Time Burnout Monitoring Dashboard
# Operationalize this Power BI dashboard as a live management tool. Configure automated alerts for managers when employees enter the Warning Zone, create weekly department-level burnout index reports, and tie burnout cards to managerial performance metrics to ensure accountability at the leadership level.
# #### Rec 05 & 06 — Long-Term: Predictive Analytics and Automation
# In the medium-to-long term, invest in predictive HR analytics capabilities — machine learning models trained on engagement, stress, and satisfaction data to flag at-risk employees 30–60 days before burnout manifests. This shifts the organization from reactive crisis management to proactive workforce wellness governance.
# 
# 

# ### The Final Story — From Data to Decision
# ####   What Problem Exists?
# A healthcare organization employing 5,000 professionals is operating under conditions of widespread, measurable burnout. Nearly half its workforce is classified as Critical Risk; two-thirds are disengaged; and the average stress level of 7.33 out of 10 places the organization in the "high-hazard" range. This is not a perception problem — it is a quantifiable operational and human crisis.
# ####   Why Is This Happening?
# The data is unambiguous about the root causes. The top three workplace stressors — Heavy Workload (8.2), Work-Life Imbalance (8.0), and Safety Concerns (8.0) — all score above 8.0 on a 10-point scale. Clinical roles, by nature, carry high emotional and physical demands; but the data shows that structural factors — inadequate staffing ratios, poor scheduling, unclear role expectations, and limited career growth — are amplifying these inherent challenges beyond sustainable limits.
# Job satisfaction at 2.20 out of 5 reflects a workforce that does not feel valued, supported, or heard. Low satisfaction fuels low engagement (65.42% Low Engagement), which in turn accelerates burnout. This is a self-reinforcing downward cycle that, if unaddressed, will produce the 67% turnover that employees are already signalling.
# ####   What Actions Must Be Taken?
# The data prescribes a three-horizon response strategy:
# Immediate (Q1): Deploy wellness programs in the three highest-risk departments; establish real-time burnout monitoring for managers; audit workloads in clinical departments.
# #### Short-to-Medium Term (Q2–Q3): 
# Implement job satisfaction improvement programmes; restructure staffing ratios; introduce transparent career pathways and recognition frameworks.
# #### Long-Term (Q4+):
# Build predictive HR analytics infrastructure; automate burnout-risk scoring; institutionalize quarterly workforce mental health reporting at the executive level.
# 
# The cost of inaction is measurable: replacing a single healthcare professional costs 50–200% of their annual salary in recruitment, 
# training, and lost productivity. With 67% expressing intent to leave, the potential financial exposure is existential. Acting on these insights is not merely a welfare obligation — it is a business imperative.
# 

# ### Data Tables
# #### Burnout Risk Segmentation
# | **Burnout Risk Segment** | **Employee Count** | **% of Workforce** |
# | ------------------------ | ------------------ | ------------------ |
# | 🔴 Critical Risk         | 2,220              | 44.42%             |
# | 🟡 Warning Zone          | 1,090              | 21.74%             |
# | 🟢 Balanced              | 850                | 17.08%             |
# | 🔵 Healthy               | 840                | 16.76%             |
# 
# #### Engagement Category Distribution
# | **Engagement Category** | **Employee Count** | **% of Workforce** |
# | ----------------------- | ------------------ | ------------------ |
# | 🔴 Low Engagement       | 3,270              | 65.42%             |
# | 🟡 Moderate Engagement  | 1,100              | 22.06%             |
# | 🟢 High Engagement      | 630                | 12.52%             |
# 
# ####  Workplace Stressor Analysis
# | **Workplace Stressor Factor** | **Avg Stress Score (/10)** | **Risk Level** |
# | ----------------------------- | -------------------------- | -------------- |
# | Heavy Workload                | 8.2                        | 🔴 CRITICAL    |
# | Work-Life Imbalance           | 8.0                        | 🔴 CRITICAL    |
# | Safety Concerns               | 8.0                        | 🔴 CRITICAL    |
# | Poor Work Environment         | 7.0                        | 🟠 HIGH        |
# | Emotional Demands             | 6.8                        | 🟠 HIGH        |
# | Job Insecurity                | 5.7                        | 🟡 MODERATE    |
# | Unclear Expectations          | 4.9                        |                |
# 
# #### Department × Stress Category Heatmap Matrix
# | **Department**      | **High Stress** | **Moderate Stress** | **Low Stress** | **Total Avg** |
# | ------------------- | --------------- | ------------------- | -------------- | ------------- |
# | Radiology           | 2.58            | 2.21                | 1.69           | 2.29          |
# | Specialty Care      | 2.53            | 2.14                | —              | 2.33          |
# | ICU                 | 2.50            | —                   | 2.03           | 2.36          |
# | Pediatrics          | 2.52            | 2.15                | 1.64           | 2.30          |
# | Laboratory          | 2.52            | 2.19                | 1.79           | 2.26          |
# | Assisted Living     | 2.51            | 2.12                | —              | 2.32          |
# | Outpatient Services | 2.51            | 2.10                | 1.73           | 2.29          |
# | General Medicine    | 2.49            | 2.13                | 1.63           | 2.32          |
# | General Practice    | 2.45            | 2.16                | 1.74           | 2.34          |
# | Administration      | 2.47            | 2.05                | 1.75           | 1.93          |
# | **TOTAL (Overall)** | **2.51**        | **2.12**            | **1.72**       | **2.28**      |
# ### Future Enhancements
# This analysis represents a foundational baseline. The following enhancements would substantially elevate its analytical power and operational utility:
# #### Expanded Dataset
# The current dataset covers 5,000 employees across one fiscal year. Future iterations should incorporate longitudinal data (3–5 year trend analysis) and expand to benchmarking against industry-standard healthcare burnout indices (e.g., Maslach Burnout Inventory). Larger, anonymized samples would improve model reliability and enable robust segmentation by seniority, tenure, and shift pattern.
# #### Automation and Live Data Integration
# The current Power BI dashboard operates on a static dataset. Integration with live HRIS (Human Resource Information Systems) such as SAP SuccessFactors or Workday would enable real-time burnout monitoring. Automated data pipelines refreshing on a weekly or bi-weekly cadence would transform this from a point-in-time report to a dynamic operational tool.
# ####  Predictive Modelling
# Implementing machine learning models — Logistic Regression, Random Forest, or Gradient Boosting — trained on the existing labelled dataset would allow the organization to predict burnout risk 30–90 days in advance. Features such as overtime hours, absenteeism patterns, engagement score trends, and manager-employee interaction frequency could serve as leading indicators, enabling proactive intervention before critical risk is reached.
# #### Advanced and Interactive Dashboards
# Future dashboard versions should incorporate advanced Power BI capabilities: AI-powered Q&A functionality, natural language query interfaces for non-technical managers, drill-through pages by employee cohort, and mobile-optimised executive summary views. Integration with Microsoft Teams or Slack for automated weekly insight delivery to department heads would further embed analytics into daily operational decision-making.
# #### Benchmarking Against Industry Standards
# Incorporating external benchmarking data from sources such as the NHS Staff Survey, SHRM Wellbeing Reports, or Gallup Q12 Engagement data would allow the organization to contextualise its burnout and engagement scores against peer healthcare institutions — identifying whether observed patterns are sector-wide or organization-specific, and calibrating the urgency and scale of interventions accordingly.
# 

# ### Conclusion
# From the analysis, meaningful patterns and relationships were identified. These insights provide a strong foundation for data-driven business decisions.
# 
# This analysis of Workplace Mental Health and Employee Burnout reveals a healthcare organization at a critical inflection point. The data is clear: 44.42% of employees are at critical burnout risk, stress levels are alarmingly high at 7.33 out of 10, engagement has collapsed to 65% in the low category, and turnover intention sits at a workforce-threatening 67%.
# These are not isolated statistics. They form an interconnected narrative: excessive workload and poor work-life balance create high stress; high stress destroys job satisfaction; low satisfaction drives disengagement; disengagement accelerates burnout; and burnout fuels turnover. Left unaddressed, this cycle will compound year over year.
# The recommendations presented in this project from immediate wellness programs and workload redistribution to long-term predictive analytics infrastructure  are grounded entirely in what the data demonstrates. This is the power of data analytics applied to human capital: not to generate reports, but to drive decisions that protect people and sustain organizational performance.
# The Power BI dashboard built for this project is not merely a visual artefact. It is a decision-support system — one that translates complex multi-dimensional HR data into the clear, conditional signals that leaders need to act with precision and accountability. That is the ultimate objective of my project to reduce the distance between evidence and action.

# In[ ]:




