---
title: EDA MASTERCLASS
categories: [DATA SCIENCE]

tags : DATA,DATA DCIENCE, EDA


---
#EXPLORATORY DATA ANALYSIS TITANIC DATASET WEEK 3 ASSIGNMENT

add Codeadd Markdown
# Import libraries
import pandas as pd  # Data manipulation
import numpy as np  # Numerical computations
import matplotlib.pyplot as plt  # Static plots
import seaborn as sns  # Statistical plots
import missingno as msno  # Missing data visualization

# Configuring Seaborn plot aesthetics
sns.set_theme(style='darkgrid', context='notebook')

import warnings
warnings.filterwarnings("ignore")
add Codeadd Markdown
Previewing the Dataset and loading

The first step is to get a quick look at the dataset’s contents using .head(). This function shows the first 5 rows by default:

Why Use .head()

It provides a quick overview of the dataset's structure and values. You can spot-check for potential anomalies or unexpected values (e.g., nulls, typos, negative values

add Codeadd Markdown
# Load Data
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
gender = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
add Codeadd Markdown
# Preview the first 5 rows of the dataset
train.head()
add Codeadd Markdown
# Preview the first 20 rows of the dataset
train.head(20)
add Codeadd Markdown
The .shape attribute returns a tuple containing the number of rows and columns

add Codeadd Markdown
# Get the number of rows and columns
print(f'The dataset has {train.shape[0]} rows and {train.shape[1]} columns.')
add Codeadd Markdown
# Get an overview of the dataset’s columns and their data types
train.info()
add Codeadd Markdown
# Converting data types
train['Survived'] = train['Survived'].astype('category')
train['Pclass'] = train['Pclass'].astype('category')
train['Sex'] = train['Sex'].astype('category')
train['Cabin'] = train['Cabin'].astype('category')
train['Embarked'] = train['Embarked'].astype('category')
add Codeadd Markdown
train.info()
add Codeadd Markdown
Statistical Summary of Numerical Columns The .describe() function generates summary statistics for numerical columns

add Codeadd Markdown
# Summary statistics for numerical columns
train.describe().T
add Codeadd Markdown
# Drop the PassengerId column
train = train.drop(columns=['PassengerId'])
add Codeadd Markdown
# Drop the name column
train = train.drop(columns=['Name'])
add Codeadd Markdown
# Summary statistics for numerical columns
train.describe().T
add Codeadd Markdown
Column Names

To list all column names in the dataset, use .columns

add Codeadd Markdown
# List column names
train.columns
add Codeadd Markdown
i used .columns To list all column names in the dataset,

add Codeadd Markdown
# Count the unique values in each column
train.nunique()
add Codeadd Markdown
# check for duplicates
train.duplicated().sum()
add Codeadd Markdown
Common Issues You Might Spot in Initial Data Exploration

Missing Values: Some columns may have missing data (e.g., Cabin in Titanic).
Outliers: Some columns may contain unusually high or low values.
Irrelevant Features: Columns like PassengerId may not contribute to the analysis.
Data Type Mismatches: Columns with dates may be stored as strings, or numerical data may be stored as text.
add Codeadd Markdown
checking for missing values and visualizing them

add Codeadd Markdown
# Visualize missing data using missingno library
import missingno as msno
msno.bar(train)
add Codeadd Markdown
# Count the number of missing values in each column
missing_values = train.isnull().sum().sort_values(ascending=False)
missing_percentage = (missing_values / len(train)) * 100
print(pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage}))
add Codeadd Markdown
Dropping Missing Data

i dropped columns with high percentage of missing values that are

critical to your analysis.
and When removing the data won’t significantly reduce the size of your dataset.
add Codeadd Markdown
# Drop a column with too many missing values
train = train.drop(columns=['Cabin'])

# Or If a row has multiple missing values, you can drop it:
# train = train.dropna()

add Codeadd Markdown
fill in the missing data by imputing

add Codeadd Markdown
# Fill missing values in the 'Age' column with the mean age
train['Age'].fillna(train['Age'].mean(), inplace=True)

# Fill missing values in the 'Fare' column with the median
train['Fare'].fillna(train['Fare'].median(), inplace=True)

# Fill missing values in the 'Embarked' column with the most common value (mode)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
add Codeadd Markdown
Instead of dropping or imputing missing values, i created a new column to flag where data was missing

This method is useful when missingness itself might be meaningful. For example, passengers with missing Cabin values might have been in a different section of the ship.

add Codeadd Markdown
# Create a new column indicating missing values for 'Cabin'
# train['Cabin_missing_flag'] = train['Cabin'].isnull().astype(int)
add Codeadd Markdown
Univariate Analysis

Univariate analysis involves examining one variable at a timeto understand:

The distribution of the variable (normal, skewed, etc.).
The central tendency (mean, median, mode).
The spread of the data (range, variance, standard deviation).
add Codeadd Markdown
# Histogram for Age
plt.figure(figsize=(8, 5))
sns.histplot(train['Age'].dropna(), bins=30, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
add Codeadd Markdown
The KDE plot shows the probability density function of a numerical column

add Codeadd Markdown
# KDE Plot for Fare
plt.figure(figsize=(8, 5))
sns.kdeplot(train['Fare'], shade=True)
plt.title('KDE Plot of Fare')
plt.xlabel('Fare')
plt.show()
add Codeadd Markdown
Boxplot (Detecting Outliers)

Boxplots visualize the minimum, lower quartile (25%), median, upper quartile (75%), and maximum values:

add Codeadd Markdown
# Boxplot for Fare
plt.figure(figsize=(8, 5))
sns.boxplot(x=train['Fare'])
plt.title('Boxplot of Fare')
plt.show()

add Codeadd Markdown
Countplot Countplots are used to count the frequency of each category in a column

add Codeadd Markdown
# Countplot for Embarked
plt.figure(figsize=(8, 5))
sns.countplot(x='Embarked', data=train, palette='pastel')
plt.title('Countplot of Embarked')
plt.xlabel('Embarkation Port')
plt.ylabel('Count')
plt.show()
add Codeadd Markdown
Pie Chart

add Codeadd Markdown
# Pie chart for Sex distribution
train['Sex'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(6, 6), colors=['#ff9999', '#66b3ff'])
plt.title('Sex Distribution')
plt.show()

add Codeadd Markdown
# Frequency count of unique values in the 'Pclass' column
print(train['Pclass'].value_counts())
add Codeadd Markdown
Summary of Univariate Analysis

In this section, i have:

Used histograms, KDE plots, and boxplots for numerical columns to understand distributions and outliers.
Used countplots and pie charts to explore categorical columns.
Used .value_counts() to summarize the frequency of categories.
add Codeadd Markdown
Bivariate Analysis

Bivariate analysis involves exploring the relationship between two variables. It helps answer questions such as

Does the Fare change depending on the Pclass?
Are younger passengers more likely to survive on the Titanic?
Does the Embarked location affect survival rate?
Scatter Plot

Scatter plots are used to visualize the relationship between two numerical variables:

Interpretation

Clusters of points may indicate subgroups (e.g., younger passengers may have paid lower fares).
Hue (color) shows how survival correlates with fare and age.
add Codeadd Markdown
# Scatter plot for Age vs Fare
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Age', y='Fare', data=train, hue='Survived', palette='coolwarm')
plt.title('Scatter Plot of Age vs Fare (Colored by Survived)')
plt.show()
add Codeadd Markdown
Correlation Heatmap

A correlation heatmap shows the strength and direction of relationships between numerical variables:

add Codeadd Markdown
# Correlation heatmap for numerical columns only
plt.figure(figsize=(8, 6))
numerical_columns = train.select_dtypes(include=['int64', 'float64']).columns  # Select only numerical columns
sns.heatmap(train[numerical_columns].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

add Codeadd Markdown
Numerical vs Categorical Analysis

Boxplot

Boxplots are great for visualizing the distribution of numerical values grouped by categories

add Codeadd Markdown
# Boxplot of Fare grouped by Pclass
plt.figure(figsize=(8, 5))
sns.boxplot(x='Pclass', y='Fare', data=train, palette='Set2')
plt.title('Boxplot of Fare by Pclass')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.show()
add Codeadd Markdown
Violin Plot

A violin plot is similar to a boxplot but also shows the density of the data

add Codeadd Markdown
# Violin plot of Age grouped by Survived
plt.figure(figsize=(8, 5))
sns.violinplot(x='Survived', y='Age', data=train, split=True, palette='muted')
plt.title('Violin Plot of Age by Survival')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Age')
plt.show()
add Codeadd Markdown
Categorical vs Categorical Analysis Grouped Bar Plot

Grouped bar plots show the count or proportion of one category for each level of another category

add Codeadd Markdown
# Grouped bar plot of Survived vs Embarked
plt.figure(figsize=(8, 5))
sns.countplot(x='Embarked', hue='Survived', data=train, palette='pastel')
plt.title('Survival Counts by Embarked Port')
plt.xlabel('Embarked Port')
plt.ylabel('Count')
plt.show()
add Codeadd Markdown
Mosaic plots show the proportion of categories across different groups

add Codeadd Markdown
# Install required library for mosaic plot
# !pip install statsmodels
from statsmodels.graphics.mosaicplot import mosaic
from itertools import product

# Mosaic plot of Pclass vs Survived
plt.figure(figsize=(10, 10))
mosaic(train, ['Pclass', 'Survived'], title='Mosaic Plot of Pclass vs Survived')
plt.show()
add Codeadd Markdown


Multivariate Analysis

Multivariate analysis involves exploring relationships between three or more variablessimultaneously. This helps answer complex questions, such as:

How do Pclass, Age, and Fare jointly affect survival?
Are survival rates different for Embarked locations when considering Pclass?
By examining multiple variables at once, we can detect interactions combined effects hidden patterns that may not be visible in bivariate analysis.

A pair plot shows scatter plots and histograms for all numerical variable pairs:

add Codeadd Markdown
# Pair plot for numerical columns
plt.figure(figsize=(10, 10))
sns.pairplot(train, hue='Survived', diag_kind='kde', palette='coolwarm')
plt.show()
add Codeadd Markdown
FacetGrid creates multiple subplots for different subsets of data based on categorical variables

add Codeadd Markdown
# FacetGrid for Age distribution by Survived and Pclass
g = sns.FacetGrid(train, col='Survived', row='Pclass', height=4, aspect=1.5)
g.map(sns.histplot, 'Age', kde=True)
plt.show()
add Codeadd Markdown
A heatmap shows the correlation between multiple numerical variables

What the Heatmap Shows Strong positive correlations (closer to +1) indicate that variables increase together.

Strong negative correlations (closer to -1) indicate that as one variable increases, the other decreases.
Correlation values close to zero indicate no strong relationship.
add Codeadd Markdown
# Heatmap of numerical features only
plt.figure(figsize=(8, 6))
numerical_columns = train.select_dtypes(include=['int64', 'float64']).columns  # Select only numerical columns
sns.heatmap(train[numerical_columns].corr(), annot=True, cmap='Blues', fmt='.2f')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()
add Codeadd Markdown
3D Scatter Plot

A 3D scatter plot helps visualize three numerical variables at once:

add Codeadd Markdown
# 3D scatter plot for Age, Fare, and Survived
import plotly.express as px

fig = px.scatter_3d(train, x='Age', y='Fare', z='Survived', color='Pclass', size='Fare', opacity=0.7)
fig.update_traces(marker=dict(line=dict(width=0)))
fig.update_layout(title='3D Scatter Plot: Age vs Fare vs Survived')
fig.show()
add Codeadd Markdown
Outlier Detection and Handling Outliers are data points that significantly differ from the rest of the data. They may represent:

True anomalies(e.g., very wealthy passengers on the Titanic who paid exorbitant fares).
Data entry errors(e.g., an age of 500 years due to a typo).
Special cases that need deeper investigation (e.g., survivors who broke typical survival patterns).
add Codeadd Markdown
# Boxplot for Fare to detect outliers
plt.figure(figsize=(8, 5))
sns.boxplot(x=train['Fare'], palette='pastel')
plt.title('Boxplot of Fare')
plt.show()
add Codeadd Markdown
Detecting Outliers Using Z-Score

A boxplot shows the minimum, lower quartile (Q1), median, upper quartile (Q3), and maximum values, highlighting outliers as dots.

add Codeadd Markdown
# Function to detect outliers using Z-score
from scipy.stats import zscore

def detect_outliers_zscore(data, threshold=3):
    z_scores = zscore(data.dropna())  # Drop NaN to avoid errors
    outliers = data[(abs(z_scores) > threshold)]
    return outliers

# Detect outliers in the 'Age' column
outliers_age = detect_outliers_zscore(train['Age'])
print(f'Number of outliers in Age: {len(outliers_age)}')

add Codeadd Markdown
Detecting Outliers Using IQR (Interquartile Range)

The IQR method identifies outliers as data points that fall below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR

add Codeadd Markdown
# Function to detect outliers using IQR
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers

# Detect outliers in the 'Fare' column using IQR
outliers_fare = detect_outliers_iqr(train['Fare'])
print(f'Number of outliers in Fare: {len(outliers_fare)}')

add Codeadd Markdown
Handling Outliers

Once outliers are detected, i handled them using the following approaches

Remove Outliers: Remove rows containing outliers
Cap Outliers:Cap values at the upper and lower bounds
Impute Outliers: Replace outliers with mean or median values
Leave Outliers:In some cases (e.g., fraud detection, rare event analysis), outliers contain meaningful information and should be kept.
add Codeadd Markdown
# Remove outliers in the 'Fare' column
#train = train[(df['Fare'] >= train['Fare'].quantile(0.25) - 1.5 * (train['Fare'].quantile(0.75) - train['Fare'].quantile(0.25))) & 
        #(train['Fare'] <= train['Fare'].quantile(0.75) + 1.5 * (train['Fare'].quantile(0.75) - train['Fare'].quantile(0.25)))]
add Codeadd Markdown
# Cap outliers in the 'Fare' column
#train['Fare'] = train['Fare'].clip(lower=train['Fare'].quantile(0.05), upper=train['Fare'].quantile(0.95))
add Codeadd Markdown
# Impute outliers with the median
#train['Fare'] = train['Fare'].mask((train['Fare'] < train['Fare'].quantile(0.05)) | (train['Fare'] > train['Fare'].quantile(0.95)), train['Fare'].median())
add Codeadd Markdown
Target Variable Exploration: Understanding Survived

The target variable (Survived) in the Titanic dataset is what we’re trying to predict—it shows whether a passenger survived (1) or did not survive (0). Exploring the target variable helps us understand:

How balanced or imbalanced the dataset is.
What factors (like age, gender, class, or embarkation point) may influence survival.
Countplot

A countplot is useful for visualizing the distribution of categories in the target variable

add Codeadd Markdown
# Countplot for Survived
plt.figure(figsize=(8, 5))
sns.countplot(x='Survived', data=train, palette='Set2')
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()
add Codeadd Markdown
Survival Rate by numerical columns

Let’s visualize survival rates across different age groups using KDE plots

add Codeadd Markdown
# KDE Plot for Age by Survival Status
plt.figure(figsize=(8, 5))
sns.kdeplot(train[train['Survived'] == 1]['Age'], shade=True, label='Survived', color='green')
sns.kdeplot(train[train['Survived'] == 0]['Age'], shade=True, label='Did Not Survive', color='red')
plt.title('Age Distribution by Survival Status')
plt.xlabel('Age')
plt.legend()
plt.show()
add Codeadd Markdown
Survival Rate by categorical columns

Survival rates may differ significantly between males and females. Let’s visualize this relationship:

Interpretation:

The plot shows survival counts grouped by gender.
Titanic survival famously followed the "women and children first" protocol, so you may see higher survival rates for females.
add Codeadd Markdown
# Countplot for Survived grouped by Gender
plt.figure(figsize=(8, 5))
sns.countplot(x='Survived', hue='Sex', data=train, palette='muted')
plt.title('Survival Rate by Gender')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

add Codeadd Markdown
# Calculate survival rate by gender
gender_survival_rate = train.groupby('Sex')['Survived'].apply(lambda x: (x == 1).mean() * 100)
print(gender_survival_rate)

add Codeadd Markdown
Combined Analysis (Gender, Class, and Survival)

To explore survival rates based on multiple variables at once (e.g., gender and passenger class):

Interpretation:

This plot shows survival counts broken down by gender and passenger class. Look for patterns like:

High survival counts for first-class females.
Low survival counts for third-class males.
add Codeadd Markdown
# Grouped bar plot for survival by Gender and Class
plt.figure(figsize=(10, 6))
sns.countplot(x='Pclass', hue='Sex', data=train[train['Survived'] == 1], palette='Set1')
plt.title('Survivors by Gender and Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survivor Count')
plt.show()

add Codeadd Markdown
Common Insights During Target Variable Exploration Class Imbalance:

There may be a large difference in the number of survivors vs non-survivors, indicating an imbalanced dataset. Feature Importance:

Features like Pclass, Sex, and Age may have a strong influence on survival. Some features (like Ticket) may not show clear trends. Combined Effects:

The survival rate for first-class females may be much higher than for third-class males due to priority in lifeboats.

add Codeadd Markdown
Summary of Target Variable Exploration

In this section, i have:

Visualized the distribution of the target variable (Survived) using countplots and bar plots.
Explored survival rates based on gender and age.
Used combined plots to detect interaction effects.
add Codeadd Markdown
9. Key Insights and Reporting ** Purpose of the Key Insights Section**

The goal of this section is to summarize findings from the EDA and highlight important insights. This summary serves as a foundation for data preprocessing, feature selection, and model building. Key Insights from EDA 1. Data Overview

Total number of rows (records) and columns (features) in the dataset.
Types of features (numerical, categorical, datetime, etc.).
Identify features that may need special handling (e.g., text, identifiers). 2. Missing Values
Identify features with significant missing values (e.g., >50%).
Decide how to handle missing values:
Drop features with excessive missing data if they add minimal value.
Impute missing values using mean, median, mode, or more advanced techniques.
Flag missing values by creating indicator columns if missingness itself is informative.
Key Question: Are there patterns in the missing data (e.g., missingness concentrated in certain categories)?

3. Univariate Analysis

Numerical Columns:

Check for distributions (normal, skewed, bimodal).
Identify outliers that may distort statistics.
Calculate key summary statistics (mean, median, mode, range, IQR).
Categorical Columns:

Check the frequency distribution of categories (e.g., countplots).
Identify imbalances (e.g., if one category dominates the feature).
Key Question: Do the distributions suggest data preprocessing steps (e.g., normalization, encoding)?

4. Bivariate Analysis

Numerical vs Numerical:

Detect correlations using scatter plots and correlation matrices.
Identify relationships that may be useful for feature interactions (e.g., strong positive/negative correlations).
Numerical vs Categorical:

Use boxplots and violin plots to compare distributions across categories.
Identify features with significantly different distributions across target classes.
Categorical vs Categorical:

Use grouped bar plots or mosaic plots to check the distribution of categories across different groups.
Key Question: Are there clear relationships between features and the target variable?

5. Multivariate Analysis

Use multivariate techniques (e.g., pair plots, FacetGrids) to visualize relationships across three or more variables.
Check for interaction effects (e.g., does a feature behave differently depending on another feature?).
Identify combinations of features that may explain the target variable better than individual features alone.
Key Question: Do feature interactions reveal important patterns?

6. Outliers

Identify outliers using boxplots, Z-score, or IQR methods.
Decide how to handle outliers:
Remove: Drop outliers if they represent errors or distort analysis.
Cap: Replace extreme values with upper/lower bounds.
Impute: Replace outliers with a more reasonable value (e.g., median).
Key Question: Are outliers meaningful (e.g., rare events) or errors?

** Key Decisions for Preprocessing and Modeling**

Based on the insights gathered during EDA, you can make the following decisions:

feature Selection and Engineering:
Remove irrelevant features (e.g., IDs).
Consider creating new features based on combinations of existing features (e.g., ratios, time features).
Handling Missing Data:
Impute missing values with appropriate methods (mean, median, mode, or predictive imputation).
Create missing value indicators if the missingness is informative.
Handling Imbalanced Classes:
Use stratified sampling to ensure balanced train-test splits.
Consider resampling techniques (over-sampling or under-sampling) if necessary.
Outlier Treatment:
Remove or cap outliers based on their impact on the analysis.
5 feature Scaling:
Decide whether to scale numerical features (e.g., standardization or normalization) based on the chosen model.
Final Thoughts

EDA provides valuable insights into the structure and contents of the dataset, highlighting potential preprocessing steps and feature selection strategies.

add Codeadd Markdown
