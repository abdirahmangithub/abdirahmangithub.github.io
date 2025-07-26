---
title: data wrangling
categories: [data science,data]

tags : data wrangling, data


---
 data wrangling

add Codeadd Markdown
# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub
shivamb_netflix_shows_path = kagglehub.dataset_download('shivamb/netflix-shows')

print('Data source import complete.')

Data source import complete.
add Codeadd Markdown
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
/kaggle/input/netflix-shows/netflix_titles.csv
add Codeadd Markdown
DATA SCIENCE PROJECT: DATA WRANGLING the project showcases for my walkthrough on datawrangling using python on netflix dataset.

STEPS TO FOLLOW:

DISCOVERY to understand data, its existing format and quality issues to be addressed.

STRUCTURING to understand the structure and normalize or standerdize the format.

CLEANING

remove duplicate * remove irrelevant information * handle missing value * handle outliers ENRICHING

VALIDATING

PUBLISHING

add Codeadd Markdown
step1 DISCOVERY

add Codeadd Markdown
#import the data in a pandas dataframe
df= pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")

# aquick overview of the data
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8807 entries, 0 to 8806
Data columns (total 12 columns):
 #   Column        Non-Null Count  Dtype 
---  ------        --------------  ----- 
 0   show_id       8807 non-null   object
 1   type          8807 non-null   object
 2   title         8807 non-null   object
 3   director      6173 non-null   object
 4   cast          7982 non-null   object
 5   country       7976 non-null   object
 6   date_added    8797 non-null   object
 7   release_year  8807 non-null   int64 
 8   rating        8803 non-null   object
 9   duration      8804 non-null   object
 10  listed_in     8807 non-null   object
 11  description   8807 non-null   object
dtypes: int64(1), object(11)
memory usage: 825.8+ KB
add Codeadd Markdown
# Number of rows and columns
print("Shape of the dataset (R x C):", df.shape)

Shape of the dataset (R x C): (8807, 12)
add Codeadd Markdown
# List of all column names
print("Columns in the dataset:\n", df.columns.tolist())
Columns in the dataset:
 ['show_id', 'type', 'title', 'director', 'cast', 'country', 'date_added', 'release_year', 'rating', 'duration', 'listed_in', 'description']
add Codeadd Markdown

# Data types of each column
print("Data types:\n", df.dtypes)
Data types:
 show_id         object
type            object
title           object
director        object
cast            object
country         object
date_added      object
release_year     int64
rating          object
duration        object
listed_in       object
description     object
dtype: object
add Codeadd Markdown
# Group and Count of missing (null) values in each column
print("Missing values per column:\n", df.isnull().sum())
Missing values per column:
 show_id            0
type               0
title              0
director        2634
cast             825
country          831
date_added        10
release_year       0
rating             4
duration           3
listed_in          0
description        0
dtype: int64
add Codeadd Markdown
# Group and Count of duplicate rows
print("Number of duplicate rows:", df.duplicated().sum())

Number of duplicate rows: 0
add Codeadd Markdown
STEP 2 STRUCTURING

add Codeadd Markdown
# Convert 'date_added' to datetime
df['date_added'] = pd.to_datetime(df['date_added'],format='mixed')


add Codeadd Markdown
# Separate 'duration' into numeric value and unit (e.g., '90 min' → 90, 'min')
df[['duration_value', 'duration_unit']] = df['duration'].str.extract(r'(\d+)\s*(\w+)')
add Codeadd Markdown
# Convert duration_value to numeric
df['duration_value'] = pd.to_numeric(df['duration_value'])

add Codeadd Markdown
# View Resulting columns
print(df[['duration_value', 'duration_unit']])

      duration_value duration_unit
0               90.0           min
1                2.0       Seasons
2                1.0        Season
3                1.0        Season
4                2.0       Seasons
...              ...           ...
8802           158.0           min
8803             2.0       Seasons
8804            88.0           min
8805            88.0           min
8806           111.0           min

[8807 rows x 2 columns]
add Codeadd Markdown
STEP3 CLEANING

add Codeadd Markdown
# Drop description column because it will not be used
df = df.drop(columns=['description'])

add Codeadd Markdown
# Impute Director values by using relationship between cast and director

# List of Director-Cast pairs and the number of times they appear
df['dir_cast'] = df['director'] + '---' + df['cast']
counts = df['dir_cast'].value_counts() #counts unique values
filtered_counts = counts[counts >= 3] #checks if repeated 3 or more times
filtered_values = filtered_counts.index #gets the values i.e. names
lst_dir_cast = list(filtered_values) #convert to list
dict_direcast = dict()
for i in lst_dir_cast :
    director,cast = i.split('---')
    dict_direcast[director]=cast
for i in range(len(dict_direcast)): 
    df.loc[(df['director'].isna()) & (df['cast'] == list(dict_direcast.items())[i][1]),'director'] = list(dict_direcast.items())[i][0]

# Assign Not Given to all other director fields
df.loc[df['director'].isna(),'director'] ='NOT GIVEN'
add Codeadd Markdown
#Use directors to fill missing countries
directors = df['director']
countries = df['country']
#pair each director with their country use zip() to get an iterator of tuples
pairs = zip(directors, countries)
# Convert the list of tuples into a dictionary
dir_cntry = dict(list(pairs))

# Director matched to Country values used to fill in null country values
for i in range(len(dir_cntry)):    
    df.loc[(df['country'].isna()) & (df['director'] == list(dir_cntry.items())[i][0]),'country'] = list(dir_cntry.items())[i][1]
# Assign Not Given to all other country fields
df.loc[df['country'].isna(),'country'] = 'Not Given'

# Assign Not Given to all other fields
df.loc[df['cast'].isna(),'cast'] = 'Not Given'

add Codeadd Markdown
# dropping other row records that are null
df.drop(df[df['date_added'].isna()].index,axis=0,inplace=True)
df.drop(df[df['rating'].isna()].index,axis=0,inplace=True)
df.drop(df[df['duration'].isna()].index,axis=0,inplace=True)

add Codeadd Markdown
#Errors
# check if there are any added_dates that come before release_year
import datetime as dt
sum(df['date_added'].dt.year < df['release_year'])
df.loc[(df['date_added'].dt.year < df['release_year']),['date_added','release_year']]
# sample some of the records and check that they have been accurately replaced
df.iloc[[1551,1696,2920,3168]]
#Confirm that no more release_year inconsistencies
sum(df['date_added'].dt.year < df['release_year'])
14
add Codeadd Markdown
STEP 4 VALIDATION

add Codeadd Markdown
# Remove any columns you may have added during wrangling
# We already dropped 'dir_cast' in the previous step, but including this as a reminder
if 'dir_cast' in df.columns:
    df.drop(columns=['dir_cast'], inplace=True)
add Codeadd Markdown
# Ensure each column has the correct data type
print("\nData types after validation checks:")
print(df.dtypes)

Data types after validation checks:
show_id                   object
type                      object
title                     object
director                  object
cast                      object
country                   object
date_added        datetime64[ns]
release_year               int64
rating                    object
duration                  object
listed_in                 object
duration_value           float64
duration_unit             object
dtype: object
add Codeadd Markdown
# Verify specific column data types
if pd.api.types.is_datetime64_any_dtype(df['date_added']):
    print("\n'date_added' column is datetime.")
else:
    print("\n'date_added' column is NOT datetime. Further investigation needed.")

if pd.api.types.is_numeric_dtype(df['duration_value']):
     print("'duration_value' column is numeric.")
else:
    print("'duration_value' column is NOT numeric. Further investigation needed.")


'date_added' column is datetime.
add Codeadd Markdown
# Check for records with release year before a reasonable start year (e.g., 1900)
# You mentioned 1997, let's check for any release years before that.
anomalous_release_years = df[df['release_year'] < 1997]
if not anomalous_release_years.empty:
    print(f"\nFound {len(anomalous_release_years)} records with release_year before 1997:")
    display(anomalous_release_years[['title', 'release_year', 'date_added']].head()) # Display first few anomalous records
else:
    print("\nNo records found with release_year before 1997.")


Found 412 records with release_year before 1997:
title	release_year	date_added
7	Sankofa	1993	2021-09-24
22	Avvai Shanmughi	1996	2021-09-21
41	Jaws	1975	2021-09-16
42	Jaws 2	1978	2021-09-16
43	Jaws 3	1983	2021-09-16
add Codeadd Markdown
# Ensure no important fields are still missing
important_fields = ['title', 'director', 'cast', 'country', 'date_added', 'release_year', 'rating', 'duration', 'duration_value', 'duration_unit']
missing_values_after_cleaning = df[important_fields].isnull().sum()
print("\nMissing values in important fields after cleaning:")
print(missing_values_after_cleaning[missing_values_after_cleaning > 0])

Missing values in important fields after cleaning:
Series([], dtype: int64)
add Codeadd Markdown
# Sample a few rows to check visually
print("\nSample of 5 rows after validation:")
display(df.sample(5))

Sample of 5 rows after validation:
show_id	type	title	director	cast	country	date_added	release_year	rating	duration	listed_in	duration_value	duration_unit
94	s95	Movie	Show Dogs	Raja Gosnell	Will Arnett, Ludacris, Natasha Lyonne, Stanley...	United Kingdom, United States	2021-09-08	2018	PG	90 min	Children & Family Movies, Comedies	90.0	min
1697	s1698	TV Show	Survivor	NOT GIVEN	Jeff Probst, Jerri Manthey, Colby Donaldson, R...	United States	2020-11-15	2014	TV-14	2 Seasons	Reality TV	2.0	Seasons
7146	s7147	Movie	Jugaad	Susannah Heath-Eves	Not Given	Not Given	2018-09-15	2017	TV-14	52 min	Documentaries	52.0	min
2042	s2043	TV Show	Cleverman	NOT GIVEN	Hunter Page-Lochard, Rob Collins, Deborah Mail...	Australia, New Zealand, United States	2020-09-05	2017	TV-MA	1 Season	TV Dramas, TV Sci-Fi & Fantasy, TV Thrillers	1.0	Season
1971	s1972	Movie	Bhaji In Problem	Smeep Kang	Gippy Grewal, Gurpreet Guggi, Ragini Khanna, O...	India	2020-09-19	2013	TV-14	117 min	Comedies, International Movies	117.0	min
add Codeadd Markdown
# Reset the Index
df_reset = df.reset_index(drop=True)
print("\nDataFrame index has been reset.")

DataFrame index has been reset.
add Codeadd Markdown
#Publish
# Save as CSV 
df.to_csv('/kaggle/working/cleaned_netflix.csv', index=False)
add Codeadd Markdown



  git commit -m "update "  