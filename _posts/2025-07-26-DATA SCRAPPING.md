---
title: data scrapping
categories: [data science, data]

tags : data, data scrapping

---
ABDIRAHMAN ABDULLAHI ISMAIL.
WEB SCRAPING PROJECT


#importing libraries that i will be needing for web scraping

from bs4 import BeautifulSoup
import requests
import pandas as pd


# New section

## New section

i Used the requests library to get the HTML content of the webpage i wanted to scrape. i made a GET request to the website's URL and store the response in a variable:
i then Used BeautifulSoup to parse the HTML content and make it easier to extract data.






url = "https://www.scrapethissite.com/pages/forms/"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
print(soup)

url

# i used find_all function with soup to get the table content and saved the result to object, hockeytable
hockeytable =soup.find_all("table", class_="table")

print (hockeytable)



# here i find all column heading content and printed it in form of list
titles= soup.find_all("th")
hockeytable1 = [title.text.strip()for title in titles]
print(hockeytable1)


#Save the column headings onto a Pandas DataFrame

df =pd.DataFrame(columns=hockeytable1)
df




#First get all rows
table_data = soup.find_all('tr')


#Extract the data row by row, then loop through each while stripping and saving data into the DataFrame

for row in table_data[1:]:
  raw_data = row.find_all('td')
  individualrawdata = [data.text.strip() for data in raw_data]
  length = len(df)
  df.loc[length] =  individualrawdata

#Inspect the resulting DataFrame
df


# mount google drive to save the file directly to my Google Drive
from google.colab import drive
drive.mount('/content/drive')

#Save to a .csv file in google drive

df.to_csv('/content/drive/My Drive/Copy of aai webscrapping.csv', index=False)






### **New section**