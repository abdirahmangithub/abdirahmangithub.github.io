---
title: web scraping
categories: [data science, data analysis,data wrangling]

tags : data , data wrangling


---
## Web Scraping and Data Handling in Python

## Introduction
As trainees of DATA and ARTIFICIAL INTELLIGENCE, My first weeks assignment was to start extracting data using web scraping for analysis. I was totally new to the tools we were introduced to. I had never written Python code before and had not created a Colab account. However, it was a very effective way of learning by seeing actual results.
The link to the website i did scrape on is (https://www.scrapethissite.com/pages/forms/).

Our tutors have taken us through and supported us, they also helped me gain insight on soft skills that prepared me as a responsible being.
The objectives of the assignment were:
Practical Python coding on Jupiter Notebooks hosted on Google Colab
Use requests and BeautifulSoup to extract data from a web page.
Parse and clean the extracted data.
Store structured data into a Pandas DataFrame.
Export the final dataset to a .csv file.
Tasks Completed
I started by opening file at colab, a hosted jupyter notebook service . and named the file as abdirahman webscrapping.ipynb

 
I started the web scrapping coding by first importing python libraries like 
Request
Beautifulsoup  and
Pandas

I define the url of the website to be scrapped.
I then used the requests library to get the HTML content of the webpage i wanted to scrape. i made a GET request to the website's URL and store the response in a variable:
 i then Used BeautifulSoup to parse the HTML content and make it easier to extract data.





Then use find_all  function with soup to get the table content and saved the result to object,hockeytable  
And  printed the table content using print function





 Sample of result is shown here 



Then i started finding all column heading content and printed it in form of list.



And sample of its result is shown as follows



Saved the column headings into pandas DataFrame



And called the dataframe to get result sample as follows







Then got all raw content data

Then extracted the data row by row, looped through each raw  while stripping and saving data into the dataFrames
for row in table_data[1:]:
  raw_data = row.find_all('td')
  individualrawdata = [data.text.strip() for data in raw_data]
  length = len(df)
  df.loc[length] =  individualrawdata



Inspected the resulting dataframe to get result sample as follows



i then mounted google drive before converting my file to csv and saved it directly to my google drive

Result was 


Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).




Finally save the file as .csv file in google drive and put index to false so that it does not get included in my csv file





csv Result in my google drive was as follows



## Conclusion
This week I gained a good grounding on the introductory concepts relating to data science and artificial intelligence. I am getting a better understanding that I can build on as we work on more advanced concepts in later weeks. I have posted my write up on my blog linkedln and I look forward to building a portfolio that I can showcase on my CV as I look for jobs in Data and AI.
All credit goes to my teachers, salute you.




Here is the link to the code


Link to Code: https://colab.research.google.com/drive/1hyIXleHlOkQHaLXoG96AZuOyyjXodkD8?usp=sharing
Link to csv file .
https://drive.google.com/file/d/14oT-xRRrBOhfaM_sXpHUZtdKRzJ0BPPx/view?usp=sharing


