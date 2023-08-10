#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install bs4')


# In[5]:


get_ipython().system('pip install requests')


# In[6]:


from bs4 import BeautifulSoup
import requests


# In[7]:


page=requests.get('https://www.imdb.com/search/title/?genres=action&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=f11158cc-b50b-4c4d-b0a2-40b32863395b&pf_rd_r=XZ8X52H1R40B7KG5SNZ9&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_1')


# In[8]:


page


# In[9]:


soup = BeautifulSoup(page.content)
soup


# Scraping the data

# In[11]:


SNo=soup.find('span',class_="lister-item-index unbold text-primary")


# In[12]:


SNo.text


# In[13]:


SNo=[] #emplty list
for i in soup.find_all('span',class_="lister-item-index unbold text-primary"):
 SNo.append(i.text)   


# In[14]:


SNo


# In[15]:


MovieName=[] #emplty list
for i in soup.find_all('h3',class_="lister-item-header"):
 MovieName.append(i.text)   


# In[16]:


MovieName


# In[17]:


year=[] #emplty list
for i in soup.find_all('span',class_="lister-item-year text-muted unbold"):
 year.append(i.text)    


# In[18]:


year


# In[19]:


Duration=[] #emplty list
for i in soup.find_all('span',class_="runtime"):
 Duration.append(i.text)    


# In[20]:


Duration


# In[21]:


ratings=[] #emplty list
for i in soup.find_all('div',class_="inline-block ratings-imdb-rating"):
 ratings.append(i.text)    


# In[22]:


ratings


# In[ ]:


#Scraping Director name -
 cast = store.find("p", class_ = '')
 cast = cast.text.replace('\n', '').split('|')
 cast = [x.strip() for x in cast]
 cast = [cast[i].replace(j, "") for i,j in enumerate(["Director:", "Stars:"])
 director_name.append(cast[0])


# In[2]:


#Creating an empty list, so that we can append the values -
# CSv - 1:
Sno = []
movie_name = []
director_name = []
duration = []
year = []
rating = []
metascore = []
#CSV - 2:
Stars = []
votes = []
genre = []
gross = []
popularity = []
certifcation = []


# In[10]:


#Storing the meaningfull required data in the variable
movie_data = soup.findAll('div', attrs= {'class': "lister-item mode-advanced"})
# <div class=>
# movie_name


# In[ ]:


#Calling molvie-name one-by-one using for loop and storing it in movie-name list
#For data-set -1 ->
for store in movie_data:
 name = store.h3.a.text
 movie_name.append(name) 
 
 serial = store.h3.find('span', class_ = "lister-item-index unbold text-primary")
 #print(serial)
 Sno.append(serial)
 runtime = store.p.find('span', class_ = 'runtime').text.replace(' min', '')
 duration.append(runtime) 
 
 #Scraping Director name -
 cast = store.find("p", class_ = '')
 cast = cast.text.replace('\n', '').split('|')
 cast = [x.strip() for x in cast]
 cast = [cast[i].replace(j, "") for i,j in enumerate(["Director:", "Stars:"])
 director_name.append(cast[0])
 
 year_of_release = store.h3.find('span', class_ = "lister-item-year text-mute
 #print(year_of_release)
 year.append(year_of_release)
 
 rate = store.find('div', class_ = 'inline-block ratings-imdb-rating').text.r
 rating.append(rate)
 
 meta = store.find('span', class_ = 'metascore').text.replace(' ', '') if st
 metascore.append(meta)
 
#For data-set -2 ->
 #Cast Details -- Stars
 cast = store.find("p", class_ = '')
 cast = cast.text.replace('\n', '').split('|')
 cast = [x.strip() for x in cast]
 cast = [cast[i].replace(j, "") for i,j in enumerate(["Director:", "Stars:"])
 Stars.append([x.strip() for x in cast[1].split(",")]) 
 
 #Since, gross and votes have same attributes, that's why we had created a co
 value = store.find_all('span', attrs = {'name': 'nv'}) 
 vote = value[0].text
 votes.append(vote)
 
 genre_list = store.p.find('span', class_ = "genre")
 #print("I am genre_list", genre_list)
 genre.append(genre_list)
 grosses = value[1].text if len(value) >1 else '*****'
 gross.append(grosses)
 
 #Popularity --
 rate = store.find('div', class_ = 'inline-block ratings-imdb-rating').text.r
 popularity.append(rate)
 
 #Description of the Movies -- Not explained in the Video, But you will figur
 #describe = store.find_all('p', class_ = 'text-muted')
 #description_ = describe[1].text.replace('\n', '') if len(describe) >1 else 
 #description.append(description_)
#Certifcation --
 certificate_list = store.p.find('span', class_ = "certificate")
 #print("I am certificate_list", certificate_list)
 certifcation.append(certificate_list)


# In[143]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# In[154]:


mov=pd.read_csv("movie_data (2).csv")


# In[155]:


mov


# In[ ]:





# In[8]:


import sqlite3


# In[9]:


db=sqlite3.connect("movie_data.csv_database.db")


# In[10]:


cur=db.cursor()


# In[11]:


with open('movie_data.csv','r') as file:
    total_records=0
    for row in file 
    cur.execute("insert into 'movie_2.csv' values(?,?,?,?,?,?)",row.split(","))
    
    total_records+=1
    print (total_records ,"Records inserted")   


# In[13]:


result=cur.execute("select * from movie_data.csv")
for row in result:
    print(row)


# EDA Analysis

# In[156]:


mov.head()


# In[157]:


mov.shape


# In[158]:


mov.dtypes


# In[159]:


mov.columns


# In[160]:


mov.info()


# In[161]:


mov.isnull().sum()


# # Missing data are present in director2,3,4,5,6 and  star 2 , 3,4 and genre 2,3 and certification

# In[162]:


mov.describe()


# In[83]:


# Now filling up the null values


# In[163]:


p=mov.replace(np.nan,0)


# In[164]:


mov.info()


# In[165]:


# so nan values are not present now


# In[166]:


# as we don't need director and star columns so we drop it.


# In[167]:


mov.Gross_collection.value_counts()


# In[168]:


sns.countplot(x="Director1",data=mov)
print(mov["Director1"].value_counts())


# In[169]:


sns.countplot(x="Certification",data=mov)
print(mov["Certification"].value_counts())


# In[170]:


sns.countplot(x="Genre1",data=mov)
print(mov["Genre1"].value_counts())


# In[171]:


sns.countplot(x="Genre2",data=mov)
print(mov["Genre2"].value_counts())


# In[172]:


sns.countplot(x="Genre3",data=mov)
print(mov["Genre3"].value_counts())


# In[173]:


# Bivariate analysis


# In[174]:


plt.scatter(mov["Ratings"],mov["Gross_collection"])
plt.show()


# In[175]:


plt.scatter(mov["Metascore"],mov["Gross_collection"])
plt.show()


# In[176]:


sns.distplot(mov["Gross_collection"],bins=10,kde=False)


# In[177]:


sns.distplot(mov["Votes"],bins=10,kde=False)


# In[178]:


sns.lmplot("Gross_collection","Ratings",data=mov,hue="Genre1")


# In[179]:


sns.lmplot("Gross_collection","Ratings",data=mov,hue="Genre2")


# In[180]:


mov.drop(["Sno","Director2","Director3","Director4","Director5","Director6","Star2","Star3","Star4","Genre2","Genre3","Certification"],axis=1,inplace=True)


# In[181]:


mov


# In[182]:


mov.dropna()


# In[183]:


from sklearn.preprocessing import LabelEncoder


# In[184]:


lb=LabelEncoder()


# In[185]:


mov["Director1"]=lb.fit_transform(mov["Director1"])


# In[186]:


mov["Star1"]=lb.fit_transform(mov["Star1"])


# In[187]:


mov["Genre1"]=lb.fit_transform(mov["Genre1"])


# In[188]:


mov.skew()


# In[189]:


mov.drop(["Year"],axis=1,inplace=True)


# In[191]:


mov.drop(["Movie_name"],axis=1,inplace=True)


# In[192]:


#votes is skewed so it should be removed
mov["Votes"]=np.cbrt(mov["Votes"])


# In[193]:


mov.dtypes


# In[194]:


mov.describe()


# # let us checking the corralation

# In[195]:


mov


# In[196]:


movcor=mov.corr()
movcor


# In[197]:


sns.heatmap(movcor)


# In[198]:


# creating the model.


# In[199]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split


# In[200]:


x = mov.drop('Gross_collection', axis=1).copy()
x


# In[212]:


y = mov['Gross_collection'].copy()
y


# In[213]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)


# In[214]:


x_train.shape


# In[215]:


y_train.shape


# In[216]:


x_test.shape


# In[217]:


y_test.shape


# In[218]:


lm=LinearRegression()


# In[219]:


lm.fit(x_train,y_train)


# In[220]:


lm.coef_


# In[221]:


lm.intercept_


# In[222]:


lm.score(x_train,y_train)


# In[223]:


# Here we can see that model is 100% accurate


# In[224]:


# predict the value


# In[225]:


pred=lm.predict(x_test)


# In[226]:


print("predicted result price:",pred)


# In[227]:


print("actual price",y_test)


# In[228]:


print("Error")
print("Mean Absolute Error:",mean_absolute_error(y_test,pred))
print("Mean Squared Error:",mean_squared_error(y_test,pred))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred)))


# In[231]:


from sklearn.metrics import r2_score
print(r2_score(y_test,pred))


# In[ ]:





# In[232]:


x = mov.drop('Votes', axis=1).copy()
x


# In[233]:


y = mov['Votes'].copy()
y


# In[256]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)


# In[257]:


lm=LinearRegression()


# In[258]:


lm.fit(x_train,y_train)


# In[259]:


lm.coef_


# In[260]:


lm.intercept_


# In[261]:


lm.score(x_train,y_train)


# In[262]:


# here we can see that model is not accurate. Now applying regressor to improve accuracy score.


# In[263]:


from sklearn.linear_model import Lasso


# In[264]:


ls=Lasso(alpha=.0001)


# In[265]:


ls.fit(x_train,y_train)
print(ls.score(x_train,y_train))
predlasso=ls.predict(x_test)
print(mean_squared_error(y_test,predlasso))


# In[266]:


from sklearn.metrics import r2_score
print(r2_score(y_test,pred))


# In[267]:


# voting model is not fitting


# In[ ]:




