
Introduction
Airbnb currently is considered as the market leaders when it comes to the question of vaccation rental,lodging, and toursism activities. 
This blog aims at understanding the positive and negative outlook for the Airbnb rental property as well as the host who owns those rental property. This analysis will assist the Airbnb as well
as the hosts to improve upon certain shortcomings that the guests have pointed and on the other hand utilize the positive view as the standards for the current as well as for any prospective Airbnb hosts. 
Secondly, this blog will be performing an idepth analysis on the rental home types which are preferred by the tourist while booking Airbnb in Seatle area  
and lastly we will be performing an analysis to understand the earning potential for Airbnb hosts in different time frames and in different parts of Seatle .We will be utilzing the customers comments for understanding
amenities pattern, property type for understanding the tourist preferred proprty type when it comes to booking Airbnb vaccation homes in Seatle area.




We will answering three core questions in this blogs, which are as follow:

What are the key shortcomings\area for improvements that the guests at any Seatle Rental Property have pointed out?
What are the rental property types for the Airbnb Seatle area gets the highest booking ?
What are the time frmaes and area in Seatle that generates highest revenue? 

# this is for Airbnb Seatle data analysis and this work would be one of the outcome of the Udicity data scientist nanodegree project 
# Importing all the necessary libraries and packages. We will be using pythons nltk package and subsequent libraries for parsing the review column for natural language process 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
import seaborn as sns



## Loading and saving Airbnb Seatle Listings, Reviews and Calendar data in individual dataframes 

AIRBNB_Seatle_Listings = pd.read_csv("\\Udacity\\Udacity_AIRBNB_Project\\Seatle_AIRBNB\\listings.csv")
AIRBNB_Seatle_Reviews = pd.read_csv("\\Udacity\\Udacity_AIRBNB_Project\\Seatle_AIRBNB\\reviews.csv")
AIRBNB_Seatle_Calendar = pd.read_csv("\\Udacity\\Udacity_AIRBNB_Project\\Seatle_AIRBNB\\calendar.csv")

Business Understanding, Data Understanding and Data Preprocessing

Airbnb Inc, is an American Corporation headquarterd in Sanfransisco California is an online platform which allows its users to book vaccation homes on rent for their short-term stay 
at a destination of their choice worldwide. As of July, 20222 Airbnb have a strong network of more than 2.9 millions with almost 14,000 additional hosts gets added every month.https://www.stratosjets.com/blog/airbnb-statistics/#:~:text=How%20many%20hosts%20does%20Airbnb,platform%20each%20month%20in%202022.  
The prime source of revenue for Airbnb comes from the commission fees of online booking that a users books. The revenue for Airbnb can thus be increased by finding ways that can speed up the magnitude of potential hosts on website 
and retaining those current customes who might be thinking of going off from Airbnb platform. This blog will be looking into the customer 'notes' metrics and try to identify which amenities customer might be responding to on the most posive manner that can encourage other hosts as well to add those amenities in thier current property for 
customer retention. Further, this blog will be looking at the various property types that gets booked maximun by the users and the earning potential for that property type that will act as an incentive for any prospective host to host their property in Seatle Area.


## Data Preprocessing
## For understanding the customer sentiments we will be utilizing the comments field from the Airbnb_Seatle_Reviews dataframe.
## we will be performing Natural Language Processing method to parse out the customers comments to understand their views for individual property in Seatle

# Creating a copy of AIRBNB_Seatle_Reviews data for processing comments column
Comments = AIRBNB_Seatle_Reviews['comments'].copy()

## Dropping any missing values from the comments column
Comments = Comments.dropna()
Comments = Comments.replace('[\!""@#$%^&*-+={:>/?\r,]', '', regex=True).astype(str) 


# we need to download all the nltk tool kits befor we start working on the customer reviews sentiments\n",
nltk.download(["stopwords","pros_cons","tagsets","opinion_lexicon","vader_lexicon"]) # for matching against positive and negative words 
words = words.words('en') ## this is for matching against our list of words later
Unwanted_Words = nltk.corpus.stopwords.words("english") # for filtering out any unwanted english words 

# Data Preprocessing
# For understanding the customer sentiments we will be utilizing the comments field from the Airbnb_Seatle_Reviews dataframe.
# we will be performing Natural Language Processing method to parse out the customers comments to understand their views for individual property in Seatle
# Creating empty dataframes for appending and subprocessing steps
Comments_New = []
Positive_Reviews = []
Negative_Reviews = []
Negative_Word_Tokenize = list()
Positive_Word_Tokenize = list()
Negative_Words_Tagged = list()
Positive_Words_Tagged = list()
All_Neg_Words = []
All_Pos_Words = []
Top_Negative_words = []
Top_Negative_words_counts = []
Top_Positive_words = []
Top_Positive_words_counts = []


# For the purpose of sentimental analysis, I used NLTK pretrained Sentimental Analyzer. This algorithim analyzer is pretrained and provides output in the short period of time with high efficiency

Comments = AIRBNB_Seatle_Reviews['comments'].copy() # Creating a copy of AIRBNB_Seatle_Reviews data for processing comments column

Comments = Comments.dropna() # Dropping any missing values from the comments column since we do not want to impute values for comments

for com in Comments:
    Comments_New.append(re.findall('[A-Z][^A-Z]*', com)) # Replacing any special characters from the comments string

 # This script will loop over every string from the list of cleansed string and will score by running the nltk automated sentiment analyzer and saving the output in Positive and Negative Data frames
for strings in Comments_New:
    for string in strings:
        sia = SentimentIntensityAnalyzer()                    
        if sia.polarity_scores(string)['neg'] > .20 : # I put a threshold of .20 negetive values so that the algorithim only pick up strings which are trully negetive 
            Negative_Reviews.append({"Sia_Polarity_Score": sia.polarity_scores(string) ,"Comments": string})  
        else:
            Positive_Reviews.append({"Sia_Polarity_Score": sia.polarity_scores(string) ,"Comments": string})  

# Looping through Negative and Positive Reviews and Saving them in a Tokenized word list
for i in Negative_Reviews:
    if not i['Comments'].lower() in Unwanted_Words: ## filtering the unwanted or stop words
        Negative_Word_Tokenize.append(word_tokenize(i['Comments'])) # tokenizing the string into word list
                
for j in Positive_Reviews:
    if not j['Comments'] in Unwanted_Words:## filtering the unwanted or stop words
        Positive_Word_Tokenize.append(word_tokenize(j['Comments']))   # tokenizing the string into word list

# Looping through the tokenized list of negative and positive words and saving them in a tagged list 
for neg_words in Negative_Word_Tokenize: # tagging the negative words 
    Negative_Words_Tagged.append(nltk.pos_tag(neg_words))
        
for pos_words in Positive_Word_Tokenize:
    Positive_Words_Tagged.append(nltk.pos_tag(pos_words))# tagging the positive words 

# I utilized "Verb,Present Participle" to filter out the words representing negative and positive views of the guests in their comments
for tag_words in Negative_Words_Tagged:
    for tag_word in tag_words:
        if 'NN' in tag_word:
            All_Neg_Words.append(tag_word[0])
            
for tag_words in Positive_Words_Tagged:
    for tag_word in tag_words:
        if 'NN' in tag_word:
            All_Pos_Words.append(tag_word[0])


colors = cm.rainbow(np.linspace(0, 1, 10))
fig_1 = plt.figure()
plt.title('10 Negative Review Words With count')
plt.xlabel('Top_Negative_Review_words')
plt.ylabel('counts')
plt.barh(Top_Negative_words,Top_Negative_words_counts, color=colors)
plt.show()

positive_counted_words = collections.Counter(All_Pos_Words)
for letter, count in positive_counted_words.most_common(10):
    Top_Positive_words.append(letter)
    Top_Positive_words_counts.append(count)
    print(letter,count)


colors = cm.rainbow(np.linspace(0, 1, 10))
fig = plt.figure()
plt.title('10 Positive Review Words With count')
plt.xlabel('Top_Positive_Review_words')
plt.ylabel('counts')
plt.barh(Top_Positive_words,Top_Positive_words_counts, color=colors)
plt.show()


#Question 2
# What are the rental property types for the Airbnb Seatle area gets the highest booking 
# In order to understand the most preffered proprty types by the customers in Seatle Area we will be looking at the proprty types on the basis of review score ratings\n",
# and the amount of reviews per month that these rental properties have got.  

AIRBNB_Seatle_Property_Types = AIRBNB_Seatle_Listings.filter(["id","property_type","review_scores_rating","reviews_per_month"], axis=1).copy()

## Identifying columns that have highe missing value percentages
AIRBNB_Seatle_Property_Types.isna().sum()/len(AIRBNB_Seatle_Property_Types) *100 
    
# Below figure clearly shows that review_scores_rating and reviews_per_month contains the higest percentages for missing values.\n",

id                       0.000000
property_type            0.026192
review_scores_rating    16.946045
reviews_per_month       16.422211
dtype: float64


# Since we will be utilzing both these columns for further analysis we will be imputing values for missing rows 
# For imputing the missing values for review_scores_rating and reviews_per_month columns, we utilzied multiple imputation chained equation (MICE) method with linear regression Baysian Ridge.\n",
# This method will help in imputing values for missing rows in a scaled and effective way.
    
# Creating Mice Imputer
Mice_Imputer = IterativeImputer(estimator=linear_model.BayesianRidge(),n_nearest_features=None, imputation_order='ascending')    
## Fitting the imputed values for both the missing columns",
AIRBNB_Seatle_Property_Types['review_scores_rating'] = Mice_Imputer.fit_transform(AIRBNB_Seatle_Property_Types['review_scores_rating'].to_frame())
AIRBNB_Seatle_Property_Types['reviews_per_month'] = Mice_Imputer.fit_transform(AIRBNB_Seatle_Property_Types['reviews_per_month'].to_frame())
    
# Before we perform any calculation on the imputed columns we need to change the data types as "int64" else it will output in 'NAN' values 
    
AIRBNB_Seatle_Property_Types['review_scores_rating'] = AIRBNB_Seatle_Property_Types['review_scores_rating'].astype("int64")
AIRBNB_Seatle_Property_Types['reviews_per_month'] = AIRBNB_Seatle_Property_Types['reviews_per_month'].astype("int64")

#Calculating the mean values for 'review score rating' for each rental property type
AIRBNB_Seatle_Property_Review_Score = AIRBNB_Seatle_Property_Types.groupby(["property_type"])['review_scores_rating'].sum()/len(AIRBNB_Seatle_Property_Types['review_scores_rating'])
AIRBNB_Seatle_Property_Review_Score

property_type
Apartment          42.042954
Bed & Breakfast     0.922734
Boat                0.199057
Bungalow            0.328706
Cabin               0.529335
Camper/RV           0.324515
Chalet              0.043740
Condominium         2.264274
Dorm                0.047145
House              42.985595
Loft                0.997381
Other               0.552907
Tent                0.123625
Townhouse           2.960189
Treehouse           0.074908
Yurt                0.026192
Name: review_scores_rating, dtype: float64

#Below output as per review scores rating clearly suggests that the property type such as "Appartment" ,"House" and "Townhouse"  remains the most prefferred rental property for travellers in Seattle"
AIRBNB_Seatle_Property_Types.groupby(['property_type']).boxplot(column='review_scores_rating',subplots=False,rot=60,figsize=(20,10))
plt.figure()


#Question 3:
##  How much Airbnb homes are earning in certain time frames and areas.

AIRBNB_Seatle_Listing_Price = AIRBNB_Seatle_Calendar.copy() ## creating a copy from Calendar df
    
AIRBNB_Seatle_Listing_Price['available'] = np.where(AIRBNB_Seatle_Listing_Price["available"] == "t",1,0 ) ## encoding boolean values 
    
AIRBNB_Seatle_Listing_Price =   AIRBNB_Seatle_Listing_Price.rename(columns ={"listing_id": "id"}) ## for keeping consistency btween  dataframes
    
AIRBNB_Seatle_Listing_Price['price'] = AIRBNB_Seatle_Listing_Price['price'].replace('[\\$,+]', '', regex=True).astype(float)
    
    
# Imputing missig values for price column
Mean_Imputer = SimpleImputer(strategy='mean')
AIRBNB_Seatle_Listing_Price['price']  = Mean_Imputer.fit_transform(AIRBNB_Seatle_Listing_Price['price'].values.reshape(-1,1))
AIRBNB_Seatle_Listing_Price['price'] = round(AIRBNB_Seatle_Listing_Price['price'],2)
AIRBNB_Seatle_Listing_Price['month'] = pd.DatetimeIndex(AIRBNB_Seatle_Listing_Price['date']).month_name()
   
## Joining Seatle Listing and Calendar Dataframe 
Airbnb_Seatle_Revenue = pd.merge(AIRBNB_Seatle_Listings,AIRBNB_Seatle_Listing_Price,how='inner',on='id')

## Filtering only columns that we will use for analysis
Airbnb_Seatle_Revenue = Airbnb_Seatle_Revenue.filter(['id','property_type','neighbourhood_cleansed','price_y','available','month','availability_365']).copy()
    
## Renaming some of the column names
Airbnb_Seatle_Revenue = Airbnb_Seatle_Revenue.rename(columns = {'property_type':'Property_Type','price_y':'Price_Per_Day','availability_365': 'Available_Yearly','neighbourhood_cleansed':'Neighbourhood','month':'Month'})
# Created a new calculated field based on the price per day of a rental property based on the availabilty 
Airbnb_Seatle_Revenue['Total_Rental'] = Airbnb_Seatle_Revenue['Price_Per_Day'] * Airbnb_Seatle_Revenue['available']
Airbnb_Seatle_Revenue['Total_Rental'] = round(Airbnb_Seatle_Revenue['Total_Rental'],2)    
#
Airbnb_Seatle_Revenue.groupby(['Month'])['Total_Rental'].mean()
                                                                                  
 Month
April         89.683700
August        97.181255
December     100.958803
February      82.308439
January       69.683499
July          95.377854
June          99.453597
March         91.233047
May           94.281823
November      96.879911
October       95.444659
September     96.612092
Name: Total_Rental, dtype: float64


## Airbnb Seatle Revenue by Months
Airbnb_Seatle_Revenue.groupby(['Month']).boxplot(column='Total_Rental',subplots=False,rot=60,figsize=(20,10))
plt.figure()                                                                                    