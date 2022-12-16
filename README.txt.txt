
Introduction
Airbnb currently is considered as the market leaders when it comes to the question of vaccation rental,lodging, and toursism activities. 
This blog aims at understanding the positive and negative outlook for the Airbnb rental property as well as the host who owns those rental property. This analysis will assist the Airbnb as well
as the hosts to improve upon certain shortcomings that the guests have pointed and on the other hand utilize the positive view as the standards for the current as well as for any prospective Airbnb hosts. 
Secondly, this blog will be performing an idepth analysis on the rental home types which are preferred by the tourist while booking Airbnb in Seatle area  
and lastly we will be performing an analysis to understand the earning potential for Airbnb hosts in different time frames and in different parts of Seatle .We will be utilzing the customers comments for understanding
amenities pattern, property type for understanding the tourist preferred proprty type when it comes to booking Airbnb vaccation homes in Seatle area.




We will answering three core questions in this project, which are as follow:

1.What are the key shortcomings\area for improvements that the guests at any Seatle Rental Property have pointed out?
2.What are the rental property types for the Airbnb Seatle area gets the highest booking ?
3.What are the time frmaes and area in Seatle that generates highest revenue? 

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

                                   Data Gathering 


 Loading and saving Airbnb Seatle Listings, Reviews and Calendar data in individual dataframes 

AIRBNB_Seatle_Listings = pd.read_csv("\\Udacity\\Udacity_AIRBNB_Project\\Seatle_AIRBNB\\listings.csv")
AIRBNB_Seatle_Reviews = pd.read_csv("\\Udacity\\Udacity_AIRBNB_Project\\Seatle_AIRBNB\\reviews.csv")
AIRBNB_Seatle_Calendar = pd.read_csv("\\Udacity\\Udacity_AIRBNB_Project\\Seatle_AIRBNB\\calendar.csv")

Business Understanding, Data Understanding and Data Preprocessing 
Airbnb Inc, is an American Corporation headquarterd in Sanfransisco California is an online platform which allows its users to book vaccation homes on rent for their short-term stay at a destination of their choice worldwide. As of July, 20222 Airbnb have a strong network of more than 2.9 millions with almost 14,000 additional hosts gets added every month.https://www.stratosjets.com/blog/airbnb-statistics/#:~:text=How%20many%20hosts%20does%20Airbnb,platform%20each%20month%20in%202022. The prime source of revenue for Airbnb comes from the commission fees of online booking that a users books. The revenue for Airbnb can thus be increased by finding ways that can speed up the magnitude of potential hosts on website and retaining those current customes who might be thinking of going off from Airbnb platform. This blog will be looking into the customer 'notes' metrics and try to identify which amenities customer might be responding to on the most posive manner that can encourage other hosts as well to add those amenities in thier current property for customer retention. Further, this blog will be looking at the various property types that gets booked maximun by the users and the earning potential for that property type that will act as an incentive for any prospective host to host their property in Seatle Area.

Data Preprocessing
For understanding the customer sentiments we will be utilizing the comments field from the Airbnb_Seatle_Reviews dataframe.
We will be performing Natural Language Processing method to parse out the customers comments to understand their views for individual property in Seatle



Creating a copy of AIRBNB_Seatle_Reviews data for processing comments column
Comments = AIRBNB_Seatle_Reviews['comments'].copy()

Dropping any missing values from the comments column
Comments = Comments.dropna()
Comments = Comments.replace('[\!""@#$%^&*-+={:>/?\r,]', '', regex=True).astype(str) 


Some of the nltk libraries imported for performing sentimental analysis
nltk.download(["stopwords","pros_cons","tagsets","opinion_lexicon","vader_lexicon"]) # for matching against positive and negative words 
words = words.words('en') ## this is for matching against our list of words later
Unwanted_Words = nltk.corpus.stopwords.words("english") # for filtering out any unwanted english words 

# Data Preprocessing
# For understanding the customer sentiments we will be utilizing the comments field from the Airbnb_Seatle_Reviews dataframe.
# we will be performing Natural Language Processing method to parse out the customers comments to understand their views for individual property in Seatle
# Creating empty dataframes for appending and subprocessing steps
Top_Negative_words = list()
Top_Positive_words = list()
TopTenPositiveWords = list()
TopTenPositiveWordsCounts = list()
TopTenNegativeWords = list()
TopTenNegativeWordsCounts = list()


For the purpose of sentimental analysis, I used NLTK pretrained Sentimental Analyzer. This algorithim analyzer is pretrained and provides output in the short period of time with high efficiency

Top_Neg_Words and Top Pos_Words
This functions will perform the following task: -Dropping any missing values from the comments column since we do not want to impute values for comments -Act as parser for data cleansing and manipulation on comments column -Creates, calculates and parse out the Sentemental Analyzer score values. I have established a threshold value of .50 for negative reviews as this was found to be picking out the most reasonable negative words used in the comments. and positive scores. This score was the minimum score established for filtering out the negative from positive sentimental strings -Function further creates tokenized positive and negative strings from the comments column -Function will create an individual wordstags list from the positive and negative tokenized strings created in Seattle comments function -Function will filter out positve and negative sentimental words and further will append them in All positive and All Negative words list. I have used verb adverb (VBG) criterea to understand guests positive and negative comments words. -These funcions will create a list for negative letters along with their individual count. It further will parse out the top ten negative letters and their counts.

Unique_Words
This function will find the similar words that fell on both positive and negative words lis -It then deletes those common words from one both positive and negative words list -The function then picks up the 10 most common positive and negative words along with their counts from the rest of the positive and negative lists

Pos_Words_Visual

This function will initiall create a Positive Dataframe from the TopTenPositiveWords list Dataframes

Finally the functions will create a horizontal bar visualization for Top Ten Positive Words Neg_Words_Visual

This function will initiall create a Negative Dataframe from the TopTenNegativeWords list Dataframes

Finally the functions will create a horizontal bar visualization for Top Ten Negative Words

Note-These functions can be run together and seperatly, I preferred running them individually so that the output can be discussed after every function run

Output Analysis
Negative Review Words Some of negative words that are used by the guests as part of negative reviews are Starving, depressing, stealing, struggling, Flushing, Shaking, realizing, Disgusting, Celling, and Poisoning. 
Though the number of positive words outnumbers the number of negative words by large but Airbnb being a service sector should not overlook even a single negative reviews used by any of its guests.
Words such as Starving, Stealing, Flushing, and Celling are the four words that caught my attention. Words like Struggling and Stealing directly focuses on the bad experience that the guests experienced by the hosts personality
whereas words such as Flushing and Celling pertains to the bad state of host's rental property. These words reflects that the guests feels cheated by their hosts for their stay at Seattle Airbnb. Negative reviews like 
these should be avoided at all cost as as it can hurt Airbnb reputation in the long run.



Question 2
What are the rental property types for the Airbnb Seatle area gets the highest booking ?
In order to understand the most preffered proprty types by the customers in Seatle Area we will be rating property types on the the mean values for the review score ratings that a particular property type have received
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

Output Analysis:

As the reveiew scores rating suggested and the visualization also certifies,that property type such as "Appartment" ,"House" and "Townhouse" are the most prefferred rental property for travellers in Seattle area.
This analysis can be utilized by Airbnb in Seattle to push for Apartments, House and Townhouse owners to start hosting their property if they have not already done so. This can prove to be a substantial push in increasing the 
host footfall on Airbnb Seattle platform.

Revenue by Neighbourhood


#Question 3:
How much Airbnb homes are earning in certain time frames and areas.?
Filtering only columns that we will use for analysis. Total Rental value have been calculated by Price Per Day multiplied by Number of Days there was availability in a particular rental property. 
This is a hypothetical situation that a rental property will be occupied by guests on days on which its available but the liklihood for a renatl property to make rental income is very likely on days 
when its available than when its not available.


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

Output Analysis
Months of September, October, November, and December is considered to highest revenue generator while January tends prove to be a slowest month in terms of rental income for Airbnb Seattle. 
February seems to be a month when business starts getting its traction until the end of the year when business really booms for Airbnb in the Seattle Area.       


Revenue by Neighbourhood
Output Analysis Output signifies that areas such as Windermere, Alki,Belltown,and Westlake are amongst areas with high mean reantal income where as areas such as Atlantic,Adams, West Woodland, and Whittier Heights are some of
 the areas that are lowest in terme of mean reantal income generation. Airbnb along with the hosts from areas with high rental income should have an information sessions to dicuss and find ways and come up with a strategy that can 
focus on assist in increasing rental income growth

Conclusion
This article leaves with three main conclusion that might help Airbnb Seattle if taken into consideration:
. Negative review words such as Lying, questioning, intimidating should be taken into consideration and appropriate actions must be taken 
. Positive words such as Amazing, Sparkling, Charming,Comforting, and Loving should be taken as words of pride and must be looked as the standards to aim for other hosts
. Owners for Apartments, Townhouse, and Houses must be reached out in Seatle area and be conviced to start hosting for Airbnb if not already 
. Appropriate strategy should be made for increasing the rental income in the month of January 
. Airbnb Seattle should work with the hosts in areas such as Windermere, Westlake, Alki, and Arbor Heights along with the low  performing areas like Atlantic, Adams, West Woodland and Whittier Heights for understanding and working towards revenue stabilizing throughout the Seattle.                                                                          