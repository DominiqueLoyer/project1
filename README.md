# US Election 2020 Trump Vs. Biden \

## Final version Notebook
```python

!pip install tweepy
1. Authenticate to Twitter
# Import tweepy to work with the twitter API
import tweepy as tw

# Import numpy and pandas to work with dataframes
import numpy as np
import pandas as pd

# Import seaborn and matplotlib for viz
from matplotlib import pyplot as plt
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''
# Authenticate
auth = tw.OAuthHandler(consumer_key, consumer_secret)
# Set Tokens
auth.set_access_token(access_token, access_token_secret)
# Instantiate API
api = tw.API(auth, wait_on_rate_limit=True)
2. Get Tweets
hashtag = "#presidentialdebate"
query = tw.Cursor(api.search, q=hashtag).items(1000)
tweets = [{'Tweet':tweet.text, 'Timestamp':tweet.created_at} for tweet in query]
print(tweets)
df = pd.DataFrame.from_dict(tweets)
df.head()
trump_handle = ['DonaldTrump', 'Donald Trump', 'Donald', 'Trump', 'Trump\'s']
biden_handle = ['JoeBiden', 'Joe Biden', 'Joe', 'Biden', 'Biden\'s']
def identify_subject(tweet, refs):
    flag = 0 
    for ref in refs:
        if tweet.find(ref) != -1:
            flag = 1
    return flag

df['Trump'] = df['Tweet'].apply(lambda x: identify_subject(x, trump_handle)) 
df['Biden'] = df['Tweet'].apply(lambda x: identify_subject(x, biden_handle))
df.head(10)
3. Preprocess
# Import stopwords
import nltk
from nltk.corpus import stopwords

# Import textblob
from textblob import Word, TextBlob
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = stopwords.words('english')
custom_stopwords = ['RT', '#PresidentialDebate']
def preprocess_tweets(tweet, custom_stopwords):
    processed_tweet = tweet
    processed_tweet.replace('[^\w\s]', '')
    processed_tweet = " ".join(word for word in processed_tweet.split() if word not in stop_words)
    processed_tweet = " ".join(word for word in processed_tweet.split() if word not in custom_stopwords)
    processed_tweet = " ".join(Word(word).lemmatize() for word in processed_tweet.split())
    return(processed_tweet)

df['Processed Tweet'] = df['Tweet'].apply(lambda x: preprocess_tweets(x, custom_stopwords))
df.head()
print('Base review\n', df['Tweet'][0])
print('\n------------------------------------\n')
print('Cleaned and lemmatized review\n', df['Processed Tweet'][0])
4. Calculate Sentiment
# Calculate polarity
df['polarity'] = df['Processed Tweet'].apply(lambda x: TextBlob(x).sentiment[0])
df['subjectivity'] = df['Processed Tweet'].apply(lambda x: TextBlob(x).sentiment[1])
df[['Processed Tweet', 'Biden', 'Trump', 'polarity', 'subjectivity']].head()
display(df[df['Trump']==1][['Trump','polarity','subjectivity']].groupby('Trump').agg([np.mean, np.max, np.min, np.median]))
df[df['Biden']==1][['Biden','polarity','subjectivity']].groupby('Biden').agg([np.mean, np.max, np.min, np.median])
5. Visualise
biden = df[df['Biden']==1][['Timestamp', 'polarity']]
biden = biden.sort_values(by='Timestamp', ascending=True)
biden['MA Polarity'] = biden.polarity.rolling(10, min_periods=3).mean()

trump = df[df['Trump']==1][['Timestamp', 'polarity']]
trump = trump.sort_values(by='Timestamp', ascending=True)
trump['MA Polarity'] = trump.polarity.rolling(10, min_periods=3).mean()
trump.head()
repub = 'red'
demo = 'blue'
fig, axes = plt.subplots(2, 1, figsize=(13, 10))

axes[0].plot(biden['Timestamp'], biden['MA Polarity'])
axes[0].set_title("\n".join(["Biden Polarity"]))
axes[1].plot(trump['Timestamp'], trump['MA Polarity'], color='red')
axes[1].set_title("\n".join(["Trump Polarity"]))

fig.suptitle("\n".join(["Presidential Debate Analysis"]), y=0.98)

plt.show()
 

```
## Final version
```Python
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install tweepy"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. Authenticate to Twitter"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import tweepy to work with the twitter API\n",
    "import tweepy as tw\n",
    "\n",
    "# Import numpy and pandas to work with dataframes\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "# Import seaborn and matplotlib for viz\n",
    "from matplotlib import pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "consumer_key = ''\n",
    "consumer_secret = ''\n",
    "access_token = ''\n",
    "access_token_secret = ''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Authenticate\n",
    "auth = tw.OAuthHandler(consumer_key, consumer_secret)\n",
    "# Set Tokens\n",
    "auth.set_access_token(access_token, access_token_secret)\n",
    "# Instantiate API\n",
    "api = tw.API(auth, wait_on_rate_limit=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2. Get Tweets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "hashtag = \"#presidentialdebate\"\n",
    "query = tw.Cursor(api.search, q=hashtag).items(1000)\n",
    "tweets = [{'Tweet':tweet.text, 'Timestamp':tweet.created_at} for tweet in query]\n",
    "print(tweets)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.DataFrame.from_dict(tweets)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "trump_handle = ['DonaldTrump', 'Donald Trump', 'Donald', 'Trump', 'Trump\\'s']\n",
    "biden_handle = ['JoeBiden', 'Joe Biden', 'Joe', 'Biden', 'Biden\\'s']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def identify_subject(tweet, refs):\n",
    "    flag = 0 \n",
    "    for ref in refs:\n",
    "        if tweet.find(ref) != -1:\n",
    "            flag = 1\n",
    "    return flag\n",
    "\n",
    "df['Trump'] = df['Tweet'].apply(lambda x: identify_subject(x, trump_handle)) \n",
    "df['Biden'] = df['Tweet'].apply(lambda x: identify_subject(x, biden_handle))\n",
    "df.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 3. Preprocess"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import stopwords\n",
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "\n",
    "# Import textblob\n",
    "from textblob import Word, TextBlob"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "nltk.download('stopwords')\n",
    "nltk.download('wordnet')\n",
    "stop_words = stopwords.words('english')\n",
    "custom_stopwords = ['RT', '#PresidentialDebate']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocess_tweets(tweet, custom_stopwords):\n",
    "    processed_tweet = tweet\n",
    "    processed_tweet.replace('[^\\w\\s]', '')\n",
    "    processed_tweet = \" \".join(word for word in processed_tweet.split() if word not in stop_words)\n",
    "    processed_tweet = \" \".join(word for word in processed_tweet.split() if word not in custom_stopwords)\n",
    "    processed_tweet = \" \".join(Word(word).lemmatize() for word in processed_tweet.split())\n",
    "    return(processed_tweet)\n",
    "\n",
    "df['Processed Tweet'] = df['Tweet'].apply(lambda x: preprocess_tweets(x, custom_stopwords))\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('Base review\\n', df['Tweet'][0])\n",
    "print('\\n------------------------------------\\n')\n",
    "print('Cleaned and lemmatized review\\n', df['Processed Tweet'][0])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 4. Calculate Sentiment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate polarity\n",
    "df['polarity'] = df['Processed Tweet'].apply(lambda x: TextBlob(x).sentiment[0])\n",
    "df['subjectivity'] = df['Processed Tweet'].apply(lambda x: TextBlob(x).sentiment[1])\n",
    "df[['Processed Tweet', 'Biden', 'Trump', 'polarity', 'subjectivity']].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "display(df[df['Trump']==1][['Trump','polarity','subjectivity']].groupby('Trump').agg([np.mean, np.max, np.min, np.median]))\n",
    "df[df['Biden']==1][['Biden','polarity','subjectivity']].groupby('Biden').agg([np.mean, np.max, np.min, np.median])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Visualise"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "biden = df[df['Biden']==1][['Timestamp', 'polarity']]\n",
    "biden = biden.sort_values(by='Timestamp', ascending=True)\n",
    "biden['MA Polarity'] = biden.polarity.rolling(10, min_periods=3).mean()\n",
    "\n",
    "trump = df[df['Trump']==1][['Timestamp', 'polarity']]\n",
    "trump = trump.sort_values(by='Timestamp', ascending=True)\n",
    "trump['MA Polarity'] = trump.polarity.rolling(10, min_periods=3).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "trump.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "repub = 'red'\n",
    "demo = 'blue'\n",
    "fig, axes = plt.subplots(2, 1, figsize=(13, 10))\n",
    "\n",
    "axes[0].plot(biden['Timestamp'], biden['MA Polarity'])\n",
    "axes[0].set_title(\"\\n\".join([\"Biden Polarity\"]))\n",
    "axes[1].plot(trump['Timestamp'], trump['MA Polarity'], color='red')\n",
    "axes[1].set_title(\"\\n\".join([\"Trump Polarity\"]))\n",
    "\n",
    "fig.suptitle(\"\\n\".join([\"Presidential Debate Analysis\"]), y=0.98)\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
```
# version 1 \

```python
from tweepy.streaming import StreamListener  
from tweepy import OAuthHandler  
from tweepy import Stream  
   
import twitter_credentials  
   
# # # # TWITTER STREAMER # # # #  
class TwitterStreamer():  
    """  
    Class for streaming and processing live tweets.    """    def __init__(self):  
        pass  
  
    def stream_tweets(self, fetched_tweets_filename, hash_tag_list):  
        # This handles Twitter authetification and the connection to Twitter Streaming API  
        listener = StdOutListener(fetched_tweets_filename)  
        auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)  
        auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)  
        stream = Stream(auth, listener)  
  
        # This line filter Twitter Streams to capture data by the keywords:   
stream.filter(track=hash_tag_list)  
  
  
# # # # TWITTER STREAM LISTENER # # # #  
class StdOutListener(StreamListener):  
    """  
    This is a basic listener that just prints received tweets to stdout.    """    def __init__(self, fetched_tweets_filename):  
        self.fetched_tweets_filename = fetched_tweets_filename  
  
    def on_data(self, data):  
        try:  
            print(data)  
            with open(self.fetched_tweets_filename, 'a') as tf:  
                tf.write(data)  
            return True  
        except BaseException as e:  
            print("Error on_data %s" % str(e))  
        return True  
            
    def on_error(self, status):  
        print(status)  
  
   
if __name__ == '__main__':  
   
    # Authenticate using config.py and connect to Twitter Streaming API.  
    hash_tag_list = ["donal trump", "hillary clinton", "barack obama", "bernie sanders"]  
    fetched_tweets_filename = "tweets.txt"  
  
    twitter_streamer = TwitterStreamer()  
    twitter_streamer.stream_tweets(fetched_tweets_filename, hash_tag_list)
```
