import pandas as pd
import snscrape.modules.twitter as sntwitter
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import tqdm
from tqdm import tqdm

# create dataframe containing date, user and tweet
queries = ["Fianna Fail min_faves:150 until:2022-12-31 since:2020-01-01",
           "Fine Gael min_faves:150 until:2022-12-31 since:2020-01-01",
           "Sinn Fein min_faves:150 until:2022-12-31 since:2020-01-01"]
tweets = []

for query in queries:
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        tweets.append([tweet.date, tweet.username, tweet.content, " ".join(query.split()[:2])])

df_tweets = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet', "Party"])

# group by month of the year
df_tweets['Month'] = df_tweets['Date'].dt.strftime('%Y-%m')

# create model
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# create polarity function that passes through the model
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores).tolist()
    return scores

# create 3 columns for our scores
df_tweets = df_tweets.assign(Negative = None, Neutral = None, Positive = None)

for i, row in tqdm(df_tweets.iterrows(), total=len(df_tweets)):
    try:
      text = row['Tweet']
      scores = polarity_scores_roberta(text)
      df_tweets.loc[i, 'Negative'] = scores[0]
      df_tweets.loc[i, 'Neutral'] = scores[1]
      df_tweets.loc[i, 'Positive'] = scores[2]
    except RuntimeError:
        print(f'Broke for id {i}')

df_tweets = df_tweets.astype({'Negative':'float','Neutral':'float','Positive':'float'}) # convert them to floats

# find average by month for each scores
df_month = df_tweets.groupby(['Month', 'Party']).mean()
df_month = df_month.pivot_table(index = 'Month',
                                columns = 'Party',
                                values = ['Negative', 'Neutral', 'Positive'])
df_month.columns = [' '.join(col) for col in df_month.columns.values]
