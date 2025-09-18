import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
#nltk.download("wordnet")
#nltk.download('vader_lexicon')

#we are performing basic NLP first
# we create a function it'll become easy to use

def sentiment_analysis(text):
    # making all lower
    text=text.lower()
    # tokenization
    text=word_tokenize(text)
    # performing stop words
    nltk_stop_words=set(stopwords.words("english"))
    #my_stop_word={"is","are"}
    #nltk_stop_words.update(my_stop_word)
    for i in text:
        if i in nltk_stop_words:
            count_i=text.count(i)
            for j in range(count_i):
                text.remove(i)

    # NOW WE are performing stemming
    stem=PorterStemmer() # it give us normal root words
    text=[stem.stem(w) for w in text]

    # NOW WE are performing lemmatizer
    lemma=WordNetLemmatizer() # It give us root like gives,given,gived as one word give
    text=[lemma.lemmatize(w) for w in text]

    # joining the text so that we are removing the brackets for all sentences
    text1=" ".join(text)
    return text1 # this should return the value

### now all the basic NLP operations are done lets move on to sentiment analysis
# for that we create another function
def nlp_sentiment_analysis(text):
    score=analyser.polarity_scores(text)
    #print("SCORE  :")
    #print(score)

    sentiment=1 if score["pos"] > 0.2 else 0
    return sentiment

data=pd.read_csv("C:/Users/user/Downloads/amazon.csv")
#print(data)
# calling the function
data["reviewText"]=data["reviewText"].apply(sentiment_analysis)
print(data)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser=SentimentIntensityAnalyzer()

data["Predicted"]=data["reviewText"].apply(nlp_sentiment_analysis)

# Now we will predict the score using machine learning
from sklearn.metrics import confusion_matrix,accuracy_score

cm= confusion_matrix(data['Positive'],data['Predicted'])
print(cm)
## we will also see the accuracy score
print("Accuracy score = ",accuracy_score(data['Positive'],data['Predicted']))


#--------------------------------------------------------------------------------------------------------------------
## now let's perform for our own text by that we can know does the model is working well or not
# we can use another function for it become easy to understand


def own_sentiment_analyze(text):
    score=analyser1.polarity_scores(text)
    print(score)

    sentiment=1 if score["pos"] > 0.1 else 0
    return sentiment

from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser1 = SentimentIntensityAnalyzer()

text1="excellent"
score=own_sentiment_analyze(text1)
print(score)
text2="poor"
score=own_sentiment_analyze(text2)
print(score)
text3="average"
score=own_sentiment_analyze(text3)
print(score)
text4="product was not good but i like it"
score=own_sentiment_analyze(text4)
print(score)
text5="excellent excellent excellent excellent"
score=own_sentiment_analyze(text5)
print(score)
text6="poor  poor poor poor poor poor "
score=own_sentiment_analyze(text6)
print(score)

## OUR MODEL IS RUNNING PERFECTLY
# THANK YOU....