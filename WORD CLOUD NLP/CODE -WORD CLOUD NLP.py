
## GENERATING WORD CLOUD USING NLP

link="https://storymirror.com/read/english/story/the-amazons-2-chapter-5/sr35f518"
import nltk
#we are accessing a website : urlopen() gets us the access to any webpage
from urllib.request import urlopen
html_content=urlopen(link).read()
#print(html_content)

# if we want organize the content in neat way we use bs4

from bs4 import BeautifulSoup
soup=BeautifulSoup(html_content)
#print(soup)

#here we are removing the script and style from the html code
for s in soup(["script","style"]):
    s.extract()
#print(soup)

text=soup.get_text()
text=" ".join(text.split())
#print(text)
text=text.lower()
#print(text)

# word tokenize
from nltk.tokenize import word_tokenize
text=word_tokenize(text)

#stopword removal
from nltk.corpus import stopwords
nltk_stop_words=set(stopwords.words("english"))
my_stop_word={"|",".",",","-","1","2","3","4","5","6","7","8","9","0"}# we can give our own stop words to remove after removing regular way. if anything you don't like just add this
nltk_stop_words.update(my_stop_word)
for i in text:
    if i in nltk_stop_words:
        count_i=text.count(i)
        for j in range(count_i):
            text.remove(i)
#print(text)

# print the word cloud
# install wordcloud - pip install wordcloud
# we are visualizing

from wordcloud import wordcloud, WordCloud
import matplotlib.pyplot as plt
wc=WordCloud(max_words=100,background_color="BLACK").generate(" ".join(text))
plt.imshow(wc)
plt.show()
print(text)