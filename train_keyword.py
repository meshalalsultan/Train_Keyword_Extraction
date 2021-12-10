import numpy as np # linear algebra
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer 

cv_text = '''Profile 
Business-minded data scientist with a demonstrated ability to deliver valuable insights via data analytics and advanced data-driven methods,
With deep understanding of neural networks and the ability to explain technical topics,data, machine learning, and statistics.
Passionate about explaining data science to non-technical business audiences,
I have experience using stats and machine-learning to find useful insights in data.

Professional Experience
Analyzed and Predict Stock Price for next day:
Developed and trained Artificial Recurrent Neural network (RNN) on stock dataframe and tested it with loss matrix and accuracy to predict next day price.

Built Twitter and News Sentiment App: 
Create Algorithm with TensorFlow predict the polarity of textual data or sentiments like Positive, Neural, and negative.

Predict What Products Customers Likely Want
Boosted Machine learning to Predict what products customers likely want and then optimize a targeted campaign using advanced decision optimization in a Jupyter notebook. 

Analyzed Public Twitter Profile: 
Established Deeper personality insight on several dimensions, including traits , values , need and consumer preferences . Enhanced with chart plot.

Technical Skills 
Development Machine Learning: 
Classification, Regression, Clustering, Future Engineering, TensorFlow, Transfer Learning, Restful API, Terminal Bash

Programming Language:
Python (Scikit-Learn, NumPy, Pandas, TensorFlow, Matplotlib, SciPy), SQL, Microsoft, HTML, JavaScript, Dart, Excel, Flutter

Statics Methods:
Time Series, Regression Models, Hypothesis Testing.

Visualization:
Tableau, Oracle DVD.

Development Platform:
Web Browser, IOS & Android Apps .

Tools:
Anaconda, Jupyter Notebook, IDE Editor, Android Studio.

Selected Coursework:
Stochastic Gradient Decent, Liner Algebra, Theory, Probity and Statics, A/B Testing

Education 
•	Studying Master Degree in Data Scientist in University of Colorado Bolder
•	Faculty of Business Studies Accountant Diploma 2001-2004		CERTIFICATION 
	
Tensorflow Certified Developer: Googel.com

IBM Applied AI Specialization :

NLP Natural Language Processing with python :  Udemy
Building Ai Powered Chatbot : IBM
Browser Based Model with TensorFlow: deeplearning.io
Machine Learning with Python: Coursera.
Augmented Data Visualization
 
LANGUAGES
Arabic, English

ADDITIONAL INTEREST
Karate
Swimming
Travel 
'''






df = pd.read_csv('papers.csv')


stop_words = set(stopwords.words('english'))
##Creating a list of custom stopwords
new_words = ["fig","figure","image","sample","using", 
             "show", "result", "large", 
             "also", "one", "two", "three", 
             "four", "five", "seven","eight","nine"]
stop_words = list(stop_words.union(new_words))

def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    ##Convert to list from string
    text = text.split()
    
    # remove stopwords
    text = [word for word in text if word not in stop_words]

    # remove words less than three letters
    text = [word for word in text if len(word) >= 3]

    # lemmatize
    lmtzr = WordNetLemmatizer()
    text = [lmtzr.lemmatize(word) for word in text]
    
    return ' '.join(text)

docs = df['paper_text'].apply(lambda x:pre_process(x))

'''
Using TF-IDF
TF-IDF stands for Text Frequency Inverse Document Frequency. The importance of each word increases in proportion to the number of times a word appears in the document (Text Frequency – TF) but is offset by the frequency of the word in the corpus (Inverse Document Frequency – IDF).

Using the tf-idf weighting scheme, the keywords are the words with the highest TF-IDF score. For this task, I’ll first use the CountVectorizer method in Scikit-learn to create a vocabulary and generate the word count:
'''

from sklearn.feature_extraction.text import CountVectorizer
#docs = cv_text.tolist()
#create a vocabulary of words, 
cv=CountVectorizer(max_df=0.95,         # ignore words that appear in 95% of documents
                   max_features=10000,  # the size of the vocabulary
                   ngram_range=(1,3)    # vocabulary contains single words, bigrams, trigrams
                  )
word_count_vector=cv.fit_transform(cv_text)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

# get feature names
feature_names=cv.get_feature_names()

def get_keywords(idx, docs):

    #generate tf-idf for the given document
    tf_idf_vector=tfidf_transformer.transform(cv.transform([docs[idx]]))

    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())

    #extract only the top n; n here is 10
    keywords=extract_topn_from_vector(feature_names,sorted_items,10)
    
    return keywords

def print_results(idx,keywords, df):
    # now print the results
    print("\n=====Title=====")
    print(df['title'][idx])
    print("\n=====Abstract=====")
    print(df['abstract'][idx])
    print("\n===Keywords===")
    for k in keywords:
        print(k,keywords[k])
idx=941
keywords=get_keywords(idx, docs)
print_results(idx,keywords, df)