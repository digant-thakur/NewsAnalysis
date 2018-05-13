import re
import operator
import numpy as np
import tensorflow as tf
from newsapi_fetcher import NewsFetcher

numDimensions = 300
maxSeqLength = 250
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000
wordsList = np.load('training_data/wordsList.npy').tolist()
wordsList = [word.decode('UTF-8') for word in wordsList]


def lstm_model(News):
    news_dict = News.get_news_dict()
    news_list = list(news_dict.keys())
    wordVectors = np.load('training_data/wordVectors.npy')
    tf.reset_default_graph()

    labels = tf.placeholder(tf.float32, [batchSize, numClasses])
    input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

    data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(wordVectors, input_data)

    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('models'))

    positiveCount = 0
    negativeCount = 0
    positiveNews = {}
    negativeNews = {}

    for article in news_list:
        inputText = article
        inputMatrix = getSentenceMatrix(inputText)
        predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
        if predictedSentiment[0] > predictedSentiment[1]:
            positiveNews[article] = predictedSentiment[0]
            positiveCount = positiveCount + 1
        else:
            negativeNews[article] = predictedSentiment[1]
            negativeCount = negativeCount + 1
    print("Positive {} Negative {}".format(positiveCount, negativeCount))

    sorted_positive = sorted(positiveNews.items(), key = operator.itemgetter(1), reverse=True)[0:3]
    sorted_negative = sorted(negativeNews.items(), key = operator.itemgetter(1), reverse=True)[0:3]
    sorted_positive = list(list(zip(*sorted_positive))[0])
    sorted_negative = list(list(zip(*sorted_negative))[0])
    news_payload = return_url_headline(News, sorted_positive, sorted_negative)
    payload = [positiveCount, negativeCount, news_payload, News.urlToImage]
    return payload
    
def cleanSentences(string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


def getSentenceMatrix(sentence):
    arr = np.zeros([batchSize, maxSeqLength])
    sentenceMatrix = np.zeros([batchSize, maxSeqLength], dtype='int32')
    cleanedSentence = cleanSentences(sentence)
    split = cleanedSentence.split()
    for indexCounter, word in enumerate(split):
        try:
            sentenceMatrix[0, indexCounter] = wordsList.index(word)
        except ValueError:
            sentenceMatrix[0, indexCounter] = 399999  # Vector for unkown words
    return sentenceMatrix


def return_url_headline(News, positive_top3, negative_top3):
    positive_news = {}
    negative_news = {}
    for page in positive_top3:
        positive_news[page] = News.news[page]
    for page in negative_top3:
        negative_news[page] = News.news[page]
    return {'positiveNews':positive_news,'negativeNews':negative_news}

def get_sentiment(news_topic,no_of_weeks = 1):
    News = NewsFetcher()
    News.set_start_date_from_now(no_of_weeks)
    News.get_news(news_topic, sort='relevancy', domainName='firstpost.com')
    result = lstm_model(News)
    return result

