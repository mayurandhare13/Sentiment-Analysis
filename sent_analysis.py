import numpy as np 
import pandas as pd 
import bz2
import os
import string 
import keras
import tensorflow
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import time
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

print("------A-------")
print(os.listdir("amazonreviews"))

trainfile = bz2.BZ2File('amazonreviews/train.ft.txt.bz2','r')
lines = trainfile.readlines()

print(lines[1])

docSentimentList=[]
def getDocumentSentimentList(docs,splitStr='__label__'):
    for i in range(len(docs)):
        #print('Processing doc ',i,' of ',len(docs))
        text=str(lines[i])
        #print(text)
        splitText=text.split(splitStr)
        secHalf=splitText[1]
        text=secHalf[2:len(secHalf)-1]
        sentiment=secHalf[0]
        #print('First half:',secHalf[0],'\nsecond half:',secHalf[2:len(secHalf)-1])
        docSentimentList.append([text,sentiment])
    print('Done!!')
    return docSentimentList

docSentimentList=getDocumentSentimentList(lines[:1000000],splitStr='__label__')

train_df = pd.DataFrame(docSentimentList,columns=['Text','Sentiment'])
print(train_df.head())

print("------B-------")
#Text Preprocessing
train_df['Sentiment'][train_df['Sentiment']=='1'] = 0
train_df['Sentiment'][train_df['Sentiment']=='2'] = 1

print(train_df['Sentiment'].value_counts())

train_df['word_count'] = train_df['Text'].str.lower().str.split().apply(len)
print(train_df.head())


def remove_punc(s):
    table = str.maketrans({key: None for key in string.punctuation})
    return s.translate(table)

train_df['Text'] = train_df['Text'].apply(remove_punc)
print(train_df.shape)
print(train_df.head())

print(len(train_df['word_count'][train_df['word_count']<=25]))

train_df1 = train_df[:][train_df['word_count']<=25]
print(train_df1.head())

print(train_df1['Sentiment'].value_counts())

#saving
train_df1.to_csv("train_df1.csv", sep=',', encoding='utf-8')

print("------C-------")
#CountVectorized Representation

st_wd = text.ENGLISH_STOP_WORDS
c_vector = CountVectorizer(stop_words = st_wd,min_df=.0001,lowercase=1)
X_counts = c_vector.fit_transform(train_df1['Text'].values)
# this return DTM
print(X_counts)

y = train_df1['Sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(X_counts, y, test_size=0.1, random_state=139)


X_train = X_train.todense()
X_test = X_test.todense()

y_train = y_train.astype('int')
y_test = y_test.astype('int')


print(X_train.shape)

# Logistic Regression
print("=== Logistic Regression ===")
start = time.process_time()

LR_classifier = LogisticRegression(C=10, penalty='l1', multi_class='ovr', verbose=1)
LR_classifier.fit(X_train, y_train)
LR_test_score = LR_classifier.score(X_test, y_test)
print(LR_test_score)

LR_train_score = LR_classifier.score(X_train, y_train)
print(LR_train_score)

end = time.process_time()
print("total time taken LR Search: {} min".format((end - start) / 60))

# Naive bayes
print("=== NB ===")
start = time.process_time()

nb_clf = MultinomialNB()
nb_clf.fit(X_train, y_train)

#predicted = nb_clf.predict(X_test)
NB_test_score = nb_clf.score(X_test, y_test)
print(NB_test_score)

NB_train_score = nb_clf.score(X_train, y_train)
print(NB_train_score)

end = time.process_time()
print("total time taken: Multinomial NB {} min".format((end - start) / 60))

# Linear SVM
print("=== Linear SVM ===")

start = time.process_time()
svm_linear_clf = LinearSVC(C=1.0, verbose=1, multi_class="ovr")
svm_linear_clf.fit(X_train, y_train)
svm_linear_test_score = svm_linear_clf.score(X_test, y_test)
print(svm_linear_test_score)
svm_linear_train_score = svm_linear_clf.score(X_train, y_train)
print(svm_linear_train_score)
print(svm_linear_clf.predict(X_test[[332]]))
print(y_test[332])

end = time.process_time()
print("total time taken Linear SVM: {} min".format((end - start) / 60))

# KNN
print("=== KNN ===")
start = time.process_time()

KN_classifier = KNeighborsClassifier(n_neighbors=5, p=1, weights='distance')
KN_classifier.fit(X_train, y_train)

print(KN_classifier.predict(X_test[[332]]))

KN_test_score = KN_classifier.score(X_test, y_test)
print(KN_test_score)
KN_train_score = KN_classifier.score(X_train, y_train)
print(KN_train_score)
end = time.process_time()
print("total time taken KNN Search: {} min".format((end - start) / 60))


# Random Forest
print("=== Random Forest ===")
start = time.process_time()

RF_classifier = RandomForestClassifier(n_estimators=10, max_depth=50, criterion='entropy')
RF_classifier.fit(X_train, y_train)

print(RF_classifier.predict(X_test[[332]]))
print(y_test[332])

RF_test_score = RF_classifier.score(X_test, y_test)
print(RF_test_score)

RF_train_score = RF_classifier.score(X_train, y_train)
print(RF_train_score)

end = time.process_time()
print("total time taken Random Forest Search: {} min".format((end - start) / 60))

# SVM
print("=== Poly Kernel SVM ===")
start = time.process_time()
svm_clf = SVC(C = 10, kernel='poly', verbose=True)
svm_clf.fit(X_train, y_train)

print(svm_clf.predict(X_test[[332]]))
print(y_test[332])

svm_test_score = svm_clf.score(X_test, y_test)
print(svm_test_score)


svm_train_score = svm.score(X_train, y_train)
print(svm_train_score)

end = time.process_time()
print("total time taken for Multi SVM: {} min".format((end - start) / 60))

# visualization
import matplotlib.pyplot as plt

N = 6

train_acc = [RF_train_score, svm_linear_train_score,svm_train_score, NB_train_score, KN_train_score, LR_train_score]
test_acc = [RF_test_score, svm_linear_train_score,svm_test_score, NB_test_score, KN_test_score, LR_test_score]

train_acc = [i * 100 for i in train_acc]
test_acc = [i * 100 for i in test_acc]
ind = np.arange(N)    # the x locations for the groups
width = 0.25       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, train_acc, width)
p2 = plt.bar(ind, test_acc, width,
             bottom=train_acc)

plt.ylabel('Scores')
plt.title('Scores by algorithms and train_test')
plt.xticks(ind, ('Random Forest', 'Linear SVM','SVM', 'Naive Bayes', 'KNN', 'Logistic Regression'), rotation='vertical')
plt.yticks(np.arange(0, 200, 10))
plt.margins(0.2)
plt.legend((p1[0], p2[0]), ('train acc', 'test acc'))
plt.subplots_adjust(bottom=0.15)
plt.savefig('algo_train_test_acc.png')
plt.close()


#Grid Search
print("=== Grid Search ===")
start = time.process_time()

parameters = {'n_neighbors':[1, 5, 9], 'p':[1, 2], 'weights':('uniform', 'distance')}
K_neighbor = KNeighborsClassifier()
clf = GridSearchCV(K_neighbor, parameters, verbose=True)
clf.fit(X_train, y_train)
grid_KN_test_score = clf.score(X_test, y_test)
print(grid_KN_test_score)

grid_KN_train_score = clf.score(X_train, y_train)
print(grid_KN_train_score)

end = time.process_time()
print("total time taken Grid Search: {} min".format((end - start) / 60))


N = 7
train_acc = (svm_linear_train_score, RF_train_score, svm_train_score, NB_train_score, KN_train_score, LR_train_score, grid_KN_train_score)
test_acc = (svm_linear_test_score, RF_test_score, svm_test_score, NB_test_score, KN_test_score, LR_test_score, grid_KN_test_score)

train_acc = [i * 100 for i in train_acc]
test_acc = [i * 100 for i in test_acc]

ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, train_acc, width)
p2 = plt.bar(ind, test_acc, width,
             bottom=train_acc)

plt.ylabel('Scores')
plt.title('Scores by algorithms and train_test')
plt.xticks(ind, ('Linear SVM','Random Forest', 'Crammer-Singer SVM', 'Multinomial NB', 'KNN', 'Logistic Regression', 'Grid Search KNN'))
plt.yticks(np.arange(0, 200, 10))
plt.legend((p1[0], p2[0]), ('train acc', 'test acc'))
plt.savefig('algo_train_test_acc_final.png')
plt.close()
