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
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron


#Read Dataset
print(os.listdir("amazonreviews"))

trainfile = bz2.BZ2File('amazonreviews/train.ft.txt.bz2','r')
lines = trainfile.readlines()

print(lines[1])

'''
Split dataset into Labels and Text
'''

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
'''
Text Preprocessing
1) Replacing labels `1` to `0` and `2` to `1`
2) Make all text to lowercase so that it will count as single word at tokenization.
3) Remove punctuations
4) Remove stop words
all above will halp in feature extraction. so that we will have less features for training.
'''
train_df['Sentiment'][train_df['Sentiment']=='1'] = 0
train_df['Sentiment'][train_df['Sentiment']=='2'] = 1

print(train_df['Sentiment'].value_counts())

#lowercasing
train_df['word_count'] = train_df['Text'].str.lower().str.split().apply(len)
print(train_df.head())


#remove punctuations
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

#train_df1 = pd.read_csv("train_df1.csv", sep=',', encoding='utf-8')
print("------C-------")

#remove stop words CountVectorized Representation
st_wd = text.ENGLISH_STOP_WORDS
c_vector = CountVectorizer(stop_words = st_wd,min_df=.0001,lowercase=1)
X_counts = c_vector.fit_transform(train_df1['Text'].values)
# X_counts is Document Term Matrix(DTM)
print(X_counts)

y = train_df1['Sentiment'].values


#train and test spliting for checking accuracies
X_train, X_test, y_train, y_test = train_test_split(X_counts, y, test_size=0.2, random_state=139)


X_train = X_train.todense()
X_test = X_test.todense()

y_train = y_train.astype('int')
y_test = y_test.astype('int')

print(X_train.shape)
# Algorithms

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

# Decision Tree
print("=== Decision Tree ===")
start = time.process_time()

DT_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=10, splitter='best')
DT_classifier.fit(X_train, y_train)

print(DT_classifier.predict(X_test[[332]]))

DT_test_score = DT_classifier.score(X_test, y_test)
print(DT_test_score)
DT_train_score = DT_classifier.score(X_train, y_train)
print(DT_train_score)
end = time.process_time()
print("total time taken Decision Tree: {} min".format((end - start) / 60))


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

# Perceptron
print("=== Perceptron===")
start = time.process_time()
per_clf = Perceptron(penalty='l1', verbose=1)
per_clf.fit(X_train, y_train)

print(per_clf.predict(X_test[[332]]))
print(y_test[332])

per_test_score = per_clf.score(X_test, y_test)
print(per_test_score)


per_train_score = per_clf.score(X_train, y_train)
print(per_train_score)

end = time.process_time()
print("total time taken for Perceptron: {} min".format((end - start) / 60))

# visualization
import matplotlib.pyplot as plt

N = 6

train_acc = [RF_train_score, svm_linear_train_score,per_train_score, NB_train_score, DT_train_score, LR_train_score]
test_acc = [RF_test_score, svm_linear_train_score,per_test_score, NB_test_score, DT_test_score, LR_test_score]

train_acc = [i * 100 for i in train_acc]
test_acc = [i * 100 for i in test_acc]
ind = np.arange(N)    # the x locations for the groups
width = 0.25       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, train_acc, width)
p2 = plt.bar(ind, test_acc, width, bottom=train_acc)

plt.ylabel('Scores')
plt.title('Scores by algorithms and train_test')
plt.xticks(ind, ('Random Forest', 'Linear SVM', 'Perceptron', 'Naive Bayes', 'Decision Tree', 'Logistic Regression'), rotation=45)
plt.yticks(np.arange(0, 200, 10))
plt.margins(0.2)
plt.legend((p1[0], p2[0]), ('train acc', 'test acc'))
plt.subplots_adjust(bottom=0.15)
plt.savefig('algo_train_test_acc.png')
plt.close()


print("------CNN-------")
# Keras CNN Model

model1=  Sequential()
model1.add(Dense(1000,input_shape=(8915,),activation='relu'))
model1.add(Dense(1,activation='sigmoid'))

model1.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

hist = model1.fit(X_train,y_train,epochs=6,batch_size=128,verbose=1)

print(model1.evaluate(X_train, y_train, batch_size=128))
print(model1.evaluate(X_test, y_test, batch_size=128))


model2=  Sequential()
model2.add(Dense(1000,input_shape=(8915,),activation='relu'))
model2.add(Dense(500,activation='relu'))
model2.add(Dense(1,activation='sigmoid'))

model2.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

hist2 = model2.fit(X_train,y_train,epochs=6,batch_size=128,verbose=1)

model3=  Sequential()
model3.add(Dense(2000,input_shape=(8915,),activation='relu'))
model3.add(Dense(1000,activation='relu'))
model3.add(Dense(500,activation='relu'))
model3.add(Dense(1,activation='sigmoid'))

model3.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
hist3 = model3.fit(X_train,y_train,epochs=6,batch_size=128,verbose=1)

print(model2.evaluate(X_test, y_test, batch_size=128))
print(model3.evaluate(X_test, y_test, batch_size=128))

print("------E-------")
loss_curve = hist.history['loss']
epoch_c = list(range(len(loss_curve)))
loss_curve2 = hist2.history['loss']
epoch_c = list(range(len(loss_curve)))
loss_curve3 = hist3.history['loss']
epoch_c = list(range(len(loss_curve)))
plt.xlabel('Epochs')
plt.ylabel('Loss value')
plt.plot(epoch_c,loss_curve, color='red', marker='o',label='1 Hidden layer')
plt.plot(epoch_c,loss_curve2, color='green', marker='o',label='2 Hidden layers')
plt.plot(epoch_c,loss_curve3, color='blue',marker='o',label='3 Hidden layers')
plt.legend()
plt.savefig('loss_cnn.png')
plt.show()
plt.close()

acc_curve = hist.history['acc']
epoch_c = list(range(len(loss_curve)))
acc_curve2 = hist2.history['acc']
epoch_c = list(range(len(loss_curve)))
acc_curve3 = hist3.history['acc']
epoch_c = list(range(len(loss_curve)))
plt.xlabel('Epochs')
plt.ylabel('Accuracy value')
plt.plot(epoch_c,acc_curve, color='red', marker='o',label='1 Hidden layer')
plt.plot(epoch_c,acc_curve2, color='green', marker='o', label='2 Hidden layers')
plt.plot(epoch_c,acc_curve3, color='blue', marker='o', label='3 Hidden layers')
plt.legend()
plt.savefig('accuracies_cnn.png')
plt.show()
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
plt.xticks(ind, ('Linear SVM','Random Forest', 'Poly SVM', 'Multinomial NB', 'KNN', 'Logistic Regression', 'Grid Search KNN'), rotation = 'vertical')
plt.yticks(np.arange(0, 200, 10))
plt.margins(0.2)
plt.legend((p1[0], p2[0]), ('train acc', 'test acc'))
plt.subplots_adjust(bottom=0.15)
plt.savefig('algo_train_test_acc_final.png')
plt.close()
