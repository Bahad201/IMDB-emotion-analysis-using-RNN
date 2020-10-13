import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
from keras.datasets import imdb
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import SimpleRNN, Dense,Activation,Flatten,Dropout
(X_train, Y_train),(X_test,Y_test)=imdb.load_data(path="imdb.npz",
               num_words= None,
               skip_top= 0,
               maxlen= None,
               seed= 113,
               start_char= 1,
               oov_char= 2,
               index_from= 3)

print("Type: ",type(X_train))

print("X Train Shape: ",X_train.shape)
print("Y Train Shape: ",Y_train.shape)
#%%
# EDA
print("Y Train Values: ", np.unique(Y_train))
print("Y Test Values: ", np.unique(Y_test))

unique, counts = np.unique(Y_train,return_counts=True)
print("Y train distribution:",dict(zip(unique, counts)))


unique, counts = np.unique(Y_test,return_counts=True)
print("Y test distribution:",dict(zip(unique, counts)))


plt.figure()
sns.countplot(Y_train)
plt.xlabel("Classes")
plt.ylabel("Freq")
plt.title("Y Train")

plt.figure()
sns.countplot(Y_test)
plt.xlabel("Classes")
plt.ylabel("Freq")
plt.title("Y Test")
plt.show()

d = X_train[0]
print(d)
print(len(d))

review_len_train = []
review_len_test = []

for i , ii in zip(X_train,X_test):
    review_len_train.append(len(i))
    review_len_test.append(len(ii))
    


plt.figure()
sns.distplot(review_len_train,hist_kws = {"alpha":0.3})
sns.distplot(review_len_test,hist_kws = {"alpha":0.3})


print("Train Mean:", np.mean(review_len_train))
print("Train Median:", np.median(review_len_train))
print("Train Mean:", stats.mode(review_len_train))

# number of words

word_index = imdb.get_word_index()
print(type(word_index))


for keys, values in word_index.items():
    if values == 22:
        print(keys)
    


def whatiTSay(index):
    reverse_index = dict([(value, key) for (key,value) in word_index.items()])
    decode_review = " ".join([reverse_index.get(i-3,"!") for i in X_train[index]])
    print(decode_review)
    print(Y_train[index])
    return decode_review

decoded_review = whatiTSay(15)



#%%
#preprocessing

num_words=15000

(X_train,Y_train),(X_test,Y_test) = imdb.load_data(num_words=num_words)

maxlen = 130

X_train = pad_sequences(X_train,maxlen=maxlen)
X_test = pad_sequences(X_test,maxlen=maxlen)

print(X_train[5])


#%% 
#RECURRENT NEURAL NETWORK(RNN)

rnn = Sequential()

rnn.add(Embedding(num_words,32,input_length=maxlen))
# integerlari belirli boyutlarda yogunluk vektorlerine cevirmek icin
rnn.add(SimpleRNN(16,input_shape=(num_words,maxlen),return_sequences=False,activation='relu'))

rnn.add(Dense(8))
rnn.add(Dense(4))
rnn.add(Dropout(0.25))
rnn.add(Dense(1))
rnn.add(Activation('sigmoid'))

print(rnn.summary())

rnn.compile(loss="binary_crossentropy",optimizer = 'rmsprop', metrics = [ 'accuracy'])



history = rnn.fit(X_train,Y_train,
                  validation_data = (X_test,Y_test),
                  epochs=5,
                  batch_size=128,
                  verbose=1)


score = rnn.evaluate(X_test,Y_test)

print("Accuracy: %",score[1]*100)


plt.figure()

plt.plot(history.history["accuracy"],label = "Train  Accuracy")
plt.plot(history.history["val_accuracy"],label = "Test Accuracy")
plt.plot(history.history["loss"],label = "Train  loss")
plt.plot(history.history["val_loss"],label = "Test loss")
plt.title("acc")
plt.ylabel("acc")
plt.xlabel("epochs")
plt.legend()
plt.show()
