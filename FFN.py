# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:17:14 2023

@author: yakou
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import load_model
from tensorflow.keras import models
from tensorflow.keras import backend
from tensorflow.keras import metrics as metrics_module
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import re
import string
import random
from sklearn.model_selection import train_test_split
"""
  • • • #preprocessing
"""
data = pd.read_csv('ticaret-yorum.csv')
print(data)

tr_stop_words = pd.read_csv('tr_stop_word.txt',header=None)
print("First 5 entries:")
for each in tr_stop_words.values[:5]:
  print(each[0])

print(data.text.duplicated(keep="first").value_counts()) # duplacitons

data.drop_duplicates(subset="text",keep="first",inplace=True,ignore_index=True) #drop the duplications

topic_list = data.category.unique()
print("Topics:\n", topic_list)

print(data.isnull().sum())

print(data.describe(include='all'))

number_of_topics = len(topic_list)
print("Number of Topics: ",number_of_topics)

print(data.category.value_counts())

#the dataset is balanced.
data.category.value_counts().plot.bar(x="Topics",y="Number of Reviews",figsize=(32,6) )

data['words'] = [len(x.split()) for x in data['text'].tolist()] #word count colunm

print(data.head())

print(data['words'].describe())
print(data.groupby(['category'])['words'].describe())

min_review_size = 15
print(data[data['words']<min_review_size].count())

min_review_size = 15 
max_review_size = 40 #50
print(data.count())

data= data[data['words']>=min_review_size]

print(data[data['words']<min_review_size].count())

vocab = set()
corpus= [x.split() for x in data['text'].tolist()]
for sentence in corpus:
  for word in sentence:
    vocab.add(word.lower())
print("Number of distinct words in raw data: ", len(vocab))

print(list(vocab)[:25])


word_freq= data.text.str.split(expand=True).stack().value_counts()
word_freq=word_freq.reset_index(name='freq').rename(columns={'index': 'word'})
top_50_frequent_words = word_freq[:50]
print(top_50_frequent_words)

for each in top_50_frequent_words['word']:
  if each in tr_stop_words.values:
    print (each) 

print(data.info())

print(data.describe())

data= data.sample(frac=1)

data["category"] = data["category"].astype('category')
data["category_id"] = data["category"].cat.codes
print(data.tail())
print(data.dtypes)

id_to_category = pd.Series(data.category.values,index=data.category_id).to_dict()
print(id_to_category)

category_to_id= {v:k for k,v in id_to_category.items()}
print(category_to_id)

print("alisveris id is " , category_to_id["alisveris"])
print("0 is for " , id_to_category[0])

number_of_categories = len(category_to_id)
print("number_of_categories: ",number_of_categories)

data_size=  427230
data= data[:data_size]
data.info()

features, targets = data['text'], data['category_id']

all_train_features, test_features, all_train_targets, test_targets = train_test_split(
        features, targets,
        train_size=0.8,
        test_size=0.2,
        random_state=42,
        shuffle = True,
        stratify=targets
    )

print("All Train Data Set size: ",len(all_train_features))
print("Test Data Set size: ",len(test_features))

reduce_ratio = 0.02

reduced_train_features, _, reduced_train_targets, _ = train_test_split(
        all_train_features, all_train_targets,
        train_size=reduce_ratio,
        random_state=42,
        shuffle = True,
        stratify=all_train_targets
    )

print("Reduced Train Data Set size: ",len(reduced_train_features))
print("Test Data Set size: ",len(test_features))

train_features, val_features, train_targets, val_targets = train_test_split(
        reduced_train_features, reduced_train_targets,
        train_size=0.9,
        random_state=42,
        shuffle = True,
        stratify=reduced_train_targets
    )
print("Train Data Set size: ",len(train_features))
print("Validation Data Set size: ",len(val_features))
print("Test Data Set size: ",len(test_features))

print(train_features.values[:5])
print(train_targets.values[:5])

train_text_ds_raw = tf.data.Dataset.from_tensor_slices(
            tf.cast(train_features.values, tf.string)
) 
train_cat_ds_raw = tf.data.Dataset.from_tensor_slices(
            tf.cast(train_targets.values, tf.int64),

) 
vocab_size = 100000  
max_len = max_review_size 

@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_string):
    """ Remove html line-break tags and handle punctuation """
    no_uppercased = tf.strings.lower(input_string, encoding='utf-8')
    no_stars = tf.strings.regex_replace(no_uppercased, "\*", " ")
    no_repeats = tf.strings.regex_replace(no_stars, "devamını oku", "")    
    no_html = tf.strings.regex_replace(no_repeats, "<br />", "")
    no_digits = tf.strings.regex_replace(no_html, "\w*\d\w*","")
    no_punctuations = tf.strings.regex_replace(no_digits, f"([{string.punctuation}])", r" ")
    #remove stop words
    #no_stop_words = ' '+no_punctuations+ ' '
    #for each in tr_stop_words.values:
    #  no_stop_words = tf.strings.regex_replace(no_stop_words, ' '+each[0]+' ' , r" ")
    no_extra_space = tf.strings.regex_replace(no_punctuations, " +"," ")
    #remove Turkish chars
    no_I = tf.strings.regex_replace(no_extra_space, "ı","i")
    no_O = tf.strings.regex_replace(no_I, "ö","o")
    no_C = tf.strings.regex_replace(no_O, "ç","c")
    no_S = tf.strings.regex_replace(no_C, "ş","s")
    no_G = tf.strings.regex_replace(no_S, "ğ","g")
    no_U = tf.strings.regex_replace(no_G, "ü","u")

    return no_U
input_string = "Bu Issız Öğlenleyin de;  şunu ***1 Pijamalı Hasta***, ve  Ancak İşte Yağız Şoföre Çabucak Güvendi...Devamını oku"
print("input:  ", input_string)
output_string= custom_standardization(input_string)
print("output: ", output_string.numpy().decode("utf-8"))

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size - 1,
    output_mode="int", #tf-idf / int / binary / count
    output_sequence_length=max_len,
)

vectorize_layer.adapt(train_text_ds_raw)
vocab = vectorize_layer.get_vocabulary() 

print("vocab has the ", len(vocab)," entries")
print("vocab has the following first 10 entries")
for word in range(10):
  print(word, " represents the word: ", vocab[word])
print("2 sample text preprocessing:")
for X in train_text_ds_raw.take(2):
  print(" Given raw data: " )
  print(X.numpy().decode("utf-8") )
  tokenized = vectorize_layer(tf.expand_dims(X, -1))
  print(" Tokenized and Transformed to a vector of integers: " )
  print (tokenized)
  print(" Text after Tokenized and Transformed: ")
  transformed = ""
  for each in tf.squeeze(tokenized):
    transformed= transformed+ " "+ vocab[each]
  print(transformed)  

vectorizer_model = tf.keras.models.Sequential()
vectorizer_model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
vectorizer_model.add(vectorize_layer)
vectorizer_model.summary()

# Save it
filepath = "vectorize_layer_model"
vectorizer_model.save(filepath, save_format="tf")

loaded_vectorizer_model = tf.keras.models.load_model(filepath)
# Extract the trained TextVectorization layer out of the loaded model
loaded_vectorizer_layer = loaded_vectorizer_model.layers[0]

loaded_vocab = loaded_vectorizer_layer.get_vocabulary()
print("original vocab has the ", len(vocab)," entries")
print("loaded_vectorizer_layer vocab has the ", len(loaded_vocab)," entries")
print("original vocab: ", vocab[:10])
print("loaded vocab  : ", loaded_vocab[:10])

for X in train_text_ds_raw.take(1):
  print(" Given raw data: " )
  print(X.numpy().decode("utf-8") )

  tokenized = vectorize_layer(tf.expand_dims(X, -1))
  print(" original vectorizer layer: Tokenized and Transformed to a vector of integers: " )
  print (tokenized)

  tokenized = loaded_vectorizer_layer(tf.expand_dims(X, -1))
  print(" loaded_vectorizer_layer: Tokenized and Transformed to a vector of integers: " )
  #print (tokenized.to_tensor(shape=[1, max_review_size]))
  print (tokenized)
  
  tokenized = loaded_vectorizer_model.predict(tf.expand_dims(X, -1))
  print(" loaded_vectorizer_model: Tokenized and Transformed to a vector of integers: " )
  #print (tokenized.to_tensor(shape=[1, max_review_size]))
  print (tokenized)

  print(" Text after Tokenized and Transformed: ")
  transformed = ""
  for each in tf.squeeze(tokenized):
    transformed= transformed+ " "+ vocab[each]
  print(transformed)
  
def prepare_lm_inputs_labels(text):
    text = tf.expand_dims(text, -1) 
    return tf.squeeze(vectorize_layer(text))

train_text_ds = train_text_ds_raw.map(prepare_lm_inputs_labels, 
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

for each in train_text_ds.take(1):
  print(each)
  
train_ds = tf.data.Dataset.zip(
    (       train_text_ds,
            train_cat_ds_raw
        )
) 
for X,y in train_ds.take(1):
  print("X.shape: ",X.shape, "y.shape: ", y.shape)
  print("X: ", X)
  print("y: ", y)
  input = " ".join([vocab[_] for _ in np.squeeze(X)])
  output = id_to_category[y.numpy()]
  print("input (review as text): " , input)
  print("output (category as text): " , output)

train_size = train_ds.cardinality().numpy()
print("Train size: ", train_size)

val_text_ds_raw = tf.data.Dataset.from_tensor_slices(
            tf.cast(val_features.values, tf.string)
) 
val_cat_ds_raw = tf.data.Dataset.from_tensor_slices(
            tf.cast(val_targets.values, tf.int64),

) 
val_text_ds = val_text_ds_raw.map(prepare_lm_inputs_labels, 
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_ds = tf.data.Dataset.zip(
    (       val_text_ds,
            val_cat_ds_raw
       )
) 
for X,y in val_ds.take(1):
  print("X.shape: ",X.shape, "y.shape: ", y.shape)
  print("X: ", X)
  print("y: ",y)
  input = " ".join([vocab[_] for _ in np.squeeze(X)])
  output = id_to_category[y.numpy()]
  print("input (review as text): " , input)
  print("output (category as text ): " , output)
  
test_text_ds_raw = tf.data.Dataset.from_tensor_slices(
            tf.cast(test_features.values, tf.string)
) 
test_cat_ds_raw = tf.data.Dataset.from_tensor_slices(
            tf.cast(test_targets.values, tf.int64),

) 
test_text_ds = test_text_ds_raw.map(prepare_lm_inputs_labels, 
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = tf.data.Dataset.zip(
    (       test_text_ds,
            test_cat_ds_raw
       )
) 
for X,y in test_ds.take(1):
  print("X.shape: ",X.shape, "y.shape: ", y.shape)
  print("X: ", X)
  print("y: ",y)
  input = " ".join([vocab[_] for _ in np.squeeze(X)])
  output = id_to_category[y.numpy()]
  print("input (review as text): " , input)
  print("output (category as text ): " , output)

test_size = test_ds.cardinality().numpy()
print("Test size: ", test_size)

batch_size=64
AUTOTUNE=tf.data.experimental.AUTOTUNE

train_ds=train_ds.shuffle(buffer_size=train_size)
train_ds=train_ds.batch(batch_size=batch_size,drop_remainder=True)
train_ds=train_ds.cache()
train_ds = train_ds.prefetch(AUTOTUNE)

val_ds=val_ds.shuffle(buffer_size=train_size)
val_ds=val_ds.batch(batch_size=batch_size,drop_remainder=True)
val_ds=val_ds.cache()
val_ds = val_ds.prefetch(AUTOTUNE)

test_ds=test_ds.shuffle(buffer_size=train_size)
test_ds=test_ds.batch(batch_size=batch_size,drop_remainder=True)
test_ds=test_ds.cache()
test_ds = test_ds.prefetch(AUTOTUNE)

for X, y in train_ds.take(1):
  print(X.shape, y.shape)
  print("All categories values in this batch: ", y)
  print("\nFirst sample in the batch:")
  print("\tX is: " ,X[0])
  print("\ty is: ", y[0].numpy)
  input = " ".join([vocab[_] for _ in np.squeeze(X[0])])
  output = id_to_category[y[0].numpy()]
  print("\tinput (in text): " , input)
  print("\toutput (in category): " , output)

  print("\nSecond sample in the batch:")
  print("\tX is: " ,X[1])
  print("\ty is: ", y[1].numpy)
  input = " ".join([vocab[_] for _ in np.squeeze(X[1])])
  output = id_to_category[y[1].numpy()]
  print("\tinput (in text): " , input)
  print("\toutput (in category): " , output)

# Embedding size for each token
embed_dim = 16 
# Hidden layer size in feed forward network
feed_forward_dim = 64  
def create_model_FFN():
    inputs_tokens = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding_layer = layers.Embedding(input_dim=vocab_size, 
                                       output_dim=embed_dim, 
                                       input_length=max_len)
    x = embedding_layer(inputs_tokens)
    x = layers.Flatten()(x)
    dense_layer = layers.Dense(feed_forward_dim, activation='relu')
    x = dense_layer(x)
    x = layers.Dropout(.5)(x)
    outputs = layers.Dense(number_of_categories)(x)

    model = keras.Model(inputs=inputs_tokens, outputs=outputs, name='model_FFN')
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric_fn  = tf.keras.metrics.SparseCategoricalAccuracy()
    model.compile(optimizer="adam", loss=loss_fn, metrics=metric_fn)  
    return model
  
model_FFN=create_model_FFN()

print(model_FFN.summary())
tf.keras.utils.plot_model(model_FFN,show_shapes=True)
history=model_FFN.fit(train_ds, validation_data=val_ds ,verbose=2, epochs=7)

plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('model_FFN accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_FFN loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

tf.keras.models.save_model(model_FFN, 'MultiClassTextClassification_FFN')

loss, accuracy = model_FFN.evaluate(test_ds)
print("Test accuracy: ", accuracy)

preds = model_FFN.predict(test_ds)
preds = preds.argmax(axis=1)
print("preds...:",preds)

actuals = test_ds.unbatch().map(lambda x,y: y)  
actuals=list(actuals.as_numpy_iterator())

from sklearn import metrics
print(metrics.classification_report(actuals, preds, digits=4))

from sklearn.metrics import confusion_matrix
# Creating  a confusion matrix, 
# which compares the y_test and y_pred
cm = confusion_matrix(actuals, preds)
cm_df = pd.DataFrame(cm, index = id_to_category.values() ,columns = id_to_category.values())
print(cm_df)

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(22,22))
ax = sns.heatmap(cm_df/np.sum(cm_df), annot=True, fmt='.1%', cmap='Blues')

ax.set_title('Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted')
ax.set_ylabel('Actual');

ax.xaxis.set_ticklabels(id_to_category.values())
ax.yaxis.set_ticklabels(id_to_category.values())

## Display the visualization of the Confusion Matrix.
plt.show()



