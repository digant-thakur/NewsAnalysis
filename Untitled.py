
# coding: utf-8

# In[3]:


import keras 
import numpy as np
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Flatten
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[38]:


data = pd.read_csv('asanew.csv')


# In[39]:


data.head()


# In[40]:


def cut(str):
    return str.split()[0]


# In[41]:


data['code']=data['code'].apply(cut)


# In[42]:


#data['code']


# In[43]:


data['target'] = data.code.astype('category').cat.codes


# In[44]:


data['num_words'] = data.text.apply(lambda x : len(x.split()))


# In[45]:


num_class = len(np.unique(data.code.values))
y = data['target'].values


# In[46]:


y.shape


# In[47]:


MAX_LENGTH = 100
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data.text.values)
post_seq = tokenizer.texts_to_sequences(data.text.values)
post_seq_padded = pad_sequences(post_seq, maxlen=MAX_LENGTH)


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(post_seq_padded, y, test_size=0.3)


# In[28]:


vocab_size = len(tokenizer.word_index) + 1


# In[29]:


inputs = Input(shape=(MAX_LENGTH, ))
embedding_layer = Embedding(vocab_size,
                            128,
                            input_length=MAX_LENGTH)(inputs)
x = Flatten()(embedding_layer)
x = Dense(32, activation='relu')(x)

predictions = Dense(num_class, activation='softmax')(x)
model = Model(inputs=[inputs], outputs=predictions)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

model.summary()


# In[30]:


filepath="weights-simple.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = model.fit([X_train], batch_size=64, y=to_categorical(y_train), verbose=1, validation_split=0.25, 
          shuffle=True, epochs=1, callbacks=[checkpointer])


# In[18]:



inputs = Input(shape=(MAX_LENGTH, ))
embedding_layer = Embedding(vocab_size,
                            128,
                            input_length=MAX_LENGTH)(inputs)

x = LSTM(64)(embedding_layer)
x = Dense(32, activation='relu')(x)
predictions = Dense(num_class, activation='softmax')(x)
model = Model(inputs=[inputs], outputs=predictions)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

model.summary()


# In[ ]:


filepath="weights.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = model.fit([X_train], batch_size=64, y=to_categorical(y_train), verbose=1, validation_split=0.25, 
          shuffle=True, epochs=10, callbacks=[checkpointer])

