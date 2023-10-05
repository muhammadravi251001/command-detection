import pandas as pd
import numpy as np
import h5py
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer

from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Dense, LSTM, BatchNormalization, Dropout, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy as CC
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.initializers import he_uniform, glorot_uniform
from tensorflow.keras.metrics import AUC
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model

nltk.download('punkt')
nltk.download("stopwords")

# Import ATIS Datasets
df_sentences_atis_train = pd.read_csv('https://raw.githubusercontent.com/moore3930/SlotRefine/main/data/atis/train/seq.in',
                                header=None, names=["sentences"])

df_sentences_atis_test = pd.read_csv('https://raw.githubusercontent.com/moore3930/SlotRefine/main/data/atis/test/seq.in',
                                header=None, names=["sentences"])

df_label_atis_train = pd.read_csv('https://raw.githubusercontent.com/moore3930/SlotRefine/main/data/atis/train/label', 
                            header=None, names=["label"])

df_label_atis_test = pd.read_csv('https://raw.githubusercontent.com/moore3930/SlotRefine/main/data/atis/test/label', 
                            header=None, names=["label"])

# Exploratory Data Analysis (EDA)
df_count_label_atis = df_label_atis_train.append(df_label_atis_test).groupby('label')['label'].count() \
                            .reset_index(name='count') \
                            .sort_values(['count'], ascending=False)
                            
df_count_label_atis.plot.bar(x='label', y='count', rot=90, 
                             figsize=(14,5), 
                             title="Distribution of Label Counts in ATIS Data")

# Preprocessing
y_encoder= OneHotEncoder().fit(np.array(df_label_atis_train.append(df_label_atis_test)['label']).reshape(-1,1))
ytr_encoded= y_encoder.transform(np.array(df_label_atis_train['label']).reshape(-1,1)).toarray()
yts_encoded= y_encoder.transform(np.array(df_label_atis_test['label']).reshape(-1,1)).toarray()

df_sentences_atis_train["lower_text"] = df_sentences_atis_train["sentences"].map(lambda x: x.lower())
df_sentences_atis_test["lower_text"] = df_sentences_atis_test["sentences"].map(lambda x: x.lower())

# Tokenization
df_sentences_atis_train["tokenized"] = df_sentences_atis_train["lower_text"].map(word_tokenize)
df_sentences_atis_test["tokenized"] = df_sentences_atis_test["lower_text"].map(word_tokenize)

# Stopword Removal
def remove_stop(strings, stop_list):
    classed= [s for s in strings if s not in stop_list]
    return classed

stop= stopwords.words("english")
stop_punc= list(set(punctuation))+ stop

df_sentences_atis_train["selected"]= df_sentences_atis_train["tokenized"].map(lambda df: remove_stop(df, stop_punc))
df_sentences_atis_test["selected"]= df_sentences_atis_test["tokenized"] .map(lambda df: remove_stop(df, stop_punc))

# Stemming
def normalize(text):
    return " ".join(text)

stemmer= PorterStemmer()

df_sentences_atis_train["stemmed"]= df_sentences_atis_train["selected"].map(lambda xs: [stemmer.stem(x) for x in xs])
df_sentences_atis_train["normalized"]= df_sentences_atis_train["stemmed"].apply(normalize)

df_sentences_atis_test["stemmed"]= df_sentences_atis_test["selected"].map(lambda xs: [stemmer.stem(x) for x in xs])
df_sentences_atis_test["normalized"]= df_sentences_atis_test["stemmed"].apply(normalize)

# Tokenization with Keras Tokenizer
tokenizer= Tokenizer(num_words= 10000)
tokenizer.fit_on_texts(df_sentences_atis_train["normalized"])

tokenized_train= tokenizer.texts_to_sequences(df_sentences_atis_train["normalized"])
tokenized_test= tokenizer.texts_to_sequences(df_sentences_atis_test["normalized"])

# Padding Sequences
train_padded= pad_sequences(tokenized_train, maxlen= 20, padding= "pre")
test_padded= pad_sequences(tokenized_test, maxlen= 20, padding= "pre")

# Transforming Text to 3D Matrix
def transform_x(data, tokenizer):
    output_shape= [data.shape[0],
                  data.shape[1],
                  len(tokenizer.word_index.keys())]
    results= np.zeros(output_shape)
    
    for i in range(data.shape[0]):
        for ii in range(data.shape[1]):
            results[i, ii, data[i,ii]-1]= 1
    return results

xtr_transformed = transform_x(train_padded, tokenizer)
xts_transformed = transform_x(test_padded, tokenizer)

class LSTMModel(object):
    
    def build_model(self, input_dim, output_shape, steps, dropout_rate, kernel_regularizer, bias_regularizer):
        input_layer= Input(shape= (steps, input_dim))
        
        # Make lstm_layer
        lstm= LSTM(units= steps)(input_layer)
        dense_1= Dense(output_shape, kernel_initializer= he_uniform(),
                       bias_initializer= "zeros", 
                       kernel_regularizer= l2(l= kernel_regularizer),
                       bias_regularizer= l2(l= bias_regularizer))(lstm)
        x= BatchNormalization()(dense_1)
        x= relu(x)
        x= Dropout(rate= dropout_rate)(x)
        o= Dense(output_shape, kernel_initializer= glorot_uniform(),
                 bias_initializer= "zeros", 
                 kernel_regularizer= l2(l= kernel_regularizer), 
                 bias_regularizer= l2(l= bias_regularizer))(x)
        o= BatchNormalization()(o)
        output= softmax(o, axis= 1)
        
        loss= CC()
        metrics= AUC()
        optimizer= Adam()
        self.model= Model(inputs= [input_layer], outputs= [output])
        self.model.compile(optimizer= optimizer, loss= loss, metrics= [metrics])
    
    def train(self, x, y, validation_split, epochs):
        self.model.fit(x, y, validation_split= validation_split, epochs= epochs)
        
    def predict(self, x):
        return self.model.predict(x)
    
    def save(self, name):
        return self.model.save('{}'.format(name))

    def load(self):
        return self.load_model('model-LSTM.h5')

    def summary(self):
        return self.model.summary()

steps = xtr_transformed.shape[1]
dim = xtr_transformed.shape[2]
output_shape = ytr_encoded.shape[1]

model= LSTMModel()
model.build_model(input_dim= dim,
                  output_shape= output_shape,
                  steps= steps, 
                  dropout_rate= 0.5, 
                  bias_regularizer= 0.0001, 
                  kernel_regularizer= 0.3)

model.train(xtr_transformed, ytr_encoded,
           0.01, 150)

model.save('model-LSTM-ATIS.h5')

new_model = load_model('model-LSTM-ATIS.h5')

model.summary()

def evaluate_and_analyze_model(model, xtr_transformed, xts_transformed, y_encoder, df_label_atis_train, df_label_atis_test):
    # Evaluate the model and generate classification reports
    prediction_train = y_encoder.inverse_transform(model.predict(xtr_transformed))
    prediction_test = y_encoder.inverse_transform(model.predict(xts_transformed))

    print("Classification Report for Training Data:")
    print(classification_report(df_label_atis_train["label"], prediction_train))

    print("Classification Report for Test Data:")
    print(classification_report(df_label_atis_test["label"], prediction_test))

    # Create DataFrames for predictions
    df_prediction_train = pd.DataFrame(prediction_train, columns=['Predicted Label'])
    df_prediction_test = pd.DataFrame(prediction_test, columns=['Predicted Label'])

    # Concatenate predictions for both training and test data
    df_all_prediction = df_prediction_train.append(df_prediction_test)

    # Save predictions to an Excel file
    df_all_prediction.to_excel("predicted-atis.xlsx")

# Call the function to evaluate and analyze the model
evaluate_and_analyze_model(model, xtr_transformed, xts_transformed, y_encoder, df_label_atis_train, df_label_atis_test)

# Importing SNIPS dataset
df_sentences_snips_train = pd.read_csv('https://raw.githubusercontent.com/moore3930/SlotRefine/main/data/snips/train/seq.in',
                                       header=None, names=["sentences"])

df_sentences_snips_test = pd.read_csv('https://raw.githubusercontent.com/moore3930/SlotRefine/main/data/snips/test/seq.in',
                                      header=None, names=["sentences"])

df_label_snips_train = pd.read_csv('https://raw.githubusercontent.com/moore3930/SlotRefine/main/data/snips/train/label',
                                   header=None, names=["label"])

df_label_snips_test = pd.read_csv('https://raw.githubusercontent.com/moore3930/SlotRefine/main/data/snips/test/label',
                                  header=None, names=["label"])

# Exploratory Data Analysis (EDA)
df_count_label_snips = df_label_snips_train.append(df_label_snips_test).groupby('label')['label'].count() \
                            .reset_index(name='count') \
                            .sort_values(['count'], ascending=False)

df_count_label_snips.plot.bar(x='label', y='count', rot=90,
                             figsize=(14,5),
                             title="Distribution of Label Counts in Snips Data")

# Preprocessing
y_encoder = OneHotEncoder().fit(np.array(df_label_snips_train.append(df_label_snips_test)['label']).reshape(-1,1))

ytr_encoded = y_encoder.transform(np.array(df_label_snips_train['label']).reshape(-1,1)).toarray()
yts_encoded = y_encoder.transform(np.array(df_label_snips_test['label']).reshape(-1,1)).toarray()

df_sentences_snips_train["lower_text"] = df_sentences_snips_train["sentences"].map(lambda x: x.lower())
df_sentences_snips_test["lower_text"] = df_sentences_snips_test["sentences"].map(lambda x: x.lower())

# Tokenization
df_sentences_snips_train["tokenized"] = df_sentences_snips_train["lower_text"].map(word_tokenize)
df_sentences_snips_test["tokenized"] = df_sentences_snips_test["lower_text"].map(word_tokenize)

# Stopword Removal
def remove_stop(strings, stop_list):
    return [s for s in strings if s not in stop_list]

stop = stopwords.words("english")
stop_punc = list(set(punctuation)) + stop

df_sentences_snips_train["selected"] = df_sentences_snips_train["tokenized"].map(lambda df: remove_stop(df, stop_punc))
df_sentences_snips_test["selected"] = df_sentences_snips_test["tokenized"].map(lambda df: remove_stop(df, stop_punc))

# Stemming
stemmer = PorterStemmer()

def normalize(text):
    return " ".join(text)

df_sentences_snips_train["stemmed"] = df_sentences_snips_train["selected"].map(lambda xs: [stemmer.stem(x) for x in xs])
df_sentences_snips_train["normalized"] = df_sentences_snips_train["stemmed"].apply(normalize)

df_sentences_snips_test["stemmed"] = df_sentences_snips_test["selected"].map(lambda xs: [stemmer.stem(x) for x in xs])
df_sentences_snips_test["normalized"] = df_sentences_snips_test["stemmed"].apply(normalize)

# Tokenization and Padding
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df_sentences_snips_train["normalized"])

tokenized_train = tokenizer.texts_to_sequences(df_sentences_snips_train["normalized"])
tokenized_test = tokenizer.texts_to_sequences(df_sentences_snips_test["normalized"])

train_padded = pad_sequences(tokenized_train, maxlen=20, padding="pre")
test_padded = pad_sequences(tokenized_test, maxlen=20, padding="pre")

# Transforming data into a 3D matrix
def transform_x(data, tokenizer):
    output_shape = [data.shape[0], data.shape[1], len(tokenizer.word_index)]
    results = np.zeros(output_shape)
    
    for i in range(data.shape[0]):
        for ii in range(data.shape[1]):
            results[i, ii, data[i, ii] - 1] = 1
    return results

xtr_transformed = transform_x(train_padded, tokenizer)
xts_transformed = transform_x(test_padded, tokenizer)

# Modeling

# Model Building
class LSTMModel(object):
    
    def build_model(self, input_dim, output_shape, steps, dropout_rate, kernel_regularizer, bias_regularizer):
        input_layer = Input(shape=(steps, input_dim))
        
        lstm = LSTM(units=steps)(input_layer)
        dense_1 = Dense(output_shape, kernel_initializer=he_uniform(),
                       bias_initializer="zeros", 
                       kernel_regularizer=l2(l=kernel_regularizer),
                       bias_regularizer=l2(l=bias_regularizer))(lstm)
        x = BatchNormalization()(dense_1)
        x = relu(x)
        x = Dropout(rate=dropout_rate)(x)
        o = Dense(output_shape, kernel_initializer=glorot_uniform(),
                 bias_initializer="zeros", 
                 kernel_regularizer=l2(l=kernel_regularizer), 
                 bias_regularizer=l2(l=bias_regularizer))(x)
        o = BatchNormalization()(o)
        output = softmax(o, axis=1)
        
        loss = CC()
        metrics = AUC()
        optimizer = Adam()
        self.model = Model(inputs=[input_layer], outputs=[output])
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    
    def train(self, x, y, validation_split, epochs):
        self.model.fit(x, y, validation_split=validation_split, epochs=epochs)
        
    def predict(self, x):
        return self.model.predict(x)
    
    def save(self, name):
        return self.model.save('{}'.format(name))

    def load(self):
        return self.load_model('model-LSTM.h5')

    def summary(self):
        return self.model.summary()

# Model Training
steps = xtr_transformed.shape[1]
dim = xtr_transformed.shape[2]
output_shape = ytr_encoded.shape[1]

model = LSTMModel()
model.build_model(input_dim=dim,
                  output_shape=output_shape,
                  steps=steps, 
                  dropout_rate=0.5, 
                  bias_regularizer=0.0001, 
                  kernel_regularizer=0.3)

model.train(xtr_transformed.astype(np.uint8), ytr_encoded,
            0.01, 30)

model.save('model-LSTM-SNIPS.h5')

new_model = load_model('model-LSTM-SNIPS.h5')

# Evaluation & Analysis
prediction = y_encoder.inverse_transform(model.predict(xtr_transformed))
print(classification_report(df_label_snips_train["label"], prediction))

prediction_test = y_encoder.inverse_transform(model.predict(xts_transformed))
print(classification_report(df_label_snips_test["label"], prediction_test))

df_prediction = pd.DataFrame(prediction, columns=['Predicted Label'])
df_prediction_test = pd.DataFrame(prediction_test, columns=['Predicted Label'])

df_all_prediction = df_prediction.append(df_prediction_test)
df_all_prediction.to_excel("predicted-snips.xlsx")
