import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from urllib.request import urlretrieve
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer, TFBertModel

# Import Datasets
df_sentences_snips_train = pd.read_csv('https://raw.githubusercontent.com/moore3930/SlotRefine/main/data/snips/train/seq.in',
                               header=None, names=["sentences"])

df_sentences_snips_test = pd.read_csv('https://raw.githubusercontent.com/moore3930/SlotRefine/main/data/snips/test/seq.in',
                               header=None, names=["sentences"])

df_sentences_snips_valid = pd.read_csv('https://raw.githubusercontent.com/moore3930/SlotRefine/main/data/snips/valid/seq.in',
                               header=None, names=["sentences"])

df_slot_snips_train = pd.read_csv('https://raw.githubusercontent.com/moore3930/SlotRefine/main/data/snips/train/seq.out', 
                            header=None, names=["slot"])

df_slot_snips_test = pd.read_csv('https://raw.githubusercontent.com/moore3930/SlotRefine/main/data/snips/test/seq.out', 
                            header=None, names=["slot"])

df_slot_snips_valid = pd.read_csv('https://raw.githubusercontent.com/moore3930/SlotRefine/main/data/snips/valid/seq.out', 
                            header=None, names=["slot"])

# Download vocab.slot
path = Path("vocab.slot")
if not path.exists():
    print("Downloading vocab.slot ...")
    urlretrieve("https://github.com/ogrisel/slot_filling_and_intent_detection_of_SLU/blob/master/data/snips/vocab.slot" + "?raw=true", path)

# Preprocessing
# Install transformers library
# get_ipython().system('pip install transformers')

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

first_sentence = df_sentences_snips_train.iloc[0]['sentences']

# Print first sentence
print(first_sentence)

# Tokenize the first sentence
print(tokenizer.tokenize(first_sentence))

# Encode sentence to id
print(tokenizer.encode(first_sentence))

# Decode the encoded sentence
print(tokenizer.decode(tokenizer.encode(first_sentence)))

# Load the BERT tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

# Calculate and visualize the distribution of token lengths in the training data
train_sequence_lengths = [len(tokenizer.encode(text)) for text in df_sentences_snips_train['sentences']]
plt.hist(train_sequence_lengths, bins=30)
plt.title(f'Max sequence length in train: {max(train_sequence_lengths)}')
plt.xlabel('Length')
plt.ylabel('Count')
plt.show()

# Display the vocabulary size of the BERT tokenizer
print(f'Vocabulary size: {tokenizer.vocab_size} words.')

# Load and process slot names
slot_names = []
slot_names += Path('vocab.slot').read_text('utf-8').strip().splitlines()

slot_map = {}
for label in slot_names:
    slot_map[label] = len(slot_map)

# Encode input sentences into token IDs and attention masks
def encode_sentences(tokenizer, text_sequences, max_length):
    token_ids = np.zeros(shape=(len(text_sequences), max_length), dtype=np.int32)
    for i, text_sequence in enumerate(text_sequences):
        encoded = tokenizer.encode(text_sequence)
        token_ids[i, 0:len(encoded)] = encoded
    attention_masks = (token_ids != 0).astype(np.int32)
    return {'input_ids': token_ids, 'attention_mask': attention_masks}

encoded_train = encode_sentences(tokenizer, df_sentences_snips_train['sentences'], 55)
encoded_validation = encode_sentences(tokenizer, df_sentences_snips_valid['sentences'], 55)
encoded_test = encode_sentences(tokenizer, df_sentences_snips_test['sentences'], 55)

# Encode slot labels for slot filling
def encode_slot_labels(text_sequences, slot_names, tokenizer, slot_map, max_length):
    encoded = np.zeros(shape=(len(text_sequences), max_length), dtype=np.int32)
    for i, (text_sequence, word_labels) in enumerate(zip(text_sequences, slot_names)):
        encoded_labels = []
        for word, word_label in zip(text_sequence.split(), word_labels.split()):
            tokens = tokenizer.tokenize(word)
            encoded_labels.append(slot_map[word_label])
            expand_label = word_label.replace("B-", "I-")
            if not expand_label in slot_map:
                expand_label = word_label
            encoded_labels.extend([slot_map[expand_label]] * (len(tokens) - 1))
        encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
    return encoded

slot_train = encode_slot_labels(df_sentences_snips_train['sentences'], df_slot_snips_train['slot'], tokenizer, slot_map, 55)
slot_validation = encode_slot_labels(df_sentences_snips_valid['sentences'], df_slot_snips_valid['slot'], tokenizer, slot_map, 55)
slot_test = encode_slot_labels(df_sentences_snips_test['sentences'], df_slot_snips_test['slot'], tokenizer, slot_map, 55)

class SlotFillingModel(tf.keras.Model):
    def __init__(self, slot_num_labels=None, model_name="bert-base-uncased", dropout_prob=0.1):
        super().__init__(name="joint_intent_slot")
        self.bert = TFBertModel.from_pretrained(model_name)
        self.dropout = Dropout(dropout_prob)
        self.slot_classifier = Dense(slot_num_labels, name="slot_classifier")
        
    def call(self, inputs, **kwargs):
        sequence_output, _ = self.bert(inputs, **kwargs).values()
        sequence_output = self.dropout(sequence_output, training=kwargs.get("training", False))
        slot_logits = self.slot_classifier(sequence_output)
        return slot_logits
    
    def save(self, name):
        return self.model.save('{}'.format(name))

model = SlotFillingModel(slot_num_labels=len(slot_map))

opt = Adam(learning_rate=3e-5, epsilon=1e-08)
losses = [SparseCategoricalCrossentropy(from_logits=True)]
metrics = [SparseCategoricalAccuracy('accuracy')]
model.compile(optimizer=opt, loss=losses, metrics=metrics)

history = model.fit(
    encoded_train, slot_train,
    validation_data=(encoded_validation, slot_validation),
    epochs=1, batch_size=32)

# Save Model
tf.saved_model.save(model, 'model-slot-filling-SNIPS')

# Load Model
new_model = tf.saved_model.load('model-slot-filling-SNIPS')

# Evaluation
model.evaluate(encoded_test, slot_test)
model.summary()
slot_test.shape

def show_predictions(text, tokenizer, model, slot_names):
    inputs = tf.constant(tokenizer.encode(text))[None, :]  # batch_size = 1
    outputs = model(inputs)
    slot_logits = outputs
    slot_ids = slot_logits.numpy().argmax(axis=-1)[0, 1:-1]
    slot_prediction = ""
    for token, slot_id in zip(tokenizer.tokenize(text), slot_ids):
        slot_prediction += f"{token}:{slot_names[slot_id]} "
    return slot_prediction

# Example of classification
print(show_predictions('Will it snow tomorrow in Paris?',
                 tokenizer, model, slot_names))

df_training = df_sentences_snips_train.append(df_sentences_snips_valid)
df_all = df_training.append(df_sentences_snips_test)

all_predictions = []
for idx, row in df_all.iterrows():
    all_predictions.append(show_predictions(row['sentences'],
                 tokenizer, model, slot_names))
    
print(len(all_predictions))

df_all_predictions = pd.DataFrame(all_predictions, columns=['slot mapping'])
df_all_predictions.to_excel("slot_filling-snips.xlsx")
