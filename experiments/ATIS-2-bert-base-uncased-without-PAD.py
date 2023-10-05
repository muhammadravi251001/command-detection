import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from urllib.request import urlretrieve
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from google.colab import files
from tensorflow.keras.models import load_model

# Load dataset
df_sentences_atis_train = pd.read_csv('https://raw.githubusercontent.com/moore3930/SlotRefine/main/data/atis/train/seq.in',
                                      header=None, names=["sentences"])

df_sentences_atis_test = pd.read_csv('https://raw.githubusercontent.com/moore3930/SlotRefine/main/data/atis/test/seq.in',
                                     header=None, names=["sentences"])

df_sentences_atis_valid = pd.read_csv('https://raw.githubusercontent.com/moore3930/SlotRefine/main/data/atis/valid/seq.in',
                                      header=None, names=["sentences"])

df_slot_atis_train = pd.read_csv('https://raw.githubusercontent.com/moore3930/SlotRefine/main/data/atis/train/seq.out',
                                 header=None, names=["slot"])

df_slot_atis_test = pd.read_csv('https://raw.githubusercontent.com/moore3930/SlotRefine/main/data/atis/test/seq.out',
                                header=None, names=["slot"])

df_slot_atis_valid = pd.read_csv('https://raw.githubusercontent.com/moore3930/SlotRefine/main/data/atis/valid/seq.out',
                                 header=None, names=["slot"])

# Check if 'vocab.slot' exists, otherwise download it
vocab_path = Path("vocab.slot")
if not vocab_path.exists():
    print("Downloading vocab.slot ...")
    urlretrieve("https://github.com/ogrisel/slot_filling_and_intent_detection_of_SLU/blob/master/data/atis-2/vocab.slot" + "?raw=true",
                vocab_path)

# Preprocessing
# Install the 'transformers' package if not installed
# Import other necessary packages
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

# Tokenize the first sentence
first_sentence = df_sentences_atis_train.iloc[0]['sentences']
print(first_sentence)
tokenized_sentence = tokenizer.tokenize(first_sentence)
encoded_sentence = tokenizer.encode(first_sentence)

# Decoding an encoded sentence
decoded_sentence = tokenizer.decode(tokenizer.encode(first_sentence))

# Calculate and plot the sequence length distribution
train_sequence_lengths = [len(tokenizer.encode(text)) for text in df_sentences_atis_train['sentences']]
plt.hist(train_sequence_lengths, bins=30)
plt.title(f'Max sequence length: {max(train_sequence_lengths)}')
plt.xlabel('Length')
plt.ylabel('Count')
plt.show()

# Print the vocabulary size
print(f'Vocabulary size: {tokenizer.vocab_size} words.')

# Encoding the dataset
def encode_dataset(tokenizer, text_sequences, max_length):
    token_ids = np.zeros(shape=(len(text_sequences), max_length), dtype=np.int32)
    for i, text_sequence in enumerate(text_sequences):
        encoded = tokenizer.encode(text_sequence)
        token_ids[i, 0:len(encoded)] = encoded
    attention_masks = (token_ids != 0).astype(np.int32)
    
    return {'input_ids': token_ids, 'attention_mask': attention_masks}

encoded_train = encode_dataset(tokenizer, df_sentences_atis_train['sentences'], 65)
encoded_validation = encode_dataset(tokenizer, df_sentences_atis_valid['sentences'], 65)
encoded_test = encode_dataset(tokenizer, df_sentences_atis_test['sentences'], 65)

# Define slot names and create a slot map
slot_names = []
slot_names += Path('vocab.slot').read_text('utf-8').strip().splitlines()

slot_map = {}
for label in slot_names:
    slot_map[label] = len(slot_map)

# Encode token labels
def encode_token_labels(text_sequences, slot_names, tokenizer, slot_map, max_length):
    encoded = np.zeros(shape=(len(text_sequences), max_length), dtype=np.int32)
    for i, (text_sequence, word_labels) in enumerate(zip(text_sequences, slot_names)):
        encoded_labels = []
        for word, word_label in zip(text_sequence.split(), word_labels.split()):
            tokens = tokenizer.tokenize(word)
            encoded_labels.append(slot_map[word_label])
            expand_label = word_label.replace("B-", "I-")
            if expand_label not in slot_map:
                expand_label = word_label
            encoded_labels.extend([slot_map[expand_label]] * (len(tokens) - 1))
        encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
    return encoded

slot_train = encode_token_labels(df_sentences_atis_train['sentences'], df_slot_atis_train["slot"], tokenizer, slot_map, 65)
slot_validation = encode_token_labels(df_sentences_atis_valid['sentences'], df_slot_atis_valid["slot"], tokenizer, slot_map, 65)
slot_test = encode_token_labels(df_sentences_atis_test['sentences'], df_slot_atis_test["slot"], tokenizer, slot_map, 65)

# Modelling
class SlotFillingModel(tf.keras.Model):
    def __init__(self, slot_num_labels=None, model_name="bert-base-uncased", dropout_prob=0.1):
        super().__init__(name="joint_intent_slot")
        self.bert = TFBertModel.from_pretrained(model_name)
        self.dropout = Dropout(dropout_prob)
        self.slot_classifier = Dense(slot_num_labels, name="slot_classifier")

    def call(self, inputs, **kwargs):
        sequence_output, _ = self.bert(inputs, **kwargs)
        
        # (batch_size, max_length, output_dim)
        sequence_output = self.dropout(sequence_output, training=kwargs.get("training", False))
        slot_logits = self.slot_classifier(sequence_output)
        return slot_logits

    def save(self, name):
        return self.save_model('{}'.format(name))


model = SlotFillingModel(slot_num_labels=len(slot_map))

opt = Adam(learning_rate=3e-5, epsilon=1e-08)
losses = SparseCategoricalCrossentropy(from_logits=True)
metrics = [SparseCategoricalAccuracy('accuracy')]
model.compile(optimizer=opt, loss=losses, metrics=metrics)

history = model.fit(
    encoded_train, slot_train,
    validation_data=(encoded_validation, slot_validation),
    epochs=2, batch_size=32)

# Save Model
import h5py
model.save('model-slot-filling-ATIS')

# Load Model
new_model = load_model('model-slot-filling-ATIS')

files.download('model-slot-filling-ATIS')

# Evaluation
evaluation_result = model.evaluate(encoded_test, slot_test)
print("Evaluation Result:", evaluation_result)

model.summary()

def show_predictions(text, tokenizer, model, slot_names):
    inputs = tf.constant(tokenizer.encode(text))[None, :]  # batch_size = 1
    outputs = model(inputs)
    slot_logits = outputs
    slot_ids = slot_logits.numpy().argmax(axis=-1)[0, 1:-1]
    slot_prediction = ""
    for token, slot_id in zip(tokenizer.tokenize(text), slot_ids):
        slot_prediction += f"{token}:{slot_names[slot_id]} "
    return slot_prediction

# Combine training, validation, and test data
df_training = pd.concat([df_sentences_atis_train, df_sentences_atis_valid, df_sentences_atis_test])

all_predictions = []
for idx, row in df_training.iterrows():
    all_predictions.append(show_predictions(row['sentences'], tokenizer, model, slot_names))

print(len(all_predictions))

df_all_predictions = pd.DataFrame(all_predictions, columns=['slot mapping'])

# Save predictions to Excel
df_all_predictions.to_excel("slot_filling-atis.xlsx")
