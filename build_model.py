from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from Data_preprocessing import *
from gensim.models import Word2Vec
from keras import Input, Model
from keras.activations import softmax
from keras.layers import Embedding, LSTM, Dense
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras_preprocessing.text import Tokenizer
from matplotlib import pyplot as plt
import time
from tensorflow.keras.callbacks import TensorBoard


########################################################################################################################
############################################# MODEL BUILDING AND TRAINING ###########################################################
########################################################################################################################

Name = "ChatBotModel-{}".format(int(time.time()))
checkPointModel = "Model-{}.h5".format(int(time.time()))
modelName = "model{}.h5".format(int(time.time()))
filepath ="Saved_models_checkpoints/{}".format(checkPointModel)

es = EarlyStopping(monitor='accuracy', mode='max', verbose=1, patience=9)
M_checkP = ModelCheckpoint(filepath=filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
tensorboard = TensorBoard(log_dir='logs/{}'.format(Name))

target_regex = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n\'0123456789'
tokenizer = Tokenizer(filters=target_regex)
tokenizer.fit_on_texts(questions + answers)
VOCAB_SIZE = len(tokenizer.word_index) + 1
print('Vocabulary size : {}'.format(VOCAB_SIZE))

tokenized_questions = tokenizer.texts_to_sequences(questions)
maxlen_questions = max([len(x) for x in tokenized_questions])
encoder_input_data = pad_sequences(tokenized_questions,
                                   maxlen=maxlen_questions,
                                   padding='post')

print(encoder_input_data.shape)

tokenized_answers = tokenizer.texts_to_sequences(answers)
maxlen_answers = max([len(x) for x in tokenized_answers])
decoder_input_data = pad_sequences(tokenized_answers,
                                   maxlen=maxlen_answers,
                                   padding='post')
print(decoder_input_data.shape)

for i in range(len(tokenized_answers)):
    tokenized_answers[i] = tokenized_answers[i][1:]
padded_answers = pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
decoder_output_data = to_categorical(padded_answers, VOCAB_SIZE)

print(decoder_output_data.shape)

enc_inputs = Input(shape=(None,))
enc_embedding = Embedding(VOCAB_SIZE, 200, mask_zero=True)(enc_inputs)
_, state_h, state_c = LSTM(200, return_state=True)(enc_embedding)
enc_states = [state_h, state_c]

dec_inputs = Input(shape=(None,))
dec_embedding = Embedding(VOCAB_SIZE, 200, mask_zero=True)(dec_inputs)
dec_lstm = LSTM(200, return_state=True, return_sequences=True)
dec_outputs, _, _ = dec_lstm(dec_embedding, initial_state=enc_states)
dec_dense = Dense(VOCAB_SIZE, activation=softmax)
output = dec_dense(dec_outputs)

model = Model([enc_inputs, dec_inputs], output)
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

######################## Training phase######################

#history =model.fit([encoder_input_data, decoder_input_data],
          #decoder_output_data,
         # validation_split=0.33,
          #callbacks=[es,M_checkP,tensorboard],
          #batch_size=32,
          #epochs=1000)
#plt.figure(0)
#plt.plot(history.history['loss'], label='training loss')
#plt.plot(history.history['val_loss'], label='validation loss')
#plt.title("Loss")
#plt.xlabel("Epochs")
#plt.ylabel("loss")
#plt.legend()
#plt.show()

#plt.figure(1)
#plt.plot(history.history['accuracy'], label='training accuracy ')
#plt.plot(history.history['val_accuracy'],label='validation accuracy')
#plt.title("accuracy")
#plt.xlabel("Epochs")
#plt.ylabel("accu")
#plt.legend()
#plt.show()

######################## ######################
model.load_weights('Saved_models_checkpoints/Model-1622045047.h5')

def make_inference_models():
    dec_state_input_h = Input(shape=(200,))
    dec_state_input_c = Input(shape=(200,))
    dec_states_inputs = [dec_state_input_h, dec_state_input_c]
    dec_outputs, state_h, state_c = dec_lstm(dec_embedding,
                                             initial_state=dec_states_inputs)
    dec_states = [state_h, state_c]
    dec_outputs = dec_dense(dec_outputs)
    dec_model = Model(
        inputs=[dec_inputs] + dec_states_inputs,
        outputs=[dec_outputs] + dec_states)
    print('Inference decoder:')
    dec_model.summary()
    print('Inference encoder:')
    enc_model = Model(inputs=enc_inputs, outputs=enc_states)
    enc_model.summary()
    return enc_model, dec_model


def str_to_tokens(sentence: str):
    words = sentence.lower().split()
    tokens_list = list()
    for current_word in words:
        result = tokenizer.word_index.get(current_word, '')
        if result != '':
            tokens_list.append(result)
    return pad_sequences([tokens_list],
                         maxlen=maxlen_questions,
                         padding='post')


enc_model, dec_model = make_inference_models()


