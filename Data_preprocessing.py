import io
import os
import re
import numpy as np
import yaml


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

########################################################################################################################
########################################### DATA PREPARATION ###########################################################
########################################################################################################################

dir_path = 'chatbot_nlp/data'
files_list = os.listdir(dir_path + os.sep)


def clean_text(text_to_clean):
    res = text_to_clean.lower()
    res = re.sub(r"i'm", "i am", res)
    res = re.sub(r"he's", "he is", res)
    res = re.sub(r"she's", "she is", res)
    res = re.sub(r"it's", "it is", res)
    res = re.sub(r"that's", "that is", res)
    res = re.sub(r"what's", "what is", res)
    res = re.sub(r"where's", "where is", res)
    res = re.sub(r"how's", "how is", res)
    res = re.sub(r"\'ll", " will", res)
    res = re.sub(r"\'ve", " have", res)
    res = re.sub(r"\'re", " are", res)
    res = re.sub(r"\'d", " would", res)
    res = re.sub(r"\'re", " are", res)
    res = re.sub(r"won't", "will not", res)
    res = re.sub(r"can't", "cannot", res)
    res = re.sub(r"n't", " not", res)
    res = re.sub(r"n'", "ng", res)
    res = re.sub(r"'bout", "about", res)
    res = re.sub(r"'til", "until", res)
    res = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", res)
    return res


questions = list()
answers = list()
for filepath in files_list:
    stream = open(dir_path + os.sep + filepath, 'rb')
    docs = yaml.safe_load(stream)
    conversations = docs['conversations']
    for con in conversations:
        if len(con) > 2:
            questions.append(con[0])
            ans = ''
            for rep in con[1:]:
                ans += ' ' + rep
            answers.append(ans)
        elif len(con) > 1:
            questions.append(con[0])
            answers.append(con[1])
answers_with_tags = list()
for i in range(len(answers)):
    if type(answers[i]) == str:
        answers_with_tags.append(answers[i])
    else:
        questions.pop(i)
answers = list()
for i in range(len(answers_with_tags)):
    answers.append('<START> ' + answers_with_tags[i] + ' <END>')
print(len(questions))
print(len(answers))
