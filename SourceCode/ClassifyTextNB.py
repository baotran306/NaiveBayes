import numpy as np
import string
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def clear_specific_character(s):
    translator = str.maketrans("", "", string.punctuation)
    return s.translate(translator)


# Preprocessing
def tokenize(s):
    """
    Transform input string to a list of standardized string ignored all specific character and
    ENGLISH_STOP_WORDS like a, an, the, in, at ...
    Example
    x_data = tokenize("abdj 12312 AASDASD    92192325 234@#%#$% asdf!@#  1@#$sda in at")
    return: ['abdj', '12312', 'aasdasd', '92192325', '234', 'asdf', '1sda']
    """
    new_s = clear_specific_character(s).lower().strip()
    token = re.split("\W+", new_s)
    return [w for w in token if w not in ENGLISH_STOP_WORDS]


def read_file(file_name):
    input_text = []
    labels = []
    with open(file_name) as f:
        for line in f:
            if line.startswith("ham"):
                labels.append(0)
                input_text.append(tokenize(line[3:]))
            else:
                labels.append(1)
                input_text.append(tokenize(line[4:]))
    return input_text, labels


def get_word_counts(words):
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0.0) + 1.0
    return word_counts


# Create model to fit and predict
class NBTextClassifier:
    def __init__(self):
        self.vocab = set()
        self.probs = {}
        self.word_counts = {}
        self.prior_classes = {}
        self.class_sample_count = {}
        self.classes = 0

    def fit(self, x, y):
        samples = len(y)
        self.classes = np.unique(y)
        # computes prior probs
        for c in self.classes:
            self.prior_classes[c] = np.log(sum(y == c) / samples)
            self.class_sample_count[c] = sum(y == c)
            self.word_counts[c] = {}

        for text, label in zip(x, y):
            counts = get_word_counts(text)
            for word, count in counts.items():
                if word not in self.vocab:
                    self.vocab.add(word)
                for c in self.classes:
                    if word not in self.word_counts[c].keys():
                        self.word_counts[c][word] = 0.0
                    self.word_counts[c][word] += count

    def predict(self, x):
        result = []
        for raw in x:
            class_probs = self.__compute_class_probs(raw)
            c = max(class_probs, key=class_probs.get)
            result.append(c)
        return ["ham" if x == 0 else "spam" for x in result]

    def __compute_class_probs(self, x):
        p_of_class = {}
        counts = get_word_counts(tokenize(x))
        for c in self.classes:
            p_of_class[c] = self.prior_classes[c]
            for word in counts.keys():
                if word not in self.vocab:
                    continue
                p_of_class[c] += np.log((self.word_counts[c].get(word, 0.0) + 1.0) /
                                        (self.class_sample_count[c] + len(self.vocab)))
        return p_of_class


def set_label_text_out(input_str, out_labels):
    result = []
    for text, label in zip(input_str, out_labels):
        result.append(label + ": " + text)
    return result


if __name__ == '__main__':
    x_data, y_data = read_file('../Data/SMSSpamCollection.txt')
    print(x_data[:5])
    model = NBTextClassifier()
    model.fit(x_data, y_data)
    test_predict_text = [' @# Hello my name is Tran Quoc Bao :vv. Nice to meet you :DD',
                         'Luv ya 3000',
                         'Discount in 3 hours ##@. Call phone ^%%3 22 number 0982371827 to book new room.  ',
                         'Free entry in 2 a wkly comp to see football match Chelsea vs Real 21st May 2020.'
                         ' Text FA to 87121 to receive entry question(std txt rate)s apply ',
                         'Is &_ that seriously how you spell his name? it\'s free :)',
                         'WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!'
                         ' To claim call 09061701461. Claim code KL341. Valid 12 hours only.']
    print("------prediction------")
    ans = model.predict(test_predict_text)
    print(ans)
    for res in set_label_text_out(test_predict_text, ans):
        print(res)
