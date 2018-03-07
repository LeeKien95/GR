import menpo.io as mio
import os
from sklearn import svm
import numpy as np


class ChangeVector:
    def __init__(self, facs = [], landmarkChange = [], emotion_label = 0):
        self.landmarkChange = landmarkChange
        self.facs = facs
        self.emotion_label = emotion_label

def process_input_image(image, crop_proportion=0.2, max_diagonal=400):
    if image.n_channels == 3:
        image = image.as_greyscale()
    image = image.crop_to_landmarks_proportion(crop_proportion)
    d = image.diagonal()
    if d > max_diagonal:
        image = image.rescale(float(max_diagonal) / d)
    return image


class Emotion:
    def __init__(self, name, facs_required, criteria):
        self.name = name
        self.facs_required = facs_required
        self.criteria = criteria
    
    def criteria(self, facs_input):
        return True
    
    def score(self,facs_input = []):
        if(self.criteria(facs_input) == True):
            max = 0
            for required in self.facs_required:
                au_count = 0
                for facs in facs_input:
                    if facs in required:
                        au_count += 1
                if au_count/float(len(required)) >= max:
                    max = au_count/float(len(required))
            return max
        else:
            return 0
    
def angry_criteria(facs_input):
    if(23 in facs_input):
        return True
    return False

def disgus_criteria(facs_input):
    if(9 in facs_input or 10 in facs_input):
        return True
    return False

def fear_criteria(facs_input):
    if(1 in facs_input and 2 in facs_input and 3 in facs_input):
        return True
    return False

def surprise_criteria(facs_input):
    if(1 in facs_input and 2 in facs_input):
        return True
    if(5 in facs_input):
        return True
    return False

def sadness_criteria(facs_input):
    return True

def happy_criteria(facs_input):
    if(12 in facs_input):
        return True
    return False

def contempt_criteria(facs_input):
    if(14 in facs_input):
        return True
    return False

happy = Emotion('happy', [[6,12]], happy_criteria)
sadness = Emotion('sadness', [[1,4,5], [6,15], [1,4,15]], sadness_criteria)
surprise = Emotion('surprise', [[1,2,5,26]], surprise_criteria)
fear = Emotion('fear', [[1,2,4,5,7,20,26]], fear_criteria)
angry = Emotion('angry', [[4,5,7,23]], angry_criteria)
disgust = Emotion('disgust', [[9,15,16], [10,15,16]], disgus_criteria)
contempt = Emotion('contempt', [[12,14]], contempt_criteria)

emotions = [happy, sadness, surprise, fear, angry, disgust, contempt]

