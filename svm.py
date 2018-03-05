import menpo.io as mio
import os
from sklearn import svm
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

path_to_svm_training_database = '/Programing/GR/Code/CK+/aam-images/**/**/**/*'
path_to_facs = '/Programing/GR/Code/CK+/FACS/'
path_to_emotions = '/Programing/GR/Code/CK+/Emotion/'
path_to_svm_testing_database = "/Programing/GR/Code/CK+/test-aam-images/**/**/**/*"


class ChangeVector:
    def __init__(self, facs = [], landmarkChange = [], emotion = 0):
        self.landmarkChange = landmarkChange
        self.facs = facs
        self.emotion = emotion

def process(image, crop_proportion=0.2, max_diagonal=400):
    if image.n_channels == 3:
        image = image.as_greyscale()
    image = image.crop_to_landmarks_proportion(crop_proportion)
    d = image.diagonal()
    if d > max_diagonal:
        image = image.rescale(float(max_diagonal) / d)
    return image

#process changingVector, reduce dimension from 2x68 to 1x68, by process PCA
def pca(changeVector):
    X = np.array(changeVector)
    pca_model = PCA(n_components=1)
    return pca_model.fit_transform(X)


training_images = mio.import_images(path_to_svm_training_database, verbose=True)
training_images = training_images.map(process)



#create training data
count = 0;
svm_training_data = []
while(count < len(training_images)):
    file_path = str(training_images[count].path).split("\\")
    facs_path = path_to_facs + file_path[6] + '/' + file_path[7]
    gt_emotion = file_path[7]
    facs_path = facs_path + '/' + os.listdir(facs_path)[0]
    fi = open(facs_path, 'r')
    data_facs = []
    for line in fi: # read rest of lines
        for x in line.split():
            if(int(float(x)) not in data_facs and int(float(x))!= 0):
                data_facs.append(int(float(x)))
    data_facs.sort()
    fi.close()
    
    landmarkChange = []
    landmark_neutral = training_images[count].landmarks['PTS'].lms.points
    landmark_perk = training_images[count + 1].landmarks['PTS'].lms.points
    for i in range(0,68):
        landmarkChange.append([landmark_perk[i][0] - landmark_neutral[i][0], landmark_perk[i][1] - landmark_neutral[i][1]])

    svm_training_data.append(ChangeVector(data_facs, landmarkChange, gt_emotion))   
    count = count + 2
    

facs = []
#create facs array
for data in svm_training_data:
    for facs_code in data.facs:
        if(int(facs_code) not in facs and int(facs_code)!= 0):
            facs.append(facs_code)
facs.sort()

#create model for each action unit in facs[]
models = []
au_models_score = []
for au in facs:
    x_training = []
    y_label = []
    #create label array
    for data in svm_training_data:
        if(au in data.facs):
            y_label.append(1)
        else:
            y_label.append(0)
        #create training data: 1x68 array, result of PCA process
        vector = []
        for tmp in data.landmarkChange:
            vector.append(tmp[0])
            vector.append(tmp[1])
        x_training.append(vector)
    clf = svm.LinearSVC()
    clf.fit(x_training, y_label)
    au_models_score.append(clf.score(x_training, y_label))
    models.append(clf)

#create testing data
svm_testing_data = []
testing_images = mio.import_images(path_to_svm_testing_database, verbose=True)
testing_images = testing_images.map(process)

count = 0;
while(count < len(testing_images)):
    file_path = str(testing_images[count].path).split("\\")
    facs_path = path_to_facs + file_path[6] + '/' + file_path[7]
    gt_emotion = file_path[7]
    facs_path = facs_path + '/' + os.listdir(facs_path)[0]
    fi = open(facs_path, 'r')
    data_facs = []
    for line in fi: # read rest of lines
        for x in line.split():
            if(int(float(x)) not in data_facs and int(float(x)) != 0):
                data_facs.append(int(float(x)))
    #print(array)
    fi.close()
    
    landmarkChange = []
    landmark_neutral = testing_images[count].landmarks['PTS'].lms.points
    landmark_perk = testing_images[count + 1].landmarks['PTS'].lms.points
    for i in range(0,68):
        landmarkChange.append([landmark_perk[i][0] - landmark_neutral[i][0], landmark_perk[i][1] - landmark_neutral[i][1]])
    
    svm_testing_data.append(ChangeVector(data_facs, landmarkChange, gt_emotion))   
    count = count + 2

# regresssion model
wrong_predict = 0
au_score = []
for data in svm_testing_data:
    print('#####')
    local_wrong_predict = 0
    local_accurate_predict = 0
    tmp = []
    predict = []
    
    for vector in data.landmarkChange:
        tmp.append(vector[0])
        tmp.append(vector[1])
        
    for model in models:
        if(model.predict([tmp]) >= 0.5):
            predict.append([facs[models.index(model)], model.predict([tmp])[0]])
            #print(facs[models.index(model)])
            if(facs[models.index(model)] not in data.facs):
                local_wrong_predict += 1
            else: 
                local_accurate_predict += 1
        else:
            if(facs[models.index(model)] in data.facs):
                local_wrong_predict += 1
    print(predict)
    print(local_accurate_predict)
    au_score.append(float(local_accurate_predict)/float(len(data.facs)))
    print("---")
    data.facs.sort()
    for gt_facs in data.facs:
        print([gt_facs, models[facs.index(gt_facs)].predict([tmp])[0]])
    wrong_predict += local_wrong_predict
    
# print(wrong_predict)
# print(sum(au_score)/float(len(au_score)))

#emotion models
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

result = []
for data in svm_testing_data:
    tmp = []
    facs_predict = []
    for vector in data.landmarkChange:
        tmp.append(vector[0])
        tmp.append(vector[1])
        
    for model in models:
        if(model.predict([tmp]) >= 0.5):
            facs_predict.append(facs[models.index(model)])
            #print(facs[models.index(model)])
    emotion_predict = []
    for emotion in emotions:
        emotion_predict.append([emotion.name, emotion.score(facs_predict)])
    result.append([emotion_predict, data.emotion])

log_path = '/Programing/GR/Code/Python/log/'
import time
import json
ts = int(time.time())
with open(log_path + ts + '.txt', "w+", 'w+') as outfile:
    json.dump(data, outfile)




