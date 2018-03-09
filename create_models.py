#svm
import menpo.io as mio
import os
from sklearn import svm
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import Models
from Models import ChangeVector
import time
import json

path_to_svm_training_database = '/Programing/GR/Code/CK+/aam-images/**/**/**/*'

# class ChangeVector:
#     def __init__(self, facs = [], landmarkChange = [], emotion = 0):
#         self.landmarkChange = landmarkChange
#         self.facs = facs
#         self.emotion = emotion

def process_input_image(image, crop_proportion=0.2, max_diagonal=400):
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
training_images = training_images.map(process_input_image)

path_to_facs = '/Programing/GR/Code/CK+/FACS/'
path_to_emotions = '/Programing/GR/Code/CK+/Emotion/'

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
path_to_svm_testing_database = "/Programing/GR/Code/CK+/test-aam-images/**/**/**/*"
testing_images = mio.import_images(path_to_svm_testing_database, verbose=True)
testing_images = testing_images.map(process_input_image)

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

    
#evaluate trained model with test data and get score
#print('#######')
#print('Sccore: ')
for au in facs:
    x_training = []
    y_label = []
    #create label array
    for data in svm_testing_data:
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
    #print(models[facs.index(au)].score(x_training, y_label))

#regresssion model
wrong_predict = 0
possitive_au_predict_score = []
for data in svm_testing_data:
    #print('#####')
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
    #print(predict)
    #print(local_accurate_predict)
    possitive_au_predict_score.append(float(local_accurate_predict)/float(len(data.facs)))
    #print("---")
    data.facs.sort()
    #for gt_facs in data.facs:
        #print([gt_facs, models[facs.index(gt_facs)].predict([tmp])[0]])
    wrong_predict += local_wrong_predict
    
#print(wrong_predict)
#print(sum(au_score)/float(len(au_score)))

#emotion prediction model
emotions = Models.emotions
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
    result.append([emotion_predict, data.emotion_label])

#save model
import pickle
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
save_object(models, '/Programing/GR/Code/Python/models/au_models.pkl')
save_object(facs, '/Programing/GR/Code/Python/models/facs.pkl')


#create log file
log_path = '/Programing/GR/Code/Python/log/'

ts = int(time.time())
output = {'n_train': len(training_images), 'n_test': len(testing_images), 
    'au_models_score' : au_models_score, 'possitive_au_predict_score': sum(possitive_au_predict_score)/float(len(possitive_au_predict_score)),'facs': facs, 'au_accuracy': sum(au_models_score)/float(len(au_models_score)),  }
    
with open(log_path + 'system_log' + str(ts) + '.txt', "w+") as outfile:
    json.dump(output, outfile,sort_keys=True, indent=4, separators=(',', ': '))