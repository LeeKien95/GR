import Models

def getEmotionPredict(landmarkChange):
	models = Models.getAuModels()
	facs = Models.getFacsList()
	facs_predict = []
		
	for model in models:
		if(model.predict([landmarkChange]) >= 0.5):
			facs_predict.append(facs[models.index(model)])
			#print(facs[models.index(model)])

	emotions = Models.emotions
	emotion_predict = {}
	for emotion in emotions:
		emotion_predict[emotion.name] = emotion.score(facs_predict)

	return emotion_predict