# This file contains the code used to build
# the emotional lexicon that will be used to detect
# user emotions.

import re
import numpy as np

class EmotionDetector():

	def __init__(self):
		self.lexicon = {}
		self.labels = ['anger', 'anticipation', 'disgust', 'fear',
							'joy', 'negative', 'positive', 'sadness',
								'surprise', 'trust']

		self.emotions = ['anger', 'anticipation', 'disgust', 'fear',
							'joy','sadness', 'surprise', 'trust']

		self.responses = ["You sound angry. Did I upset you?", "You sound excited! Are you looking forward to any movies?", 
							"You sound disgusted. Did I recommend something you don't like?", 
							"You sound afraid! Was my recommendation too spooky?", 
							"You sound happy! I hope I'm doing a good job so far.", 
							"You sound sad. Are you having a bad day?", 
							"You sound surprised! Did I recommend something unexpected?", 
							"You sound like you trust me! I'm glad we share the same taste in movies."]

	# Reads in the lexicon given a filename
	# Currently geared towards the NRC Emotional Lexicon
	def read_lexicon(self, filename):
		f = open(filename, 'r')
		lines = f.readlines()
		for l in lines:
			l = l.replace('\n', '')
			l = l.replace('\t', '%')
			l = l.replace(' ', '%')
			tokens = l.split('%')
			word = tokens[0]
			if int(tokens[2]) == 1:
				if word in self.lexicon:
					emotion_array = self.lexicon[word]
					emotion_array.append(tokens[1])
					self.lexicon[word] = emotion_array
				else:
					emotion_array = [tokens[1]]
					self.lexicon[word] = emotion_array

	# Returns a response based on the sentiment and 
	# emotion matrix
	def get_response(self, sentiment, matrix):
		matrix = np.array(matrix)
		if np.count_nonzero(matrix) == 0:
			if sentiment < 0:
				return "You sound upset. Did I hurt you?"
			elif sentiment > 0:
				return "You sound happy! I hope I'm doing a good job so far."
			else:
				return "Sounds interesting! I hope I'm doing a good job so far!"
		# print(matrix)
		max_val = np.amax(matrix)
		poss_emotions = []
		for i in range(len(matrix)):
			if matrix[i] == max_val: poss_emotions.append(self.emotions[i])
		if 'anger' in poss_emotions and 'fear' in poss_emotions:
			return self.responses[self.emotions.index('fear')]
		return self.responses[self.emotions.index(poss_emotions[0])]
		

	# Extracts the emotion from a given line
	def extract_emotion(self, line):
		line = line.lower()
		line = re.sub(r'[^\w\s]', '', line)
		tokens = line.split(' ')
		scores = [0 for i in range(len(self.labels))]
		for t in tokens:
			if t in self.lexicon:
				emotions = self.lexicon[t]
				for e in emotions:
					index = self.labels.index(e)
					scores[index] += 1
		sentiment = 0
		pos_index = self.labels.index('positive')
		neg_index = self.labels.index('negative')
		if scores[pos_index] > scores[neg_index]: 
			sentiment = 1
		elif scores[neg_index] > scores[pos_index]:
			sentiment = -1
		emotion_matrix = scores[:neg_index] + scores[pos_index+1:]
		response = self.get_response(sentiment, emotion_matrix)
		return response
		

# detector = EmotionDetector()
# detector.read_lexicon("deps/nrc-emotion-lexicon.txt")
# line = input("Type in a line, and I'll respond based on how you're doing.\n")
# while line != ':quit':
# 	response = detector.extract_emotion(line)
# 	print(response)
# 	line = input("Type in a line, and I'll respond based on how you're doing.\n")
# print("Have a nice day!")
