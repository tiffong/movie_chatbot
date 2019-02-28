# This file contains the code used to build
# the emotional lexicon that will be used to detect
# user emotions.

import re

class EmotionDetector():

	def __init__(self):
		self.lexicon = {}
		self.emotions = ['anger', 'anticipation', 'disgust', 'fear',
							'joy', 'negative', 'positive', 'sadness',
								'surprise', 'trust']

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

	def get_response(self, rating, matrix):
		pass

	# Extracts the emotion from a given line
	def extract_emotion(self, line):
		line = re.sub(r'[^\w\s]', '', line)
		tokens = line.split(' ')
		scores = [0 for i in range(len(self.emotions))]
		for t in tokens:
			if t in self.lexicon:
				emotions = self.lexicon[t]
				for e in emotions:
					index = self.emotions.index(e)
					scores[index] += 1
		

detector = EmotionDetector()
detector.read_lexicon("deps/nrc-emotion-lexicon.txt")
detector.extract_emotion('I am so upset and angry at you!!')
