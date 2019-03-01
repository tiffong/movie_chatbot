import nltk

labels = ['anger', 'disgust', 'fear',
          'joy','sadness',
          'surprise']

keep_words = ['angry', 'sad', 'happy', 'mad', 'disgusting',
				'surprised', 'asshole', 'abandoned', 'afraid',
				'scared', 'screaming', 'yelling', 'yell']

f = open("deps/nrc_condensed_lexicon.txt", 'r')
condensed_file = open("deps/new_lexicon.txt", 'w')
lines = f.readlines()
lines_read = 0
for line in lines:
	l = line
	l = l.replace('\n', '')
	l = l.replace('\t', '%')
	l = l.replace(' ', '%')
	tokens = l.split('%')

	text = nltk.word_tokenize(line)
	tags = nltk.pos_tag(text)

	if int(tokens[2]) > 0 and tags[0][1] != 'NN' and \
		tags[0][1] != 'VB' and tags[0][1] != 'NNS':
		if tokens[0] in labels or tokens[0] in keep_words:
			condensed_file.write(line)
			continue
		if lines_read % 5 != 0:
			 condensed_file.write(line)
	lines_read += 1


condensed_file.close()
f.close()