f = open("deps/nrc-emotion-lexicon.txt", 'r')
condensed_file = open("deps/nrc_condensed_lexicon.txt", 'w')
lines = f.readlines()
for line in lines:
	l = line
	l = l.replace('\n', '')
	l = l.replace('\t', '%')
	l = l.replace(' ', '%')
	tokens = l.split('%')
	if int(tokens[2]) > 0:
		condensed_file.write(line)
condensed_file.close()
f.close()