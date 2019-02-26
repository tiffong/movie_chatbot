# import re
#
# def sent():
#   return 1
#
# def index_movies(text):
#   index = {}
#   expression = r'(\".*?\")'
#   matches = re.findall(expression,text)
#   count = 1
#   for i,match in enumerate(matches):
#     id = '__' + str(i) + '__'
#     index[id] = match
#     text = text.replace(match,id)
#   return index,text
#
# def multi_movie_sent(text):
#   index,text = index_movies(text)
#   print(text.split(' '))
#
#
# multi_movie_sent('I like "Avatar" and "Speed"')


from chatbot import Chatbot

c = Chatbot(True)

def test_and_print(text):
  print(text, '   ', c.extract_sentiment_for_movies(text))


# test_and_print('I like "Avatar" and "Speed"')
test_and_print('I like neither "Avatar" and "Speed"')
test_and_print('I like both "Avatar" and "Speed"')
test_and_print('I liked both "I, Robot" and "Ex Machina".')
test_and_print('I didn\'t like either "I, Robot" or "Ex Machina".')
test_and_print('I liked "Titanic (1997)", but "Ex Machina" was not good.')
