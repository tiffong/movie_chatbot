from chatbot import Chatbot

c = Chatbot(True)

def test_and_print(text):
  print(text, '   ', c.extract_sentiment_for_movies(text))


test_and_print('I like "Avatar" and "Speed" a lot')
test_and_print('I like neither "Avatar" and "Speed"')
test_and_print('I like both "Avatar" and "Speed"')
test_and_print('I liked both "I, Robot" and "Ex Machina".')
test_and_print('I didn\'t like either "I, Robot" or "Ex Machina".')
test_and_print('I liked "Titanic (1997)", but "Ex Machina" was not good.')
test_and_print('I liked "Titanic (1997)", but not "Ex Machina"')
test_and_print('I liked "Titanic (1997)" although "Ex Machina" is terrible')
test_and_print('I liked "movie1", "movie2","movie3","movie4" and "movie5" but "movie6" is terrible. ')  #Todo: more than one sentence at a time???
