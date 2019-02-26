from chatbot import Chatbot

c = Chatbot(True)

def test_and_print(text):
  print(text, c.extract_sentiment(text))

test_and_print('I hate "The Matrix"')
test_and_print('I dislike "The Matrix"')
test_and_print('I saw "The Matrix"')


test_and_print('I like "The Matrix" a lot')
test_and_print('I love "The Matrix"')

test_and_print('I really hate "The Matrix"')
test_and_print('I really dislike "The Matrix"')
test_and_print('I really saw "The Matrix"')
test_and_print('I really like "The Matrix"')
test_and_print('I really love "The Matrix"')

test_and_print('I loved "Zootopia"')
test_and_print('"Zootopia" was terrible.')
test_and_print('I really reeally liked "Zootopia"!!!')