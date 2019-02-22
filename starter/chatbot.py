# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import movielens

import numpy as np
import re
from PorterStemmer import PorterStemmer
from heapq import nlargest
import random


class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
      # The chatbot's default name is `moviebot`. Give your chatbot a new name.
      self.name = 'moviebot'

      self.creative = creative

      # This matrix has the following shape: num_movies x num_users
      # The values stored in each row i and column j is the rating for
      # movie i by user j
      self.titles, ratings = movielens.ratings()
      self.articles = ['A', 'An', 'The']

      sentiment = movielens.sentiment()
      self.porterStemmer = PorterStemmer()
      self.sentiment = {}
      for word in sentiment:
          self.sentiment[self.porterStemmer.stem(word)] = sentiment[word]

      self.negation_words = ['no','not','neither','hardly','barely','doesnt','isnt','wasnt','shouldnt','wouldnt',
                             'couldnt','wont',  'cant','dont','didnt','nor','ni','werent']

      #############################################################################
      # TODO: Binarize the movie ratings matrix.                                  #
      #############################################################################

      # Binarize the movie ratings before storing the binarized matrix.
      self.ratings = self.binarize(ratings)

      self.positive_responses = ["I am glad you liked \"{}\".",
                                 "So you enjoyed the film \"{}\".",
                                 "So you enjoyed the movie \"{}\". Good to know.",
                                 "I understand that \"{}\" was an enjoyable movie for you.",
                                 "You liked the movie \"{}\".",
                                 "It's nice to hear that you enjoyed \"{}\".",
                                 "\"{}\" was a good film for you. ",
                                 "Great! I understand that you thought \"{}\" was good."]  # 5 of each
      self.negative_responses = ["Sorry you didn't enjoy \"{}\".",
                                 "So you did not like the film \"{}\".",
                                 "I see that \"{}\" was not a good movie for you.",
                                 "You did not think the movie \"{}\" was good.",
                                 "It's sad to hear that you did not enjoy \"{}\".",
                                 "\"{}\" was a bad film for you.",
                                 "Ok. I understand that you disliked \"{}\"."]  # 5 of each
      self.neutral_responses = ["Sorry. I did not get that.",
                                "I did not understand.",
                                "I could not make out what you meant by that."]
      self.asking_for_more_responses = ["Tell me your opinion on another film.",
                                        "What is a another film you liked or disliked?",
                                        "Can you give me another movie?",
                                        "I need another one of your film preferences.",
                                        "Can you describe to me another of your movie reactions?"]
      self.announcing_recommendation_responses = ["I have enough information to give you a recommendation.",
                                                  "That's enough movies for me to recommend to you a new one.",
                                                  "I can now recommend a new movie for you."]
      self.recommendation_templates = ["I recommend that you watch \"{}\".",
                                       "I suggest that you check out the film \"{}\".",
                                       "I believe that you would enjoy \"{}\".",
                                       "\"{}\" would be a good film for you to watch."]

      self.user_sentiment = np.zeros(len(self.titles))

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

    #############################################################################
    # 1. WARM UP REPL                                                           #
    #############################################################################

    def greeting(self):
      """Return a message that the chatbot uses to greet the user."""
      #############################################################################
      # TODO: Write a short greeting message                                      #
      #############################################################################

      greeting_message = "Hi! I'm MovieBot! Ready to watch some new movies? First I'll find out your taste in movies. Tell about a movie you've seen and how you liked it."

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return greeting_message

    def goodbye(self):
      """Return a message that the chatbot uses to bid farewell to the user."""
      #############################################################################
      # TODO: Write a short farewell message                                      #
      #############################################################################

      goodbye_message = "Bye!"

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return goodbye_message


    ###############################################################################
    # 2. Modules 2 and 3: extraction and transformation                           #
    ###############################################################################

    def process(self, line):
      """Process a line of input from the REPL and generate a response.

      This is the method that is called by the REPL loop directly with user input.

      You should delegate most of the work of processing the user's input to
      the helper functions you write later in this class.

      Takes the input string from the REPL and call delegated functions that
        1) extract the relevant information, and
        2) transform the information into a response to the user.

      Example:
        resp = chatbot.process('I loved "The Notebok" so much!!')
        print(resp) // prints 'So you loved "The Notebook", huh?'

      :param line: a user-supplied line of text
      :returns: a string containing the chatbot's response to the user input
      """
      #############################################################################
      # TODO: Implement the extraction and transformation in this method,         #
      # possibly calling other functions. Although modular code is not graded,    #
      # it is highly recommended.                                                 #
      #############################################################################
      

      if self.creative:
        response = "I processed {} in creative mode!!".format(line)
      
      else:
        
        #the movies that the user inputted
        movies = self.extract_titles(format(line))

        sentence_sentiment = self.extract_sentiment(format(line))

        if len(movies) > 1:
          response = "Please tell me about one movie at a time. Go ahead."
        elif len(movies) == 1:
          movie_indices = self.find_movies_by_title(movies[0])
          if len(movie_indices) == 0:
            response = '"' + movies[0] + '" is not a valid movie. Please tell me about a movie that exists.'
          elif sentence_sentiment == 0:
            print("here")
            response = "here" + random.choice(self.neutral_responses) + '  Please give me information about a movie.'
          else:
            if sentence_sentiment == -1:
              self.user_sentiment[movie_indices[0]] = -1
              response = random.choice(self.negative_responses)
              response = response.replace('{}', movies[0])
            else:
              self.user_sentiment[movie_indices[0]] = 1
              response = random.choice(self.positive_responses)
              response = response.replace('{}', movies[0])
            if np.count_nonzero(self.user_sentiment) < 5:
              response += '\n' + random.choice(self.asking_for_more_responses)
            else:
              recommendation = self.recommend(self.user_sentiment, self.ratings, k=1, creative=False)  # TODO : recommend a movie
              recommended_movie_index = recommendation[0]
              movie_title = self.titles[recommended_movie_index][0]
              movie_title = movie_title.split(' (')[0]
              response = random.choice(self.announcing_recommendation_responses)
              response += '\n' + random.choice(self.recommendation_templates).replace('{}', movie_title)
        else:
            response = str(len(movies)) + random.choice(self.neutral_responses) + '  Please give me information about a movie.'

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return response

    def extract_titles(self, text):
      """Extract potential movie titles from a line of text.

      Given an input text, this method should return a list of movie titles
      that are potentially in the text.

      - If there are no movie titles in the text, return an empty list.
      - If there is exactly one movie title in the text, return a list
      containing just that one movie title.
      - If there are multiple movie titles in the text, return a list
      of all movie titles you've extracted from the text.

      Example:
        potential_titles = chatbot.extract_titles('I liked "The Notebook" a lot.')
        print(potential_titles) // prints ["The Notebook"]

      :param text: a user-supplied line of text that may contain movie titles
      :returns: list of movie titles that are potentially in the text
      """
      titles = re.findall('"(.+?)"', text)
      return titles

    def find_movies_by_title(self, title):
      """ Given a movie title, return a list of indices of matching movies.

      - If no movies are found that match the given title, return an empty list.
      - If multiple movies are found that match the given title, return a list
      containing all of the indices of these matching movies.
      - If exactly one movie is found that matches the given title, return a list
      that contains the index of that matching movie.

      Example:
        ids = chatbot.find_movies_by_title('Titanic')
        print(ids) // prints [1359, 1953]

      :param title: a string containing a movie title
      :returns: a list of indices of matching movies
      """
      indices = []

      title_split = title.split(' ')
      if re.fullmatch('\([0-9]{4}\)', title_split[len(title_split) - 1]):
        if title_split[0] in self.articles:
          title = ''
          for i in range(1, len(title_split) - 1):
            title += title_split[i]
            if i < len(title_split) - 2: title += ' '
          title +=', ' + title_split[0]
          title += ' ' + title_split[len(title_split) - 1]
        for i in range(len(self.titles)):
          curr_title = self.titles[i][0]
          if title == curr_title:
            indices.append(i)
      else: 
        if title_split[0] in self.articles:
          title = ''
          for i in range(1, len(title_split)):
            title += title_split[i]
            if i < len(title_split) - 1: title += ' '
          title += ', ' + title_split[0]
        for i in range(len(self.titles)):
          curr_title = self.titles[i][0]
          movie_name = curr_title.split(' (')
          if title == movie_name[0]:
            indices.append(i)
      return indices


    def extract_sentiment(self, text):
      """Extract a sentiment rating from a line of text.

      You should return -1 if the sentiment of the text is negative, 0 if the
      sentiment of the text is neutral (no sentiment detected), or +1 if the
      sentiment of the text is positive.

      As an optional creative extension, return -2 if the sentiment of the text
      is super negative and +2 if the sentiment of the text is super positive.

      Example:
        sentiment = chatbot.extract_sentiment('I liked "The Titanic"')
        print(sentiment) // prints 1

      :param text: a user-supplied line of text
      :returns: a numerical value for the sentiment of the text
      """

      text = re.sub(r'[^\w\s]', '', text)  # removing punctuation
      text = text.lower()  # lowercase
      words = text.split(' ') # getting individual words

      score = 0
      negate = False
      for word in words:
          word = self.porterStemmer.stem(word)
          if word in self.negation_words:
              negate = True
          elif word in self.sentiment:
              sentiment = self.sentiment[word]
              if sentiment == 'pos':
                  if negate:
                      score -= 1
                  else:
                      score += 1
              else:
                  if negate:
                      score += 1
                  else:
                      score -= 1
              negate = False
      if score > 0:
          return 1
      elif score < 0:
          return -1
      else:
          return 0

    def extract_sentiment_for_movies(self, text):
      """Creative Feature: Extracts the sentiments from a line of text
      that may contain multiple movies. Note that the sentiments toward
      the movies may be different.

      You should use the same sentiment values as extract_sentiment, described above.
      Hint: feel free to call previously defined functions to implement this.

      Example:
        sentiments = chatbot.extract_sentiment_for_text('I liked both "Titanic (1997)" and "Ex Machina".')
        print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

      :param text: a user-supplied line of text
      :returns: a list of tuples, where the first item in the tuple is a movie title,
        and the second is the sentiment in the text toward that movie
      """
      pass

    def find_movies_closest_to_title(self, title, max_distance=3):
      """Creative Feature: Given a potentially misspelled movie title,
      return a list of the movies in the dataset whose titles have the least edit distance
      from the provided title, and with edit distance at most max_distance.

      - If no movies have titles within max_distance of the provided title, return an empty list.
      - Otherwise, if there's a movie closer in edit distance to the given title 
        than all other movies, return a 1-element list containing its index.
      - If there is a tie for closest movie, return a list with the indices of all movies
        tying for minimum edit distance to the given movie.

      Example:
        chatbot.find_movies_closest_to_title("Sleeping Beaty") # should return [1656]

      :param title: a potentially misspelled title
      :param max_distance: the maximum edit distance to search for
      :returns: a list of movie indices with titles closest to the given title and within edit distance max_distance
      """

      pass

    def disambiguate(self, clarification, candidates):
      """Creative Feature: Given a list of movies that the user could be talking about
      (represented as indices), and a string given by the user as clarification 
      (eg. in response to your bot saying "Which movie did you mean: Titanic (1953) 
      or Titanic (1997)?"), use the clarification to narrow down the list and return 
      a smaller list of candidates (hopefully just 1!)

      - If the clarification uniquely identifies one of the movies, this should return a 1-element
      list with the index of that movie.
      - If it's unclear which movie the user means by the clarification, it should return a list
      with the indices it could be referring to (to continue the disambiguation dialogue).

      Example:
        chatbot.disambiguate("1997", [1359, 2716]) should return [1359]
      
      :param clarification: user input intended to disambiguate between the given movies
      :param candidates: a list of movie indices
      :returns: a list of indices corresponding to the movies identified by the clarification
      """
      pass


    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    def binarize(self, ratings, threshold=2.5):
      """Return a binarized version of the given matrix.

      To binarize a matrix, replace all entries above the threshold with 1.
      and replace all entries at or below the threshold with a -1.

      Entries whose values are 0 represent null values and should remain at 0.

      :param x: a (num_movies x num_users) matrix of user ratings, from 0.5 to 5.0
      :param threshold: Numerical rating above which ratings are considered positive

      :returns: a binarized version of the movie-rating matrix
      """
      #############################################################################
      # TODO: Binarize the supplied ratings matrix.                               #
      #############################################################################

      # The starter code returns a new matrix shaped like ratings but full of zeros.
      binarized_ratings = np.zeros_like(ratings)

      for i in range(len(ratings)): #row
        for j in range(len(ratings[0])): #column
          
          value = 0
          rating = ratings[i][j]
          
          if(rating == 0):
            value = 0
          elif (rating > threshold):
            value = 1
          elif(rating <= threshold):  
            value = -1

          binarized_ratings[i][j] = value

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return binarized_ratings


    def similarity(self, u, v):
      """Calculate the cosine similarity between two vectors.

      You may assume that the two arguments have the same shape.

      :param u: one vector, as a 1D numpy array
      :param v: another vector, as a 1D numpy array

      :returns: the cosine similarity between the two vectors
      """
      if(np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0):
        return 0
      else:
        similarity = np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))#

      #############################################################################
      # TODO: Compute cosine similarity between the two vectors.
      #############################################################################
      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return similarity


    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
      """Generate a list of indices of movies to recommend using collaborative filtering.

      You should return a collection of `k` indices of movies recommendations.

      As a precondition, user_ratings and ratings_matrix are both binarized.

      Remember to exclude movies the user has already rated!

      :param user_ratings: a binarized 1D numpy array of the user's movie ratings
      :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
        `ratings_matrix[i, j]` is the rating for movie i by user j
      :param k: the number of recommendations to generate
      :param creative: whether the chatbot is in creative mode

      :returns: a list of k movie indices corresponding to movies in ratings_matrix,
        in descending order of recommendation
      """

      #######################################################################################
      # TODO: Implement a recommendation function that takes a vector user_ratings          #
      # and matrix ratings_matrix and outputs a list of movies recommended by the chatbot.  #
      #                                                                                     #
      # For starter mode, you should use item-item collaborative filtering                  #
      # with cosine similarity, no mean-centering, and no normalization of scores.          #
      #######################################################################################

      # Populate this list with k movie indices to recommend to the user.
      
      #for each movie i in the dataset
      num_movies = np.size(ratings_matrix,0)
      #print(num_movies)

      #get index of movies that ARE rated by user
      watched = set()
      rating_index = []
      for i in range(len(user_ratings)):
        if (user_ratings[i] == 1 or user_ratings[i] == -1):
          rating_index.append(i)
          watched.add(i)
      rating_index = sorted(rating_index)

      #rxi
      recommendation_ratings = np.zeros(num_movies)
      #user_rxi_scores = np.zeros()
      rxi = 0

      #for each movie in the dataset
      for i in range(num_movies):
        movie_i = ratings_matrix[i] #ratings of all users for this movie 
        #print(movie_i)

        #for each rating the user gave
        for j in range(len(rating_index)):
          user_movie_index = rating_index[j] #getting index of movies that were rated by user in user_ratings

          #user rated movie
          movie_j = ratings_matrix[user_movie_index]

          sim = self.similarity(movie_i, movie_j) #similarity between random movie i and user rated movie j
          user_rating = user_ratings[user_movie_index]

          rxi += sim*user_rating
        #print(rxi)
        #print(i)

        if (i in watched):
          rxi = 0

        recommendation_ratings[i] = rxi #add the recommendation for that specific movie
        rxi = 0

      #for movie in range(len(recommendation_ratings)):
      recommendations = np.argsort(recommendation_ratings)[-k:]
      reversed_arr = recommendations[::-1]
      #print(type(recommendations))
      #recommendations = np.flip(recommendations, k)

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return reversed_arr.tolist()

    #############################################################################
    # 4. Debug info                                                             #
    #############################################################################

    def debug(self, line):
      """Return debug information as a string for the line string from the REPL"""
      # Pass the debug information that you may think is important for your
      # evaluators
      debug_info = 'debug info'
      return debug_info


    #############################################################################
    # 5. Write a description for your chatbot here!                             #
    #############################################################################
    def intro(self):
      """Return a string to use as your chatbot's description for the user.

      Consider adding to this description any information about what your chatbot
      can do and how the user can interact with it.
      """
      return """
      Your task is to implement the chatbot as detailed in the PA6 instructions.
      Remember: in the starter mode, movie names will come in quotation marks and
      expressions of sentiment will be simple!
      Write here the description for your own chatbot!
      """


if __name__ == '__main__':
  print('To run your chatbot in an interactive loop from the command line, run:')
  print('    python3 repl.py')

# # Test zone
# chatbot = Chatbot()
# # # movies = chatbot.extract_titles('I liked "The Titanic" and "Hello World" "hi world"')
# # # print(movies)
# indices = chatbot.find_movies_by_title('The American President')
# print(indices)
