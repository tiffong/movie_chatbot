#!/usr/bin/env python

# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
#
# Original Python code by Ignacio Cases (@cases)
#
# See: https://docs.python.org/3/library/cmd.html
######################################################################
import argparse
import cmd
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

# Uncomment me to see which REPL commands are being run!
# logger.setLevel(logging.DEBUG)

from chatbot import Chatbot

# Modular ASCII font from http://patorjk.com/software/taag/
HEADER = """Welcome to Stanford CS124's
 _______  _______         ___
|       ||   _   |       |   |
|    _  ||  |_|  | ____  |   |___
|   |_| ||       ||____| |    _  |
|    ___||       |       |   | | |
|   |    |   _   |       |   |_| |
|___|    |__| |__|       |_______|
 _______  __   __  _______  _______  _______  _______  _______  __
|       ||  | |  ||   _   ||       ||  _    ||       ||       ||  |
|       ||  |_|  ||  |_|  ||_     _|| |_|   ||   _   ||_     _||  |
|       ||       ||       |  |   |  |       ||  | |  |  |   |  |  |
|      _||       ||       |  |   |  |  _   | |  |_|  |  |   |  |__|
|     |_ |   _   ||   _   |  |   |  | |_|   ||       |  |   |   __
|_______||__| |__||__| |__|  |___|  |_______||_______|  |___|  |__|
"""

pa6_description = 'Simple Read-Eval-Print-Loop that handles the input/output part of the conversational agent'

class REPL(cmd.Cmd):
    """Simple REPL to handle the conversation with the chatbot."""
    prompt = '> '
    doc_header = ''
    misc_header = ''
    undoc_header = ''
    ruler = '-'

    def __init__(self, creative=False):
      super().__init__()

      self.chatbot = Chatbot(creative=creative)
      self.name = self.chatbot.name
      self.bot_prompt = '\001\033[96m\002%s> \001\033[0m\002' % self.name

      self.greeting = self.chatbot.greeting()
      self.intro = self.chatbot.intro() + '\n' + self.bot_prompt + self.greeting
      self.debug = False

    def cmdloop(self, intro=None):
      logger.debug('cmdloop(%s)', intro)
      return super().cmdloop(intro)

    def preloop(self):
      logger.debug('preloop(); Chatbot %s created and loaded', self.chatbot)
      print(HEADER)
      self.debug_chatbot = False

    def postloop(self):
      goodbye = self.chatbot.goodbye()
      print(self.bot_says(goodbye))

    def onecmd(self, s):
      logger.debug('onecmd(%s)', s)
      if s:
        return super().onecmd(s)
      else:
        return False  # Continue processing special commands.

    def emptyline(self):
      logger.debug('emptyline()')
      return super().emptyline()

    def default(self, line):
      logger.debug('default(%s)', line)
      # Stop processing commands if the user enters :quit
      if line == ":quit":
        return True
      else:
        response = self.chatbot.process(line)
        print(self.bot_says(response))

    def precmd(self, line):
      logger.debug('precmd(%s)', line)
      return super().precmd(line)

    def postcmd(self, stop, line):
      logger.debug('postcmd(%s, %s)', stop, line)

      if line == ':quit':
        return True
      elif line.lower() == 'who are you?':
        self.do_SECRET(line)
      elif ':debug on' in line.lower():
        print('enabling debug...')
        self.debug_chatbot = True
      elif ':debug off' in line.lower():
        print('disabling debug...')
        self.debug_chatbot = False

      # Debugging the chatbot
      if self.debug_chatbot:
        print(self.chatbot.debug(line))

      return super().postcmd(stop, line)

    def bot_says(self, response):
      return self.bot_prompt + response

    def do_PROMPT(self, line):
      """Set the interactive prompt."""
      self.prompt = line + ': '

    def do_SECRET(self, line):
      """Could it be... a secret message?"""
      story = """A long time ago, in a remote land, a young developer named Alberto Caso managed to build an ingenious and mysterious chatbot... Now it's your turn!"""
      print(story)


def process_command_line():
  parser = argparse.ArgumentParser(description=pa6_description)
  parser.add_argument('--creative', dest='creative', action='store_true', default=False, help='Enables creative mode')
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = process_command_line()
  repl = REPL(creative=args.creative)
  repl.cmdloop()
