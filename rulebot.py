import re
import random
class RuleBot:
  ###Potential Negative Responses
  negative_responses=("no","nope","nah","naw","not a chance","sorry")
  ###Exit Conversation keywords
  exit_commands=("quit","pause","exit","goodbye","bye","later")
  #Random Starter Questions
  random_questions=(
      "Why are you here?",
      "Are there many humans like you?",
      "What do you consume for sustenance?",
      "Is there intelligent life on this planet?",
      "Does earth have a leader?",
      "What planets have you visited?",
      "What technology do you have on this planet?"
  )

  def __init__(self):
    self.alienbabble={'describe_planet_intent':r'.*\s*your planet.*',
                      'answer_why_intent':r'why\sare.*',
                      'about vit':r'.*\s*vit'
    }
  def greet(self):
    self.name=input("What is your name?\n")
    will_help=input(
        f"Hi {self.name}, I am Rule-Bot. Will you help me learn about your planet?\n")
    if will_help in self.negative_responses:
      print("Ok, have a nice earth day!")
      return
    self.chat()

  def make_exit(self,reply):
    for command in self.exit_commands:
      if reply==command:
        print("Okay, have a nice earth day!")
        return True
  
  def chat(self):
    reply=input(random.choice(self.random_questions)).lower()
    while not self.make_exit(reply):
      reply=input(self.match_reply(reply))
  
  def match_reply(self, reply):
    for key, value in self.alienbabble.items():
      intent=key
      regex_pattern=value
      found_match=re.match(regex_pattern,reply)
      if found_match and intent=='describe_planet_intent':
        return self.describe_planet_intent()
      elif found_match and intent=='answer_why_intent':
        return self.answer_why_intent()
      elif found_match and intent=='about vit':
        return self.about_vit()
    if not found_match:
      return self.no_match_intent()
    
  def describe_planet_intent(self):
    responses=("My planet is a utopia of diverse organisms and species.\n",
               "I am from Opidipus, the capital of the Wayward Galaxies.\n ")
    return random.choice(responses)

  def answer_why_intent(self):
    responses=("I come in peace\n","I am here to collect data on your planet and it's inhabitants\n","I heard the coffee is good\n")
    return random.choiceas(responses)

  def about_vit(self):
    responses=("VIT was established with the aim of providing quality higher education on par with international standards\n","It persistently seeks and adopts innovative methods to improve the quality of higher education on a consistent basis\n","The campus has a cosmopolitan atmosphere with students from all corners of the globe\n", "Experienced and learned teachers are strongly encouraged to nurture the students\n","The global standards set at VIT in the field of teaching and research spur us on in our relentless pursuit of excellence\n")
    return random.choice(responses)

  def no_match_intent(self):
    responses=(
        "Please tell me more.\n","Tell me more!\n","Why do you say that?\n","I see. Can you ellaborate?\n",
        "Interesting, can you tell me more?\n","I see. How do you think?\n","Why?\n",
        "How do you think I feel when you say that?\n")
    return random.choice(responses)
    

AlienBot= RuleBot()
AlienBot.greet()
