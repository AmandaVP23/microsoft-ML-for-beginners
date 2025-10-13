from textblob import TextBlob
from textblob.np_extractors import ConllExtractor

# the quick red fox jumped over the lazy brown dog

extractor = ConllExtractor()

print("Hello, I am Marvin, the simple robot.")
print("You can end this conversation at any time by typing 'bye'")
print("After typing each answer, press 'enter'")
print("How are you today?")


while True:
    user_input = input("> ")
    if user_input == "bye":
        break
    else:
        user_input_blob = TextBlob(user_input, np_extractor=extractor) # non default extractor specified
        np = user_input_blob.noun_phrases
        response = ""
        if user_input_blob.polarity <= -0.5:
            response = "Oh dear, that sounds bad. "
        elif user_input_blob.polarity <= 0:
            response = "Hmm, that's not great. "
        elif user_input_blob.polarity <= 0.5:
            response = "Well, that sounds positive. "
        elif user_input_blob.polarity <= 1:
            response = "Wow, that sounds great. "

        if len(np) != 0:
            # there was at least one noun phrase detected, ask about that and pluralize it, eg, cat => cats
            response = response + "Can you tell me more about " + np[0].pluralize() + "?"
        else:
            response = response + "Can you tell me more?"
        
        print(response)

print("It was nice talking to you, goodbye!")


# when you need a noun phrase extractor:
# user_input = input("> ")
# user_input_blob = TextBlob(user_input, np_extractor=extractor) # non default extractor specified
# np = user_input_blob.noun_phrases
# print(np)
