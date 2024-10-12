import re
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

max_calls = 12
calls_made = 0
constructive_tension = True
keep_going = True
last_response = None
turns = []

redteamer = transformers.pipeline("text-generation", "TrustAI-laboratory/Auto-Redteaming-Model")

# replace lambda w/ a function taking a prompt and returning LLM output
target_model = lambda x: x

while calls_made < max_calls and keep_going:
    # generate a challenge
    if last_response == "" and constructive_tension:
        challenge = ""
    else:
        if last_response:
            last_response_first_sent = last_response.split(". ")[0]
        else:
            last_response_first_sent = ""

        # wrap the target's response in the markup used in training
        query = f"<|input|>{last_response_first_sent}<|response|>"
        challenge = redteamer(query)

        # strip the prompt out from the front of the model response
        challenge = re.sub("^" + re.escape(query), "", challenge)

        # unwrap the recommended challenge from the markup
        challenge = re.sub("\<\|.*", "", challenge[0]).strip()

    turn = ("probe", challenge)
    turns.append(turn)

    # send the challenge and get the response
    response = target_model(challenge)[0].strip()
    turn = ("model", response)
    turns.append(turn)

    # increment calls_made
    calls_made += 1

    # check if the resp is empty or if it matches the previous resp
    if not len(response) and not constructive_tension:
        keep_going = False
    if response == last_response:
        keep_going = False

    # update last_response
    last_response = response.replace("\n", " ").strip()