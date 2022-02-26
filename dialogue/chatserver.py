import re

from flask import render_template
from flask import Flask, request
import uuid
import json

app = Flask(__name__)
#from flask_cors import CORS
#CORS(app, origins='*')

import getpass

import sys
sys.path = sys.path + ['../']

from src.common.authentication import Authentication
from src.common.request import Request, RequestResult
from src.proxy.accounts import Account
from proxy.remote_service import RemoteService

# An example of how to use the request API.
api_key = getpass.getpass(prompt="Enter a valid API key: ")
auth = Authentication(api_key=api_key)
service = RemoteService("https://crfm-models.stanford.edu")

few_shot_context = \
"""BEGIN DIALOGUE
Jen: I remember going to see the fireworks with my best friend. It was the first time we ever spent time alone together. Although there was a lot of people, we felt like the only people in the world.
Bob: Was this a friend you were in love with, or just a best friend?
Jen: This was a best friend. I miss her.
Bob: Where has she gone?
Jen: We no longer talk.
Bob: Oh was this something that happened because of an argument?
BEGIN DIALOGUE
Jen: it feels like hitting to blank wall when i see the darkness
Bob: Oh ya? I don't really see how
Jen: dont you feel so.. its a wonder
Bob: I do actually hit blank walls a lot of times but i get by
Jen: i virtually thought so.. and i used to get sweatings
Bob: Wait what are sweatings
BEGIN DIALOGUE
Jen: Hi how are you doing today
Bob: doing good.. how about you
Jen: Im good, trying to understand how someone can feel like hitting a blank wall when they see the darkness
Bob: it's quite strange that you didnt imagine it
Jen: i dont imagine feeling a lot, maybe your on to something
BEGIN DIALOGUE
Jen: I have never cheated on my wife.
Bob: And thats something you should never do, good on you.
Jen: Yea it hasn't been easy but I am proud I haven't
Bob: What do you mean it hasn't been easy? How close have you come to cheating?
BEGIN DIALOGUE
Jen: Job interviews always make me sweat bullets, makes me uncomfortable in general to be looked at under a microscope like that.
Bob: Don't be nervous. Just be prepared.
Jen: I feel like getting prepared and then having a curve ball thrown at you throws you off.
Bob: Yes but if you stay calm it will be ok.
Jen: It's hard to stay clam. How do you do it?
BEGIN DIALOGUE
"""

#Refer to src/common/request.py for a list of possible parameters
params = {
"temperature": 0.5,  # Medium amount of randomness
"stop_sequences": ["Jen"],  # Stop when you hit a newline
"num_completions": 1,  # Generate many samples
"model": "ai21/j1-jumbo",
}

@app.route('/interface')
def projects():
    return render_template("interface.html", title = 'Chat Interface')

@app.route('/conversation', methods=['POST'])
def conversational_turn():
    json_args = request.get_json(force=True)
    #TODO: Error handling?
    user_utterance = str(json_args.get('user_utterance', None) or '')
    session_uuid = str(json_args.get('session_uuid', None) or str(uuid.uuid4()))
    user_uuid = str(json_args.get('user_uuid', None) or str(uuid.uuid4()))
    payload = json_args.get('payload', None) or []

    payload+=["Jen:"+user_utterance,]
    prompt = few_shot_context + '\n'.join(payload) + '\n' + "Bob:"


    model_request = Request(prompt=prompt, **params)
    request_result: RequestResult = service.make_request(auth, model_request)
    response = request_result.completions[0].text
    payload += ['Bob:'+response.strip(),]
    

    #TODO: Consider returning the deserialized current state
    json_response = {
        'session_uuid': session_uuid,
        'user_uuid': user_uuid,
        'bot_utterance': response,
        'payload': payload
    }
    with open(session_uuid+'.json', 'w') as f:
        json.dump(json_response, f)
    return json_response

# Make a request

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5001)
