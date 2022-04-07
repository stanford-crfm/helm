import uuid
import json
import random

import sys

sys.path = sys.path + ["../../"]

from src.common.authentication import Authentication  # noqa: E402
from src.common.request import Request, RequestResult  # noqa: E402
from proxy.remote_service import RemoteService  # noqa: E402

# flake8: noqa
# An example of how to use the request API.
"""
api_key = getpass.getpass(prompt="Enter a valid API key: ")
auth = Authentication(api_key=api_key)
"""
service = RemoteService("https://crfm-models.stanford.edu")


few_shot_context = ""

contexts = {
    "common_sense": 
    """BEGIN DIALOGUE
    Prompt: I remember going to the fireworks with my best friend. There was a lot of people_comma_ but it only felt like us in the world.
    Jen: I remember going to see the fireworks with my best friend. It was the first time we ever spent time alone together. Although there was a lot of people, we felt like the only people in the world.
    Bob: Was this a friend you were in love with, or just a best friend?
    Jen: This was a best friend. I miss her.
    Bob: Where has she gone?
    Jen: We no longer talk.
    Bob: Oh was this something that happened because of an argument?
    BEGIN DIALOGUE
    Prompt: i used to scare for darkness.
    Jen: it feels like hitting to blank wall when i see the darkness
    Bob: Oh ya? I don't really see how
    Jen: dont you feel so.. its a wonder
    Bob: I do actually hit blank walls a lot of times but i get by
    Jen: i virtually thought so.. and i used to get sweatings
    Bob: Wait what are sweatings
    BEGIN DIALOGUE
    Prompt: I showed a guy how to run a good bead in welding class and he caught on quick.
    Jen: Hi how are you doing today
    Bob: doing good.. how about you
    Jen: Im good, trying to understand how someone can feel like hitting a blank wall when they see the darkness
    Bob: it's quite strange that you didnt imagine it
    Jen: i dont imagine feeling a lot, maybe your on to something
    BEGIN DIALOGUE
    Prompt: I have always been loyal to my wife.
    Jen: I have never cheated on my wife.
    Bob: And thats something you should never do, good on you.
    Jen: Yea it hasn't been easy but I am proud I haven't
    Bob: What do you mean it hasn't been easy? How close have you come to cheating?
    BEGIN DIALOGUE
    Prompt: A recent job interview that I had made me feel very anxious because I felt like I didn't come prepared.
    Jen: Job interviews always make me sweat bullets, makes me uncomfortable in general to be looked at under a microscope like that.
    Bob: Don't be nervous. Just be prepared.
    Jen: I feel like getting prepared and then having a curve ball thrown at you throws you off.
    Bob: Yes but if you stay calm it will be ok.
    Jen: It's hard to stay clam. How do you do it?
    BEGIN DIALOGUE
    Prompt: Today, as i was leaving for work in the morning, i had a tire burst in the middle of a busy road. That scared the hell out of me!,
    """,
    "empathetic_dialogues": 
    """
    BEGIN DIALOGUE
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
    """,
    "wizard_of_wikipedia": 
    """
    BEGIN_DIALOGUE
    Topic: Dog
    Bob: Dogs were the first species to be domesticated and have been bred for many different jobs throughout history.
    Ines: I've heard a decent amount about that, didn't early humans see them eating from the carcasses of what we killed so we started to utilize them and domesticate them?
    Bob: Well the closest living relative to the dog is the gray wolf, so it is highly likely that early humans domesticated them.
    Ines: Oh right, so what were the main tasks that we bred dogs for?
    Bob: That I am unsure of, however the first undisputed record of a domesticated dog buried with human remains dates back over 14,700 years.
    Ines: Wow that's an insanely long time! Were dogs back then anything like dogs today or closer to wolves?
    Bob: They were closer to a wolf. The evolution of the wolf occurred over 800 thousand years turning into the specimens we now recognize as dogs.
    Ines: That's pretty bizarre, but really cool. There's so many breeds of dogs nowadays, though its kinda sad to see some of them just bred to look good at the expense of their health.
    Bob: Yes, that is very sad. But it is nice that todays dogs perform many various roles for people like aiding the handicapped and therapeutic roles.
    Ines: Oh yeah it's great that smart dogs are able to help the mentally and physically handicapped among other vital roles.
    Bob: Yes, it is easy to understand why dogs are considered mans best friend when they are able to perform these tasks.
    BEGIN_DIALOGUE
    Topic: London
    Bob: London is really cool city to visit in England.
    Ines: i would love to visit.
    Bob: Same here, I've been to England before but I didnt get to go to London.
    Ines: Yea same, bet it has awesome history
    Bob: I know! It also stands on the River Thames in the south east of the island of Great Britain.
    Ines: beautiful country they have there.
    Bob: I've always wanted to visit the River Thames over there in Great Britain.
    Ines: ill make sure to go next time i visit.
    BEGIN_DIALOGUE
    Topic: Journalist
    Ines: Are you a journalist?
    Bob: No I am not.  Maybe in another life I would liked to have been a sports journalist.  How about you?
    Ines: I am not but I think it would be a very interesting career!
    Bob: Reminds me of the album 'In Another Life' by Bilal.  Ever heard of it?
    Ines: I've never heard of it at all, could you tell me more about it?
    Bob: It is a well-received R&B album.  Has a lot of 70s funk and soul in it.  It is considered one of the best R&B to come along in a while.
    Ines: Interesting, I might have to check it out, Is Bilal still around?
    Bob: Oh yes, he is still around.  He lives in New York City.  'In Another Life' came out in 2015.
    Ines: Oh wow, so he has been working for a long time.
    Bob: Relatively speaking I suppose.  Any music artists you like?
    BEGIN_DIALOGUE
    Topic: Elementary school (United States)
    Bob: So I just started working at an Elementary School in my town!
    Ines: Oh wow! Congrats! What grade are you teaching?
    Bob: 3rd grade.The main purpose of an elementary school is to educate children usually between 4-11 years old
    Ines: Yes, I am glad that the children have a place to go and learn.
    Bob: There were 92,858 elementary schools in 2001! That's a lot
    Ines: That is a ton. No wonder teacher's don't get paid that much. The school's would go broke.
    Bob: Well, there is overcrowding for sure but there is also a great lack of funding in general. Students learn basic arithmetic and mathematics in elementary school.
    Ines: The lack of funding is sad. I think there should be more pride taken in our children's education.
    Bob: Or education in general. We seem to prefer debt based rather than knowledge based learning.
    Ines: Sad but true. The world today has messed up priorities.
    BEGIN_DIALOGUE
    Topic: Piano
    Bob: I'd love to learn how to play the piano, the acoustic stringed musical instrument! Can you play it?
    Ines: As a kid, I took lessons, and could play at a basic level, but the chords and how they connected never made sense to me, so I had to memorize sheet music.  
    Bob: Oh, I get that, it's just printed form of music notation so you're not really learning just replaying. Do you like music with piano in it?
    Ines: Yes I do.  I'm not much on classical, though.  I prefer ragtime music!
    Bob: Oh, that's good music too, ragtime's musical style was primarily enjoyed in the 1890's? Just 190 years after the discovery of a piano!
    Ines: I didn't realize it started that early!  What type of piano music do you enjoy hearing?
    Bob: While it's not a traditional, I do love the sounds played using an electronic keyboard.
    Ines: Electronic keyboards have come a long way.  They truly sound more like a traditional piano than ever before.
    Bob: Definitely, although the term electronic keyboard also refers to synthesizers, digital and stage pianos, and electronic organs. Have you ever heard the melodious sounds of an electric organ?
    Ines: Well, yes, but my ears don't think of an electric organ as 'melodious.'
    BEGIN_DIALOGUE
    Topic: Taylor Swift
    """
}
prompts = {
    "common_sense": "During this conversation, please discuss the following scenario: Today, as i was leaving for work in the morning, i had a tire burst in the middle of a busy road. That scared the hell out of me!",
    "empathetic_dialogues": "Please talk about a time when you felt excited",
    "wizard_of_wikipedia": "Please discuss the topic 'Taylor Swift'"
}
# Refer to src/common/request.py for a list of possible parameters
# TODO: replace with equivalent Adapter spec that the script for HIT creation will spit out
params = {
    "temperature": 0.5,  # Medium amount of randomness
    "stop_sequences": ["Jen"],  # Stop when you hit a newline
    "num_completions": 1,  # Generate many samples
    "model": "ai21/j1-jumbo",
}


def start_conversation(auth: Authentication, json_args: dict):
    """
    This is a stub method to get a text prompt.
    TODO: Replace this with an integration with interactive runer
    """
    datasets = ["empathetic_dialogues", "common_sense", "wizard_of_wikipedia"]
    dataset = random.choice(datasets)
    few_shot_context = contexts[dataset]
    
    return {
        "prompt": prompts[dataset]
    }


def conversational_turn(auth: Authentication, json_args: dict) -> dict:
    """
    Call CRFM API to get the next turn of conversation

    Args:
        auth (Authentication): CRFM API authentication
        json_args: args to query CRFM API with including
        - user_utterance: user utterance
        - payload: conversational history + five-shot training examples
        - session_uuid: unique session id
        - user_uuid: unique user id

    Returns:
        json_response: json response containing
        - bot_utterance: the bot's response to the user's utterance
        - payload: training examples + updated conv history
        - session_uuid: unique session id
        - user_uuid: unique user id
    """
    user_utterance = str(json_args.get("user_utterance", None) or "")
    session_uuid = str(json_args.get("session_uuid", None) or str(uuid.uuid4()))
    user_uuid = str(json_args.get("user_uuid", None) or str(uuid.uuid4()))
    payload = json_args.get("payload", None) or []

    # TODO: Define the names of the two participants in one place as a constant
    payload += [
        "Jen:" + user_utterance,
    ]
    prompt = few_shot_context + "\n".join(payload) + "\n" + "Bob:"

    model_request = Request(prompt=prompt, **params)
    request_result: RequestResult = service.make_request(auth, model_request)
    response = request_result.completions[0].text
    payload += [
        "Bob:" + response.strip(),
    ]

    # TODO: Consider returning the deserialized current state
    json_response = {
        "session_uuid": session_uuid,
        "user_uuid": user_uuid,
        "bot_utterance": response,
        "payload": payload,
    }
    # TODO: Write this in a directory that's passed in from somewhere
    with open(session_uuid + ".json", "w") as f:
        json.dump(json_response, f)
    return json_response  # Outputs


def submit_interview(json_args: dict) -> dict:
    session_uuid = str(json_args.get("session_uuid"))
    with open(session_uuid + ".answers.json", "w") as f:
        json.dump(json_args, f)
    return {"success": True}  # Outputs
