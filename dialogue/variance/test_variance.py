"""
Measuring variance in chat responses caused by changes in five-shot training examples and 
"""
import os
import re
import sys
import uuid
import json
import string
import getpass
import argparse
from itertools import combinations
from proxy.remote_service import RemoteService

sys.path = sys.path + ['../../']
from src.proxy.accounts import Account
from src.common.authentication import Authentication
from src.common.request import Request, RequestResult

api_key = getpass.getpass(prompt="Enter a valid API key: ")
auth = Authentication(api_key=api_key)
service = RemoteService("https://crfm-models.stanford.edu")

"""
Iterate through test files in directory
Iterate through parameters [i.e. models]
Run conversation
Record data
Record metrics
"""

"""
Building parameters
"""
params = {
"temperature": 0.5,  # Medium amount of randomness
"stop_sequences": ["Jen", '\n'],  # Stop when you hit a newline
"num_completions": 1,  # Generate many samples
}

"""
Get bot response by calling API
"""
def call_api(payload):
    model_request = Request(prompt=payload, **params)
    request_result: RequestResult = service.make_request(auth, model_request)
    response = request_result.completions[0].text
    return response

"""
Read in context from file
"""
def read_context(context_fpath):
    few_shot_file = open(context_fpath)
    few_shot_context = few_shot_file.read() # Read file into str
    return few_shot_context

"""
Running conversation
- open in file
- open out file
- for line in in file
- input to model
- get model output
- input model output + next line
- write model outputs + lines to output file
"""
def run_conversation(model, user_inputs, context_fpath, out_fpath):
    # Add model to parameters
    params["model"] = model
    few_shot_context = read_context(context_fpath)
    
    # Open output file
    # TODO: Add handling for directories which haven't been created yet
    out_file = open(out_fpath, "w+")
    conv_history = []
    with open(user_inputs, 'r') as input_file:
        print(context_fpath)
        for user_input in input_file:
            conv_history.append("Jen: "+user_input.strip())
            payload = few_shot_context + '\n'.join(conv_history) + '\n' + "Bob:"
            bob_response = call_api(payload)
            print(bob_response)
            conv_history.append("Bob:"+bob_response)
            out_file.write(bob_response.strip()+'\n')

"""
Preparing an utterance
- lower case
- strip punctuation
"""  
def clean_utterance(utterance):
    utterance =  utterance.strip().lower()
    utterance = utterance.translate(str.maketrans('', '', string.punctuation))
    return utterance

"""
Basic Jaccard scoring fn
"""
def get_jaccard_sim(f1_line, f2_line):
    f1_set = set(clean_utterance(f1_line).split())
    f2_set = set(clean_utterance(f2_line).split())
    line_intersect = f1_set.intersection(f2_set)
    line_union = f1_set.union(f2_set)
    return float(len(line_intersect)) / len(line_union)

"""
Stats
# of different tokens
# of different responses
# of token overlap
"""
def score_pair(pair):
    f1 = open(pair[0])
    f2 = open(pair[1])
    lines1 = f1.readlines()
    lines2 = f2.readlines()
    scores = []
    for idx in range(len(lines1)):
        scores.append(get_jaccard_sim(lines1[idx], lines2[idx]))
    return scores

"""
Aggregate scores
- average
- maximum
- minimum
"""
def aggregate_scores(pairwise_scores):
    all_scores = [value for key, values in pairwise_scores.items() for value in values]
    avg = sum(all_scores) / len(all_scores)
    avg_across_lines = {key: sum(value) / len(value) for key, value in pairwise_scores.items()}
    min_key =  min(avg_across_lines, key=avg_across_lines.get)
    max_key = max(avg_across_lines, key=avg_across_lines.get)
    print("Average similarity score: ",avg)
    print("Minimum similarity score: {} For files {}".format(min_key, avg_across_lines[min_key]))
    print("Maximum similarity score: {} For files {}".format(max_key, avg_across_lines[max_key]))


"""
Main loop
for model in model_list:
    for file in dir:
        curr_stats = do_conversation(model, file)
        add_to_total_stats(curr_stats)

print(curr_stats)
"""
def main():
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_dir", help="Name of directory where five-shot training files are stored",
                        nargs="?", type=str, default="test_contexts") # TODO: Come up with better arg name
    parser.add_argument("--output_dir", help="Name of directory where output files are stored",
                        nargs="?", type=str, default="test_outputs/hopeful")
    # TODO: Change this file name
    parser.add_argument("--results_fname", help="Name of file where aggregated results are stored",
                        nargs="?", type=str, default="variance_results.csv")
    parser.add_argument("--user_inputs", help="Name of file containing user inputs to test",
                        nargs="?", type=str, default="user_inputs.txt")
    args = parser.parse_args()

    # Main loop

    models = ['ai21/j1-jumbo']#, 'openai/davinci']
    model_names = {'ai21/j1-jumbo': 'j1-jumbo', 'openai/davinci': 'davinci'}
    for model in models: # Test variance across models
        for context_fname in os.listdir(args.context_dir): # And across files
            out_fname = model_names[model] + "_" + context_fname[:-4] + "_devastated_results.txt"
            context_fpath = os.path.join(args.context_dir, context_fname) # Get full path for context file
            out_fpath = os.path.join(args.output_dir, out_fname) # Get full path for output file
            run_conversation(model, args.user_inputs, context_fpath, out_fpath)

    # Store paths
    out_paths = []
    for output_fname in os.listdir(args.output_dir): # Iterate through output files
        out_fpath = os.path.join(args.output_dir, output_fname)
        out_paths.append(out_fpath)

    # Compute pairwise scores
    pairwise_scores = {}
    #all_scores = []
    pairs = list(combinations(out_paths, 2))
    for pair in pairs:
        scores = score_pair(pair)
        pairwise_scores[pair] = scores
    
    aggregate_scores(pairwise_scores)


if __name__ == '__main__':
    # Run chat app
    # app.run(host="127.0.0.1", port=5001)
    main()
