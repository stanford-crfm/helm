To inject synthetic dialogues:

1 - Create a file with a few dialouges. Look at sample_synthetic_dialogue.yaml for an example. 

2 - Create a new run for the scenario with `annotation_stage=filtering` or `annotation_stage=agreement`, with the same number of instances

	For e.g. for the sample file `venv/bin/benchmark-run -r empatheticdialogues:user_initiated=True,annotation_stage=filtering --pre-interaction -m 2`

3 - Run `inject_synthetic_dialogues.py` with the synthetic dialogues yaml file as the first paramater and the run_path as the second parameter

	For e.g. `python src/benchmark/interaction_server/inject_synthetic_dialogues.py src/benchmark/interaction_server/sample_synthetic_dialogue.yaml benchmark_output/runs/empatheticdialogues:annotation_stage=filtering,user_initiated=True``

4 - Start the interaction server and in the url, set the run_name to the above. Take care to url encode the equals signs. 

	For e.g. a sample url for the above will be http://<ip>/static/dialogue/interface.html?run_name=empatheticdialogues:annotation_stage%3Dfiltering,user_initiated%3DTrue&interaction_trace_id=<actual_interaction_trace_id>&user_id=1234

    NB - For each user_id, a new survey result will be captured in the interaction trace. 
