from dataclasses import replace
import re

from benchmark.adapter import InteractionTrace, InteractiveAdapter, RequestState, UserInput, AdapterSpec


class DialogueAdapter(InteractiveAdapter):
    def __init__(self, user_initiated, user_name: str, agent_name: str):
        super().__init__(user_initiated)
        self.user_name = user_name
        self.agent_name = agent_name

    def adapt_user_input_string(self, inp: str) -> str:
        """Adapts user input string by prepending user_name"""
        inp = inp.strip()
        adapted_utterance = self.user_name + ": <span class=\"conversation_utterance\">\"" + inp + "\"</span>"
        return adapted_utterance

    def postprocess_initial_request(
        self, initial_request_state: RequestState, adapter_spec: AdapterSpec
    ) -> RequestState:
        if self.user_initiated:

            print("Before postprocessing")
            print(initial_request_state.request.prompt)
            new_prompt = re.sub(
                adapter_spec.input_prefix + ".*(?=" + adapter_spec.output_prefix + ")",
                "",
                initial_request_state.request.prompt,
            )
            new_request = replace(initial_request_state.request, prompt=new_prompt)
            initial_request_state = replace(initial_request_state, request=new_request)
            print("After postprocessing")
            print(initial_request_state.request.prompt)
        return initial_request_state

    def agent_prompt(self) -> str:
        agent_prompt = self.agent_name + ": <span class=\"conversation_utterance\">\""
        return agent_prompt

    def initial_lm_request(self, initial_request_state: RequestState) -> RequestState:
        initial_prompt = initial_request_state.request.prompt
        new_prompt = initial_prompt + self.agent_prompt()
        new_request = replace(initial_request_state.request, prompt=new_prompt)
        new_request_state = replace(initial_request_state, request=new_request)
        return new_request_state

    def adapt_user_input(self, interaction_trace: InteractionTrace, user_input: UserInput) -> RequestState:
        adapted_user_input = self.adapt_user_input_string(user_input.input)
        assert len(interaction_trace.trace) > 0
        last_request_state = interaction_trace.trace[-1].request_state
        last_prompt = last_request_state.request.prompt
        last_response = ""
        if last_request_state.result and len(last_request_state.result.completions) > 0:
            last_response = last_request_state.result.completions[0].text
        new_prompt = last_prompt + last_response + "\"</span>\n" + adapted_user_input + self.agent_prompt()
        new_request = replace(last_request_state.request, prompt=new_prompt)
        new_request_state = replace(last_request_state, request=new_request)
        return new_request_state
