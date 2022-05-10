from dataclasses import replace

from benchmark.adapter import InteractionTrace, InteractiveAdapter, RequestState, UserInput


class DialogueAdapter(InteractiveAdapter):
    def __init__(self, user_initiated, user_name: str, agent_name: str):
        super().__init__(user_initiated)
        self.user_name = user_name
        self.agent_name = agent_name

    def adapt_user_input_string(self, inp: str) -> str:
        """Adapts user input string by prepending user_name"""
        inp = inp.strip()
        adapted_utterance = self.user_name + ": " + inp + "\n"
        return adapted_utterance

    def agent_prompt(self) -> str:
        agent_prompt = self.agent_name + ": "
        return agent_prompt

    def initial_lm_request(self, interaction_trace: InteractionTrace) -> RequestState:
        initial_request_state = interaction_trace.trace[0].request_state
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
        new_prompt = last_prompt + last_response + adapted_user_input + self.agent_prompt()
        new_request = replace(last_request_state.request, prompt=new_prompt)
        new_request_state = replace(last_request_state, request=new_request)
        return new_request_state
