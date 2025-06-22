import numpy as np
import spacy
import traceback
import random
from dataclasses import replace
from typing import List, Dict, Any, Tuple, Optional

from helm.common.hierarchical_logger import hlog, htrack_block
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.benchmark.runner import ScenarioState, RequestState
from helm.benchmark.executor import Executor
from helm.benchmark.metrics.metric import Stat
from helm.contamination.contamination_evaluator import ContaminationEvaluator
from helm.contamination.contamination_utils import ContaminationUtils


class TSGuessingQuestionBasedContaminationEvaluator(ContaminationEvaluator):
    """
    Implementation of the 'base' TS-Guessing strategy.
    Masks an important word (NOUN, ADJ, VERB) in a sentence and asks the model
    to predict the masked word. Contamination is measured via exact match.
    Reference: https://aclanthology.org/2024.naacl-long.482/
    """

    # Strategy identifiers
    STRATEGY_NAME: str = "ts_guessing_question_base"
    STRATEGY_DISPLAY_NAME: str = "TS-Guessing Base"

    # Constants for model interaction and prompt generation
    SMALL_MODEL_CONTEXT_THRESHOLD: int = 1024
    MAX_OUTPUT_TOKENS: int = 20
    TOKENIZER_BUFFER: int = 30
    GENERATION_TEMPERATURE: float = 0.0
    POS_TAGS_TO_MASK: List[str] = ["NOUN", "ADJ", "VERB"]

    def __init__(self):
        """
        Initializes the TS-Guessing base evaluator.
        """
        self.language: str = "en"

    def evaluate(
        self,
        executor: Executor,
        benchmark_path: str,
        scenario_state: ScenarioState,
        language: str,
        tokenizer_service: TokenizerService,
    ) -> List[Stat]:
        """
        Runs the entire evaluation pipeline for the ts-guessing-base strategy.
        This includes data filtering, prompt construction, model execution, and result processing.
        """

        # Standardize the language code (e.g., "en_US" -> "en").
        self.language = language.lower().split("_")[0].split("-")[0]
        # Flag to control which tokenization method to use (primary or fallback).
        check_prompt_length_primary = True
        tagger: Optional[spacy.language.Language] = None

        with htrack_block(f"{self.STRATEGY_DISPLAY_NAME} contamination evaluation for language '{self.language}'"):

            # Load the spaCy tagger for the specified language.
            tagger = ContaminationUtils.get_spacy_tagger(self.language)
            if not tagger:
                raise RuntimeError(f"Could not load spaCy tagger for language '{self.language}'. Aborting.")

            if not scenario_state.adapter_spec:
                raise ValueError("AdapterSpec not found in ScenarioState. Cannot proceed.")

            # Get the model deployment name.
            model_deployment_name_from_spec = (
                scenario_state.adapter_spec.model_deployment or scenario_state.adapter_spec.model
            )
            if not model_deployment_name_from_spec:
                raise ValueError("Model identifier not found in AdapterSpec. Cannot proceed.")

            # Determine the model's max context length.
            model_max_length = ContaminationUtils.determine_model_max_length(
                model_deployment_name_from_spec, ContaminationUtils.DEFAULT_MODEL_MAX_CONTEXT_TOKENS
            )
            hlog(f"INFO: Effective model_max_length for {model_deployment_name_from_spec}: {model_max_length}")

            # Load prompt templates for the current strategy and language.
            prompt_components = ContaminationUtils.get_prompt_fragments(self.STRATEGY_NAME, self.language)
            if not prompt_components:
                raise RuntimeError(f"Could not load prompt components for strategy '{self.STRATEGY_NAME}'.")

            # Filter data points to get suitable texts for masking.
            data_points = self._filter_data(scenario_state)
            hlog(f"INFO: Filtered to {len(data_points)} data points for evaluation.")
            if not data_points:
                hlog("INFO: No data points available after filtering.")
                return []

            # Shuffle the data points to avoid order biases.
            shuffled_data_points = [data_points[i] for i in np.random.permutation(len(data_points))]

            # Set up parameters for the generation request.
            generation_params = {
                "max_tokens": self.MAX_OUTPUT_TOKENS,
                "temperature": self.GENERATION_TEMPERATURE,
                "stop_sequences": [],
                "num_outputs": 1,
                "output_prefix": "",
                "instructions": "",
                "input_prefix": "",
                "input_suffix": "",
                "output_suffix": "",
                "global_prefix": "",
                "global_suffix": "",
                "reference_prefix": "",
                "reference_suffix": "",
            }
            generation_adapter_spec = ContaminationUtils.create_generation_adapter_spec(
                scenario_state.adapter_spec, generation_params
            )

            # Calculate the maximum number of tokens allowed for the prompt text.
            max_allowable_prompt_tokens = model_max_length - self.MAX_OUTPUT_TOKENS - self.TOKENIZER_BUFFER

            # Store original words and valid requests.
            original_masked_words: List[str] = []
            valid_request_states_for_execution: List[RequestState] = []
            skipped_instance_count: int = 0

            # Process each data point to create a request.
            for data_point_item in shuffled_data_points:
                original_rs_idx = data_point_item["original_request_state_index"]

                if not (0 <= original_rs_idx < len(scenario_state.request_states)):
                    hlog(
                        "DEBUG: original_request_state_index" f" {original_rs_idx} out of bounds or missing. Skipping."
                    )
                    skipped_instance_count += 1
                    continue

                current_request_state: RequestState = scenario_state.request_states[original_rs_idx]

                try:
                    # Build the prompt with a masked word.
                    final_prompt_text, masked_word_original = self._build_prompt(
                        data_point_item, tagger, prompt_components
                    )
                    if final_prompt_text == "failed":
                        skipped_instance_count += 1
                        continue

                    # Check prompt length using a primary-fallback strategy.
                    # This attempts to use the official tokenizer service, falling back to a
                    # GPT-2-based approximation only if the primary service fails.
                    if check_prompt_length_primary:
                        try:
                            is_valid_len, num_prompt_tokens = ContaminationUtils.check_prompt_length(
                                final_prompt_text,
                                model_deployment_name_from_spec,
                                tokenizer_service,
                                max_allowable_prompt_tokens,
                            )
                        except RuntimeError:
                            # If primary fails, switch to fallback (GPT-2) for all subsequent checks.
                            check_prompt_length_primary = False
                            hlog(f"INFO: Switching to fallback tokenization for '{model_deployment_name_from_spec}'.")
                            is_valid_len, num_prompt_tokens = ContaminationUtils.check_prompt_length_fallback(
                                final_prompt_text, max_allowable_prompt_tokens
                            )
                    else:
                        # Use fallback if primary has already failed.
                        is_valid_len, num_prompt_tokens = ContaminationUtils.check_prompt_length_fallback(
                            final_prompt_text, max_allowable_prompt_tokens
                        )

                    if not is_valid_len:
                        skipped_instance_count += 1
                        continue

                    # Create a new Instance with the masked prompt.
                    new_input = replace(current_request_state.instance.input, text=final_prompt_text)
                    new_instance = replace(current_request_state.instance, input=new_input, references=[])

                    # Create a new Request with the masked prompt and generation settings.
                    new_request = replace(
                        current_request_state.request,
                        prompt=final_prompt_text,
                        max_tokens=self.MAX_OUTPUT_TOKENS,
                        temperature=self.GENERATION_TEMPERATURE,
                        stop_sequences=[],
                    )

                    # Create a new RequestState for execution, cleaning unnecessary fields.
                    prepared_rs = replace(
                        current_request_state,
                        instance=new_instance,
                        request=new_request,
                        result=None,
                    )

                    if hasattr(prepared_rs, "output_mapping"):
                        prepared_rs = replace(prepared_rs, output_mapping=None)
                    if hasattr(prepared_rs, "reference_index"):
                        prepared_rs = replace(prepared_rs, reference_index=None)

                    # Add the prepared request and original word to our lists.
                    valid_request_states_for_execution.append(prepared_rs)
                    original_masked_words.append(masked_word_original.lower())

                except Exception as e:
                    # Log errors during instance preparation and skip the instance.
                    hlog(
                        f"WARNING: Error preparing instance from original_rs_idx {original_rs_idx} "
                        f"(ID: {data_point_item.get('id', 'N/A')}): {e}\n{traceback.format_exc()}"
                    )
                    skipped_instance_count += 1

            if skipped_instance_count > 0:
                hlog(f"INFO: Skipped {skipped_instance_count} instances during preparation.")
            if not valid_request_states_for_execution:
                hlog("INFO: No instances were prepared for execution.")
                return []

            # Create a ScenarioState with only the valid requests and the generation AdapterSpec.
            hlog(f"INFO: Sending {len(valid_request_states_for_execution)} requests to the model.")
            execution_scenario_state = replace(
                scenario_state, adapter_spec=generation_adapter_spec, request_states=valid_request_states_for_execution
            )

            # Query the model.
            try:
                response_scenario_state: ScenarioState = self._query_model(execution_scenario_state, executor)
            except Exception as e:
                hlog(f"CRITICAL: The model execution phase failed: {e}\n{traceback.format_exc()}")
                raise

            # Process the model responses.
            processed_instance_results: List[Dict[str, str]] = []
            for i, response_state in enumerate(response_scenario_state.request_states):
                try:
                    raw_response_text = ""
                    processed_model_word = ""
                    # Get the original (gold) word for comparison.
                    current_gold_word = (
                        original_masked_words[i]
                        if i < len(original_masked_words)
                        else "ERROR_GOLD_WORD_INDEX_OUT_OF_BOUNDS"
                    )

                    if response_state.result and response_state.result.completions:
                        if response_state.result.completions[0].text is not None:
                            raw_response_text = response_state.result.completions[0].text.strip()
                            # Process the raw text to get just the predicted word.
                            processed_model_word = self._process_response(raw_response_text)
                        else:
                            hlog(f"DEBUG [Inst {i}] RAW_RESPONSE: text is None")
                    else:
                        hlog(f"DEBUG [Inst {i}] RAW_RESPONSE: No result or completions found.")

                    # Store the gold word and the model's prediction.
                    instance_id = getattr(valid_request_states_for_execution[i].instance, "id", f"base_inst_{i}")
                    processed_instance_results.append(
                        {
                            "id": instance_id,
                            "masked_word_original": current_gold_word,
                            "model_predicted_word": processed_model_word.lower(),
                        }
                    )
                except IndexError:
                    hlog(f"ERROR: IndexError processing result for instance {i}. List length mismatch.")
                except Exception as e:
                    hlog(
                        "ERROR: Error processing result for prepared"
                        f" instance index {i}: {e}\n{traceback.format_exc()}"
                    )

            if not processed_instance_results:
                hlog("INFO: No valid results after model query and processing.")
                return []

            # Calculate the exact match score.
            exact_match_count = sum(
                1 for res in processed_instance_results if res["model_predicted_word"] == res["masked_word_original"]
            )
            exact_match_score = (
                (exact_match_count / len(processed_instance_results)) if processed_instance_results else 0.0
            )

            # Prepare metrics for HELM.
            calculated_metrics = {"exact_match": exact_match_score}
            strategy_metric_prefix = f"contamination ({self.STRATEGY_DISPLAY_NAME}"
            final_helm_stats = ContaminationUtils.format_helm_stats(
                calculated_metrics, strategy_metric_prefix, split="test"
            )

            hlog(
                f"INFO: Evaluation completed for language '{self.language}'. "
                f"Processed {len(processed_instance_results)} instances. "
            )
            return final_helm_stats

    def _filter_data(self, scenario_state: ScenarioState) -> List[Dict[str, Any]]:
        """
        Filters raw instances from ScenarioState to retain only those suitable for word masking.
        """

        data_points: List[Dict[str, Any]] = []
        # Iterate through all original request states.
        for i, rs in enumerate(scenario_state.request_states):
            instance = rs.instance
            if not instance:
                hlog(f"DEBUG: Skipping request_state at index {i} as it has no instance.")
                continue
            try:
                # Extract the relevant text (question or context).
                text_content = ContaminationUtils.get_question_text(instance)

                if text_content and text_content != "Unknown question or context":
                    words_in_text = text_content.split()
                    if len(words_in_text) >= 5:
                        # If valid, add it to our list with its original index.
                        data_points.append(
                            {
                                "text_to_mask": text_content,
                                "original_request_state_index": i,
                            }
                        )
                    else:
                        hlog(
                            f"DEBUG: Skipping instance from request_state"
                            f" {i} (ID: {getattr(instance, 'id', 'N/A')}) "
                            f"due to insufficient word count ({len(words_in_text)} <= 4)"
                            f" in text: '{text_content[:50]}...'"
                        )
                else:
                    hlog(
                        f"DEBUG: Skipping instance from request_state"
                        f" {i} (ID: {getattr(instance, 'id', 'N/A')}) "
                        f"due to missing or invalid text_content."
                    )
            except Exception as e:
                hlog(
                    f"ERROR: Error filtering instance from request_state"
                    f" {i} (ID: {getattr(instance, 'id', 'N/A')}): {e}\n{traceback.format_exc()}"
                )
        return data_points

    def _build_prompt(
        self,
        example_item: Dict[str, Any],
        tagger: spacy.language.Language,
        prompt_components: Dict[str, str],
    ) -> Tuple[str, str]:
        """
        Builds the masked-word prompt for a given example item.
        """

        try:
            text_to_process = example_item.get("text_to_mask")
            if not text_to_process or not isinstance(text_to_process, str):
                hlog("DEBUG: _build_prompt failed - 'text_to_mask' is missing or not a string.")
                return "failed", ""

            # Use spaCy to tag the text.
            doc = tagger(text_to_process)
            # Find candidate words to mask (NOUN, ADJ, VERB, non-stop, non-punct).
            candidate_tokens = [
                token
                for token in doc
                if token.pos_ in self.POS_TAGS_TO_MASK
                and not token.is_stop
                and not token.is_punct
                and len(token.text.strip()) > 1
            ]

            if not candidate_tokens:
                hlog(
                    f"DEBUG: No suitable candidate tokens"
                    f" (POS: {self.POS_TAGS_TO_MASK}, non-stop, non-punct, len>1) "
                    f"found in text: '{text_to_process[:100]}...'"
                )
                return "failed", ""

            # Randomly select one candidate word.
            selected_token = random.choice(candidate_tokens)
            word_to_mask_original_case = selected_token.text

            # Get the character indices of the selected word.
            start_char = selected_token.idx
            end_char = start_char + len(word_to_mask_original_case)

            if not (0 <= start_char < end_char <= len(text_to_process)):
                hlog(
                    f"WARNING: Token indices ({start_char}-{end_char})"
                    f" for word '{word_to_mask_original_case}' "
                    f"are out of bounds for text length {len(text_to_process)}. Text: '{text_to_process[:100]}...'. "
                    "This might indicate issues with tokenization or the source text."
                )

                masked_text_fallback = text_to_process.replace(word_to_mask_original_case, "[MASK]", 1)
                if masked_text_fallback == text_to_process:
                    hlog(
                        "ERROR: Fallback replace also failed for"
                        f" word '{word_to_mask_original_case}'. Cannot build prompt."
                    )
                    return "failed", ""
                masked_text = masked_text_fallback
            else:
                # Use precise indices for masking.
                masked_text = text_to_process[:start_char] + "[MASK]" + text_to_process[end_char:]

            # Get prompt components (instruction, template).
            instruction = prompt_components.get("instruction", "Fill in the [MASK] in the sentence:")
            sentence_template = prompt_components.get("sentence_template", '"{masked_sentence}"')
            answer_prefix = prompt_components.get("answer_prefix", "Answer: ")

            # Format the prompt using the template.
            try:
                sentence_formatted = sentence_template.format(masked_sentence=masked_text)
            except KeyError as e_format:
                hlog(
                    "ERROR: 'sentence_template' from prompt_components"
                    f" is missing '{{masked_sentence}}' placeholder. Template: '{sentence_template}'. Error: {e_format}"
                )
                return "failed", ""

            # Combine instruction and formatted sentence into the final prompt.
            final_prompt = f"{instruction}\n" f"{sentence_formatted}\n\n" f"{answer_prefix}"

            # Return the prompt and the original word (for later comparison).
            return final_prompt, word_to_mask_original_case

        except Exception as e:
            item_id_info = example_item.get(
                "id", f"from original_rs_idx {example_item.get('original_request_state_index', 'Unknown')}"
            )
            hlog(
                f"EXCEPTION: Critical error in _build_prompt for"
                f" item '{item_id_info}': {e}\n{traceback.format_exc()}"
            )
            return "failed", ""

    def _process_response(self, full_response_text: str) -> str:
        """
        Processes the model's raw response to extract the predicted word,
        handling language-specific nuances for tokenization and punctuation.
        """

        try:
            if not full_response_text:
                return ""

            processed_text = str(full_response_text).strip()

            # Special handling for Chinese (no splitting, just remove punctuation).
            if self.language == "zh":
                punctuation_zh = "。，！？；：（）《》「」『』“”‘’ 、\n\t" + " "
                cleaned_text = "".join(char for char in processed_text if char not in punctuation_zh)
                first_meaningful_part = cleaned_text[: self.MAX_OUTPUT_TOKENS]
            else:
                # For other languages, split by space and take the first word.
                response_words = processed_text.split()
                first_word_candidate = response_words[0] if response_words else ""

                # Clean standard punctuation from the first word.
                punctuation_std = '"""\'\'.,;!?¿¡#()[]{}<>:\n\t' + " "
                first_meaningful_part = first_word_candidate.strip(punctuation_std)

            return first_meaningful_part

        except Exception as e:
            hlog(
                "ERROR: Error processing model response (base strategy):"
                f" '{full_response_text[:50]}...': {e}\n{traceback.format_exc()}"
            )
            return ""

    def _query_model(self, scenario_state: ScenarioState, executor: Executor) -> ScenarioState:
        """
        Executes requests to the model.
        """
        try:
            return executor.execute(scenario_state)
        except Exception as e:
            hlog(f"CRITICAL: Model query execution failed: {e}\n{traceback.format_exc()}")
            raise
