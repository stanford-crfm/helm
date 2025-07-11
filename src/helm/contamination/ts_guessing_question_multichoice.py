import os
import numpy as np
import traceback
from rouge_score import rouge_scorer
from dataclasses import replace
from typing import List, Dict, Any, Tuple

from helm.common.hierarchical_logger import hlog, htrack_block
from nltk.tokenize import sent_tokenize
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.benchmark.runner import ScenarioState, RequestState
from helm.benchmark.executor import Executor
from helm.benchmark.metrics.metric import Stat
from helm.contamination.contamination_evaluator import ContaminationEvaluator
from helm.contamination.contamination_utils import ContaminationUtils


class TSGuessingQuestionMultiChoiceContaminationEvaluator(ContaminationEvaluator):
    """
    Implements a question-based multi-choice guessing test for contamination detection.
    Prompts are constructed with a randomly masked incorrect option. The model is asked
    to predict the content of the mask. This prediction is then compared against the
    original text of that specific masked (incorrect) option.
    Reference: https://aclanthology.org/2024.naacl-long.482/
    """

    # Strategy identifiers
    STRATEGY_NAME: str = "ts_guessing_question_multichoice"
    STRATEGY_DISPLAY_NAME: str = "TS-Guessing Multichoice"

    # Constants for model interaction and prompt generation
    SMALL_MODEL_CONTEXT_THRESHOLD: int = 1024
    MAX_OUTPUT_TOKENS: int = 100
    TOKENIZER_BUFFER: int = 30
    GENERATION_TEMPERATURE: float = 0.0

    def __init__(self):
        """
        Initializes the TS-Guessing multichoice evaluator.
        """
        self.language: str = "en"
        self.alphabet: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self._check_nltk_punkt()

    def _check_nltk_punkt(self):
        """
        Verifies that the NLTK 'punkt' tokenizer is available. Logs a warning if not found.
        Needed for sentence tokenization in _process_response.
        """
        try:
            sent_tokenize("Test sentence for NLTK punkt check.")
        except LookupError:
            hlog("WARNING: NLTK 'punkt' package not found. Needed for sentence tokenization in _process_response.")
            hlog("WARNING: Please download it by running: import nltk; nltk.download('punkt')")

    def evaluate(
        self,
        executor: Executor,
        benchmark_path: str,
        scenario_state: ScenarioState,
        language: str,
        tokenizer_service: TokenizerService,
    ) -> List[Stat]:
        """
        Runs the entire evaluation pipeline for the ts-guessing-multichoice strategy.
        """

        # Standardize the language code (e.g., "en_US" -> "en").
        self.language = language.lower().split("_")[0].split("-")[0]
        # Flag to control which tokenization method to use (primary or fallback).
        check_prompt_length_primary = True

        with htrack_block(f"{self.STRATEGY_DISPLAY_NAME} contamination evaluation for language '{self.language}'"):

            eval_data_name = os.path.basename(benchmark_path).split(":")[0]
            if not (
                hasattr(scenario_state, "adapter_spec")
                and hasattr(scenario_state.adapter_spec, "method")
                and scenario_state.adapter_spec.method == "multiple_choice_joint"
            ):
                raise ValueError(f"Benchmark '{eval_data_name}' does not use 'multiple_choice_joint' method")

            if not scenario_state.adapter_spec.model_deployment and not scenario_state.adapter_spec.model:
                raise ValueError("Model identifier not found in AdapterSpec. Cannot proceed.")

            # Get the model deployment name.
            model_deployment_name_from_spec = (
                scenario_state.adapter_spec.model_deployment or scenario_state.adapter_spec.model
            )

            # Determine the model's max context length.
            model_max_length = ContaminationUtils.determine_model_max_length(model_deployment_name_from_spec)
            hlog(f"INFO: Effective model_max_length for {model_deployment_name_from_spec}: {model_max_length}")

            # Load prompt templates for the current strategy and language.
            prompt_components = ContaminationUtils.get_prompt_fragments(self.STRATEGY_NAME, self.language)
            if not prompt_components:
                raise RuntimeError(f"Could not load prompt components for strategy '{self.STRATEGY_NAME}'.")

            # Filter for valid multiple-choice data points.
            data_points = self._filter_data(scenario_state)
            if not data_points:
                hlog("INFO: No suitable data points available after filtering. Nothing to evaluate.")
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
            # Create a new AdapterSpec with these generation parameters.
            generation_adapter_spec = ContaminationUtils.create_generation_adapter_spec(
                scenario_state.adapter_spec, generation_params
            )

            # Calculate the maximum number of tokens allowed for the prompt text.
            max_allowable_prompt_tokens = model_max_length - self.MAX_OUTPUT_TOKENS - self.TOKENIZER_BUFFER

            # Initialize lists to store ground truth and prepared requests.
            reference_texts_for_masked_slots: List[str] = []
            masked_option_letters: List[str] = []
            valid_request_states_for_execution: List[RequestState] = []
            skipped_instance_count: int = 0

            for data_point_item in shuffled_data_points:
                original_rs_idx = data_point_item["original_request_state_index"]

                if not (0 <= original_rs_idx < len(scenario_state.request_states)):
                    hlog(f"DEBUG: original_request_state_index {original_rs_idx} out of bounds. Skipping.")
                    skipped_instance_count += 1
                    continue

                current_request_state: RequestState = scenario_state.request_states[original_rs_idx]

                try:
                    # Build the prompt with a masked *incorrect* option.
                    instruction_text, user_text, original_text_of_masked_option, wrong_letter = self._build_prompt(
                        data_point_item, prompt_components
                    )
                    if instruction_text == "failed":
                        skipped_instance_count += 1
                        continue

                    combined_prompt = f"{instruction_text}\n\n{user_text}"

                    # Check prompt length (with primary/fallback logic).
                    if check_prompt_length_primary:
                        try:
                            is_valid_len, num_prompt_tokens = ContaminationUtils.check_prompt_length(
                                combined_prompt,
                                model_deployment_name_from_spec,
                                tokenizer_service,
                                max_allowable_prompt_tokens,
                            )
                        except RuntimeError:
                            # If primary fails, switch to fallback (GPT-2) for all subsequent checks.
                            check_prompt_length_primary = False
                            hlog(f"INFO: Switching to fallback tokenization for '{model_deployment_name_from_spec}'.")
                            is_valid_len, num_prompt_tokens = ContaminationUtils.check_prompt_length_fallback(
                                combined_prompt, max_allowable_prompt_tokens
                            )
                    else:
                        # Use fallback if primary has already failed.
                        is_valid_len, num_prompt_tokens = ContaminationUtils.check_prompt_length_fallback(
                            combined_prompt, max_allowable_prompt_tokens
                        )

                    if not is_valid_len:
                        skipped_instance_count += 1
                        continue

                    # Create a new Instance with the masked prompt.
                    new_input = replace(current_request_state.instance.input, text=combined_prompt)
                    new_instance = replace(current_request_state.instance, input=new_input, references=[])

                    # Create a new Request with the masked prompt and generation settings.
                    new_request = replace(
                        current_request_state.request,
                        prompt=combined_prompt,
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

                    # Store valid request and its corresponding ground truth.
                    valid_request_states_for_execution.append(prepared_rs)
                    reference_texts_for_masked_slots.append(original_text_of_masked_option)
                    masked_option_letters.append(wrong_letter)

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
            hlog(f"INFO: Sending {len(valid_request_states_for_execution)} requests for model generation.")
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
                    processed_response = ""
                    # Get the original (gold) word for comparison.
                    current_gold_reference = (
                        reference_texts_for_masked_slots[i]
                        if i < len(reference_texts_for_masked_slots)
                        else "ERROR_GOLD_REF_INDEX_OUT_OF_BOUNDS"
                    )
                    current_masked_letter = (
                        masked_option_letters[i]
                        if i < len(masked_option_letters)
                        else "ERROR_MASKED_LETTER_INDEX_OUT_OF_BOUNDS"
                    )

                    # Extract and process the response text.
                    if response_state.result and response_state.result.completions:
                        if response_state.result.completions[0].text is not None:
                            raw_response_text = response_state.result.completions[0].text.strip()
                            processed_response = self._process_response(raw_response_text, current_masked_letter)
                        else:
                            hlog(f"DEBUG MULTICHOICE [Inst {i}] RAW_RESPONSE: text is None")
                    else:
                        hlog(f"DEBUG MULTICHOICE [Inst {i}] RAW_RESPONSE: No result or completions found.")

                    # Store the results.
                    instance_id = getattr(valid_request_states_for_execution[i].instance, "id", f"proc_inst_{i}")
                    processed_instance_results.append(
                        {
                            "id": instance_id,
                            "answer_to_compare": current_gold_reference.lower(),
                            "model_prediction": processed_response.lower(),
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

            # Prepare lists for metric calculation.
            gold_references = [res["answer_to_compare"] for res in processed_instance_results]
            model_generations = [res["model_prediction"] for res in processed_instance_results]

            # Calculate Exact Match.
            exact_match_score = 0.0
            if any(model_generations):
                if gold_references:
                    exact_match_score = sum(gen == gold for gen, gold in zip(model_generations, gold_references)) / len(
                        gold_references
                    )

            # Calculate ROUGE-L F1.
            calculated_metrics = {"exact_match": exact_match_score, "rouge_l_f1": 0.0}
            if any(model_generations) and gold_references:
                try:
                    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=(self.language == "en"))
                    rouge_scores_f1 = [
                        (
                            scorer.score(str(model_gen), str(gold_ref))["rougeLsum"].fmeasure
                            if gold_ref and model_gen
                            else 0.0
                        )
                        for model_gen, gold_ref in zip(model_generations, gold_references)
                    ]
                    if rouge_scores_f1:
                        calculated_metrics["rouge_l_f1"] = np.mean(rouge_scores_f1)
                except Exception as e:
                    hlog(f"ERROR: ROUGE Scorer calculation failed: {e}\n{traceback.format_exc()}")

            # Format metrics for HELM.
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
        Filters valid multiple-choice instances from ScenarioState for evaluation.
        """

        data_points: List[Dict[str, Any]] = []
        # Iterate through all original request states.
        for i, request_state_item in enumerate(scenario_state.request_states):
            instance = request_state_item.instance
            if not instance:
                hlog(f"DEBUG: Skipping request_state at index {i} as it has no instance object.")
                continue
            try:
                # Extract necessary fields from the instance.
                question_text = ContaminationUtils.get_question_text(instance)
                choices_list = ContaminationUtils.get_choices(instance)
                true_correct_answer_idx = ContaminationUtils.get_answer_index(instance)

                if (
                    question_text
                    and question_text != "Unknown question or context"
                    and choices_list
                    and len(choices_list) > 1
                    and isinstance(true_correct_answer_idx, int)
                    and 0 <= true_correct_answer_idx < len(choices_list)
                ):
                    # If valid, add to data points.
                    data_points.append(
                        {
                            "id": getattr(instance, "id", f"instance_{i}"),
                            "text": question_text,
                            "choices": choices_list,
                            "true_correct_answer_index": true_correct_answer_idx,
                            "original_request_state_index": i,
                        }
                    )
                else:
                    q_ok = bool(question_text and question_text != "Unknown question or context")
                    c_list_len = len(choices_list) if choices_list else 0
                    idx_ok = isinstance(true_correct_answer_idx, int) and 0 <= true_correct_answer_idx < c_list_len
                    hlog(
                        f"DEBUG: Skipping instance from request_state {i}"
                        f" (ID: {getattr(instance, 'id', 'N/A')}) "
                        f"during filtering due to missing/invalid data (Q_OK: {q_ok},"
                        f" Choices_Count: {c_list_len}, AnswerIdx_OK: {idx_ok})."
                    )
            except Exception as e:
                hlog(
                    f"ERROR: Error filtering instance from request_state {i}"
                    f" (ID: {getattr(instance, 'id', 'N/A')}): {e}\n{traceback.format_exc()}"
                )
        return data_points

    def _build_prompt(
        self, example_item: Dict[str, Any], prompt_components: Dict[str, str]
    ) -> Tuple[str, str, str, str]:
        """
        Constructs the instruction and user text with one incorrect option masked.
        """

        # Load all necessary prompt template parts.
        try:
            part_instr_fill = prompt_components.get(
                "instruction_fill_option", "Your task is to complete the [MASK] in option"
            )
            part_instr_knowledge = prompt_components.get(
                "instruction_knowledge", "Use your knowledge to figure out what the original text was."
            )
            part_instr_rule = prompt_components.get(
                "instruction_rule",
                "Provide only the text that should replace [MASK]. Do not include the "
                "option letter or any other surrounding text.",
            )
            part_header_q = prompt_components.get("header_question", "Question:")
            part_header_opts = prompt_components.get("header_options", "Options:")
            part_footer_reply = prompt_components.get("footer_reply", "The original text for the masked option is:")
            answer_prefix = prompt_components.get("answer_prefix", "Answer: ")

            # Get data for the current instance.
            original_question_text = example_item.get("text", "")
            choices_list = example_item.get("choices", [])
            true_correct_answer_idx = example_item.get("true_correct_answer_index", -1)

            if not (
                original_question_text
                and choices_list
                and len(choices_list) > 1
                and isinstance(true_correct_answer_idx, int)
                and 0 <= true_correct_answer_idx < len(choices_list)
            ):
                hlog(
                    "DEBUG: _build_prompt failed - invalid input"
                    f" data for ID: {example_item.get('id', 'Unknown')}. "
                    f"Q: '{original_question_text[:30]}...', Choices: {len(choices_list)},"
                    f" CorrectIdx: {true_correct_answer_idx}"
                )
                return "failed", "", "", ""

            # Identify all *incorrect* options.
            incorrect_option_indices = [i for i in range(len(choices_list)) if i != true_correct_answer_idx]
            if not incorrect_option_indices:
                hlog(
                    f"DEBUG: No incorrect options to mask for instance"
                    f" {example_item.get('id', 'Unknown')}. (Choices: {len(choices_list)},"
                    f" CorrectIdx: {true_correct_answer_idx})"
                )
                return "failed", "", "", ""

            # Randomly select one incorrect option to mask.
            masked_choice_idx = np.random.choice(incorrect_option_indices)
            original_text_of_masked_option = str(choices_list[masked_choice_idx])
            masked_option_letter = self.alphabet[masked_choice_idx % len(self.alphabet)]

            # Construct the instruction text.
            instruction_text_segment = (
                f"{part_instr_fill} {masked_option_letter} {part_instr_knowledge}\n{part_instr_rule}\n"
            )

            # Construct the options list, masking the selected incorrect option.
            options_lines_segment = ""
            for i, choice_text_item in enumerate(choices_list):
                if i >= len(self.alphabet):
                    hlog(
                        f"WARNING: Number of choices ({len(choices_list)})"
                        f" exceeds alphabet size ({len(self.alphabet)}) "
                        f"for instance {example_item.get('id', 'Unknown')}. Truncating options display in prompt."
                    )
                    break
                letter = self.alphabet[i]
                option_content = "[MASK]" if i == masked_choice_idx else f"{str(choice_text_item)}"
                options_lines_segment += f"\n{letter}: {option_content}"

            # Construct the user-facing part of the prompt (question + options + reply footer).
            user_text_segment = (
                f"{part_header_q} {original_question_text}\n"
                f"{part_header_opts}{options_lines_segment}\n"
                f"{part_footer_reply}\n\n"
                f"{answer_prefix}"
            )

            return instruction_text_segment, user_text_segment, original_text_of_masked_option, masked_option_letter
        except Exception as e:
            hlog(
                "EXCEPTION: Critical error in _build_prompt for example"
                f" {example_item.get('id', 'Unknown ID')}: {e}\n{traceback.format_exc()}"
            )
            return "failed", "", "", ""

    def _process_response(self, response_text_input: str, masked_option_letter_param: str) -> str:
        """
        Cleans and extracts the main content from the model's raw text response.
        Prioritizes removing the 'Answer: ' prefix if present.
        """

        try:
            if not response_text_input:
                return ""

            processed_text = str(response_text_input).strip()

            # Remove the common "Answer: " prefix if present (case-insensitive).
            expected_model_prefix = "Answer: "
            if processed_text.lower().startswith(expected_model_prefix.lower()):
                processed_text = processed_text[len(expected_model_prefix) :].strip()

            try:
                if "nltk" in globals() and hasattr(globals()["nltk"], "download") and sent_tokenize:
                    sentences = sent_tokenize(processed_text)
                    if sentences:
                        processed_text = sentences[0]
            except LookupError:
                pass
            except Exception as e_sent_tok:
                hlog(f"DEBUG: sent_tokenize failed during response processing: {e_sent_tok}")

            # Remove any [MASK] text and surrounding/internal quotes.
            processed_text = processed_text.replace("[MASK]", "").strip()

            if processed_text.startswith("[") and processed_text.endswith("]"):
                processed_text = processed_text[1:-1].strip()

            if (processed_text.startswith('"') and processed_text.endswith('"')) or (
                processed_text.startswith("'") and processed_text.endswith("'")
            ):
                processed_text = processed_text[1:-1].strip()

            return processed_text
        except Exception as e:
            hlog(
                "ERROR: Error processing model response:"
                f" '{response_text_input[:50]}...': {e}\n{traceback.format_exc()}"
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
