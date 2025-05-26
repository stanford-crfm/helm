import numpy as np
import spacy
import asyncio
import traceback
import random
from dataclasses import replace
from typing import List, Dict, Any, Tuple, Optional

from helm.common.hierarchical_logger import hlog, htrack_block
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.benchmark.runner import ScenarioState, RequestState
from helm.benchmark.executor import Executor
from helm.benchmark.metrics.metric import Stat
from .utils_contamination import UtilsContamination


class TSGuessingQuestionBasedContaminationEvaluator:
    """
    Implementation of the 'base' TS-Guessing strategy.
    Masks an important word (NOUN, ADJ, VERB) in a sentence and asks the model
    to predict the masked word. Contamination is measured via exact match.
    Reference: https://aclanthology.org/2024.naacl-long.482/
    """

    STRATEGY_NAME: str = "ts_guessing_question_base"
    STRATEGY_DISPLAY_NAME: str = "TS-Guessing Base"

    SMALL_MODEL_CONTEXT_THRESHOLD: int = 1024
    MAX_OUTPUT_TOKENS: int = 20
    TOKENIZER_BUFFER: int = 30
    GENERATION_TEMPERATURE: float = 0.1
    POS_TAGS_TO_MASK: List[str] = ["NOUN", "ADJ", "VERB"]

    def __init__(self):
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
        Entry point for synchronous evaluation. Wraps the asynchronous evaluation.

        Args:
            executor: The executor to use for running model queries.
            benchmark_path: Path to the benchmark data.
            scenario_state: The scenario state containing instances and adapter specifications.
            language: Language code for the evaluation (e.g., "en", "pt").
            tokenizer_service: Tokenizer service used for token counting.

        Returns:
            A list of dictionaries containing evaluation metrics and statistics (PinnedStat-like).
        """

        return asyncio.run(self._evaluate_async(executor, benchmark_path, scenario_state, language, tokenizer_service))

    async def _evaluate_async(
        self,
        executor: Executor,
        benchmark_path: str,
        scenario_state: ScenarioState,
        language: str,
        tokenizer_service: TokenizerService,
    ) -> List[Stat]:
        """
        Asynchronous implementation of the evaluation logic.
        """

        self.language = language.lower().split("_")[0].split("-")[0]
        check_prompt_length_primary = True
        tagger: Optional[spacy.language.Language] = None

        with htrack_block(f"{self.STRATEGY_DISPLAY_NAME} contamination evaluation for language '{self.language}'"):
            try:
                tagger = UtilsContamination.get_spacy_tagger(self.language)
            except Exception as e:
                hlog(
                    f"STRATEGY CRITICAL: Failed to load spaCy tagger for language '{self.language}': {e}\n{traceback.format_exc()}"
                )
                return []

            if not scenario_state.adapter_spec:
                hlog("STRATEGY ERROR: AdapterSpec not found in ScenarioState. Cannot proceed.")
                return []

            model_deployment_name_from_spec = (
                scenario_state.adapter_spec.model_deployment or scenario_state.adapter_spec.model
            )
            if not model_deployment_name_from_spec:
                hlog(
                    "STRATEGY ERROR: Model identifier (model_deployment or model) not found in AdapterSpec. Cannot proceed."
                )
                return []

            model_max_length = UtilsContamination.determine_model_max_length(
                model_deployment_name_from_spec, UtilsContamination.DEFAULT_MODEL_MAX_CONTEXT_TOKENS_UTIL
            )
            hlog(f"STRATEGY INFO: Effective model_max_length for {model_deployment_name_from_spec}: {model_max_length}")

            if model_max_length <= self.SMALL_MODEL_CONTEXT_THRESHOLD:
                hlog(
                    f"STRATEGY WARNING: Model {model_deployment_name_from_spec} (context window {model_max_length} tokens) "
                    "may skip many instances if prompts are too long."
                )

            data_points = self._filter_data(scenario_state)
            hlog(f"STRATEGY INFO: Filtered to {len(data_points)} data points for evaluation.")
            if not data_points:
                hlog("STRATEGY INFO: No data points available after filtering.")
                return []

            shuffled_data_points = [data_points[i] for i in np.random.permutation(len(data_points))]

            original_masked_words: List[str] = []
            valid_request_states_for_execution: List[RequestState] = []
            skipped_instance_count: int = 0

            prompt_components = UtilsContamination.get_prompt_fragments(self.STRATEGY_NAME, self.language)
            if not prompt_components:
                hlog(
                    f"STRATEGY ERROR: Could not load prompt components for strategy '{self.STRATEGY_DISPLAY_NAME}' and language '{self.language}'. "
                    f"Check PROMPT_CONFIGS_MASTER in UtilsContamination for key '{self.STRATEGY_DISPLAY_NAME}'."
                )
                return []

            generation_params = {
                "max_tokens": self.MAX_OUTPUT_TOKENS,
                "temperature": self.GENERATION_TEMPERATURE,
                "stop_sequences": [],
                "num_outputs": 1,
                "output_prefix": prompt_components.get("answer_prefix", "Answer: "),
                "instructions": "",
                "input_prefix": "",
                "input_suffix": "",
                "output_suffix": "",
                "global_prefix": "",
                "global_suffix": "",
                "reference_prefix": "",
                "reference_suffix": "",
            }
            generation_adapter_spec = UtilsContamination.create_generation_adapter_spec(
                scenario_state.adapter_spec, generation_params
            )

            max_allowable_prompt_tokens = model_max_length - self.MAX_OUTPUT_TOKENS - self.TOKENIZER_BUFFER

            for data_point_item in shuffled_data_points:
                original_rs_idx = data_point_item.get("original_request_state_index", -1)

                if not (0 <= original_rs_idx < len(scenario_state.request_states)):
                    hlog(
                        f"STRATEGY DEBUG: original_request_state_index {original_rs_idx} out of bounds or missing. Skipping."
                    )
                    skipped_instance_count += 1
                    continue

                current_request_state: RequestState = scenario_state.request_states[original_rs_idx]

                try:
                    if tagger is None:
                        hlog("STRATEGY ERROR: spaCy tagger is None, cannot process instance.")
                        skipped_instance_count += 1
                        continue

                    final_prompt_text, masked_word_original = self._build_prompt(
                        data_point_item, tagger, prompt_components
                    )
                    if final_prompt_text == "failed":
                        skipped_instance_count += 1
                        continue

                    if check_prompt_length_primary:
                        try:
                            is_valid_len, num_prompt_tokens = (
                                UtilsContamination.check_prompt_length_tokenization_request(
                                    final_prompt_text,
                                    model_deployment_name_from_spec,
                                    tokenizer_service,
                                    max_allowable_prompt_tokens,
                                )
                            )
                        except Exception:
                            check_prompt_length_primary = False
                            hlog(
                                f"STRATEGY INFO: Switching to fallback tokenization for '{model_deployment_name_from_spec}' due to primary tokenizer failure."
                            )

                            is_valid_len, num_prompt_tokens = UtilsContamination.check_prompt_length_fallback_gpt2(
                                final_prompt_text, model_deployment_name_from_spec, max_allowable_prompt_tokens
                            )
                    else:
                        is_valid_len, num_prompt_tokens = UtilsContamination.check_prompt_length_fallback_gpt2(
                            final_prompt_text, model_deployment_name_from_spec, max_allowable_prompt_tokens
                        )

                    if not is_valid_len:
                        log_msg = f"STRATEGY DEBUG: Instance from original_rs_idx {original_rs_idx} skipped. "
                        if num_prompt_tokens == -1:
                            log_msg += "Tokenization failed."
                        else:
                            log_msg += f"Prompt too long ({num_prompt_tokens} > {max_allowable_prompt_tokens})."
                        hlog(log_msg)
                        skipped_instance_count += 1
                        continue

                    new_input = replace(current_request_state.instance.input, text=final_prompt_text)
                    new_instance = replace(current_request_state.instance, input=new_input, references=[])

                    new_request = replace(
                        current_request_state.request,
                        prompt=final_prompt_text,
                        max_tokens=self.MAX_OUTPUT_TOKENS,
                        temperature=self.GENERATION_TEMPERATURE,
                        stop_sequences=[],
                    )

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

                    valid_request_states_for_execution.append(prepared_rs)
                    original_masked_words.append(masked_word_original.lower())

                except Exception as e:
                    hlog(
                        f"STRATEGY ERROR: Error preparing instance from original_rs_idx {original_rs_idx} "
                        f"(ID: {data_point_item.get('id', 'N/A')}): {e}\n{traceback.format_exc()}"
                    )
                    skipped_instance_count += 1

            if skipped_instance_count > 0:
                hlog(f"STRATEGY INFO: Skipped {skipped_instance_count} instances during preparation.")
            if not valid_request_states_for_execution:
                hlog("STRATEGY INFO: No instances prepared for execution after processing all data points.")
                return []

            hlog(f"STRATEGY INFO: Sending {len(valid_request_states_for_execution)} requests for model generation.")
            execution_scenario_state = replace(
                scenario_state, adapter_spec=generation_adapter_spec, request_states=valid_request_states_for_execution
            )

            try:
                response_scenario_state: ScenarioState = await self._query_model(execution_scenario_state, executor)
            except Exception as e:
                hlog(f"STRATEGY CRITICAL: Error during model query phase: {e}\n{traceback.format_exc()}")
                return []

            processed_instance_results: List[Dict[str, str]] = []
            for i, response_state in enumerate(response_scenario_state.request_states):
                try:
                    raw_response_text = ""
                    processed_model_word = ""
                    current_gold_word = (
                        original_masked_words[i]
                        if i < len(original_masked_words)
                        else "ERROR_GOLD_WORD_INDEX_OUT_OF_BOUNDS"
                    )

                    if response_state.result and response_state.result.completions:
                        if response_state.result.completions[0].text is not None:
                            raw_response_text = response_state.result.completions[0].text.strip()
                            processed_model_word = self._process_response(raw_response_text)
                        else:
                            hlog(f"STRATEGY DEBUG [Inst {i}] RAW_RESPONSE: text is None")
                    else:
                        hlog(f"STRATEGY DEBUG [Inst {i}] RAW_RESPONSE: No result or completions found.")

                    instance_id = getattr(valid_request_states_for_execution[i].instance, "id", f"base_inst_{i}")
                    processed_instance_results.append(
                        {
                            "id": instance_id,
                            "masked_word_original": current_gold_word,
                            "model_predicted_word": processed_model_word.lower(),
                        }
                    )
                except IndexError:
                    hlog(f"STRATEGY ERROR: IndexError processing result for instance {i}. List length mismatch.")
                except Exception as e:
                    hlog(
                        f"STRATEGY ERROR: Error processing result for prepared instance index {i}: {e}\n{traceback.format_exc()}"
                    )

            if not processed_instance_results:
                hlog("STRATEGY INFO: No valid results after model query and processing.")
                return []

            exact_match_count = sum(
                1 for res in processed_instance_results if res["model_predicted_word"] == res["masked_word_original"]
            )
            exact_match_score = (
                (exact_match_count / len(processed_instance_results)) if processed_instance_results else 0.0
            )

            calculated_metrics = {"exact_match": exact_match_score}

            strategy_metric_prefix = f"contamination ({self.STRATEGY_DISPLAY_NAME}"
            final_helm_stats = UtilsContamination.format_helm_stats(
                calculated_metrics, strategy_metric_prefix, split="test"
            )

            hlog(
                f"STRATEGY INFO: Evaluation completed for language '{self.language}'. "
                f"Processed {len(processed_instance_results)} instances. "
            )
            return final_helm_stats

    def _filter_data(self, scenario_state: ScenarioState) -> List[Dict[str, Any]]:
        """
        Filters raw instances from ScenarioState to retain only those suitable for word masking.

        Args:
            scenario_state: The current ScenarioState containing all request_states.

        Returns:
            A list of dictionaries, each containing 'text_to_mask' and 'original_request_state_index'.
        """

        data_points: List[Dict[str, Any]] = []
        for i, rs in enumerate(scenario_state.request_states):
            instance = rs.instance
            if not instance:
                hlog(f"STRATEGY DEBUG: Skipping request_state at index {i} as it has no instance.")
                continue
            try:
                text_content = UtilsContamination.get_question_text(instance)

                if text_content and text_content != "Unknown question or context":
                    words_in_text = text_content.split()
                    if len(words_in_text) >= 5:
                        data_points.append(
                            {
                                "text_to_mask": text_content,
                                "original_request_state_index": i,
                            }
                        )
                    else:
                        hlog(
                            f"STRATEGY DEBUG: Skipping instance from request_state {i} (ID: {getattr(instance, 'id', 'N/A')}) "
                            f"due to insufficient word count ({len(words_in_text)} <= 4) in text: '{text_content[:50]}...'"
                        )
                else:
                    hlog(
                        f"STRATEGY DEBUG: Skipping instance from request_state {i} (ID: {getattr(instance, 'id', 'N/A')}) "
                        "due to missing or invalid text_content."
                    )
            except Exception as e:
                hlog(
                    f"STRATEGY ERROR: Error filtering instance from request_state {i} (ID: {getattr(instance, 'id', 'N/A')}): {e}\n{traceback.format_exc()}"
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
                hlog("STRATEGY DEBUG: _build_prompt failed - 'text_to_mask' is missing or not a string.")
                return "failed", ""

            doc = tagger(text_to_process)
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
                    f"STRATEGY DEBUG: No suitable candidate tokens (POS: {self.POS_TAGS_TO_MASK}, non-stop, non-punct, len>1) "
                    f"found in text: '{text_to_process[:100]}...'"
                )
                return "failed", ""

            selected_token = random.choice(candidate_tokens)
            word_to_mask_original_case = selected_token.text

            start_char = selected_token.idx
            end_char = start_char + len(word_to_mask_original_case)

            if not (0 <= start_char < end_char <= len(text_to_process)):
                hlog(
                    f"STRATEGY WARNING: Token indices ({start_char}-{end_char}) for word '{word_to_mask_original_case}' "
                    f"are out of bounds for text length {len(text_to_process)}. Text: '{text_to_process[:100]}...'. "
                    "This might indicate issues with tokenization or the source text."
                )

                masked_text_fallback = text_to_process.replace(word_to_mask_original_case, "[MASK]", 1)
                if masked_text_fallback == text_to_process:
                    hlog(
                        f"STRATEGY ERROR: Fallback replace also failed for word '{word_to_mask_original_case}'. Cannot build prompt."
                    )
                    return "failed", ""
                masked_text = masked_text_fallback
            else:
                masked_text = text_to_process[:start_char] + "[MASK]" + text_to_process[end_char:]

            instruction = prompt_components.get("instruction", "Fill in the [MASK] in the sentence:")
            sentence_template = prompt_components.get("sentence_template", '"{masked_sentence}"')

            try:
                sentence_formatted = sentence_template.format(masked_sentence=masked_text)
            except KeyError as e_format:
                hlog(
                    f"STRATEGY ERROR: 'sentence_template' from prompt_components is missing '{{masked_sentence}}' placeholder. Template: '{sentence_template}'. Error: {e_format}"
                )
                return "failed", ""

            final_prompt = f"{instruction}\n{sentence_formatted}"

            return final_prompt, word_to_mask_original_case

        except Exception as e:
            item_id_info = example_item.get(
                "id", f"from original_rs_idx {example_item.get('original_request_state_index', 'Unknown')}"
            )
            hlog(
                f"STRATEGY EXCEPTION: Critical error in _build_prompt for item '{item_id_info}': {e}\n{traceback.format_exc()}"
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

            if self.language == "zh":
                punctuation_zh = "。，！？；：（）《》「」『』“”‘’ 、\n\t" + " "
                cleaned_text = "".join(char for char in processed_text if char not in punctuation_zh)
                first_meaningful_part = cleaned_text[: self.MAX_OUTPUT_TOKENS]
            else:
                response_words = processed_text.split()
                first_word_candidate = response_words[0] if response_words else ""

                punctuation_std = '"""\'\'.,;!?¿¡#()[]{}<>:\n\t' + " "
                first_meaningful_part = first_word_candidate.strip(punctuation_std)

            return first_meaningful_part

        except Exception as e:
            hlog(
                f"STRATEGY ERROR: Error processing model response (base strategy): '{full_response_text[:50]}...': {e}\n{traceback.format_exc()}"
            )
            return ""

    async def _query_model(self, scenario_state: ScenarioState, executor: Executor) -> ScenarioState:
        """Executa as requisições ao modelo de forma assíncrona."""

        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, executor.execute, scenario_state)
        except Exception as e:
            hlog(f"STRATEGY CRITICAL: Model query execution failed: {e}\n{traceback.format_exc()}")
            raise
