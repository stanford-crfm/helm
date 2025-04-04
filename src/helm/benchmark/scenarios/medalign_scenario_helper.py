# flake8: noqa
# type: ignore
# fmt: off

import ast
import datetime
import transformers
import langchain
import langchain.prompts
import lxml.etree
import os
import pandas as pd
import re
import tiktoken

from langchain_community.retrievers import BM25Retriever
from tqdm import tqdm
from typing import Any, Dict, Optional, Union, Callable
from langchain.schema import Document
import langchain_community



def get_instructions(path_to_instructions: str) -> Dict[int, Dict[str, Any]]:
    """
    Builds map from Instruction ID to instruction details

    The needed information for creating the map is accomplished by reading
    a CSV file from the user-specified path.

    The CSV file is expected to contain at least the following columns:
    - instruction_id: The ID of the instruction.
    - question: The text of the instruction.
    - person_id: The ID of the associated patient.
    - is_selected_ehr: A flag indicating whether the instruction is selected.

    See https://stanfordmedicine.box.com/s/0om9qav2sklb9vaitn0ibye65vgbfx0e

    Parameters:
        path_to_instructions (str): Path to CSV file containing instructions.

    Returns:
        Dict[int, Dict[str, Any]]: A dictionary mapping instruction IDs to a
            dictionary containing instruction text and associated patient ID.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the CSV file does not contain the expected columns.
    """
    if not os.path.exists(path_to_instructions):
        raise FileNotFoundError(
            f"The specified file {path_to_instructions} does not exist."
        )

    instructions_df = pd.read_csv(path_to_instructions, sep='\t')
    required_columns = {
        "instruction_id",
        "question",
        "person_id",
    }
    if not required_columns.issubset(instructions_df.columns):
        raise ValueError(
            f"The CSV file is missing one or more of the required columns: {required_columns}"
        )

    selected_instructions_df = instructions_df #.query("is_selected_ehr == 'yes'")
    instructions_map = {
        row["instruction_id"]: {
            "instruction": row["question"],
            "patient_id": row["person_id"],
        }
        for _, row in selected_instructions_df.iterrows()
    }
    return instructions_map


def extract_patient_id_from_fname(fname: str) -> Optional[int]:
    """
    Extracts and returns the patient ID from a given filename.

    The function expects filenames in the format 'EHR_<patient_id>.xml',
    where <patient_id> is a sequence of digits.

    Parameters:
        fname (str): The filename from which to extract the patient ID.

    Returns:
        Optional[int]: The extracted patient ID as an integer, or None if
                    the filename doesn't match the expected format.
    """
    name=fname.split('.')[0]
    return int(name)


def get_ehrs(path_to_ehrs: str) -> Dict[int, str]:
    """
    Builds a map from Instruction ID to EHR (Electronic Health Record) timeline.

    EHR timelines are in string format and EHR files are read in from the
    user-specified directory. Each file in the directory should be named
    'EHR_<patient_id>.xml', where <patient_id> is a sequence of digits.

    See https://stanfordmedicine.box.com/s/r28wfwwude9rpjtu0szhzegmku8qv2pe

    Parameters:
        path_to_ehrs (str): The path to the directory containing the EHR files.

    Returns:
        Dict[int, str]: A dictionary mapping patient IDs to EHR timelines.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
    """
    if not os.path.isdir(path_to_ehrs):
        raise FileNotFoundError(
            f"The specified directory {path_to_ehrs} does not exist."
        )

    ehr_map = {}
    for fname in os.listdir(path_to_ehrs):
        pt_id = extract_patient_id_from_fname(fname)
        if pt_id is None:
            print(
                f"Warning: File '{fname}' does not match the expected format "
                "and will be skipped."
            )
            continue

        file_path = os.path.join(path_to_ehrs, fname)
        with open(file_path, encoding="utf-8", mode="r") as f:
            ehr = f.read()

        ehr_map[pt_id] = ehr
    return ehr_map


def get_tokenizer(tokenizer_name: str) -> Callable:
    """
    Returns a tokenizer based on the given tokenizer name.

    Parameters:
        tokenizer_name (str): The name of the tokenizer. Acceptable values are:
            - "tiktoken"
            - "chatgpt"
            - "gpt-3.5-turbo"
            - "gpt-4"
            - "gpt-4-turbo"
            - "gpt-4o"
            - "cl100k_base"
            - Any valid tokenizer name recognized by the transformers library.

    Returns:
        Callable: The tokenizer instance.
    """
    if tokenizer_name.lower() in [
        "tiktoken",
        "chatgpt",
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4o",
        "cl100k_base",
    ]:
        return tiktoken.get_encoding("cl100k_base")
    print(tokenizer_name)
    return transformers.AutoTokenizer.from_pretrained(tokenizer_name, legacy=False)


def retrieve_most_relevant_visits(ehr_visit_strs, query, target_length, tokenizer):
    """
    Retrieve and filter relevant EHR visits based on a query and target length.

    This function retrieves electronic health record (EHR) visit strings, sorts them
    by relevance using the BM25Retriever, and constructs a list of final documents
    that fit within a specified character length. The final list ensures that the
    most important visit isn't cut off and is sorted chronologically.

    Parameters:
        ehr_visit_strs (list of str): List of EHR visit strings.
        query (str): Query string to retrieve relevant visits.
        target_length (int): Maximum total token count for the final list of documents.
        tokenizer (Callable): Tokenizer that converts text to tokens (used for tracking context length)

    Returns:
        list[str]: List of EHR visit strings sorted chronologically and constrained by the target length.
    """
    ehr_visits=re.split(r'(?=</encounter>\n)',ehr_visit_strs)
    langchain_docs = [
        langchain.schema.Document(page_content=doc) for doc in ehr_visits #broken since ehr_visit_strs is one string of all visits
    ]
    # `k` is the number of documents to retrieve
    # We retrieve everything and just use the BM25Retriever to sort the documents
    retriever = langchain_community.retrievers.BM25Retriever.from_documents(
        langchain_docs, k=len(langchain_docs)
    )

    # Invoking the retriever means the most relevant documents are sorted first
    sorted_docs = retriever.invoke(query)

    # Define the regex pattern to find the start time
    # pattern = r'start="([\d/]+ [\d:]+)"'
    pattern = r'start="([\d/]+ [\d:]+ ?[APM]{0,2})"'

    docs = []
    dts = []

    # Find the startime of the document
    for doc in sorted_docs:
        doc_content = doc.page_content
        start_dt_match = re.search(pattern, doc_content)
        if start_dt_match:
            start_dt = start_dt_match.group(1)
            parsed = False
            # Try different date formats
            for fmt in (
                "%m/%d/%y %I:%M %p",
                "%m/%d/%Y %I:%M %p",
                "%m/%d/%y %H:%M",
                "%m/%d/%Y %H:%M",
            ):
                try:
                    dts.append(datetime.datetime.strptime(start_dt, fmt))
                    parsed = True
                    break
                except ValueError:
                    continue
            if not parsed:
                print(f"Error parsing date: {start_dt}")
                continue
        else:
            print(f"Start time not found., {doc_content}")
            dts.append(datetime.datetime.min)
        docs.append(doc_content)

    final_docs = []
    current_length = 0

    # Add documents until we exceed the allocated context length
    for i in range(len(docs)):
        doc_content = docs[i]
        doc_length = len(tokenizer.encode(doc_content))
        final_docs.append((dts[i], doc_content))
        current_length += doc_length
        if current_length > target_length:
            break

    # Sort final_docs chronologically
    final_docs.sort(key=lambda x: x[0])

    # Extract only the document content for the final output
    final_docs_content = [doc_content for _, doc_content in final_docs]

    return final_docs_content



def pack_and_trim_prompts(
    instructions: Dict[int, Dict[str, str]],
    ehrs: Dict[int, str],
    prompt_template: langchain.prompts.PromptTemplate,
    context_length: int,
    generation_length: int,
    tokenizer: Any,
    use_RAG: bool = True,
    verbose: bool = False,
    include_ehr: bool = True,
) -> Dict[int, str]:
    """
    Returns:
        A map from Instruction ID to prompt
    """
    prompts_map = {}
    for instruction_id in tqdm(instructions.keys()):
        instruction = instructions[instruction_id]["instruction"]
        patient_id = int(instructions[instruction_id]["patient_id"])
        relevant_ehr = ehrs[patient_id]

        # Calculate how many tokens of EHR we can include in the prompt
        num_tokens_instruction = len(tokenizer.encode(instruction))
        num_tokens_prompt_template = len(tokenizer.encode(prompt_template.template))
        if include_ehr:
            target_ehr_length = context_length - generation_length - num_tokens_prompt_template - num_tokens_instruction
        else:
            target_ehr_length = 0
        if target_ehr_length <= 0:
            prompt_with_truncated_ehr = prompt_template.format(question=instruction, ehr="")
        else:
            if use_RAG:
                # Return a list of the most relevant visit strings
                most_relevant_visits = retrieve_most_relevant_visits(
                    ehr_visit_strs=relevant_ehr,
                    query=instruction,
                    target_length=target_ehr_length,
                    tokenizer=tokenizer,
                )
                relevant_ehr = "\n".join(most_relevant_visits)

            # Do a first pass with a fast tokenizer
            fast_tokenizer = tiktoken.get_encoding("cl100k_base")
            fast_encoded = fast_tokenizer.encode(relevant_ehr)
            if len(fast_encoded) <= target_ehr_length:
                fast_encoded_truncated = fast_encoded[-(2 * target_ehr_length) :]
                fast_truncated_ehr = fast_tokenizer.decode(fast_encoded_truncated)

                # Then do a second pass with the actual tokenizer
                encoded_ehr = tokenizer.encode(fast_truncated_ehr)
                truncated_encoded_ehr = encoded_ehr[-target_ehr_length:]
                truncated_ehr = tokenizer.decode(truncated_encoded_ehr)
                prompt_with_truncated_ehr = prompt_template.format(question=instruction, ehr=truncated_ehr)

                prompts_map[instruction_id] = prompt_with_truncated_ehr

                if verbose:
                    print(prompt_with_truncated_ehr)
                    print("~" * 20)
    return prompts_map


def preprocess_prompts(
    target_context_length,
    generation_length,
    path_to_instructions,
    path_to_ehrs,
    use_RAG,
    include_ehr,
    tokenizer,
    codes_only=False,
    notes_only=False,
):
    print(
        f"\n\twith target context length = {target_context_length} "
        f"\n\twith target generation length = {generation_length} "
    )

    # FETCH INSTRUCTIONS
    print("Fetching instructions...")
    instructions = get_instructions(path_to_instructions)

    # FETCH RELEVANT EHRs #
    print("Fetching patient EHR timelines...")
    ehrs = get_ehrs(path_to_ehrs)

    # LOAD TOKENIZER #
    print("Loading tokenizer...")
    tokenizer = get_tokenizer(tokenizer)

    # CONSTRUCT & TRUNCATE PROMPTS #
    print("Constructing prompts using instructions and EHRs...")
    prompt_string="Instruction: Answer the following question based on the EHR:\n\nEHR: {ehr}\n\nQuestion: {question}\n\nAnswer:"
    prompt_template = langchain.prompts.PromptTemplate.from_template(prompt_string)
    filled_prompts = pack_and_trim_prompts(
        instructions=instructions,
        ehrs=ehrs,
        prompt_template=prompt_template,
        context_length=target_context_length,
        generation_length=generation_length,
        tokenizer=tokenizer,
        use_RAG=use_RAG,
        verbose=False,
        include_ehr=include_ehr,
    )
    assert filled_prompts, f"No prompts were found for length: {target_context_length}. Try again with a larger length."
    # SAVE CONSTRUCTED PROMPTS TO DISK
    df_rows = []
    for instruction_id in tqdm(filled_prompts.keys()):
        row = {}
        row["instruction_id"] = instruction_id
        patient_id = instructions[instruction_id]["patient_id"]
        row["patient_id"] = patient_id
        row["instruction"] = instructions[instruction_id]["instruction"]
        row["ehr"] = "".join(ehrs[patient_id])
        row["prompt"] = filled_prompts[instruction_id]
        row["context_length"] = target_context_length
        row["generation_length"] = generation_length
        df_rows.append(row)

    prompts_df = pd.DataFrame(df_rows)
    instructionid_to_prompt_map = (
        prompts_df[["instruction_id", "prompt"]].set_index("instruction_id").to_dict().get("prompt")
    )
    instructionid_to_prompt_df = (
        pd.DataFrame.from_dict(instructionid_to_prompt_map, orient="index", columns=["prompt"])
        .reset_index()
        .rename(columns={"index": "instruction_id"})
    )

    print("...Prompt construction complete")
    return instructionid_to_prompt_df


def add_reference_responses(prompts_df, path_to_reference_responses) -> pd.DataFrame:
    """
    Processes a single file for evaluation.

    Parameters:
    file_path (str): Path to the file to be processed.
    args (argparse.Namespace): Command line arguments passed to the script.

    Returns:
    pd.DataFrame: DataFrame containing the processed data.
    """
    gold_df = pd.read_csv(path_to_reference_responses)
    gold_df = gold_df.query("annotator_num == 'Annotator_1'")
    gold_df = gold_df[["instruction_id", "clinician_response"]]
    merged_df = gold_df.merge(prompts_df, on="instruction_id", how="inner")
    return merged_df


def return_dataset_dataframe(max_length: int) -> pd.DataFrame:
    target_context_length = max_length
    generation_length = 256
    path_to_instructions = "/share/pi/nigam/datasets/medalign_release_fixes/clinician-reviewed-model-responses.tsv"
    path_to_ehrs = "/share/pi/nigam/datasets/medalign_release_fixes/medalign_ehr_xml"
    path_to_reference_responses = "/share/pi/nigam/scottyf/clinician-instruction-responses.csv"
    use_RAG = False
    include_ehr = True
    tokenizer = "tiktoken"

    instructionid_to_prompt_df = preprocess_prompts(
        target_context_length=target_context_length,
        generation_length=generation_length,
        path_to_instructions=path_to_instructions,
        path_to_ehrs=path_to_ehrs,
        use_RAG=use_RAG,
        include_ehr=include_ehr,
        tokenizer=tokenizer,
    )
    medalign_dataframe = add_reference_responses(instructionid_to_prompt_df, path_to_reference_responses)
    return medalign_dataframe
