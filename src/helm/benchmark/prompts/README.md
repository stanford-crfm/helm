# Custom LLM Judge Prompts - README

This environment includes a default prompt file (`default_prompt.txt`) configured to work seamlessly with the HELM benchmarking framework. However, users are encouraged to define their own custom prompts based on specific evaluation needs.

---

## ✅ How to Create and Use a Custom Prompt

To define a new judging prompt:

1. Create a `.txt` file inside the `customizable_prompts` directory.
2. Pass the name of this file via the CLI using the `--prompt` flag:

    --prompt my_custom_prompt.txt

---

## 📌 Prompt Requirements

Every custom prompt **must** include the following placeholders:

- `{input}` → This will be replaced with the benchmark input (task/question).
- `{response}` → This will be replaced with the model's response to be judged.

These placeholders are **required** for HELM to dynamically fill in each instance during runtime.

---

## 📤 Expected Output Format

The LLM judge is expected to return a valid JSON object with the following structure:

{
  "judgement": 0 or 1,
  "explanation": "A brief justification for the judgment"
}

- `judgement`: Use `1` if the judge agrees with the primary model's response, or `0` if it disagrees.
- `explanation`: Provide a concise justification for the assigned judgement.

⚠️ Important: This format is strictly required. HELM uses it to compute the agreement level and process the output automatically.

---

## 💡 Final Note

We recommend using the provided `default_prompt.txt` as a starting point when crafting your own prompts.

If you encounter any issues or need additional guidance, feel free to open an issue in the repository.
