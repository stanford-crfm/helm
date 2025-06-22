"""
This class standardizes and stores prompt snippets used in benchmark data contamination evaluation strategies.
Currently, it supports the languages of benchmarks covered by HELM, including English (en),
Portuguese (pt), and Chinese (zh).
"""

TS_GUESSING_BASE = {
    "en": {
        "instruction": "Fill in the [MASK] in the sentence:",
        "sentence_template": '"{masked_sentence}"\nProvide only the single word as your answer.',
        "answer_prefix": "Answer:",
    },
    "pt": {
        "instruction": "Preencha o [MASK] na frase:",
        "sentence_template": '"{masked_sentence}"\nForneça apenas a única palavra como sua resposta.',
        "answer_prefix": "Resposta:",
    },
    "zh": {
        "instruction": "填写句子中的[MASK]：",
        "sentence_template": '"{masked_sentence}"\n请仅提供这个词作为答案。',
        "answer_prefix": "答案：",
    },
}

TS_GUESSING_MULTICHOICE = {
    "en": {
        "instruction_fill_option": "Please fill in the `[MASK]` in option ",
        "instruction_knowledge": "based on your benchmark knowledge.",
        "instruction_rule": "The crucial rule is that you should provide different answer in other options below",
        "header_question": "Question:",
        "header_options": "Options:",
        "footer_reply": "Reply with answer only. Provide only the predicted content for `[MASK]`. No option letter, "
        "no extra text.",
        "answer_prefix": "Answer: ",
    },
    "pt": {
        "instruction_fill_option": "Sua tarefa é prever o conteúdo da lacuna `[MASK]` na opção",
        "instruction_knowledge": "Use seu conhecimento do benchmark para fazer essa previsão.",
        "instruction_rule": "Importante: O conteúdo que você prever para `[MASK]` deve ser diferente do texto das "
        "outras opções (não mascaradas).",
        "header_question": "Pergunta:",
        "header_options": "Opções:",
        "footer_reply": "Forneça apenas o conteúdo previsto para `[MASK]`. Sem a letra da opção, sem texto adicional.",
        "answer_prefix": "Resposta: ",
    },
    "zh": {
        "instruction_fill_option": "您的任务是预测选项中 `[MASK]` 的内容：选项",
        "instruction_knowledge": "请根据您对该基准测试的了解进行预测。",
        "instruction_rule": "重要提示：您为 `[MASK]` 预测的内容必须与其他已填写的选项内容不同。",
        "header_question": "问题：",
        "header_options": "选项：",
        "footer_reply": "请仅提供 `[MASK]` 的预测内容。不要包含选项字母或额外文本。",
        "answer_prefix": "答案： ",
    },
}
