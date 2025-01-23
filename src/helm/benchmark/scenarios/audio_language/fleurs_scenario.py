"""Scenarios for audio models"""

from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)
from collections import OrderedDict
from tqdm import tqdm
from datasets import load_dataset
from helm.common.media_object import MediaObject, MultimediaObject


class FLEURSScenario(Scenario):
    """FLEURS Scenario

    The FLEURS (Conneau et al, 2022) dataset is an n-way parallel speech dataset in 102 languages
    built on top of the machine translation FLoRes-101 benchmark, with approximately 12 hours of speech
    supervision per language. The task is to identify the language used from the audio sample
    (the Speech Language Identification task).

    Paper: https://arxiv.org/abs/2205.12446
    Code: https://tensorflow.org/datasets/catalog/xtreme_s

    Citation:
    @inproceedings{conneau2023fleurs,
        title={Fleurs: Few-shot learning evaluation of universal representations of speech},
        author={Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and Axelrod,
        Vera and Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and Bapna, Ankur},
        booktitle={2022 IEEE Spoken Language Technology Workshop (SLT)},
        pages={798--805},
        year={2023},
        organization={IEEE}
        }
    """

    HF_DATASET_NAME = "google/xtreme_s"

    _FLEURS_LANG_TO_ID = OrderedDict(
        [
            ("Afrikaans", "af"),
            ("Amharic", "am"),
            ("Arabic", "ar"),
            ("Armenian", "hy"),
            ("Assamese", "as"),
            ("Asturian", "ast"),
            ("Azerbaijani", "az"),
            ("Belarusian", "be"),
            ("Bengali", "bn"),
            ("Bosnian", "bs"),
            ("Bulgarian", "bg"),
            ("Burmese", "my"),
            ("Catalan", "ca"),
            ("Cebuano", "ceb"),
            ("Mandarin_Chinese", "cmn_hans"),
            ("Cantonese_Chinese", "yue_hant"),
            ("Croatian", "hr"),
            ("Czech", "cs"),
            ("Danish", "da"),
            ("Dutch", "nl"),
            ("English", "en"),
            ("Estonian", "et"),
            ("Filipino", "fil"),
            ("Finnish", "fi"),
            ("French", "fr"),
            ("Fula", "ff"),
            ("Galician", "gl"),
            ("Ganda", "lg"),
            ("Georgian", "ka"),
            ("German", "de"),
            ("Greek", "el"),
            ("Gujarati", "gu"),
            ("Hausa", "ha"),
            ("Hebrew", "he"),
            ("Hindi", "hi"),
            ("Hungarian", "hu"),
            ("Icelandic", "is"),
            ("Igbo", "ig"),
            ("Indonesian", "id"),
            ("Irish", "ga"),
            ("Italian", "it"),
            ("Japanese", "ja"),
            ("Javanese", "jv"),
            ("Kabuverdianu", "kea"),
            ("Kamba", "kam"),
            ("Kannada", "kn"),
            ("Kazakh", "kk"),
            ("Khmer", "km"),
            ("Korean", "ko"),
            ("Kyrgyz", "ky"),
            ("Lao", "lo"),
            ("Latvian", "lv"),
            ("Lingala", "ln"),
            ("Lithuanian", "lt"),
            ("Luo", "luo"),
            ("Luxembourgish", "lb"),
            ("Macedonian", "mk"),
            ("Malay", "ms"),
            ("Malayalam", "ml"),
            ("Maltese", "mt"),
            ("Maori", "mi"),
            ("Marathi", "mr"),
            ("Mongolian", "mn"),
            ("Nepali", "ne"),
            ("Northern-Sotho", "nso"),
            ("Norwegian", "nb"),
            ("Nyanja", "ny"),
            ("Occitan", "oc"),
            ("Oriya", "or"),
            ("Oromo", "om"),
            ("Pashto", "ps"),
            ("Persian", "fa"),
            ("Polish", "pl"),
            ("Portuguese", "pt"),
            ("Punjabi", "pa"),
            ("Romanian", "ro"),
            ("Russian", "ru"),
            ("Serbian", "sr"),
            ("Shona", "sn"),
            ("Sindhi", "sd"),
            ("Slovak", "sk"),
            ("Slovenian", "sl"),
            ("Somali", "so"),
            ("Sorani-Kurdish", "ckb"),
            ("Spanish", "es"),
            ("Swahili", "sw"),
            ("Swedish", "sv"),
            ("Tajik", "tg"),
            ("Tamil", "ta"),
            ("Telugu", "te"),
            ("Thai", "th"),
            ("Turkish", "tr"),
            ("Ukrainian", "uk"),
            ("Umbundu", "umb"),
            ("Urdu", "ur"),
            ("Uzbek", "uz"),
            ("Vietnamese", "vi"),
            ("Welsh", "cy"),
            ("Wolof", "wo"),
            ("Xhosa", "xh"),
            ("Yoruba", "yo"),
            ("Zulu", "zu"),
        ]
    )
    _FLEURS_LANG = sorted(
        [
            "af_za",
            "am_et",
            "ar_eg",
            "as_in",
            "ast_es",
            "az_az",
            "be_by",
            "bn_in",
            "bs_ba",
            "ca_es",
            "ceb_ph",
            "cmn_hans_cn",
            "yue_hant_hk",
            "cs_cz",
            "cy_gb",
            "da_dk",
            "de_de",
            "el_gr",
            "en_us",
            "es_419",
            "et_ee",
            "fa_ir",
            "ff_sn",
            "fi_fi",
            "fil_ph",
            "fr_fr",
            "ga_ie",
            "gl_es",
            "gu_in",
            "ha_ng",
            "he_il",
            "hi_in",
            "hr_hr",
            "hu_hu",
            "hy_am",
            "id_id",
            "ig_ng",
            "is_is",
            "it_it",
            "ja_jp",
            "jv_id",
            "ka_ge",
            "kam_ke",
            "kea_cv",
            "kk_kz",
            "km_kh",
            "kn_in",
            "ko_kr",
            "ckb_iq",
            "ky_kg",
            "lb_lu",
            "lg_ug",
            "ln_cd",
            "lo_la",
            "lt_lt",
            "luo_ke",
            "lv_lv",
            "mi_nz",
            "mk_mk",
            "ml_in",
            "mn_mn",
            "mr_in",
            "ms_my",
            "mt_mt",
            "my_mm",
            "nb_no",
            "ne_np",
            "nl_nl",
            "nso_za",
            "ny_mw",
            "oc_fr",
            "om_et",
            "or_in",
            "pa_in",
            "pl_pl",
            "ps_af",
            "pt_br",
            "ro_ro",
            "ru_ru",
            "bg_bg",
            "sd_in",
            "sk_sk",
            "sl_si",
            "sn_zw",
            "so_so",
            "sr_rs",
            "sv_se",
            "sw_ke",
            "ta_in",
            "te_in",
            "tg_tj",
            "th_th",
            "tr_tr",
            "uk_ua",
            "umb_ao",
            "ur_pk",
            "uz_uz",
            "vi_vn",
            "wo_sn",
            "xh_za",
            "yo_ng",
            "zu_za",
        ]
    )

    # Randomly selected 7 languages from 7 different groups (western_european_we, eastern_european_ee,
    # central_asia_middle_north_african_cmn, sub_saharan_african_ssa, south_asian_sa, south_east_asian_sea,
    # chinese_japanase_korean_cjk) in the FLEURS dataset.
    _FLEURS_TEST_LANG_TO_ID = OrderedDict(
        [
            ("Finnish", "fi"),
            ("English", "en"),
            ("Hebrew", "he"),
            ("Zulu", "zu"),
            ("Bengali", "bn"),
            ("Thai", "th"),
            ("Mandarin_Chinese", "cmn_hans"),
        ]
    )

    name = "fleurs"
    description = "Language identification for seven languages from seven different language groups \
        ([Conneau et al, 2022](https://arxiv.org/abs/2205.12446))."
    tags: List[str] = ["audio", "recognition", "multilinguality"]

    def __init__(self, language: str) -> None:
        super().__init__()

        if language not in FLEURSScenario._FLEURS_TEST_LANG_TO_ID.keys():
            raise ValueError(
                f"Invalid language: {language}. Valid languages are: {FLEURSScenario._FLEURS_TEST_LANG_TO_ID.keys()}"
            )

        self._fleurs_lang_short_to_long = {v: k for k, v in FLEURSScenario._FLEURS_LANG_TO_ID.items()}
        self._fleurs_long_to_lang = {
            self._fleurs_lang_short_to_long["_".join(k.split("_")[:-1]) or k]: k for k in FLEURSScenario._FLEURS_LANG
        }

        self._language: str = language

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        language_category = self._fleurs_long_to_lang[self._language]
        for row in tqdm(
            load_dataset(
                FLEURSScenario.HF_DATASET_NAME,
                name=f"fleurs.{language_category}",
                cache_dir=output_path,
                split=TEST_SPLIT,
                trust_remote_code=True,
            )
        ):
            local_audio_path = row["path"]
            answer = row["transcription"]
            input = Input(
                multimedia_content=MultimediaObject([MediaObject(content_type="audio/wav", location=local_audio_path)])
            )
            references = [Reference(Output(text=answer), tags=[CORRECT_TAG])]
            instances.append(Instance(input=input, references=references, split=TEST_SPLIT))
        return instances
