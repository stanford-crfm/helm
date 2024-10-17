import logging

from . import kpi_edgar_metrics as kem
import helm.benchmark.scenarios.kpi_edgar_scenario as kes


def test_kem_tokenize():

    text = "This is a test ."
    exp_token_list = ["This", "is", "a", "test", "."]
    token_list = kem.tokenize(text)

    logging.debug(token_list)
    assert token_list == exp_token_list

    return


def test_kem_get_tagged_token_dict():

    token_list = ["This", "<kpi>is", "a", "test</kpi>", "."]
    exp_tagged_token_dict = {"kpi": {(1, "is"), (3, "test"), (2, "a")}, "cy": set(), "py": set(), "py1": set()}

    tagged_token_dict = kem.get_tagged_token_dict(token_list)

    logging.debug(tagged_token_dict)
    assert tagged_token_dict == exp_tagged_token_dict

    return


def test_kem_get_tagged_token_dict2():

    token_list = ["This", "<kpi>is</kpi>", "<cy>a", "test</cy>", "."]
    exp_tagged_token_dict = {"kpi": {(1, "is")}, "cy": {(2, "a"), (3, "test")}, "py": set(), "py1": set()}

    tagged_token_dict = kem.get_tagged_token_dict(token_list)

    logging.debug(tagged_token_dict)
    assert tagged_token_dict == exp_tagged_token_dict

    return


def test_kem_get_tagged_token_size_dict():

    token_list = ["This", "<kpi>is</kpi>", "<cy>a", "test</cy>", "."]
    exp_tagged_token_dict = {"kpi": {(1, "is")}, "cy": {(2, "a"), (3, "test")}, "py": set(), "py1": set()}
    exp_tagged_size_dict = {"kpi": 1, "cy": 2, "py": 0, "py1": 0}

    (tagged_token_dict, tagged_size_dict) = kem.get_tagged_token_size_dict(token_list)

    logging.debug(tagged_token_dict)
    logging.debug(tagged_size_dict)

    assert tagged_token_dict == exp_tagged_token_dict
    assert tagged_size_dict == exp_tagged_size_dict

    return


def test_kem_get_intersection():
    gold_set = {(0, "This"), (1, "is"), (2, "a"), (3, "pen")}
    pred_set = {(0, "That"), (1, "is"), (2, "a"), (3, "pen")}
    exp_inter_set = {(1, "is"), (2, "a"), (3, "pen")}
    inter_set = kem.get_intersection(gold_set, pred_set, False)

    logging.debug(inter_set)
    assert inter_set == exp_inter_set
    return


def test_kem_get_intersection_2():
    gold_set = {(0, "This"), (1, "is"), (2, "a"), (3, "pen")}
    pred_set = {(0, "That"), (1, "is"), (2, "a"), (4, "pen")}
    exp_inter_set = {(0, "is"), (0, "a"), (0, "pen")}
    inter_set = kem.get_intersection(gold_set, pred_set, True)

    logging.debug(inter_set)
    assert inter_set == exp_inter_set
    return


def test_kem_get_tagged_size_dict():

    gold_token_list = ["This", "<kpi>is</kpi>", "<cy>a", "test</cy>", "."]
    pred_token_list = ["This", "<kpi>is</kpi>", "<cy>a", "test</cy>", "."]
    # exp_tagged_token_dict = {"kpi": {(1, "is")}, "cy": {(2, "a"), (3, "test")}, "py": set(), "py1": set()}
    # exp_tagged_size_dict = {"kpi": 1, "cy": 2, "py": 0, "py1": 0}
    exp_intersection_size_dict = {"kpi": 1, "cy": 2, "py": 0, "py1": 0}

    tagged_size_dict = kem.get_tagged_size_dict(gold_token_list, pred_token_list, False)

    logging.debug(tagged_size_dict)

    assert tagged_size_dict["kpi"][2] == exp_intersection_size_dict["kpi"]
    assert tagged_size_dict["kpi"][2] == exp_intersection_size_dict["kpi"]

    return


def test_kem_get_tag_and_phrase():
    extracted = "annual revenue (kpi)"
    result = kem.get_tag_and_phrase(extracted)
    logging.debug(result)
    assert result[0] == "kpi"
    assert result[1] == "annual revenue"
    return


def test_kem_get_tag_and_phrase_2():
    extracted = "annual revenue - kpi)"
    result = kem.get_tag_and_phrase(extracted)
    logging.debug(result)
    assert result[0] == ""
    assert result[1] == ""
    return


def test_kem_get_tag_and_phrase_3():
    extracted = "annual [which is, a yearly] revenue [kpi]"
    result = kem.get_tag_and_phrase(extracted, kes.TAG_PAREN_RE)
    logging.debug(result)
    assert result[0] == "kpi"
    assert result[1] == "annual [which is, a yearly] revenue"
    return


def test_kem_tokenize_extraction():
    extracted = "annual revenue (kpi), 364 (cy)"
    result = kem.tokenize_extraction(extracted)
    logging.debug(result)
    assert result[0] == "annual revenue (kpi)"
    assert result[1] == "364 (cy)"


def test_kem_tokenize_extraction_2():
    extracted = "annual (which is a yearly) revenue [kpi], 9,364 [cy]"
    result = kem.tokenize_extraction(extracted, kes.TAG_PAREN_RE)
    logging.debug(result)
    assert result[0] == "annual (which is a yearly) revenue [kpi]"
    assert result[1] == "9,364 [cy]"


def test_kem_get_tagged_token_size_dict_extraction():
    entity_list = ["annual revenue (kpi)", "364 (cy)"]
    result = kem.get_tagged_token_size_dict_extraction(entity_list)
    logging.debug(result)
    assert result[0]["kpi"] == {(0, "annual"), (0, "revenue")}
    assert result[1]["cy"] == 1
    return


def test_kem_get_tagged_size_dict_2():

    gold_token_list = ["annual revenue (kpi)", "364 (cy)"]
    pred_token_list = ["annual revenue (kpi)", "364 (cy)"]
    # exp_tagged_token_dict = {"kpi": {(1, "is")}, "cy": {(2, "a"), (3, "test")}, "py": set(), "py1": set()}
    # exp_tagged_size_dict = {"kpi": 1, "cy": 2, "py": 0, "py1": 0}
    exp_intersection_size_dict = {"kpi": 2, "cy": 1, "py": 0, "py1": 0}

    tagged_size_dict = kem.get_tagged_size_dict(gold_token_list, pred_token_list, False, True)

    logging.debug(tagged_size_dict)

    assert tagged_size_dict["kpi"][2] == exp_intersection_size_dict["kpi"]
    assert tagged_size_dict["kpi"][2] == exp_intersection_size_dict["kpi"]

    return


def test_kem_compute_prrcf1():
    tp = 10
    tn = 60
    fp = 10
    fn = 10
    (pr, rc, f1) = kem.compute_prrcf1(tp, tn, fp, fn)
    logging.debug((pr, rc, f1))
    assert (pr, rc, f1) == (0.5, 0.5, 0.5)


def test_kem_compute_prrcf1_2():
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    (pr, rc, f1) = kem.compute_prrcf1(tp, tn, fp, fn)
    logging.debug((pr, rc, f1))
    assert (pr, rc, f1) == (0, 0, 0)


def test_kem_compute_prrcf1_3():
    tp = 10
    tn = 40
    fp = 30
    fn = 10
    (pr, rc, f1) = kem.compute_prrcf1(tp, tn, fp, fn)
    logging.debug((pr, rc, f1))
    assert (pr, rc, f1) == (0.25, 0.5, float(1) / float(3))


def test_kem_compute_tptnfpfn_adjusted():
    total_token_length = 100
    stats = [20, 40, 10]
    tp = 0.5
    tn = 0.0  # unused.
    fp = 0.75
    fn = 0.5
    tptnfpfn = kem.compute_tptnfpfn_adjusted(stats, total_token_length)
    logging.debug(tptnfpfn)
    assert tptnfpfn == (tp, tn, fp, fn)


def test_kem_compute_tptnfpfn_adjusted_1():
    total_token_length = 100
    stats = [0, 0, 0]
    tp = 0.0
    tn = 0.0  # unused.
    fp = 0.0
    fn = 1.0
    tptnfpfn = kem.compute_tptnfpfn_adjusted(stats, total_token_length)
    logging.debug(tptnfpfn)
    assert tptnfpfn == (tp, tn, fp, fn)


def test_kem_compute_tptnfpfn_modified_adjusted():
    total_token_length = 100
    stats = [20, 40, 10]
    tp = 10
    tn = 50
    fp = 30
    fn = 10
    tptnfpfn = kem.compute_tptnfpfn_modified_adjusted(stats, total_token_length)
    logging.debug(tptnfpfn)
    assert tptnfpfn == (tp, tn, fp, fn)


def test_kem_compute_adjusted_f1():

    total_token_lengh = 100
    tag_stats_dict = {"pos": [20, 20, 10], "neg": [80, 80, 70]}
    exp_macro_f1 = 0.6875

    macro_f1 = kem.compute_adjusted_f1(tag_stats_dict, total_token_lengh, kem.compute_tptnfpfn_adjusted)

    logging.debug(macro_f1)
    assert macro_f1 == exp_macro_f1
    return
