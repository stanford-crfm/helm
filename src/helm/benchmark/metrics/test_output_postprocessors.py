from helm.benchmark.metrics.output_processors import remove_thinking


def test_remove_thinking():
    assert remove_thinking("Before think<think>think</think>After think") == "Before thinkAfter think"
    assert remove_thinking("Before think<think>think") == "Before think"
    assert remove_thinking("Before think\n<think>\nthink\n</think>\nAfter think") == "Before think\n\nAfter think"
