from dataclasses import dataclass
from typing import List, Callable

from helm.benchmark.scenarios.numeracy_scenario import (
    distance_linear,
    distance_parabola,
    distance_plane,
    distance_paraboloid,
)


TOL = 1e-5  # note: different from TOL in numeracy_scenario.distance_<...> used for checking if real or complex


@dataclass(frozen=True)
class TestCase:
    rel_str: str
    point: List[int]
    dist: float


def check_test_cases(test_cases: List[TestCase], dist_func: Callable[[List[int], str], float]):
    for test_case in test_cases:
        dist = dist_func(test_case.point, test_case.rel_str)
        dist_gt = test_case.dist
        assert abs(dist - dist_gt) < TOL, f"{test_case.rel_str} {test_case.point}"
        # print(f"{test_case.rel_str} {test_case.point} Dist: {dist}\tDist GT: {dist_gt}")


def test_distance_linear():
    test_cases = [
        TestCase(
            "y = 4x + 4", [59, 201], 9.458889376416986
        ),  # https://www.wolframalpha.com/input?i=minimize+sqrt%28%28x+-+59%29%5E2+%2B+%284x+%2B+4+-+201%29%5E2%29
        TestCase("y = x + 3 ", [30, 78], 31.819805153394636),
        TestCase("y = 5x + 4", [-47, 2], 45.69505948719688),
        TestCase("y = 4x + 3", [-65, -255], 0.48507125007266594),
        TestCase("y = 4x + 3", [97, 391], 0.0),
    ]
    check_test_cases(test_cases, distance_linear)


def test_distance_parabola():
    test_cases = [
        TestCase("y = 2x^2 + x + 1", [159, 50000], 1.137499072212397),
        TestCase("y = 2x^2 + 2x + 4", [130, 28390], 11.364547837422966),
        TestCase("y = 2x^2 + x + 4", [53, 10000], 17.4468675121177),
        TestCase(
            "y = 2x^2 + 2x + 2", [35, 1], 34.36171077312826
        ),  # https://www.wolframalpha.com/input?i=minimize+%28x+-+35%29%5E2+%2B+%282x%5E2+%2B+2x+%2B+2+-+1%29%5E2
        TestCase("y = x^2 + x + 2", [197, 39008], 0.0),
    ]
    check_test_cases(test_cases, distance_parabola)


def test_distance_plane():
    test_cases = [
        TestCase(
            "z = 4x + 4y + 1", [-4, 9, 1], 3.481553119113957
        ),  # https://www.wolframalpha.com/input?i=minimize+sqrt%28%28x+%2B+4%29%5E2+%2B+%28y+-+9%29%5E2+%2B+%284x
        # +%2B+4y+%2B+1+-+1%29%5E2%29
        TestCase(
            "z = 3x + 5y + 4", [-10, 4, 3], 1.52127765851133
        ),  # https://www.wolframalpha.com/input?i=minimize+sqrt%28%28x+%2B+10%29%5E2+%2B+%28y+-+4%29%5E2+%2B+%283
        # x+%2B+5y+%2B+4+-+3%29%5E2%29
        TestCase("z = 4x + 3y + 4", [-5, 4, -7], 0.5883484054145521),
        TestCase("z = 3x + 5y + 2", [-7, 10, 0], 5.239956379316803),
        TestCase("z = 5x + 2y + 3", [-2, -1, -9], 0.0),
    ]
    check_test_cases(test_cases, distance_plane)


def test_distance_paraboloid():
    test_cases = [
        TestCase("z = x^2 + y^2 + 2", [0, 0, 2], 0.0),
        TestCase(
            "z = 2x^2 + y^2 + 2", [0, 11, 151], 1.2055445093982982
        ),  # https://www.wolframalpha.com/input?i=minimize+x%5E2+%2B+%28y+-+11%29%5E2+%2B+%28%282x%5E2+%2B+y%5E2+%2B+2%29+-+151%29%5E2  # noqa
        TestCase(
            "z = 2x^2 + 2y^2 + 2", [0, 0, 6], 1.3919410907075054
        ),  # https://www.wolframalpha.com/input?i=minimize+x%5E2+%2By%5E2+%2B+%28%282x%5E2+%2B+2y%5E2+%2B+2%29+-+6%29%5E2  # noqa
        TestCase(
            "z = x^2 + y^2 + 2", [0, 0, 20], 4.2130748865881795
        ),  # https://www.wolframalpha.com/input?i=x%5E2+%2B+y%5E2+%2B+%28%28x%5E2+%2B+y%5E2+%2B+2%29+-+20%29%5E2
        TestCase("z = 2x^2 + xy + y^2 + 4", [6, 19, 519], 0.5290904095503263),
        TestCase("z = 2x^2 + xy + 2y^2 + 3", [0, 14, 380], 0.26248531385619783),
        TestCase("z = x^2 + 2y^2 + 1", [5, 14, 4], 13.354544558906934),
        TestCase("z = x^2 + xy + 2y^2 + 4", [3, 20, 1001], 1.4206031238856873),
        TestCase("z = x^2 + xy + 2y^2 + 4", [0, 0, 55], 51.0),
        TestCase("z = x^2 + xy + 2y^2 + 4", [0, 9, 55], 3.8558889386410757),
        TestCase("z = 2x^2 + 2y^2 + 1", [8, 9, 289], 0.04158555512549898),
        TestCase("z = 2x^2 + 2y^2 + 1", [8, 9, 291], 0.0),
        TestCase("z = x^2 + 2xy + 5y^2 + 4", [0, 9, 55], 5.7150737847649244),
    ]
    check_test_cases(test_cases, distance_paraboloid)
