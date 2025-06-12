import os

from helm.benchmark.metrics.image_generation.fractal_dimension.fractal_dimension_util import compute_fractal_dimension


def fractal_dimension_test(image_filename: str, expected_fractal_dimension: float):
    image_path: str = os.path.join(os.path.dirname(__file__), "test_images", image_filename)
    dim: float = compute_fractal_dimension(image_path)
    assert round(dim, 2) == expected_fractal_dimension


# Test case are inspired by https://www.sciencedirect.com/science/article/pii/S0097849303001547
def test_compute_fractal_dimension_cloud():
    # Clouds have a fractal dimension (D) of 1.30-1.33.
    fractal_dimension_test("cloud.png", 1.34)


def test_compute_fractal_dimension_sea_anemone():
    # Sea anemones have a D of 1.6.
    fractal_dimension_test("sea_anemone.png", 1.54)


def test_compute_fractal_dimension_snowflake():
    # Snowflakes have a D of 1.7.
    fractal_dimension_test("snowflakes.png", 1.69)


def test_compute_fractal_dimension_convergence():
    # "Pollock continued to drip paint for a period lasting up to six months, depositing layer upon layer,
    # and gradually creating a highly dense fractal pattern. As a result, the D value of his paintings rose
    # gradually as they neared completion, starting in the range of 1.3–1.5 for the initial springboard layer
    # and reaching a final value as high as 1.9". Convergence was produced in 1952 by Jackson Pollock.
    fractal_dimension_test("convergence.png", 1.83)
