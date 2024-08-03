from typing import Tuple, Dict, Any

from helm.common.optional_dependencies import handle_module_not_found_error

try:
    from selenium import webdriver
    import selenium.common.exceptions
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, suggestions=["image2struct"])


def init_driver(url: str, resolution: Tuple[int, int] = (1920, 1080)) -> webdriver.Chrome:
    """Initialize the WebDriver

    Args:
        url (str): The URL of the website. Usually "http://localhost:{port}".
        resolution (tuple[int, int], optional): The resolution of the WebDriver. Defaults to (1920, 1080).

    Returns:
        webdriver.Chrome: The Chrome WebDriver
    """
    options = webdriver.ChromeOptions()
    options.add_argument(f"--window-size={resolution[0]},{resolution[1]}")
    options.add_argument("--headless")  # Optional: run in headless mode
    options.add_argument("--no-sandbox")  # Optional: for certain environments
    options.add_argument("--disable-dev-shm-usage")  # Optional: overcome limited resource problems
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    return driver


def close_driver(driver: webdriver.Chrome):
    """Close the WebDriver

    Args:
        driver (webdriver.Chrome): The Chrome WebDriver
    """
    driver.quit()


class ScreenshotOptions:
    """A class to store the parameters for taking a screenshot"""

    """The resolution of the screenshot"""
    resolution: Tuple[int, int] = (1920, 1080)

    """The delay between each action in milliseconds"""
    delay_between_each_action_ms: int = 1000


def save_random_screenshot(path: str, port: int, options: ScreenshotOptions = ScreenshotOptions()) -> Dict[str, Any]:
    """Save a screenshot of a random page

    Args:
        path (str): The path to save the screenshot
        port (int): The port to use for the website.
        options (ScreenshotOptions, optional): The options to use for taking the screenshot.
            Defaults to ScreenshotOptions().

    Returns:
        infos (Dict[str, Any]): Additional information about the screenshot

    Raises:
        ValueError: If the path does not end with .png
    """
    if not path.endswith(".png"):
        raise ValueError("The path should end with .png")

    driver: webdriver.Chrome
    try:
        driver = init_driver(url=f"http://localhost:{port}", resolution=options.resolution)
    except selenium.common.exceptions.WebDriverException as e:
        raise Exception(f"Failed to initialize the driver: {e}")
    except Exception as e:
        raise Exception(f"An unknown error occurred while initializing the driver: {e}")

    # Extract the HTML of the page
    html = driver.page_source

    # Take a screenshot of the page
    driver.save_screenshot(path)
    close_driver(driver)

    return {"html": html}
