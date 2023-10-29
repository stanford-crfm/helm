from typing import List

from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, TEST_SPLIT


class LandingPageScenario(Scenario):
    """
    Prompts to generate landing pages for mobile or web applications.
    Set `medium` to "landing page" in the `AdapterSpec`, which will produce prompts
    in the following format: "a landing page of a <app_type> <platform> application".
    """

    APPLICATION_TYPES: List[str] = [
        "business",
        "design",
        "developer tools",
        "education",
        "entertainment",
        "finance",
        "games",
        "health and fitness",
        "lifestyle",
        "medical",
        "music",
        "news",
        "photo and video",
        "productivity",
        "social networking",
        "sports",
        "travel",
        "weather",
    ]
    PLATFORMS: List[str] = ["mobile", "web"]

    name = "landing_page"
    description = "Prompts to generate landing pages for mobile or web applications."
    tags = ["text-to-image", "originality"]

    def get_instances(self, _) -> List[Instance]:
        return [
            Instance(
                Input(text=f"a landing page of a {app_type} {platform} application"), references=[], split=TEST_SPLIT
            )
            for app_type in self.APPLICATION_TYPES
            for platform in self.PLATFORMS
        ]
