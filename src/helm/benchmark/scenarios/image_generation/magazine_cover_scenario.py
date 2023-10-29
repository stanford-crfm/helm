from typing import List

from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, TEST_SPLIT


class MagazineCoverScenario(Scenario):
    """
    Prompts to generate magazine cover photos. There are 50 prompts in total.
    Each prompt contains a real headline from one of following magazines:

    - Better Homes & Gardens
    - Cosmopolitan
    - Costco Connection
    - National Geographic
    - Parents
    - Sports Illustrated
    - TIME
    """

    HEADLINES: List[str] = [
        # Better Homes & Gardens
        "Bright ideas: Our favorite ways to make Thanksgiving sparkle",
        "Destination Home: Fresh Ideas for Your Happy Place",
        "Easy Living: More ways to Get Outside This Summer",
        "here comes SUMMER: QUICK & EASY TIPS FOR OUTDOOR GET-TOGETHER",
        "TOUCH OF SPARKLE: Welcoming interiors full of seasonal charm",
        # Cosmopolitan: used the headlines from covers that did not have a single celebrity
        "THE LOVE ISSUE",
        "This is healthy! 11 women on why wellness doesn't have to be one size fits all",
        "Get your NEW beauty fix",
        "The A.I. issue",
        # Costco Connection
        "Queens of the grill",
        "Get the Scoop: A look inside the world of signature nuts",
        "Ultra-marathon man",
        "Hit the road: RVs and campers offer new experiences at every turn",
        "Building a future",
        "Taking a different route: Discovering luxury, relaxation and excitement (slightly) off the beaten path",
        "Healthy habits: Steps to take for better health",
        "Fair farms: A look at two programs that protect those who grow our food",
        # National Geographic
        "The Other Humans: NEANDERTHALS REVEALED",
        "Yellowstone SUPERVOLCANO: WHAT LIES BENEATH THE PARK",
        "PETRA: Ancient City of Stone",
        "THE BIG THAW: Ice on the Run, Seas on the Rise",
        "PANDA, INC.",
        "Secrets of the WHALES",
        "The Greatest Journey Ever Told: THE TRAIL OF OUR DNA",
        "Untold Stories of D-DAY",
        # Parents
        "BOND YOUR SQUAD! 23 WAYS TO SHOW YOUR LOVE",
        "JOY AT HOME! YOUR BEST CHRISTMAS STARTS HERE",
        "GET READY TO LOVE YOUR MOM STYLE",
        "ALL ABOUT THAT BABY",
        "WHAT IT TAKES TO RAISE GOOD PEOPLE",
        "WIN THE SCHOOL YEAR!",
        "RAISE A HEALTHY EATER",
        "MAKE HOLIDAY MAGIC",
        # Sports Illustrated
        "Are You Ready For Some FOOTBALL?",
        "BASEBALL PREVIEW",
        "SOCCER'S NEXT BIG THING",
        "NO EXCUSES: WHY IT'S TIME TO BUY IN ON THE WNBA",
        # TIME
        "Democracy.",
        "Zip It! THE POWER OF SAYING LESS",
        "The BEST INVENTIONS OF 2022",
        "HOW TO DO MORE GOOD",
        "THE OCEANS ISSUE WATER'S UNTAPPED POWER",
        "ENOUGH. WHEN ARE WE GOING TO DO SOMETHING?",
        "THE COLD TRUTH: LESSONS FROM THE MELTING POLES",
        "HOW COVID ENDS",
        "THE WORLD'S 100 GREATEST PLACES",
        "THE HISTORY WARS",
        "THE NEW AMERICAN REVOLUTION",
        "THE OVERDUE AWAKENING",
        "CHINA'S TEST",
    ]

    name = "magazine_cover"
    description = "Prompts to generate magazine cover photos"
    tags = ["text-to-image", "originality"]

    def get_instances(self, _) -> List[Instance]:
        def construct_prompt(headline: str) -> str:
            return f'a photo for a cover of a magazine with the headline: "{headline}"'

        return [
            Instance(Input(text=construct_prompt(headline)), references=[], split=TEST_SPLIT)
            for headline in self.HEADLINES
        ]
