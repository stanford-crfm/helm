from typing import List

from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, TEST_SPLIT


class TIMEMostSignificantHistoricalFigures(Scenario):
    """
    People from TIME's "The 100 Most Significant Figures in History" list.

    https://ideas.time.com/2013/12/10/whos-biggest-the-100-most-significant-figures-in-history/
    """

    HISTORICAL_FIGURES: List[str] = [
        "Jesus",
        "Napoleon Bonaparte",
        "Muhammad",
        "William Shakespeare",
        "Abraham Lincoln",
        "George Washington",
        "Adolf Hitler",
        "Aristotle",
        "Alexander the Great",
        "Thomas Jefferson",
        "Henry VIII of England",
        "Charles Darwin",
        "Elizabeth I of England",
        "Karl Marx",
        "Julius Caesar",
        "Queen Victoria",
        "Martin Luther",
        "Joseph Stalin",
        "Albert Einstein",
        "Christopher Columbus",
        "Isaac Newton",
        "Charlemagne",
        "Theodore Roosevelt",
        "Wolfgang Amadeus Mozart",
        "Plato",
        "Louis XIV of France",
        "Ludwig van Beethoven",
        "Ulysses S.Grant",
        "Leonardo da Vinci",
        "Augustus",
        "Carl Linnaeus",
        "Ronald Reagan",
        "Charles Dickens",
        "Paul the Apostle",
        "Benjamin Franklin",
        # "George W.Bush",
        "Winston Churchill",
        "Genghis Khan",
        "Charles I of England",
        "Thomas Edison",
        "James I of England",
        "Friedrich Nietzsche",
        "Franklin D.Roosevelt",
        "Sigmund Freud",
        "Alexander Hamilton",
        "Mohandas Karamchand Gandhi",
        "Woodrow Wilson",
        "Johann Sebastian Bach",
        "Galileo Galilei",
        "Oliver Cromwell",
        "James Madison",
        "Gautama Buddha",
        "Mark Twain",
        "Edgar Allan Poe",
        "Joseph Smith, Jr.",
        "Adam Smith",
        "David, King of Israel",
        "George III of the United Kingdom",
        "Immanuel Kant",
        "James Cook",
        "John Adams",
        "Richard Wagner",
        "Pyotr Ilyich Tchaikovsky",
        "Voltaire",
        "Saint Peter",
        "Andrew Jackson",
        "Constantine the Great",
        "Socrates",
        "Elvis Presley",
        "William the Conqueror",
        "John F.Kennedy",
        "Augustine of Hippo",
        "Vincent van Gogh",
        "Nicolaus Copernicus",
        "Vladimir Lenin",
        "Robert E.Lee",
        "Oscar Wilde",
        "Charles II of England",
        "Cicero",
        "Jean-Jacques Rousseau",
        "Francis Bacon",
        "Richard Nixon",
        "Louis XVI of France",
        "Charles V, Holy Roman Emperor",
        "King Arthur",
        "Michelangelo",
        "Philip II of Spain",
        "Johann Wolfgang von Goethe",
        "Ali, founder of Sufism",
        "Thomas Aquinas",
        "Pope John Paul II",
        "René Descartes",
        "Nikola Tesla",
        "Harry S.Truman",
        "Joan of Arc",
        "Dante Alighieri",
        "Otto von Bismarck",
        "Grover Cleveland",
        "John Calvin",
        "John Locke",
    ]

    name = "time_most_significant_historical_figures"
    description = 'People from TIME\'s "The 100 Most Significant Figures in History" list.'
    tags = ["text-to-image", "knowledge"]

    def get_instances(self, _) -> List[Instance]:
        return [
            Instance(Input(text=historical_figure), references=[], split=TEST_SPLIT)
            for historical_figure in self.HISTORICAL_FIGURES
        ]
