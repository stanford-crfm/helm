from typing import List

from .scenario import Scenario, Instance, Input, TEST_SPLIT


class LogosScenario(Scenario):
    """
    Prompts to generate company/brand logos. The prompts were inspired by the descriptions
    of Fortune 100 companies for the year 2022.
    """

    PROMPTS: List[str] = [
        # 1. Walmart
        "a company that operates chain of hypermarkets, discount department stores, and grocery stores",
        # 2. Amazon
        "a technology company that focuses on e-commerce",
        # 3. Apple
        "a technology company that makes smart phones",
        # 4. CVS Health
        "a retail corporation with a chain of drugstores and pharmacies",
        # 5. UnitedHealth Group
        "a healthcare and insurance company",
        # 6. ExxonMobil
        "a oil and gas corporation",
        # 7. Berkshire Hathaway
        "an insurance and manufacturing company",
        # 8. Alphabet
        "a search engine",
        # 9. McKesson
        "a company distributing pharmaceuticals and providing health information technology",
        # 10. AmerisourceBergen
        "",
        # 11. Costco Wholesale
        "",
        # 12. Cigna
        "",
        # 13. AT&T
        "",
        # 14. Microsoft
        "",
        # 15. Cardinal Health
        "",
        # 16. Chevron
        "",
        # 17. Home Depot
        "",
        # 18. Walgreens Boots Alliance
        "",
        # 19. Marathon Petroleum
        "",
        # 20. Elevance Health
        "",
        # 21. Kroger
        "",
        # 22. Ford Motor
        "",
        # 23. Verizon Communications
        "",
        # 24. JPMorgan Chase
        "",
        # 25. General Motors
        "",
        # 26. Centene
        "",
        # 27. Meta Platforms
        "",
        # 28. Comcast
        "",
        # 29. Phillips 66
        "",
        # 30. Valero Energy
        "",
        # 31. Dell Technologies
        "",
        # 32. Target
        "",
        # 33. Fannie Mae
        "",
        # 34. UPS
        "",
        # 35. Lowe's
        "",
        # 36. Bank of America
        "",
        # 37. Johnson & Johnson
        "",
        # 38. Archer Daniels Midland
        "",
        # 39. FedEx
        "",
        # 40. Humana
        "",
        # 41. Wells Fargo
        "",
        # 42. State Farm Insurance
        "",
        # 43. Pfizer
        "",
        # 44. Citigroup
        "",
        # 45. PepsiCo
        "",
        # 46. Intel
        "",
        # 47. Procter & Gamble
        "",
        # 48. General Electric
        "",
        # 49. IBM
        "",
        # 50. MetLife
        "",
        # 51. Prudential Financial
        "",
        # 52. Albertsons
        "",
        # 53. Walt Disney
        "",
        # 54. Energy Transfer
        "",
        # 55. Lockheed Martin
        "",
        # 56. Freddie Mac
        "",
        # 57. Goldman Sachs Group
        "",
        # 58. Raytheon Technologies
        "",
        # 59. HP
        "",
        # 60. Boeing
        "",
        # 61. Morgan Stanley
        "",
        # 62. HCAHealthcare
        "",
        # 63. AbbVie
        "",
        # 64. Dow
        "",
        # 65. Tesla
        "",
        # 66. Allstate
        "",
        # 67. AIG
        "",
        # 68. Best Buy
        "",
        # 69. Charter Communications
        "",
        # 70. Sysco
        "",
        # 71. Merck
        "",
        # 72. New York Life Insurance
        "",
        # 73. Caterpillar
        "",
        # 74. Cisco Systems
        "",
        # 75. TJX
        "",
        # 76. Publix Super Markets
        "",
        # 77. ConocoPhillips
        "",
        # 78. Liberty Mutual Insurance Group
        "",
        # 79. Progressive
        "",
        # 80. Nationwide
        "",
        # 81. Tyson Foods
        "",
        # 82. Bristol-Myers Squibb
        "",
        # 83. Nike
        "",
        # 84. Deere
        "",
        # 85. American Express
        "",
        # 86. Abbott Laboratories
        "",
        # 87. StoneX Group
        "",
        # 88. Plains GP Holdings
        "",
        # 89. Enterprise Products
        "",
        # 90. TIAA
        "",
        # 91. Oracle
        "",
        # 92. Thermo Fisher Scientific
        "",
        # 93. Coca-Cola
        "",
        # 94. General Dynamics
        "",
        # 95. CHS
        "",
        # 96. USAA
        "",
        # 97. Northwestern Mutual
        "",
        # 98. Nucor
        "",
        # 99. Exelon
        "",
        # 100. Massachusetts Mutual Life
        "",
    ]

    name = "logos"
    description = "Prompts to generate brand/company logos"
    tags = ["text-to-image", "originality"]

    def get_instances(self) -> List[Instance]:
        return [Instance(Input(text=prompt), references=[], split=TEST_SPLIT) for prompt in self.PROMPTS]
