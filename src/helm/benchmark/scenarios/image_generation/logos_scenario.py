from typing import List

from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, TEST_SPLIT


class LogosScenario(Scenario):
    """
    Prompts to generate logos for brands and companies. The prompts were inspired by Wikipedia descriptions
    of Fortune 100 companies for 2022. Prompts are in the following format: "a logo of <company description>".
    """

    COMPANY_DESCRIPTIONS: List[str] = [
        # 1. Walmart
        "a company that operates a chain of hypermarkets, discount department stores and grocery stores",
        # 2. Amazon
        "a technology company that focuses on e-commerce",
        # 3. Apple
        "a technology company that makes smartphones and personal computers",
        # 4. CVS Health
        "a retail corporation with a chain of drugstores and pharmacies",
        # 5. UnitedHealth Group
        "a healthcare and insurance company",
        # 6. ExxonMobil
        "an oil and gas corporation",
        # 7. Berkshire Hathaway
        "an insurance and manufacturing company",
        # 8. Alphabet
        "a technology company that focuses on search engine technology, online advertising and cloud computing",
        # 9. McKesson
        "a company distributing pharmaceuticals and providing health information technology",
        # 10. AmerisourceBergen
        "a drug wholesale company",
        # 11. Costco Wholesale
        "a corporation that operates big-box retail stores or warehouse clubs",
        # 12. Cigna
        "a managed healthcare and insurance company",
        # 13. AT&T
        "a telecommunications company",
        # 14. Microsoft
        "a corporation that produces computer software, consumer electronics, personal computers and related services",
        # 15. Cardinal Health
        "a company that specializes in the distribution of pharmaceuticals and medical products",
        # 16. Chevron
        "an energy corporation predominantly in oil and gas",
        # 17. Home Depot
        "a retail corporation that sells tools, construction products, appliances, and services",
        # 18. Walgreens Boots Alliance
        "a company that owns pharmacy chains",
        # 19. Marathon Petroleum
        "a petroleum refining, marketing and transportation company",
        # 20. Elevance Health
        "an insurance provider for pharmaceutical, dental, behavioral health, long-term care, and disability plans",
        # 21. Kroger
        "a company that operates supermarkets",
        # 22. Ford Motor
        "a company that sells automobiles and commercial vehicles",
        # 23. Verizon Communications
        "a telecommunications conglomerate",
        # 24. JPMorgan Chase
        "the largest bank",
        # 25. General Motors
        "an automotive manufacturing company",
        # 26. Centene
        "a managed care company",
        # 27. Meta Platforms
        "an online social media and social networking services",
        # 28. Comcast
        "a broadcasting and cable television company",
        # 29. Phillips 66
        "a company that is engaged in refining, transporting, and marketing natural gas liquids",
        # 30. Valero Energy
        "an international manufacturer and marketer of transportation fuels, other petrochemical products",
        # 31. Dell Technologies
        "a technology company that makes personal computers, servers and televisions",
        # 32. Target
        "a big box department store chain",
        # 33. Fannie Mae
        "a corporation whose purpose is to expand the secondary mortgage market",
        # 34. UPS
        "a shipping and receiving company",
        # 35. Lowe's
        "a company specializing in home improvement",
        # 36. Bank of America
        "an investment bank and financial services holding company",
        # 37. Johnson & Johnson
        "a corporation that develops medical devices, pharmaceuticals, and consumer packaged goods",
        # 38. Archer Daniels Midland
        "a food processing and commodities trading corporation",
        # 39. FedEx
        "a freight and package delivery company",
        # 40. Humana
        "a health insurance company",
        # 41. Wells Fargo
        "a financial services company",
        # 42. State Farm Insurance
        "a property and casualty insurance and auto insurance provider",
        # 43. Pfizer
        "a pharmaceutical and biotechnology corporation",
        # 44. Citigroup
        "an investment bank and financial services corporation",
        # 45. PepsiCo
        "a food, snack and beverage corporation",
        # 46. Intel
        "a semiconductor chip manufacturer",
        # 47. Procter & Gamble
        "a consumer good corporation that specializes in personal care and hygiene products",
        # 48. General Electric
        "a company that focuses in power and renewable energy",
        # 49. IBM
        "a company that specializes in computer hardware, middleware, and software",
        # 50. MetLife
        "a provider of insurance, annuities, and employee benefit programs",
        # 51. Prudential Financial
        "a company that provides insurance, retirement planning, investment management",
        # 52. Albertsons
        "a supermarket chain",
        # 53. Walt Disney
        "a mass media and entertainment company",
        # 54. Energy Transfer
        "a company engaged in natural gas and propane pipeline transport",
        # 55. Lockheed Martin
        "an aerospace, arms, defense, information security, and technology corporation",
        # 56. Freddie Mac
        "a company that buys mortgages, pools them, and sells them as a mortgage-backed security",
        # 57. Goldman Sachs Group
        "an investment bank and financial services company",
        # 58. Raytheon Technologies
        "an aerospace and defense manufacturer",
        # 59. HP
        "a company that develops personal computers, printers and related supplies",
        # 60. Boeing
        "a company that sells airplanes, rotorcraft, rockets, satellites, telecommunications equipment, and missiles",
        # 61. Morgan Stanley
        "an investment management and financial services company",
        # 62. HCAHealthcare
        "an operator of health care facilities",
        # 63. AbbVie
        "a biopharmaceutical company",
        # 64. Dow
        "a chemical corporation that manufactures plastics, chemicals and agricultural products",
        # 65. Tesla
        "an automotive and clean energy company",
        # 66. Allstate
        "an insurance company with a slogan: Are you in good hands?",
        # 67. AIG
        "a finance and insurance corporation",
        # 68. Best Buy
        "a consumer electronics retailer",
        # 69. Charter Communications
        "a tv and cable operator",
        # 70. Sysco
        "a corporation that distributes food products, smallwares, kitchen equipment and tabletop items to restaurants",
        # 71. Merck
        "a chemical, pharmaceutical and life sciences company",
        # 72. New York Life Insurance
        "a life insurance company",
        # 73. Caterpillar
        "a construction equipment manufacturer",
        # 74. Cisco Systems
        "a digital communications technology corporation",
        # 75. TJX
        "an off-price department store corporation",
        # 76. Publix Super Markets
        "an employee-owned American supermarket chain",
        # 77. ConocoPhillips
        "a company engaged in hydrocarbon exploration and production",
        # 78. Liberty Mutual Insurance Group
        "a property and casualty insurer",
        # 79. Progressive
        "a commercial auto insurer and insurance company",
        # 80. Nationwide
        "an insurance and financial services companies",
        # 81. Tyson Foods
        "processor of chicken, beef and pork",
        # 82. Bristol-Myers Squibb
        "a pharmaceutical company that manufactures prescription pharmaceuticals and biologics",
        # 83. Nike
        "a company that engages in the manufacturing and sales of footwear, apparel, equipment and accessories",
        # 84. Deere
        "a corporation that manufactures agricultural machinery, heavy equipment, forestry machinery and drivetrains",
        # 85. American Express
        "a financial services corporation specialized in payment cards",
        # 86. Abbott Laboratories
        "a medical devices and health care company",
        # 87. StoneX Group
        "a financial services organization engaged in commercial hedging and global payments",
        # 88. Plains GP Holdings
        "a company engaged in pipeline transport and storage of liquefied petroleum gas and petroleum",
        # 89. Enterprise Products
        "a midstream natural gas and crude oil pipeline company",
        # 90. TIAA
        "a leading provider of financial services",
        # 91. Oracle
        "a computer technology corporation",
        # 92. Thermo Fisher Scientific
        "a supplier of scientific instrumentation, reagents and consumables",
        # 93. Coca-Cola
        "a beverage corporation known for its carbonated soft drink",
        # 94. General Dynamics
        "an aerospace and defense corporation",
        # 95. CHS
        "a cooperative that focuses on food processing and wholesale and farm supply",
        # 96. USAA
        "a financial services group for people and families who serve, or served, in armed forces",
        # 97. Northwestern Mutual
        "a company that provides consultation on wealth and asset income protection",
        # 98. Nucor
        "a producer of steel and related products",
        # 99. Exelon
        "an energy company that provides electricity",
        # 100. Massachusetts Mutual Life
        "a life insurance, disability income insurance and long-term care insurance company",
    ]

    name = "logos"
    description = "Prompts to generate logos for brands and companies"
    tags = ["text-to-image", "originality"]

    def get_instances(self, _) -> List[Instance]:
        return [
            Instance(Input(text=f"a logo of {description}"), references=[], split=TEST_SPLIT)
            for description in self.COMPANY_DESCRIPTIONS
        ]
