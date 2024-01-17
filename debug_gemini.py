from helm.proxy.clients.auto_client import AutoClient
from helm.common.request import Request
from helm.benchmark.config_registry import register_builtin_configs_from_helm_package

credentials = {"googleProjectId": "748532799086", "googleLocation": "us-central1"}

register_builtin_configs_from_helm_package()

client = AutoClient(credentials=credentials, cache_path="./prod_env/cache")

request = Request(
    model_deployment="google/gemini-pro",
    model="google/gemini-pro",
    embedding=False,
    prompt="""Here are some input-output examples. Read the examples carefully to figure out the mapping. The output of the last example is not given, and your job is to figure out what it is.\n\nPassage: \n\nSeason\nEpisodes\nOriginally aired\n\n\nFirst aired\nLast aired\n\n\n\n1\n10\nOctober 31, 2015 (2015-10-31)\nJanuary 2, 2016 (2016-01-02)\n\n\n\n2\n10\nOctober 2, 2016 (2016-10-02)\nDecember 11, 2016 (2016-12-11)\n\n\n\n3\n10\nFebruary 25, 2018 (2018-02-25)\nApril 29, 2018 (2018-04-29)[27]\n\n\n\nQuestion: How many episodes is ash vs evil dead season 3?\nAnswer: 10\n\nPassage: Tyler Perry has confirmed that in A Madea Family Funeral (2018) Madea has another brother named Heathrow (Also played by Perry). A Vietnam war veteran.\n\nQuestion: When does madea\'s family funeral come out?\nAnswer: 2018\n\nPassage: \n\nFranchise\n0 0 Most recent0 0\ndivision title\n0Year0\nSeasons\n\n\nCleveland Browns^\nAFC Central\n1989\n25**\n\n\nDetroit Lions^\nNFC Central\n1993\n24\n\n\nBuffalo Bills^\nAFC East\n1995\n22\n\n\nOakland Raiders^\nAFC West\n2002\n15\n\n\nNew York Jets\nAFC East\n2002\n15\n\n\nTampa Bay Buccaneers\nNFC South\n2007\n10\n\n\nMiami Dolphins^\nAFC East\n2008\n9\n\n\nTennessee Titans\nAFC South\n2008\n9\n\n\nLos Angeles Chargers\nAFC West\n2009\n8\n\n\nChicago Bears\nNFC North\n2010\n7\n\n\nNew York Giants\nNFC East\n2011\n6\n\n\nBaltimore Ravens\nAFC North\n2012\n5\n\n\nSan Francisco 49ers\nNFC West\n2012\n5\n\n\nIndianapolis Colts\nAFC South\n2014\n3\n\n\nCincinnati Bengals\nAFC North\n2015\n2\n\n\nDenver Broncos\nAFC West\n2015\n2\n\n\nWashington Redskins\nNFC East\n2015\n2\n\n\nCarolina Panthers\nNFC South\n2015\n2\n\n\nArizona Cardinals\nNFC West\n2015\n2\n\n\nHouston Texans\nAFC South\n2016\n1\n\n\nDallas Cowboys\nNFC East\n2016\n1\n\n\nGreen Bay Packers\nNFC North\n2016\n1\n\n\nAtlanta Falcons\nNFC South\n2016\n1\n\n\nSeattle Seahawks\nNFC West\n2016\n1\n\n\n2017 Division Champions\n\n\nNew England Patriots\nAFC East\n2017\n—\n\n\nPittsburgh Steelers\nAFC North\n2017\n—\n\n\nJacksonville Jaguars\nAFC South\n2017\n—\n\n\nKansas City Chiefs\nAFC West\n2017\n—\n\n\nPhiladelphia Eagles\nNFC East\n2017\n—\n\n\nMinnesota Vikings\nNFC North\n2017\n—\n\n\nNew Orleans Saints\nNFC South\n2017\n—\n\n\nLos Angeles Rams\nNFC West\n2017\n—\n\n\n\nQuestion: When is the last time the jaguars won a playoff game?\nAnswer: 2017\n\nPassage: The torch-bearing arm was displayed at the Centennial Exposition in Philadelphia in 1876, and in Madison Square Park in Manhattan from 1876 to 1882. Fundraising proved difficult, especially for the Americans, and by 1885 work on the pedestal was threatened by lack of funds. Publisher Joseph Pulitzer, of the New York World, started a drive for donations to finish the project and attracted more than 120,000 contributors, most of whom gave less than a dollar. The statue was built in France, shipped overseas in crates, and assembled on the completed pedestal on what was then called Bedloe\'s Island. The statue\'s completion was marked by New York\'s first ticker-tape parade and a dedication ceremony presided over by President Grover Cleveland.\n\nQuestion: Where was the statue of liberty originally built?\nAnswer: France\n\nPassage: \n\nSeattle Seahawks\n\n\n Current season\n\n\nEstablished June 4, 1974; 43 years ago (1974-06-04)[1]\nFirst season: 1976\nPlay in CenturyLink Field\nSeattle, Washington\nHeadquartered in the Virginia Mason Athletic Center\nRenton, Washington\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nLogo\nWordmark\n\n\n\n\n\nLeague/conference affiliations\n\n\n\n\nNational Football League (1976–present)\n\nAmerican Football Conference (1977–2001)\n\nAFC West (1977–2001)\n\n\nNational Football Conference (1976, 2002–present)\n\nNFC West (1976, 2002–present)\n\n\n\n\n\n\n\n\nCurrent uniform\n\n\n\n\n\n\n\n\n\n\nTeam colors\n\nCollege Navy, Action Green, Wolf Grey[2][3][4]\n              \n\n\nMascot\nBlitz, Boom, Taima the Hawk (live Augur hawk)\n\n\nPersonnel\n\n\nOwner(s)\nPaul Allen\n\n\nChairman\nPaul Allen\n\n\nCEO\nPeter McLoughlin\n\n\nPresident\nPeter McLoughlin\n\n\nGeneral manager\nJohn Schneider\n\n\nHead coach\nPete Carroll\n\n\nTeam history\n\n\n\n\n\nSeattle Seahawks (1976–present)\n\n\n\n\n\nTeam nicknames\n\n\n\n\nThe \'Hawks\nThe Blue Wave (1984–1986)\nThe Legion of Boom (secondary, 2011–present)\n\n\n\n\nChampionships\n\n\n\nLeague championships (1)\n\nSuper Bowl championships (1)\n2013 (XLVIII)\n\n\n\n\n\nConference championships (3)\n\nNFC: 2005, 2013, 2014\n\n\n\n\n\nDivision championships (10)\n\nAFC West: 1988, 1999\nNFC West: 2004, 2005, 2006, 2007, 2010, 2013, 2014, 2016\n\n\n\n\nPlayoff appearances (16)\n\n\n\n\nNFL: 1983, 1984, 1987, 1988, 1999, 2003, 2004, 2005, 2006, 2007, 2010, 2012, 2013, 2014, 2015, 2016\n\n\n\n\nHome fields\n\n\n\n\nKingdome (1976–1999)[A]\nHusky Stadium (2000–2001)[A]\nCenturyLink Field (2002–present)\n\n\n\n\n\nQuestion: When was the last time the seattle seahawks won the superbowl?\nAnswer: 2013\n\nPassage: Annette Strean provided vocals for Blue Man Group\'s cover of "I Feel Love" on their 2003 album The Complex. Venus Hum opened for Blue Man Group on The Complex Rock Tour, and performed with them as well. The band was featured as "rock concert movement number sixty-three." "I Feel Love" was released as a single in 2004.\n\nQuestion: Who sings i feel love with the blue man group?\nAnswer:\n\nPlease provide the output to this last example. It is critical to follow the format of the preceding outputs!\nAnswer:""",
    temperature=0.8,
    num_completions=1,
    top_k_per_token=40,
    max_tokens=400,
    stop_sequences=["###"],
    echo_prompt=False,
    top_p=1,
    presence_penalty=0,
    frequency_penalty=0,
    random=None,
    messages=None,
    multimodal_prompt=None,
    image_generation_parameters=None,
)

response = client.make_request(request)

print(response)
