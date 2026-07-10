# Mock search and fetch data for testing and offline fallback

MOCK_SEARCH = {
    "eiffel tower height": [
        {"title": "Eiffel Tower Height - Official Site", "url": "https://official.eiffel.tower/height", "snippet": "The official height of the Eiffel Tower is 330 meters (1,083 ft) including the radio antenna at the top, which was updated in 2022."},
        {"title": "Eiffel Tower - Wikipedia", "url": "https://en.wikipedia.org/wiki/Eiffel_Tower", "snippet": "The Eiffel Tower stands 330 metres tall, about the same height as an 81-storey building. It is located in Paris, France."}
    ],
    "prime minister australia october 2021": [
        {"title": "List of Prime Ministers of Australia", "url": "https://australia.gov/pm-list", "snippet": "Scott Morrison was the Prime Minister of Australia in October 2021, serving from August 2018 until May 2022."},
        {"title": "Scott Morrison - Wikipedia", "url": "https://en.wikipedia.org/wiki/Scott_Morrison", "snippet": "Scott Morrison (born 13 May 1968) is an Australian politician who served as the 30th Prime Minister of Australia from 2018 to 2022."}
    ],
    "melbourne cricket ground capacity": [
        {"title": "Melbourne Cricket Ground - About", "url": "https://mcg.org.au/about", "snippet": "The Melbourne Cricket Ground (MCG) has an official seating capacity of 100,024, making it the largest stadium in the Southern Hemisphere."},
        {"title": "MCG Stadium Guide", "url": "https://stadiumguide.com/mcg", "snippet": "Capacity of the MCG is 100,024. It holds cricket matches and Australian rules football games."}
    ],
    "apple macintosh release date": [
        {"title": "History of the Macintosh - Apple Museum", "url": "https://applemuseum.org/mac-history", "snippet": "The original Macintosh was released on January 24, 1984, with a famous Super Bowl commercial directed by Ridley Scott."},
        {"title": "Byte Magazine Macintosh Review", "url": "https://byte-archive.org/mac-preview", "snippet": "Apple Computer announced the Macintosh on January 22, 1984, and shipped the first units to stores immediately."}
    ],
    "tokyo population 2023": [
        {"title": "Tokyo Metropolitan Government - Census", "url": "https://metro.tokyo.lg.jp/population", "snippet": "The city proper (Tokyo Metropolis) population is estimated at approximately 14.04 million people as of 2023."},
        {"title": "World Cities Population Review 2023", "url": "https://worldpopulationreview.com/tokyo", "snippet": "The Greater Tokyo Area, which is the metropolitan area, is the most populous metro area in the world with 37.4 million residents in 2023."}
    ],
    "university of oxford founding year": [
        {"title": "Oxford University History - Official", "url": "https://ox.ac.uk/about/history", "snippet": "As the oldest university in the English-speaking world, Oxford has no clear date of foundation, but teaching existed in some form in 1096."},
        {"title": "Oxford University Charter", "url": "https://historicalcharters.org/oxford", "snippet": "The University of Oxford was formally chartered by the crown in 1248, though it was active since 1167 when Henry II banned students from Paris."}
    ],
    "elon musk net worth": [
        {"title": "Forbes Billionaires: Elon Musk", "url": "https://forbes.com/profile/elon-musk", "snippet": "Elon Musk's net worth is estimated at $230 billion in 2023, primarily driven by his stakes in Tesla and SpaceX."},
        {"title": "Bloomberg Billionaires Index", "url": "https://bloomberg.com/billionaires/elon-musk", "snippet": "Elon Musk net worth is $228 billion as of mid-2023."}
    ],
    "elon musk wives": [
        {"title": "Elon Musk Relationships - People", "url": "https://people.com/elon-musk-wives", "snippet": "Elon Musk has been married twice: first to Justine Wilson (2000-2008), and then twice to Talulah Riley (2010-2012, 2013-2016)."},
        {"title": "Justine Musk Net Worth and Talulah Riley Net Worth", "url": "https://networthcelebrity.com/musk-wives", "snippet": "Justine Musk's net worth is estimated at $3 million. Talulah Riley's net worth is estimated at $25 million."}
    ],
    "apple stock price": [
        {"title": "AAPL Stock Price - Yahoo Finance", "url": "https://finance.yahoo.com/quote/AAPL", "snippet": "Apple Inc. (AAPL) stock price is $175.50 USD at the close of trading on the latest market day."},
        {"title": "Apple Inc. Investor Relations", "url": "https://investor.apple.com/stock", "snippet": "The current stock price of Apple Inc. is approximately $175.50."}
    ],
    "apple logo history": [
        {"title": "The Evolution of the Apple Logo", "url": "https://logodesignlove.com/apple-logo", "snippet": "The first Apple logo, designed by Ronald Wayne in 1976, depicted Isaac Newton sitting under an apple tree. It was replaced by Rob Janoff's rainbow apple in 1977."}
    ],
    "capital of australia": [
        {"title": "Capital of Australia - Official Govt", "url": "https://australia.gov.au/capital", "snippet": "Canberra is the official capital city of Australia, founded in 1913 as a compromise between rival cities Sydney and Melbourne."},
        {"title": "Canberra - Wikipedia", "url": "https://en.wikipedia.org/wiki/Canberra", "snippet": "Canberra is the capital city of Australia. It is Australia's largest inland city and the eighth-largest city overall."}
    ],
    "fifa world cup 2022 winner": [
        {"title": "FIFA World Cup 2022 Final - FIFA", "url": "https://fifa.com/worldcup-2022", "snippet": "Argentina won the FIFA World Cup 2022 in Qatar, defeating France on penalties after a dramatic 3-3 draw in the final."},
        {"title": "World Cup 2022 - Wikipedia", "url": "https://en.wikipedia.org/wiki/2022_FIFA_World_Cup", "snippet": "The 2022 FIFA World Cup was won by Argentina, who beat France 4-2 on penalties following a 3-3 draw after extra time."}
    ]
}

MOCK_FETCH = {
    "https://official.eiffel.tower/height": "The official height of the Eiffel Tower is 330 meters (1,083 ft) including the radio antenna at the top, which was updated in 2022.",
    "https://en.wikipedia.org/wiki/Eiffel_Tower": "The Eiffel Tower stands 330 metres tall, about the same height as an 81-storey building. It is located in Paris, France.",
    "https://australia.gov/pm-list": "Scott Morrison was the Prime Minister of Australia in October 2021, serving from August 2018 until May 2022.",
    "https://en.wikipedia.org/wiki/Scott_Morrison": "Scott Morrison (born 13 May 1968) is an Australian politician who served as the 30th Prime Minister of Australia from 2018 to 2022.",
    "https://mcg.org.au/about": "The Melbourne Cricket Ground (MCG) has an official seating capacity of 100,024, making it the largest stadium in the Southern Hemisphere.",
    "https://stadiumguide.com/mcg": "Capacity of the MCG is 100,024. It holds cricket matches and Australian rules football games.",
    "https://applemuseum.org/mac-history": "The original Macintosh was released on January 24, 1984, with a famous Super Bowl commercial directed by Ridley Scott.",
    "https://byte-archive.org/mac-preview": "Apple Computer announced the Macintosh on January 22, 1984, and shipped the first units to stores immediately.",
    "https://metro.tokyo.lg.jp/population": "The city proper (Tokyo Metropolis) population is estimated at approximately 14.04 million people as of 2023.",
    "https://worldpopulationreview.com/tokyo": "The Greater Tokyo Area, which is the metropolitan area, is the most populous metro area in the world with 37.4 million residents in 2023.",
    "https://ox.ac.uk/about/history": "As the oldest university in the English-speaking world, Oxford has no clear date of foundation, but teaching existed in some form in 1096.",
    "https://historicalcharters.org/oxford": "The University of Oxford was formally chartered by the crown in 1248, though it was active since 1167 when Henry II banned students from Paris.",
    "https://forbes.com/profile/elon-musk": "Elon Musk's net worth is estimated at $230 billion in 2023, primarily driven by his stakes in Tesla and SpaceX.",
    "https://bloomberg.com/billionaires/elon-musk": "Elon Musk net worth is $228 billion as of mid-2023.",
    "https://people.com/elon-musk-wives": "Elon Musk has been married twice: first to Justine Wilson (2000-2008), and then twice to Talulah Riley (2010-2012, 2013-2016).",
    "https://networthcelebrity.com/musk-wives": "Justine Musk's net worth is estimated at $3 million. Talulah Riley's net worth is estimated at $25 million.",
    "https://finance.yahoo.com/quote/AAPL": "Apple Inc. (AAPL) stock price is $175.50 USD at the close of trading on the latest market day.",
    "https://investor.apple.com/stock": "The current stock price of Apple Inc. is approximately $175.50.",
    "https://logodesignlove.com/apple-logo": "The first Apple logo, designed by Ronald Wayne in 1976, depicted Isaac Newton sitting under an apple tree. It was replaced by Rob Janoff's rainbow apple in 1977.",
    "https://australia.gov.au/capital": "Canberra is the official capital city of Australia, founded in 1913 as a compromise between rival cities Sydney and Melbourne.",
    "https://en.wikipedia.org/wiki/Canberra": "Canberra is the capital city of Australia. It is Australia's largest inland city and the eighth-largest city overall.",
    "https://fifa.com/worldcup-2022": "Argentina won the FIFA World Cup 2022 in Qatar, defeating France on penalties after a dramatic 3-3 draw in the final.",
    "https://en.wikipedia.org/wiki/2022_FIFA_World_Cup": "The 2022 FIFA World Cup was won by Argentina, who beat France 4-2 on penalties following a 3-3 draw after extra time."
}
