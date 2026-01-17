SECTOR_CORE = {
        "KRE": ["TFC", "PNC", "USB", "FITB", "KEY", "HBAN", "RF", "CFG", "ZION", "MTB"],  # Regional Banks
        "SMH": ["NVDA", "TSM", "AVGO", "AMD", "ASML", "MU", "AMAT", "LRCX", "KLAC", "INTC"],  # Semiconductors
        "SOXX": ["NVDA", "MU", "AMD", "AMAT", "AVGO", "LRCX", "KLAC", "NXPI", "INTC", "ASML"],  # Semiconductors
        "SPY": ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "BRK.B", "AVGO", "TSLA", "JPM"],  # S&P 500
        "XLB": ["LIN", "FCX", "NEM", "SHW", "APD", "ECL", "DOW", "DD", "NUE", "MLM"],  # Materials
        "XLC": ["META", "GOOGL", "GOOG", "NFLX", "TMUS", "VZ", "T", "DIS", "CMCSA", "CHTR"],  # Communication Services
        "XLE": ["XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "VLO", "PSX", "HAL"],  # Energy
        "XLF": ["BRK.B", "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "AXP"],  # Financials
        "XLI": ["GE", "CAT", "RTX", "BA", "LMT", "NOC", "DE", "ETN", "HON", "GD"],  # Industrials
        "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "CRM", "ORCL", "AMD", "ADBE", "CSCO", "QCOM"],  # Technology
        "XLP": ["WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "MDLZ", "CL", "KMB"],  # Consumer Staples
        "XLRE": ["PLD", "AMT", "EQIX", "WELL", "SPG", "O", "VTR", "PSA", "CCI", "AVB"],  # Real Estate
        "XLU": ["NEE", "SO", "DUK", "AEP", "EXC", "SRE", "XEL", "ED", "EIX", "PEG"],  # Utilities
        "XLV": ["LLY", "JNJ", "ABBV", "UNH", "MRK", "PFE", "TMO", "AMGN", "BMY", "MDT"],  # Health Care
        "XLY": ["AMZN", "TSLA", "HD", "MCD", "BKNG", "LOW", "NKE", "TJX", "SBUX", "RCL"],  # Consumer Discretionary
}

THEMATIC_SECTORS = {
        "HACK": ["AVGO", "CSCO", "NOC", "GD", "PANW", "CRWD", "OKTA", "NET", "FTNT", "FFIV"],  # Cybersecurity
        "IBB": ["VRTX", "GILD", "AMGN", "REGN", "ALNY", "INSM", "ARGX", "NTRA", "BIIB", "ONC"],  # Biotech (Large/Mid)
        "IGV": ["MSFT", "PLTR", "ORCL", "CRM", "APP", "INTU", "PANW", "ADBE", "CRWD", "NOW"],  # Software
        "ITA": ["GE", "RTX", "BA", "LMT", "LHX", "NOC", "HWM", "TDG", "AXON", "GD"],  # Aerospace & Defense
        "XBI": ["RVMD", "MRNA", "FOLD", "KRYS", "MIRM", "ROIV", "HALO", "PRAX", "EXEL", "INCY"],  # Biotech (Equal-weighted)
        "XME": ["HL", "AA", "UEC", "CDE", "FCX", "HCC", "RGLD", "NEM", "BTU", "CNR"],  # Metals & Mining
        "XOP": ["VG", "TPL", "CVX", "XOM", "VLO", "OXY", "COP", "MUR", "CRC", "DINO"],  # Oil & Gas E&P
}

GLOBAL_MACRO = {
        "DIA": ["GS", "CAT", "MSFT", "HD", "AXP", "SHW", "UNH", "AMGN", "V", "JPM"],  # Dow 30
        "EEM": ["TSM", "TCEHY", "SSNLF", "BABA", "RELIANCE.NS", "MEITF", "ICICIBANK.NS", "INFY", "000660.KS", "PDD"],  # Emerging Markets
        "EFA": ["ASML", "RHHBY", "NVS", "AZN", "HSBC", "SHEL", "TM", "SAP", "NESN.SW", "NOVN.SW"],  # Developed ex-US
        "EWJ": ["TM", "MUFG", "SMFG", "SONY", "HMC", "NTTYY", "MFG", "TOELY", "KDDIY", "CAJ"],  # Japan
        "EWT": ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "3711.TW", "2891.TW", "2881.TW", "2345.TW", "2382.TW", "2882.TW"],  # Taiwan
        "EWZ": ["NU", "VALE3.SA", "ITUB4.SA", "PETR4.SA", "PETR3.SA", "BBDC4.SA", "B3SA3.SA", "WEGE3.SA", "EMBR3.SA", "ABEV3.SA"],  # Brazil
        "FXI": ["TCEHY", "BABA", "MEITF", "XIACF", "JD", "BIDU", "CICHY", "IDCBY", "PNGAY", "NTES"],  # China Large-Cap
        "GLD": ["XAUUSD", "GC=F", "IAU", "BAR", "SGOL", "PHYS", "GLDM", "GOLD", "NEM", "AEM"],  # Gold
        "INDA": ["HDFCBANK.NS", "RELIANCE.NS", "ICICIBANK.NS", "INFY", "BHARTIARTL.NS", "M&M.NS", "AXISBANK.NS", "TCS.NS", "BAJFINANCE.NS", "LT.NS"],  # India
        "IWM": ["SMCI", "PLTR", "AXON", "CVNA", "RIVN", "SOFI", "RNR", "GTLB", "CELH", "CROX"],  # US Small-Cap
        "MDY": ["CIEN", "COHR", "FLEX", "CW", "LITE", "CASY", "PSTG", "ILMN", "FTI", "KTOS"],  # US Mid-Cap
        "QQQ": ["NVDA", "AAPL", "MSFT", "AMZN", "AVGO", "META", "GOOGL", "TSLA", "COST", "NFLX"],  # Nasdaq 100
        "RSP": ["SNDK", "MRNA", "ALB", "HII", "LRCX", "FCX", "MU", "KLAC", "BA", "LMT"],  # S&P 500 Equal Weight
        "SPY": ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "BRK.B", "AVGO", "TSLA", "JPM"],  # S&P 500
        "TLT": ["UST20Y+", "UST30Y", "UST25Y", "UST22Y", "UST27Y", "UST21Y", "UST24Y", "UST28Y", "UST23Y", "UST26Y"],  # US Long Treasuries
        "TQQQ": ["NVDA", "AAPL", "MSFT", "AMZN", "AVGO", "META", "GOOGL", "TSLA", "COST", "NFLX"],  # Nasdaq 100 3x
        "UUP": ["DXY", "EURUSD", "USDJPY", "GBPUSD", "USDCAD", "USDSEK", "USDCHF", "AUDUSD", "NZDUSD", "USDMXN"],  # US Dollar
        "USO": ["CL=F", "BZ=F", "XOM", "CVX", "COP", "OXY", "SLB", "MPC", "VLO", "PSX"],  # Oil
        "VGK": ["ASML", "SAP", "NVS", "AZN", "SHEL", "HSBC", "TTE", "SNY", "UL", "NOVN.SW"],  # Europe
        "VIXY": ["VX=F", "VIX", "SVXY", "UVXY", "VXX", "SPX", "ES=F", "SKEW", "MOVE", "VVIX"],  # VIX / Volatility
}

COMMODITIES = {
        "GDX": ["NEM", "AEM", "GOLD", "WPM", "AU", "FNV", "KGC", "RGLD", "BVN", "SSRM"],  # Gold Miners
        "GLD": ["XAUUSD", "GC=F", "IAU", "BAR", "SGOL", "PHYS", "GLDM", "GOLD", "NEM", "AEM"],  # Gold
        "SLV": ["XAGUSD", "SI=F", "SIVR", "PSLV", "SLVP", "SIL", "WPM", "PAAS", "HL", "AG"],  # Silver
        "UNG": ["NG=F", "EQT", "CHK", "AR", "SWN", "DVN", "CTRA", "RRC", "CNX", "COG"],  # Natural Gas
        "URA": ["CCJ", "NXE", "UEC", "LEU", "DNN", "UUUU", "SRUUF", "PALAF", "PDN.AX", "BOE.AX"],  # Uranium
        "USO": ["CL=F", "BZ=F", "XOM", "CVX", "COP", "OXY", "SLB", "MPC", "VLO", "PSX"],  # Oil
}

MAG8 = {
        "MAG8": ["MSFT", "AAPL", "GOOGL", "AMZN", "NVDA", "AVGO", "META", "TSLA"],  # Mega-cap Tech
}


CRYPTO_TICKERS = [
    "ARKB",
    "BITB",
    "BMNR",
    "BTC",
    "CLSK",
    "COIN",
    "ETHA",
    "ETHE",
    "ETHU",
    "FBTC",
    "FETH",
    "IBIT",
    "MARA",
    "MSTR",
    "RIOT",
]


SPECULATIVE_TICKERS = [
    "AMD",
    "BE",
    "CRWV",
    "HIMS",
    "HOOD",
    "IREN",
    "JPM",
    "NNE",
    "PLTR",
    "SOFI",
]

RATES_CREDIT = {
        "HYG": [
            "BLK CSH FND TREASURY SL AGENCY",
            "1261229 BC LTD 144A - 10.0 2032-04-15",
            "MEDLINE BORROWER LP 144A - 3.88 2029-04-01",
            "QUIKRETE HOLDINGS INC 144A - 6.38 2032-03-01",
            "CLOUD SOFTWARE GROUP INC 144A - 6.5 2029-03-31",
            "CLOUD SOFTWARE GROUP INC 144A - 9.0 2029-09-30",
            "DISH NETWORK CORP 144A - 11.75 2027-11-15",
            "CCO HOLDINGS LLC 144A - 5.13 2027-05-01",
            "HUB INTERNATIONAL LTD 144A - 7.25 2030-06-15",
            "WULF COMPUTE LLC 144A - 7.75 2030-10-15",
        ],  # High Yield Credit
        "IEF": [
            "TREASURY NOTE - 4.63 2035-02-15",
            "TREASURY NOTE - 4.0 2034-02-15",
            "TREASURY NOTE - 4.38 2034-05-15",
            "TREASURY NOTE (OLD) - 4.25 2035-08-15",
            "TREASURY NOTE - 4.25 2034-11-15",
            "TREASURY NOTE - 4.5 2033-11-15",
            "TREASURY NOTE - 3.88 2034-08-15",
            "TREASURY NOTE (2OLD) - 4.25 2035-05-15",
            "TREASURY NOTE - 3.88 2033-08-15",
            "TREASURY NOTE (OTR) - 4.0 2035-11-15",
        ],  # US Treasuries 7–10Y
        "JNK": [
            "1261229 BC LTD SR SECURED 144A 04/32 10",
            "ECHOSTAR CORP SR SECURED 11/29 10.75",
            "SSI US GOV MONEY MARKET CLASS",
            "CLOUD SOFTWARE GRP INC SR SECURED 144A 03/29 6.5",
            "CLOUD SOFTWARE GRP INC SECURED 144A 09/29 9",
            "MEDLINE BORROWER LP SR SECURED 144A 04/29 3.875",
            "DISH NETWORK CORP SR SECURED 144A 11/27 11.75",
            "QUIKRETE HOLDINGS INC SR SECURED 144A 03/32 6.375",
            "ASURION LLC/ASURION CO SR SECURED 144A 12/32 8",
            "CARNIVAL CORP COMPANY GUAR 144A 08/32 5.75",
        ],  # High Yield Credit
        "LQD": [
            "BLK CSH FND TREASURY SL AGENCY",
            "ANHEUSER-BUSCH COMPANIES LLC - 4.9 2046-02-01",
            "CVS HEALTH CORP - 5.05 2048-03-25",
            "META PLATFORMS INC - 5.5 2045-11-15",
            "T-MOBILE USA INC - 3.88 2030-04-15",
            "BANK OF AMERICA CORP MTN - 5.47 2035-01-23",
            "GOLDMAN SACHS GROUP INC/THE - 6.75 2037-10-01",
            "ABBVIE INC - 3.2 2029-11-21",
            "GOLDMAN SACHS GROUP INC (FXD-FRN) - 5.07 2032-01-21",
            "GOLDMAN SACHS GROUP INC/THE - 5.54 2047-01-21",
        ],  # Investment Grade Credit
        "SHY": [
            "TREASURY NOTE - 4.25 2027-03-15",
            "TREASURY NOTE - 4.0 2027-01-15",
            "TREASURY NOTE - 1.25 2028-04-30",
            "TREASURY NOTE - 4.5 2027-05-15",
            "TREASURY NOTE - 4.5 2027-04-15",
            "TREASURY NOTE - 2.63 2027-05-31",
            "TREASURY NOTE - 1.25 2028-06-30",
            "TREASURY NOTE - 4.13 2027-02-15",
            "TREASURY NOTE - 3.88 2027-03-31",
            "TREASURY NOTE - 3.75 2027-08-15",
        ],  # US Treasuries 1–3Y
        "TIP": [
            "TREASURY (CPI) NOTE - 1.88 2035-07-15",
            "TREASURY (CPI) NOTE - 2.13 2035-01-15",
            "TREASURY (CPI) NOTE - 1.88 2034-07-15",
            "TREASURY (CPI) NOTE - 1.75 2034-01-15",
            "TREASURY (CPI) NOTE - 1.13 2030-10-15",
            "TREASURY (CPI) NOTE - 1.63 2030-04-15",
            "TREASURY (CPI) NOTE - 1.13 2033-01-15",
            "TREASURY (CPI) NOTE - 1.38 2033-07-15",
            "TREASURY (CPI) NOTE - 0.63 2032-07-15",
            "TREASURY (CPI) NOTE - 1.63 2029-10-15",
        ],  # TIPS
}