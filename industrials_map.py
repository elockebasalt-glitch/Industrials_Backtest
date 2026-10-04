"""Industrials universe and sub-sector mapping.

Built from Industrials_ticker_mapping.xlsx. Bloomberg suffixes stripped and
class shares converted to the Yahoo convention (MOG/A -> MOG-A).

Groups are NOT mutually exclusive. "Electrical" in particular is a cross-cutting
theme whose members all also sit in Multis, Machinery, E&C, Power, HVAC or
Distributors. A ticker can therefore appear in several tabs, benchmarked against
a different composite in each.
"""

BROAD = "XLI"

SUBSECTORS = {
    "Multis": [
        "ALLE", "APG", "ATKR", "AYI", "DD", "EMR", "HON", "ITW", "MMM", "OTIS", "PH",
        "ROK", "RRX", "SWK", "VLTO", "VNT", "ETN", "HUBB", "NVT", "VRT", "APH", "CGNX",
        "KEYS", "TDY", "TEL", "TER", "ZBRA", "AME", "DOV", "FTV", "GGG", "IEX", "IR",
        "NDSN", "FLS", "ITT", "SOLS", "RAL", "FPS", "LFUS"
    ],
    "HVAC": [
        "AAON", "CARR", "JCI", "MOD", "TT", "SPXC", "MAIR"
    ],
    "Building Products": [
        "PNR", "XYL", "LII", "CSL", "MHK", "AOS"
    ],
    "E&C": [
        "ACM", "DY", "EME", "FIX", "FLR", "GVA", "J", "KBR", "PRIM", "PWR", "MTZ",
        "ROAD", "STRL"
    ],
    "Machinery": [
        "AGCO", "CNH", "DE", "CAT", "HRI", "MTW", "OSK", "TEX", "TTC", "URI", "WAB",
        "ATMU", "EPAC", "DCI", "GNRC", "GTES", "JBTM", "KMT", "ESAB", "LECO", "MIDD",
        "RBC", "TKR", "ALSN", "CMI", "PCAR", "CF", "EQPT", "SUNB"
    ],
    "Defense": [
        "GD", "LHX", "LMT", "NOC", "HII", "BAH", "CACI", "PSN", "ESE", "LDOS", "MRCY",
        "MOG-A", "KTOS", "KRMN", "PKE", "NEU", "MPTI"
    ],
    "Aero": [
        "ATI", "BA", "CRS", "CW", "DCO", "FTAI", "GE", "HEI", "HWM", "HXL", "RTX",
        "TDG", "TXT", "WWD", "VSEC", "HONA", "SARO", "AADX", "ARXS", "DPC", "CR"
    ],
    "Transports": [
        "UPS", "CNI", "CP", "CSX", "NSC", "UNP", "KEX", "CHRW", "EXPD", "HUBG", "R",
        "RXO", "JBHT", "KNX", "LSTR", "MRTN", "WERN", "SNDR", "ARCB", "ODFL", "SAIA",
        "TFII", "XPO", "FDX", "GXO", "FDXF"
    ],
    "Distributors": [
        "AIT", "BDC", "BLDR", "CNM", "FAST", "FERG", "GPC", "GWW", "MSM", "POOL", "QXO",
        "SITE", "WCC", "WSO"
    ],
    "Power": [
        "GEV", "XE", "INIO", "EROC", "BE", "MWH"
    ],
    "Electrical": [
        "ETN", "EMR", "HUBB", "NVT", "VRT", "ATKR", "AYI", "APH", "SPXC", "GEV", "BE",
        "PWR", "EME", "FIX", "STRL", "GNRC", "CMI", "CAT", "WCC", "GLW"
    ],
}

ALL_TICKERS = sorted({t for v in SUBSECTORS.values() for t in v})

# a ticker can belong to more than one group
TICKER_TO_GROUPS = {}
for _g, _rows in SUBSECTORS.items():
    for _t in _rows:
        TICKER_TO_GROUPS.setdefault(_t, []).append(_g)
