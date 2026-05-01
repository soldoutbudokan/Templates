"""Map Wikidata country-of-citizenship labels to TheTilt's ICC team codes.

The site renders flags from `public/flags/<code>.svg` using cricket conventions
that don't always match ISO 3166-1: WI is a federation across the Caribbean,
EN is England (not GB / UK), and we group all home-nations / Caribbean labels
under their cricket affiliation.

Used by pipeline/build_player_countries.py. Returns None for any label we
don't recognize so the caller can fall back to OVERRIDES dicts or fail loudly.
"""

# Wikidata `?citizenshipLabel` (English) → ICC team code.
COUNTRY_TO_ICC = {
    # Full-member nations
    "India": "IN",
    "Australia": "AU",
    "England": "EN",
    "United Kingdom": "EN",
    "Scotland": "EN",   # No SC flag in flags/; cricket-Scottish IPL players are vanishingly rare and Scotland is part of UK for our purposes
    "Wales": "EN",
    "New Zealand": "NZ",
    "South Africa": "ZA",
    "Pakistan": "PK",
    "Bangladesh": "BD",
    "Sri Lanka": "LK",
    "Afghanistan": "AF",
    "Zimbabwe": "ZW",
    "Ireland": "IE",

    # West Indies federation: any Caribbean Test-affiliated nation maps to WI.
    "Antigua and Barbuda": "WI",
    "Barbados": "WI",
    "Jamaica": "WI",
    "Trinidad and Tobago": "WI",
    "Guyana": "WI",
    "Saint Lucia": "WI",
    "Saint Vincent and the Grenadines": "WI",
    "Saint Kitts and Nevis": "WI",
    "Dominica": "WI",
    "Grenada": "WI",
    "Anguilla": "WI",
    "Montserrat": "WI",
    "British Virgin Islands": "WI",
    "Cayman Islands": "WI",

    # Associate nations that have appeared in IPL squads.
    "Netherlands": "NL",
    "Nepal": "NP",
    "United States": "US",
    "United States of America": "US",
}


def map_to_icc(citizenship):
    """Return ICC team code for a Wikidata citizenship label, or None if unmapped."""
    if not citizenship:
        return None
    return COUNTRY_TO_ICC.get(citizenship.strip())
