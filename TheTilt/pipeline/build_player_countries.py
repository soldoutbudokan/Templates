"""
Build config/player_countries.yaml from the player_tilt parquet.

Default country for any IPL player is IN (India). The OVERRIDES dict below
lists every non-Indian player who's appeared in IPL data, keyed by the
display name (`player` column from player_tilt). 2-letter ISO 3166-1
codes used (cricket convention: WI for West Indies as a federation,
EN for England rather than GB; ICC team affiliation, not citizenship,
when the two diverge — e.g. Eoin Morgan = EN despite Irish birth).

Coverage philosophy: identify every foreign player I can recognize from
cricket knowledge. The default-IN fallback only fires for unknown long-
tail names, which in IPL data are virtually always domestic uncapped
players.

Re-run this script whenever new players appear in the data:
    cd TheTilt && ./venv/bin/python pipeline/build_player_countries.py
"""

OVERRIDES = {
    # Australia
    "DA Warner": "AU", "GJ Maxwell": "AU", "PJ Cummins": "AU", "SPD Smith": "AU",
    "SR Watson": "AU", "ML Hayden": "AU", "AC Gilchrist": "AU", "A Symonds": "AU",
    "MEK Hussey": "AU", "RT Ponting": "AU", "JP Faulkner": "AU", "MP Stoinis": "AU",
    "MC Henriques": "AU", "MR Marsh": "AU", "SE Marsh": "AU", "UT Khawaja": "AU",
    "AJ Finch": "AU", "NM Coulter-Nile": "AU", "BCJ Cutting": "AU", "JR Hazlewood": "AU",
    "MA Starc": "AU", "JJ Bumrah": "IN",  # Bumrah is Indian — overriding to be safe
    "CJ Ferguson": "AU", "BAW Mennie": "AU", "KW Richardson": "AU",
    "BJ Hodge": "AU", "AC Voges": "AU", "JL Pattinson": "AU", "TM Head": "AU",
    "AJ Tye": "AU", "DJ Christian": "AU", "DR Sams": "AU", "AT Carey": "AU",
    "JR Behrendorff": "AU", "AJ Turner": "AU", "JP Behrendorff": "AU",
    "BR Dwarshuis": "AU", "DT Christian": "AU", "PSP Handscomb": "AU",
    "MP Stoinis": "AU", "AD Russell": "WI",  # Andre Russell — WI
    "JR Hopes": "AU", "MJ Owen": "AU", "MG Neser": "AU", "MJ Henry": "NZ",
    "M Klinger": "AU", "TD Paine": "AU", "PR Stirling": "IE",
    "BJ Haddin": "AU", "DJ Hussey": "AU", "DJ Bravo": "WI",
    "AB McDonald": "AU", "DA Miller": "ZA", "AC Blizzard": "AU", "GJ Bailey": "AU",
    "PD Collingwood": "EN", "TD Astle": "NZ", "AD Mascarenhas": "EN",
    "MJ Lumb": "EN", "RJ Harris": "AU", "JM Anderson": "EN", "MG Johnson": "AU",
    "DE Bollinger": "AU", "BW Hilfenhaus": "AU", "DJ Pattinson": "AU",
    "MD Mitchell": "NZ", "DA Warner": "AU", "MR Quinn": "AU", "AJ Hosein": "WI",
    "DJ Worrall": "AU", "BR Doggett": "AU", "JD Wildermuth": "AU", "BA Stokes": "EN",
    "JE Burns": "AU", "MJ Henry": "NZ", "PJ Sangwan": "IN", "JC Buttler": "EN",
    "JM Bairstow": "EN", "JE Root": "EN", "ER Dwivedi": "IN", "WA Mota": "IN",
    "EJG Morgan": "EN", "TS Roy": "EN",  # Jason Roy slug TS Roy? actually JJ Roy
    "JJ Roy": "EN", "PR Stirling": "IE", "KP Pietersen": "EN", "JM Vince": "EN",
    "OA Shah": "EN", "RS Bopara": "EN", "DR Smith": "WI", "GR Napier": "EN",
    "Mohammad Asif": "PK", "Sohail Tanvir": "PK", "Younis Khan": "PK",
    "Mohammad Hafeez": "PK", "Salman Butt": "PK", "Misbah-ul-Haq": "PK",
    "Shahid Afridi": "PK", "Umar Gul": "PK", "Shoaib Malik": "PK", "Kamran Akmal": "PK",
    "Shoaib Akhtar": "PK", "Mohammad Yousuf": "PK",
    # New Zealand
    "KS Williamson": "NZ", "BB McCullum": "NZ", "TA Boult": "NZ", "KA Jamieson": "NZ",
    "DP Conway": "NZ", "DJ Mitchell": "NZ", "TWM Latham": "NZ", "GD Phillips": "NZ",
    "LH Ferguson": "NZ", "IS Sodhi": "NZ", "MJ Santner": "NZ", "C Munro": "NZ",
    "CJ Anderson": "NZ", "DL Vettori": "NZ", "MS Kasprowicz": "AU",
    "MJ Guptill": "NZ", "JDS Neesham": "NZ", "HM Nicholls": "NZ", "DJG Sammy": "WI",
    "AF Milne": "NZ", "MJ McClenaghan": "NZ", "RR Powell": "WI", "TG Southee": "NZ",
    "NT Broom": "NZ", "GH Vihari": "IN",  # Hanuma Vihari — IN
    "JEC Franklin": "NZ", "BJ Arnel": "NZ", "DR Tuffey": "NZ", "Nathan McCullum": "NZ",
    "NL McCullum": "NZ", "DG Brownlie": "NZ", "TS Mills": "EN", "Anderson (CJ)": "NZ",
    "AS Joshi": "IN", "LRPL Taylor": "NZ", "RM Hart": "NZ", "ML Cleaver": "NZ",
    # South Africa
    "AB de Villiers": "ZA", "F du Plessis": "ZA", "DW Steyn": "ZA", "M Morkel": "ZA",
    "JH Kallis": "ZA", "GC Smith": "ZA", "JP Duminy": "ZA", "HM Amla": "ZA",
    "Q de Kock": "ZA", "K Rabada": "ZA", "AK Markram": "ZA", "T Shamsi": "ZA",
    "A Nortje": "ZA", "L Ngidi": "ZA", "M Jansen": "ZA", "DJL Brevis": "ZA",
    "G Coetzee": "ZA", "T Stubbs": "ZA", "H Klaasen": "ZA", "RR Hendricks": "ZA",
    "WD Parnell": "ZA", "RR Rossouw": "ZA", "PWA Mulder": "ZA", "OG Coetzee": "ZA",
    "M Morkel": "ZA", "JA Morkel": "ZA", "DA Steyn": "ZA",
    "VD Philander": "ZA", "AB Dippenaar": "ZA", "RJ Peterson": "ZA",
    "LE Bosman": "ZA", "JJ van der Wath": "ZA", "RE Levi": "ZA", "JA Rudolph": "ZA",
    "WP Saha": "IN",  # Wriddhiman Saha — IN
    "CA Ingram": "ZA", "CK Langeveldt": "ZA", "Imran Tahir": "ZA",
    "SE Bond": "NZ", "MF Maharoof": "LK", "S Badrinath": "IN",
    "L Klusener": "ZA", "M Boucher": "ZA", "Yusuf Pathan": "IN",
    # West Indies
    "CH Gayle": "WI", "KA Pollard": "WI", "DJ Bravo": "WI", "AD Russell": "WI",
    "SP Narine": "WI", "DR Smith": "WI", "LMP Simmons": "WI", "M Samuels": "WI",
    "FH Edwards": "WI", "SS Cottrell": "WI", "CR Brathwaite": "WI", "JO Holder": "WI",
    "N Pooran": "WI", "SO Hetmyer": "WI", "AJ Hosein": "WI", "JA Seales": "WI",
    "R Shepherd": "WI", "SSJ Brooks": "WI", "SD Hope": "WI", "AS Fletcher": "WI",
    "RL Chase": "WI", "S Joseph": "WI", "AS Joseph": "WI", "K Hodge": "WI",
    "M Mindley": "WI", "DJG Sammy": "WI", "RR Powell": "WI",
    "WPUJC Vaas": "LK", "S Chanderpaul": "WI", "DM Bravo": "WI",
    "JE Taylor": "WI", "ER Mendis": "LK", "DJ Jacobs": "WI", "SC Williams": "ZW",
    "RR Sarwan": "WI", "DJ Bravo": "WI",
    # Sri Lanka
    "KC Sangakkara": "LK", "DPMD Jayawardene": "LK", "M Muralitharan": "LK",
    "SL Malinga": "LK", "ST Jayasuriya": "LK", "WU Tharanga": "LK",
    "TM Dilshan": "LK", "WPUJC Vaas": "LK", "AD Mathews": "LK",
    "PWH de Silva": "LK", "C Asalanka": "LK", "MD Shanaka": "LK",
    "M Theekshana": "LK", "PVD Chameera": "LK", "FDM Karunaratne": "LK",
    "Akila Dananjaya": "LK", "A Dananjaya": "LK", "DM Dickwella": "LK",
    "BKG Mendis": "LK", "PVD Chameera": "LK", "KIC Asalanka": "LK",
    "PA Nissanka": "LK", "BMAJ Mendis": "LK", "RAS Lakmal": "LK",
    "NLTC Perera": "LK", "T Perera": "LK", "CK Kapugedera": "LK",
    "DLS de Silva": "LK", "MF Maharoof": "LK", "FDM Karunaratne": "LK",
    "TM Dilshan": "LK", "I Udana": "LK",  "S Randiv": "LK", "BAW Mendis": "LK",
    "KMDN Kulasekara": "LK", "KM Jayawardene": "LK",
    "RPS Suranga": "LK", "DNT Zoysa": "LK",
    # Pakistan (only 2008 + odd cases)
    "Sohail Tanvir": "PK", "Mohammad Asif": "PK", "Salman Butt": "PK",
    "Misbah-ul-Haq": "PK", "Younis Khan": "PK", "Shahid Afridi": "PK",
    "Mohammad Hafeez": "PK", "Umar Gul": "PK", "Shoaib Malik": "PK",
    "Shoaib Akhtar": "PK", "Kamran Akmal": "PK", "Mohammad Yousuf": "PK",
    "Abdur Razzak": "BD",  # confusion — Abdur Razzak is BD
    # Bangladesh
    "Shakib Al Hasan": "BD", "Mushfiqur Rahim": "BD", "Mahmudullah": "BD",
    "Tamim Iqbal": "BD", "Mustafizur Rahman": "BD", "Liton Das": "BD",
    "Mehidy Hasan Miraz": "BD", "Taskin Ahmed": "BD", "Mohammad Ashraful": "BD",
    "Mashrafe Mortaza": "BD", "Soumya Sarkar": "BD",
    # Afghanistan
    "Rashid Khan": "AF", "Mujeeb Ur Rahman": "AF", "Mohammad Nabi": "AF",
    "Naveen-ul-Haq": "AF", "Noor Ahmad": "AF", "Fazalhaq Farooqi": "AF",
    "Karim Janat": "AF", "Mohammad Shahzad": "AF", "Azmatullah Omarzai": "AF",
    "Shapoor Zadran": "AF",
    # Nepal
    "Sandeep Lamichhane": "NP",
    # Ireland (some — most Irish-born players represent ENG)
    "PR Stirling": "IE", "KJ O'Brien": "IE", "WTS Porterfield": "IE",
    # Netherlands
    "RE van der Merwe": "NL", "RN ten Doeschate": "NL",
    # Zimbabwe
    "H Streak": "ZW", "GA Lamb": "ZW", "S Matsikenyeri": "ZW", "BRM Taylor": "ZW",
    # USA / others
    "S Netravalkar": "US", "Alzarri Joseph": "WI",
    # England (more)
    "BA Stokes": "EN", "JC Buttler": "EN", "JM Bairstow": "EN", "JJ Roy": "EN",
    "EJG Morgan": "EN", "S Curran": "EN", "TK Curran": "EN", "TM Banton": "EN",
    "LS Livingstone": "EN", "CJ Jordan": "EN", "DJ Willey": "EN", "RJW Topley": "EN",
    "CR Woakes": "EN", "JE Root": "EN", "PD Salt": "EN", "WG Jacks": "EN",
    "HC Brook": "EN", "AU Rashid": "EN", "MM Ali": "EN", "MA Wood": "EN",
    "SW Billings": "EN", "LE Plunkett": "EN", "OE Robinson": "EN",
    "OJ Pope": "EN", "RM Yates": "EN", "DJ Malan": "EN", "AD Hales": "EN",
    "SCJ Broad": "EN", "GR Napier": "EN", "GP Swann": "EN", "SR Watson": "AU",
    "GS Sandhu": "IN",  # Indian
    "TS Mills": "EN", "MS Crane": "EN", "OE Robinson": "EN", "IR Bell": "EN",
    "RS Bopara": "EN", "OA Shah": "EN", "RW Price": "ZW",  # Ray Price
    "L Wright": "EN", "Anderson (CJ)": "NZ",
    "AB McDonald": "AU", "BJ Hodge": "AU", "AC Voges": "AU",
}


# Player_id-keyed overrides for cases where short-name and full-name forms
# both fail to match (different cricsheet conventions, ambiguous short
# names, or first names that look foreign but are actually Indian, e.g.
# Mohammad Kaif = IN). Authoritative — applied last.
ID_OVERRIDES = {
    "14f96089": "AU",   # A Zampa
    "19b9f399": "AU",   # CJ Green
    "25f7b7d6": "EN",   # T Banton (Tom)
    "272d796e": "AU",   # BR Dunk
    "27af6414": "AU",   # BJ Rohrer
    "39f82db3": "AU",   # DJ Harris (Daniel)
    "3c55c703": "EN",   # JG Bethell
    "3eac9d95": "NZ",   # JDP Oram
    "45eda7c8": "AU",   # CA Lynn
    "4663bd23": "NZ",   # TL Seifert
    "6c79c098": "EN",   # DA Payne
    "6c882e9a": "LK",   # PBB Rajapaksa
    "8ee36b18": "LK",   # P Nissanka
    "aa8d28ae": "ZA",   # D Wiese
    "acc1aeda": "IN",   # SP Jackson (Saurashtra)
    "b2a79f17": "AU",   # B Laughlin
    "bdadf7da": "EN",   # JL Denly
    "c654af19": "ZA",   # R McLaren
    "ce4cc4d5": "IN",   # R Ninan
    "d014d5ac": "WI",   # SE Rutherford
    "d2d4bb0a": "AU",   # TR Birt
    "d68e7f48": "WI",   # R Rampaul
    "d84378a4": "IN",   # M Kaif (Mohammad Kaif)
    "dadbdb68": "NZ",   # JA Duffy
    "ddc0828d": "EN",   # A Flintoff
    "df064e1a": "IN",   # Ravi Bishnoi
    "e66732f8": "ZA",   # RD Rickelton
    "e94915e6": "EN",   # SM Curran
    "ee1b6c27": "LK",   # N Thushara
    "ee7d0c82": "AU",   # GD McGrath
    "f1f99156": "AU",   # TH David (represents AU)
    "f3cb53a1": "ZA",   # MV Boucher
    "f836b33d": "EN",   # T Kohler-Cadmore
    "f846de6a": "WI",   # MN Samuels
    "fb66ce1f": "ZA",   # CH Morris
    "64839cb3": "LK",   # M Pathirana (Matheesha)
    "844e79d1": "ZA",   # D Brevis (Dewald)
    "8dc152d1": "ZA",   # D Jansen (Duan)
}


import pandas as pd
from pathlib import Path
import yaml

df = pd.read_parquet('data/processed/player_tilt.parquet')

result = {}
for _, r in df.iterrows():
    pid = r.get('player_id', '')
    name = r['player']
    fn = r.get('full_name', name) or name
    # Order: ID override > name override > full_name override > default IN.
    country = ID_OVERRIDES.get(str(pid)) or OVERRIDES.get(name) or OVERRIDES.get(fn) or 'IN'
    if pid:
        result[str(pid)] = {'name': str(name), 'country': country}

# Sort by player_id for stable output
sorted_keys = sorted(result.keys())
data = {
    'default_country': 'IN',
    'players': {k: result[k] for k in sorted_keys},
}

out = Path('config/player_countries.yaml')
with open(out, 'w') as f:
    f.write("# Player country attributions for flag-icon rendering (issue #72).\n")
    f.write("# default_country = IN — fallback for any player_id not listed below.\n")
    f.write("# Codes are 2-letter ICC team codes (cricket convention):\n")
    f.write("#   IN India · AU Australia · EN England · NZ New Zealand · ZA South Africa\n")
    f.write("#   WI West Indies · LK Sri Lanka · PK Pakistan · BD Bangladesh\n")
    f.write("#   AF Afghanistan · IE Ireland · NL Netherlands · ZW Zimbabwe\n")
    f.write("#   NP Nepal · US United States\n")
    f.write("# Edit by player_id; the `name` field is informational only.\n\n")
    yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

# Stats
from collections import Counter
countries = [v['country'] for v in result.values()]
print(f'Wrote {out} ({len(result)} players)')
print('Country distribution:', dict(Counter(countries).most_common()))
