"""
Build config/player_countries.yaml from the player_tilt parquet.

Resolution order (highest priority first):
    1. ID_OVERRIDES         — id-keyed manual overrides; the only place to put
                              ICC-vs-citizenship divergence cases (e.g. Eoin
                              Morgan = EN despite IE/GB citizenship in
                              Wikidata) and ambiguous shared-display-name IDs.
    2. Wikidata P27         — country of citizenship, fetched by
                              pipeline/download_people.py into
                              data/processed/player_citizenship.json, mapped
                              to ICC code via wikidata_country_map.py.
    3. OVERRIDES (name)     — display-name-keyed legacy fallback for players
                              Wikidata can't resolve (uncapped players, edge
                              cases, etc.).
    4. OVERRIDES (full_name) — full-name-keyed legacy fallback.
    5. Unresolved           — by default, the script exits non-zero and
                              prints the residue for review. Pass
                              --allow-default-in to fall back to IN for
                              unresolved players (legacy behavior, useful
                              for genuinely-uncapped Indian players).

Re-run this whenever new players appear in the data:
    cd TheTilt && ./venv/bin/python pipeline/download_people.py        # warms Wikidata cache
    cd TheTilt && ./venv/bin/python pipeline/build_player_countries.py # regenerates yaml

Codes are 2-letter ICC team codes (cricket convention): IN, AU, EN (not GB),
NZ, ZA, WI (federation), LK, PK, BD, AF, IE, NL, ZW, NP, US.
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


# Player_id-keyed overrides — highest priority. Use this for:
#   (a) ICC-vs-citizenship divergence where Wikidata's P27 gives the wrong
#       cricket affiliation (e.g. JC Archer Wikidata=Barbados but plays for EN;
#       Buttler/Russell/Gayle/Pollard show citizenship=Australia in Wikidata
#       presumably from BBL contracts, but their cricket affiliation is EN/WI).
#   (b) Ambiguous shared display names where (a) the Cricsheet name conflicts
#       with another player or (b) the player has no Wikidata entry at all.
#   (c) First names that look foreign but are Indian (Mohammad Kaif = IN).
ID_OVERRIDES = {
    # Originally hand-curated entries (pre-Wikidata-pipeline) — kept as-is.
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

    # ICC-vs-Wikidata divergence — Wikidata P27 gives the wrong cricket
    # affiliation. Without these, Wikidata wins and assigns the wrong country.
    "5574750c": "EN",   # JC Archer (Wikidata=Barbados; plays for EN)
    "99b75528": "EN",   # JC Buttler (Wikidata=Australia; plays for EN)
    "39f01cdb": "EN",   # KP Pietersen (Wikidata=South Africa; plays for EN)
    "11df3dc8": "EN",   # MJ Lumb (Wikidata=South Africa; played for EN)
    "3edb58fc": "EN",   # AD Mascarenhas (Wikidata=New Zealand; played for EN)
    "611926bc": "EN",   # GR Napier (Wikidata=New Zealand; played for EN)
    "e86754b2": "EN",   # TK Curran (Wikidata=South Africa; plays for EN)
    "7f048519": "EN",   # DJ Willey (Wikidata=New Zealand; plays for EN)
    "1558d83b": "IN",   # GS Sandhu (Wikidata=Australia; Indian)
    "4ba44e19": "LK",   # M Muralitharan (Wikidata=New Zealand; Sri Lankan legend)
    "9868bc75": "LK",   # BMAJ Mendis (Wikidata=Australia; Sri Lankan)
    "0f12f9df": "LK",   # NLTC Perera (Wikidata=Australia; Sri Lankan)
    "05c2ca46": "NL",   # RE van der Merwe (Wikidata=South Africa; plays for NL)
    "4ec07775": "NL",   # RN ten Doeschate (Wikidata=South Africa; played for NL)
    "df5a6881": "NZ",   # DP Conway (Wikidata=South Africa; plays for NZ)
    "64d43928": "PK",   # Sohail Tanvir (Wikidata=Australia; Pakistani)
    "33cb3411": "PK",   # Younis Khan (Wikidata=Australia; Pakistani)
    "9ab63e7b": "PK",   # Mohammad Hafeez (Wikidata=Australia; Pakistani)
    "16dfcc19": "PK",   # Umar Gul (Wikidata=Australia; Pakistani)
    "9a158001": "PK",   # Azhar Mahmood (Wikidata P27 returns NZ; ICC career was Pakistan — issue #221)
    "9f77963a": "AF",   # Gulbadin Naib (Wikidata=Pakistan; Afghan all-rounder/ex-captain — issue #211)
    "bbd41817": "WI",   # AD Russell (Wikidata=Australia; Jamaican/WI)
    "db584dad": "WI",   # CH Gayle (Wikidata=Australia; Jamaican/WI)
    "a757b0d8": "WI",   # KA Pollard (Wikidata=Australia; Trinidadian/WI)
    "9d430b40": "WI",   # SP Narine (Wikidata=Australia; Trinidadian/WI)
    "a84468fe": "WI",   # DJ Jacobs (Wikidata=South Africa; West Indian wicketkeeper)

    # Foreign players Wikidata couldn't resolve (no entry, or P2697 missing).
    # Identified manually from cricket knowledge.
    "9b6e1b3f": "AU",   # J Fraser-McGurk (Jake Fraser-McGurk)
    "9eb1455b": "AU",   # NT Ellis (Nathan Ellis)
    "0ebfb1ad": "WI",   # E Lewis (Evin Lewis)
    "d3a3e82d": "WI",   # AB Barath (Adrian Barath)
    "76388dc8": "WI",   # S Badree (Samuel Badree)
    "529eb9e0": "WI",   # OC McCoy (Obed McCoy)
    "531f0278": "WI",   # K Santokie (Krishmar Santokie)
    "b552a935": "ZA",   # AC Thomas (Alfonso Thomas)
    "f0af99a7": "ZA",   # D Ferreira (Donavon Ferreira)
    "107c26fb": "ZA",   # KT Maphaka (Kwena Maphaka)
    "60aa2db3": "ZA",   # LG Pretorius (Lhuan-dre Pretorius)
    "de7d833e": "LK",   # D Madushanka (Dilshan Madushanka)
    "5750bcb4": "LK",   # E Malinga (Eshan Malinga)
    "03a83c50": "LK",   # V Viyaskanth (Vijayakanth Viyaskanth)
    "9061a703": "IE",   # J Little (Joshua Little)
    "cb9b8664": "NZ",   # W O'Rourke (William O'Rourke)
    "75de770f": "ZW",   # T Taibu (Tatenda Taibu)
}


import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from wikidata_country_map import map_to_icc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--allow-default-in',
        action='store_true',
        help='Fall back to IN for any player not resolved via overrides or Wikidata. '
             'Legacy behavior; off by default so silent misclassifications fail loud.',
    )
    args = parser.parse_args()

    df = pd.read_parquet('data/processed/player_tilt.parquet')

    citizenship_path = Path('data/processed/player_citizenship.json')
    citizenships = {}
    if citizenship_path.exists():
        with open(citizenship_path) as f:
            citizenships = json.load(f)
        print(f'Loaded {len(citizenships)} Wikidata citizenships from {citizenship_path}')
    else:
        print(f'WARNING: {citizenship_path} not found. Run pipeline/download_people.py first '
              'to populate the Wikidata citizenship cache.')

    result = {}
    unresolved = []
    sources = Counter()
    unmapped_citizenships = Counter()

    for _, r in df.iterrows():
        pid = r.get('player_id', '')
        if not pid:
            continue
        name = r['player']
        fn = r.get('full_name', name) or name
        spid = str(pid)

        country = None
        source = None

        if spid in ID_OVERRIDES:
            country, source = ID_OVERRIDES[spid], 'id_override'

        if country is None and spid in citizenships:
            mapped = map_to_icc(citizenships[spid])
            if mapped:
                country, source = mapped, 'wikidata'
            else:
                unmapped_citizenships[citizenships[spid]] += 1

        if country is None and name in OVERRIDES:
            country, source = OVERRIDES[name], 'name_override'

        if country is None and fn in OVERRIDES:
            country, source = OVERRIDES[fn], 'fullname_override'

        if country is None:
            if args.allow_default_in:
                country, source = 'IN', 'default_in'
            else:
                unresolved.append((spid, name, fn, citizenships.get(spid, '<no wikidata>')))
                continue

        sources[source] += 1
        result[spid] = {'name': str(name), 'country': country}

    print(f'\nResolution sources: {dict(sources.most_common())}')
    if unmapped_citizenships:
        print(f'Wikidata labels with no ICC mapping (add to wikidata_country_map.py if needed):')
        for label, n in unmapped_citizenships.most_common():
            print(f'  {label!r}: {n} players')

    if unresolved:
        print(f'\nERROR: {len(unresolved)} players unresolved (no override, no Wikidata mapping):')
        for spid, name, fn, citz in unresolved:
            print(f'  {spid}  name={name!r}  full_name={fn!r}  wikidata_citz={citz!r}')
        print('\nResolve by one of:')
        print('  - Add player_id to ID_OVERRIDES in this file (preferred for ICC-vs-citizenship divergence)')
        print('  - Map the Wikidata citizenship label in pipeline/wikidata_country_map.py')
        print('  - Add display-name to OVERRIDES dict in this file')
        print('  - Re-run with --allow-default-in to fall back to IN for the residue')
        return 1

    sorted_keys = sorted(result.keys())
    data = {
        'default_country': 'IN',
        'players': {k: result[k] for k in sorted_keys},
    }

    out = Path('config/player_countries.yaml')
    with open(out, 'w') as f:
        f.write("# Player country attributions for flag-icon rendering (issue #72).\n")
        f.write("# Generated by pipeline/build_player_countries.py — do not edit by hand;\n")
        f.write("# adjust ID_OVERRIDES / OVERRIDES / wikidata_country_map.py upstream and re-run.\n")
        f.write("# Resolution order: ID_OVERRIDES > Wikidata P27 > name OVERRIDES > full_name OVERRIDES.\n")
        f.write("# default_country = IN — only used when --allow-default-in falls back during build.\n")
        f.write("# Codes are 2-letter ICC team codes (cricket convention):\n")
        f.write("#   IN India · AU Australia · EN England · NZ New Zealand · ZA South Africa\n")
        f.write("#   WI West Indies · LK Sri Lanka · PK Pakistan · BD Bangladesh\n")
        f.write("#   AF Afghanistan · IE Ireland · NL Netherlands · ZW Zimbabwe\n")
        f.write("#   NP Nepal · US United States\n\n")
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

    countries = [v['country'] for v in result.values()]
    print(f'\nWrote {out} ({len(result)} players)')
    print(f'Country distribution: {dict(Counter(countries).most_common())}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
