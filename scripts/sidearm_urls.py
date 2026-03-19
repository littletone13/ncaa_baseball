#!/usr/bin/env python3
"""
sidearm_urls.py — Canonical_id → Sidearm athletics domain mapping for NCAA D1 baseball.

Each school's roster URL follows the pattern:
    https://{domain}/sports/baseball/roster

Usage:
    from scripts.sidearm_urls import SIDEARM_URLS

    url = f"https://{SIDEARM_URLS[canonical_id]}/sports/baseball/roster"

Coverage: 308 NCAA D1 baseball programs (2026 season).
Domains verified via web search against official athletics websites.
"""
from __future__ import annotations


# canonical_id → athletics domain
# Roster URL = https://{domain}/sports/baseball/roster
SIDEARM_URLS: dict[str, str] = {
    # ── SEC ─────────────────────────────────────────────────────────────
    "BSB_ALABAMA": "rolltide.com",
    "BSB_ARKANSAS": "arkansasrazorbacks.com",
    "BSB_AUBURN": "auburntigers.com",
    "BSB_FLORIDA": "floridagators.com",
    "BSB_GEORGIA": "georgiadogs.com",
    "BSB_KENTUCKY": "ukathletics.com",
    "BSB_LSU": "lsusports.net",
    "BSB_MISSOURI": "mutigers.com",
    "BSB_OLE_MISS": "olemisssports.com",
    "BSB_OKLAHOMA": "soonersports.com",
    "BSB_SOUTH_CAROLINA": "gamecocksonline.com",
    "BSB_TENNESSEE": "utsports.com",
    "BSB_VANDERBILT": "vucommodores.com",
    "NCAA_614666": "hailstate.com",           # Mississippi St.
    "NCAA_614788": "12thman.com",              # Texas A&M
    "BSB_TEXAS": "texassports.com",

    # ── ACC ──────────────────────────────────────────────────────────────
    "BSB_BOSTON_COLLEGE": "bceagles.com",
    "BSB_CLEMSON": "clemsontigers.com",
    "BSB_DUKE": "goduke.com",
    "NCAA_614606": "seminoles.com",            # Florida St.
    "BSB_GEORGIA_TECH": "ramblinwreck.com",
    "BSB_LOUISVILLE": "gocards.com",
    "NCAA_614661": "hurricanesports.com",      # Miami (FL)
    "BSB_NORTH_CAROLINA": "goheels.com",
    "BSB_NC_STATE": "gopack.com",
    "BSB_NOTRE_DAME": "und.com",
    "NCAA_614710": "pittsburghpanthers.com",   # Pittsburgh
    "BSB_STANFORD": "gostanford.com",
    "BSB_VIRGINIA": "virginiasports.com",
    "BSB_VIRGINIA_TECH": "hokiesports.com",
    "BSB_WAKE_FOREST": "godeacs.com",
    "NCAA_614572": "calbears.com",             # California

    # ── Big Ten ──────────────────────────────────────────────────────────
    "BSB_ILLINOIS": "fightingillini.com",
    "BSB_INDIANA": "iuhoosiers.com",
    "BSB_IOWA": "hawkeyesports.com",
    "BSB_MARYLAND": "umterps.com",
    "BSB_MICHIGAN": "mgoblue.com",
    "NCAA_614662": "msuspartans.com",          # Michigan St.
    "BSB_MINNESOTA": "gophersports.com",
    "BSB_NEBRASKA": "huskers.com",
    "BSB_NORTHWESTERN": "nusports.com",
    "NCAA_614700": "ohiostatebuckeyes.com",    # Ohio St.
    "BSB_OREGON": "goducks.com",
    "NCAA_614708": "gopsusports.com",          # Penn St.
    "BSB_PURDUE": "purduesports.com",
    "BSB_RUTGERS": "scarletknights.com",
    "NCAA_614742": "usctrojans.com",           # Southern California / USC
    "BSB_UCLA": "uclabruins.com",
    "BSB_WASHINGTON": "gohuskies.com",

    # ── Big 12 ───────────────────────────────────────────────────────────
    "BSB_ARIZONA": "arizonawildcats.com",
    "NCAA_614853": "thesundevils.com",         # Arizona St.
    "BSB_BAYLOR": "baylorbears.com",
    "BSB_BYU": "byucougars.com",
    "BSB_CINCINNATI": "gobearcats.com",
    "BSB_HOUSTON": "uhcougars.com",
    "BSB_KANSAS": "kuathletics.com",
    "NCAA_614635": "kstatesports.com",         # Kansas St.
    "NCAA_614775": "okstate.com",              # Oklahoma St.
    "BSB_TCU": "gofrogs.com",
    "BSB_TEXAS_TECH": "texastech.com",
    "BSB_UCF": "ucfknights.com",
    "BSB_UTAH": "utahutes.com",
    "BSB_WEST_VIRGINIA": "wvusports.com",

    # ── American Athletic Conference ─────────────────────────────────────
    "NCAA_614676": "charlotte49ers.com",       # Charlotte
    "BSB_EAST_CAROLINA": "ecupirates.com",
    "NCAA_614604": "fausports.com",            # Fla. Atlantic
    "NCAA_614658": "gotigersgo.com",           # Memphis
    "NCAA_614718": "riceowls.com",             # Rice
    "NCAA_614740": "gousfbulls.com",           # South Fla.
    "NCAA_614762": "tulanegreenwave.com",      # Tulane
    "NCAA_614849": "uabsports.com",            # UAB
    "NCAA_614794": "goutsa.com",               # UTSA
    "NCAA_614817": "goshockers.com",           # Wichita St.

    # ── Conference USA ───────────────────────────────────────────────────
    "NCAA_614764": "dbupatriots.com",          # DBU (Dallas Baptist)
    "NCAA_614595": "bluehens.com",             # Delaware
    "NCAA_614605": "fiusports.com",            # FIU
    "NCAA_614632": "jaxstatesports.com",       # Jacksonville St.
    "NCAA_614831": "ksuowls.com",              # Kennesaw St.
    "NCAA_614645": "latechsports.com",         # Louisiana Tech
    "BSB_LIBERTY": "libertyflames.com",        # Liberty
    "NCAA_614664": "goblueraiders.com",        # Middle Tenn.
    "NCAA_614746": "missouristatebears.com",   # Missouri St.
    "NCAA_614683": "nmstatesports.com",        # New Mexico St.
    "NCAA_614728": "gobearkats.com",           # Sam Houston
    "NCAA_614815": "wkusports.com",            # Western Ky.

    # ── Sun Belt ─────────────────────────────────────────────────────────
    "NCAA_614852": "appstatesports.com",       # App State
    "NCAA_614854": "astateredwolves.com",      # Arkansas St.
    "BSB_COASTAL_CAROLINA": "goccusports.com",
    "NCAA_614612": "gseagles.com",             # Ga. Southern
    "NCAA_614613": "georgiastatesports.com",   # Georgia St.
    "NCAA_614634": "jmusports.com",            # James Madison
    "BSB_LOUISIANA": "ragincajuns.com",
    "NCAA_614652": "herdzone.com",             # Marshall
    "NCAA_614703": "odusports.com",            # Old Dominion
    "NCAA_614737": "usajaguars.com",           # South Alabama
    "NCAA_614780": "southernmiss.com",         # Southern Miss.
    "NCAA_614747": "txstatebobcats.com",       # Texas St.
    "NCAA_614797": "troytrojans.com",          # Troy
    "NCAA_614691": "ulmwarhawks.com",          # ULM

    # ── WCC (West Coast Conference) ──────────────────────────────────────
    "BSB_GONZAGA": "gozags.com",
    "NCAA_614648": "lmulions.com",             # LMU (CA)
    "NCAA_614706": "pacifictigers.com",        # Pacific
    "NCAA_614709": "pepperdinewaves.com",      # Pepperdine
    "NCAA_614711": "portlandpilots.com",       # Portland
    "NCAA_614726": "smcgaels.com",             # Saint Mary's (CA)
    "BSB_SAN_DIEGO": "usdtoreros.com",         # San Diego
    "NCAA_614732": "usfdons.com",              # San Francisco
    "NCAA_614734": "santaclarabroncos.com",    # Santa Clara
    "NCAA_614833": "goseattleu.com",           # Seattle U

    # ── Mountain West ────────────────────────────────────────────────────
    "NCAA_614798": "goairforcefalcons.com",    # Air Force
    "NCAA_614768": "gobulldogs.com",           # Fresno St.
    "NCAA_614830": "gculopes.com",             # Grand Canyon
    "NCAA_614681": "nevadawolfpack.com",       # Nevada
    "NCAA_614684": "golobos.com",              # New Mexico
    "NCAA_614730": "goaztecs.com",             # San Diego St.
    "NCAA_614733": "sjsuspartans.com",         # San Jose St.
    "NCAA_614680": "unlvrebels.com",           # UNLV
    "NCAA_614810": "wsucougars.com",           # Washington St.

    # ── Big East ─────────────────────────────────────────────────────────
    "NCAA_614564": "butlersports.com",         # Butler
    "NCAA_614590": "gocreighton.com",          # Creighton
    "NCAA_614611": "guhoyas.com",              # Georgetown
    "NCAA_614778": "redstormsports.com",       # St. John's (NY)
    "NCAA_614735": "shupirates.com",           # Seton Hall
    "NCAA_614587": "uconnhuskies.com",         # UConn
    "NCAA_614803": "villanova.com",            # Villanova
    "NCAA_614822": "goxavier.com",             # Xavier

    # ── Atlantic 10 ──────────────────────────────────────────────────────
    "NCAA_614592": "davidsonwildcats.com",     # Davidson
    "NCAA_614593": "daytonflyers.com",         # Dayton
    "NCAA_614608": "fordhamsports.com",        # Fordham
    "NCAA_614609": "gomason.com",              # George Mason
    "NCAA_614610": "gwsports.com",             # George Washington
    "NCAA_621265": "goexplorers.com",          # La Salle
    "NCAA_614656": "umassathletics.com",       # Massachusetts
    "NCAA_614717": "gorhody.com",              # Rhode Island
    "NCAA_614719": "richmondspiders.com",      # Richmond
    "NCAA_614723": "gobonnies.com",            # St. Bonaventure
    "NCAA_614724": "sjuhawks.com",             # Saint Joseph's
    "NCAA_614725": "slubillikens.com",         # Saint Louis
    "NCAA_614804": "vcuathletics.com",         # VCU

    # ── MVC (Missouri Valley Conference) ─────────────────────────────────
    "NCAA_614838": "belmontbruins.com",        # Belmont
    "NCAA_614866": "bradleybraves.com",        # Bradley
    "NCAA_614770": "gopurpleaces.com",         # Evansville
    "NCAA_614623": "goredbirds.com",           # Illinois St.
    "NCAA_614626": "gosycamores.com",          # Indiana St.
    "NCAA_614673": "goracers.com",             # Murray St.
    "NCAA_614743": "salukis.com",              # Southern Ill.
    "NCAA_614625": "uicflames.com",            # UIC
    "NCAA_614801": "valpoathletics.com",       # Valparaiso

    # ── SoCon (Southern Conference) ──────────────────────────────────────
    "NCAA_614583": "citadelsports.com",        # The Citadel
    "NCAA_614597": "etsubucs.com",             # ETSU
    "NCAA_614659": "mercerbears.com",          # Mercer
    "NCAA_614729": "samfordsports.com",        # Samford
    "NCAA_614677": "uncgspartans.com",         # UNC Greensboro
    "NCAA_614805": "vmikeydets.com",           # VMI
    "NCAA_614813": "catamountsports.com",      # Western Caro.
    "NCAA_614765": "woffordterriers.com",      # Wofford

    # ── CAA (Coastal Athletic Association) ────────────────────────────────
    "NCAA_614577": "gocamels.com",             # Campbell
    "NCAA_614827": "cofc.edu/athletics",       # Col. of Charleston
    "NCAA_614828": "elonphoenix.com",          # Elon
    "NCAA_614619": "gohofstra.com",            # Hofstra
    "NCAA_614670": "monmouthhawks.com",        # Monmouth
    "NCAA_614688": "ncataggies.com",           # N.C. A&T
    "NCAA_614692": "nuhuskies.com",            # Northeastern
    "NCAA_614751": "stonybrookathletics.com",  # Stony Brook
    "NCAA_614796": "towsontigers.com",         # Towson
    "NCAA_614678": "uncwsports.com",           # UNCW
    "NCAA_614818": "tribeathletics.com",       # William & Mary

    # ── ASUN ─────────────────────────────────────────────────────────────
    "NCAA_614858": "letsgopeay.com",           # Austin Peay
    "NCAA_614754": "bellarmineknights.com",    # Bellarmine
    "NCAA_614825": "ucasports.com",            # Central Ark.
    "NCAA_614599": "ekusports.com",            # Eastern Ky.
    "NCAA_614841": "fgcuathletics.com",        # FGCU
    "NCAA_614633": "judolphins.com",           # Jacksonville
    "NCAA_614840": "lipscombsports.com",       # Lipscomb
    "NCAA_614843": "roarlions.com",            # North Ala.
    "NCAA_614835": "unfospreys.com",           # North Florida
    "NCAA_614757": "queensathletics.com",      # Queens (NC)
    "NCAA_614781": "gohatters.com",            # Stetson
    "NCAA_614784": "gowestgeorgia.com",        # West Ga.

    # ── Southland Conference ─────────────────────────────────────────────
    "NCAA_614839": "goislanders.com",          # A&M-Corpus Christi
    "NCAA_614621": "hcuhuskies.com",           # Houston Christian
    "NCAA_614640": "lamarcardinals.com",       # Lamar University
    "NCAA_614657": "mcneesesports.com",        # McNeese
    "NCAA_614685": "unoprivateers.com",        # New Orleans
    "NCAA_614774": "geauxcolonels.com",        # Nicholls
    "NCAA_614696": "nsudemons.com",            # Northwestern St.
    "NCAA_614741": "lionsports.net",           # Southeastern La.
    "NCAA_614750": "sfajacks.com",             # SFA
    "NCAA_614836": "uiwcardinals.com",         # UIW
    "NCAA_614707": "goutrgv.com",              # UTRGV

    # ── Big South ────────────────────────────────────────────────────────
    "NCAA_614860": "caborathletics.com",       # Charleston So.
    "NCAA_614829": "gwusports.com",            # Gardner-Webb
    "NCAA_614766": "highpointpanthers.com",    # High Point
    "NCAA_614643": "longwoodlancers.com",      # Longwood
    "NCAA_614832": "gobluehose.com",           # Presbyterian
    "NCAA_614716": "radfordathletics.com",     # Radford
    "NCAA_614674": "uncabulldogs.com",         # UNC Asheville
    "NCAA_614837": "upstatespartans.com",      # USC Upstate
    "NCAA_614819": "winthropeagles.com",       # Winthrop

    # ── Summit League ────────────────────────────────────────────────────
    "NCAA_614690": "gobison.com",              # North Dakota St.
    "NCAA_614693": "uncbears.com",             # Northern Colo.
    "NCAA_614679": "omavs.com",                # Omaha
    "NCAA_614776": "orugoldeneagles.com",      # Oral Roberts
    "NCAA_614739": "gojacks.com",              # South Dakota St.
    "NCAA_614785": "goTommies.com",            # St. Thomas (MN)

    # ── Horizon League ───────────────────────────────────────────────────
    "NCAA_614695": "nkunorse.com",             # Northern Ky.
    "NCAA_614699": "goldengrizzlies.com",      # Oakland
    "NCAA_614820": "mkepanthers.com",          # Milwaukee
    "NCAA_614821": "wsuraiders.com",           # Wright St.
    "NCAA_614824": "ysusports.com",            # Youngstown St.

    # ── MAC (Mid-American Conference) ─────────────────────────────────────
    "NCAA_614787": "gozips.com",               # Akron
    "NCAA_614859": "ballstatesports.com",      # Ball St.
    "NCAA_614865": "bgsufalcons.com",          # Bowling Green
    "NCAA_614581": "cmuchippewas.com",         # Central Mich.
    "NCAA_614600": "emueagles.com",            # Eastern Mich.
    "NCAA_614637": "kentstatesports.com",      # Kent St.
    "NCAA_614660": "miamiredhawks.com",        # Miami (OH)
    "NCAA_614694": "niuhuskies.com",           # NIU
    "NCAA_614701": "ohiobobcats.com",          # Ohio
    "NCAA_614795": "utrockets.com",            # Toledo
    "NCAA_614816": "wmubroncos.com",           # Western Mich.

    # ── Big West ─────────────────────────────────────────────────────────
    "NCAA_614565": "gopoly.com",               # Cal Poly
    "NCAA_614566": "gorunners.com",            # CSU Bakersfield
    "NCAA_614567": "fullertontitans.com",      # Cal St. Fullerton
    "NCAA_614568": "longbeachstate.com",       # Long Beach St.
    "NCAA_614569": "gomatadors.com",           # CSUN
    "NCAA_614573": "ucdavisaggies.com",        # UC Davis
    "NCAA_614574": "ucirvinesports.com",       # UC Irvine
    "NCAA_614576": "gohighlanders.com",        # UC Riverside
    "BSB_UC_SANTA_BARBARA": "ucsbgauchos.com",
    "NCAA_614755": "ucsdtritons.com",          # UC San Diego
    "NCAA_614618": "hawaiiathletics.com",      # Hawaii

    # ── SWAC (Southwestern Athletic Conference) ───────────────────────────
    "NCAA_614846": "aamusports.com",           # Alabama A&M
    "NCAA_614847": "bamastatesports.com",      # Alabama St.
    "NCAA_614851": "alcornsports.com",         # Alcorn
    "NCAA_614834": "uapblionsroar.com",        # Ark.-Pine Bluff
    "NCAA_614862": "bcuathletics.com",         # Bethune-Cookman
    "NCAA_614603": "famuathletics.com",        # Florida A&M
    "NCAA_614771": "gogrambling.com",          # Grambling
    "NCAA_614631": "gojsutigers.com",          # Jackson St.
    "NCAA_614667": "mvsu.edu/athletics",       # Mississippi Val.
    "NCAA_614712": "pvpanthers.com",           # Prairie View
    "NCAA_614745": "gojagsports.com",          # Southern U.
    "NCAA_614790": "tsutigers.com",            # Texas Southern

    # ── OVC (Ohio Valley Conference) ──────────────────────────────────────
    "NCAA_614598": "eiupanthers.com",          # Eastern Ill.
    "NCAA_614756": "lindenwoodlions.com",      # Lindenwood
    "NCAA_614856": "lrtrojans.com",            # Little Rock
    "NCAA_614671": "msueagles.com",            # Morehead St.
    "NCAA_614779": "gosoutheast.com",          # Southeast Mo. St.
    "NCAA_614758": "gousi.com",                # Southern Ind.
    "NCAA_614752": "ttusports.com",            # Tennessee Tech
    "NCAA_614753": "utmsports.com",            # UT Martin
    "NCAA_614814": "goleathernecks.com",       # Western Ill.
    "NCAA_614744": "siuecougars.com",          # SIUE

    # ── Patriot League ───────────────────────────────────────────────────
    "NCAA_614763": "goarmywestpoint.com",      # Army West Point
    "NCAA_614563": "bucknellbison.com",        # Bucknell
    "NCAA_614620": "goholycross.com",          # Holy Cross
    "NCAA_614639": "goleopards.com",           # Lafayette
    "NCAA_614641": "lehighsports.com",         # Lehigh
    "NCAA_614799": "navysports.com",           # Navy

    # ── NEC (Northeast Conference) ────────────────────────────────────────
    "NCAA_614579": "ccsudevils.com",           # Central Conn. St.
    "NCAA_614588": "coppinstatesports.com",    # Coppin St.
    "NCAA_614594": "dsuhornets.com",           # Delaware St.
    "NCAA_614602": "fduknights.com",           # FDU
    "NCAA_614760": "lemoynedolphins.com",      # Le Moyne
    "NCAA_614772": "liuathletics.com",         # LIU
    "NCAA_614783": "gomercyhurst.com",         # Mercyhurst
    "NCAA_614655": "umeshawks.com",            # UMES
    "NCAA_615065": "newhavenchargers.com",     # New Haven
    "NCAA_614687": "nsuspartans.com",          # Norfolk St.
    "NCAA_614759": "stonehillskyhawks.com",    # Stonehill
    "NCAA_614808": "wagner.edu/athletics",     # Wagner

    # ── America East ─────────────────────────────────────────────────────
    "NCAA_614863": "bubearcats.com",           # Binghamton
    "NCAA_614869": "bryantbulldogs.com",       # Bryant
    "NCAA_614649": "goblackbears.com",         # Maine
    "NCAA_614682": "njithighlanders.com",      # NJIT
    "NCAA_614850": "ualbanysports.com",        # UAlbany
    "NCAA_614647": "goriverhawks.com",         # UMass Lowell
    "NCAA_614653": "umbcretrievers.com",       # UMBC

    # ── MAAC (Metro Atlantic Athletic Conference) ─────────────────────────
    "NCAA_614578": "gogriffs.com",             # Canisius
    "NCAA_614601": "fairfieldstags.com",       # Fairfield
    "NCAA_614629": "ionagaels.com",            # Iona
    "NCAA_614650": "gojaspers.com",            # Manhattan
    "NCAA_614651": "goredfoxes.com",           # Marist
    "NCAA_614826": "merrimackathletics.com",   # Merrimack
    "NCAA_614672": "mountathletics.com",       # Mount St. Mary's
    "NCAA_614686": "purpleeagles.com",         # Niagara
    "NCAA_614715": "quinnipiacbobcats.com",    # Quinnipiac
    "NCAA_614720": "gobroncs.com",             # Rider
    "NCAA_614722": "sacredheartpioneers.com",  # Sacred Heart
    "NCAA_614727": "saintpeterspeacocks.com",  # Saint Peter's
    "NCAA_614736": "sienasaints.com",          # Siena

    # ── Ivy League ───────────────────────────────────────────────────────
    "NCAA_614868": "brownbears.com",           # Brown
    "NCAA_614586": "gocolumbialions.com",      # Columbia
    "NCAA_614589": "cornellbigred.com",        # Cornell
    "NCAA_614591": "dartmouthsports.com",      # Dartmouth
    "NCAA_614617": "gocrimson.com",            # Harvard
    "NCAA_614777": "pennathletics.com",        # Penn
    "NCAA_614713": "goprincetontigers.com",    # Princeton
    "NCAA_614823": "yalebulldogs.com",         # Yale

    # ── WAC (Western Athletic Conference) ─────────────────────────────────
    "NCAA_614845": "acusports.com",            # Abilene Christian
    "NCAA_614844": "cbulancers.com",           # California Baptist
    "NCAA_614570": "hornetsports.com",         # Sacramento St.
    "NCAA_614761": "tarletonsports.com",       # Tarleton St.
    "NCAA_614792": "utamavs.com",              # UT Arlington
    "NCAA_614786": "utahtech.com/athletics",   # Utah Tech
    "NCAA_614842": "gouvu.com",                # Utah Valley

    # ── DI Independent ───────────────────────────────────────────────────
    "NCAA_614704": "osubeavers.com",           # Oregon St.
}


def get_roster_url(canonical_id: str) -> str | None:
    """Return the full roster URL for a canonical_id, or None if not mapped."""
    domain = SIDEARM_URLS.get(canonical_id)
    if domain is None:
        return None
    return f"https://{domain}/sports/baseball/roster"


def get_all_roster_urls() -> dict[str, str]:
    """Return {canonical_id: full_roster_url} for all mapped teams."""
    return {cid: f"https://{domain}/sports/baseball/roster"
            for cid, domain in SIDEARM_URLS.items()}


if __name__ == "__main__":
    import sys
    print(f"Total teams mapped: {len(SIDEARM_URLS)}", file=sys.stderr)

    # Optionally load canonical teams to check coverage
    try:
        import pandas as pd
        canon = pd.read_csv("data/registries/canonical_teams_2026.csv", dtype=str)
        all_cids = set(canon["canonical_id"].dropna())
        mapped = set(SIDEARM_URLS.keys())
        missing = all_cids - mapped
        print(f"Canonical teams: {len(all_cids)}", file=sys.stderr)
        print(f"Mapped: {len(mapped & all_cids)}", file=sys.stderr)
        print(f"Missing: {len(missing)}", file=sys.stderr)
        if missing:
            print("\nMissing teams:", file=sys.stderr)
            for cid in sorted(missing):
                row = canon[canon["canonical_id"] == cid].iloc[0]
                print(f"  {cid}: {row['team_name']} ({row['conference']})", file=sys.stderr)
    except Exception:
        pass
