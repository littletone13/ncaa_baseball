# The Odds API: Regions and Bookmakers (Historical Odds)

The **historical odds** endpoint accepts the same `regions` parameter as current odds. Any region that returns bookmakers for a sport for **current** odds will return the same bookmakers for **historical** odds (same snapshot logic).

## Regions valid for NCAA baseball (empirically checked)

Checked via current odds API for `baseball_ncaa`, markets=`h2h`:

| Region   | Valid for NCAA baseball? | Bookmakers that return odds |
|----------|--------------------------|-----------------------------|
| **us**   | **Yes**                  | DraftKings, BetMGM, Bovada, FanDuel, BetRivers, etc. |
| **us2**  | **Yes**                  | Bally Bet, betPARX, theScore Bet, Hard Rock Bet, etc. |
| **uk**   | **Yes**                  | **Betway**, **888sport** (sport888) |
| **eu**   | **Yes**                  | **1xBet** (onexbet), **888sport** (sport888) |
| **au**   | **No**                   | Events returned but **no AU bookmakers** offer NCAA baseball (0 books in response). |
| **us_dfs** | Unknown (player props) | Not checked. |
| **us_ex**  | Unknown                 | Not checked. |

So for **historical** NCAA baseball odds you can use **us**, **us2**, **uk**, and **eu**. Adding **uk** or **eu** gives you a few extra books (Betway, 888sport, 1xBet) in addition to the main US set. **au** is not useful for this sport.

---

## Bookmakers by region (API region key → bookmaker keys)

Source: [the-odds-api.com/sports-odds-data/bookmaker-apis.html](https://the-odds-api.com/sports-odds-data/bookmaker-apis.html).  
“Historical odds” = same as current; if a book appears for current odds for a sport, it can appear for historical.

### US (us)

| Bookmaker key   | Bookmaker  |
|-----------------|------------|
| betonlineag     | BetOnline.ag |
| betmgm          | BetMGM |
| betrivers       | BetRivers |
| betus           | BetUS |
| bovada          | Bovada |
| williamhill_us  | Caesars (paid) |
| draftkings      | DraftKings |
| fanatics        | Fanatics (paid) |
| fanduel         | FanDuel |
| lowvig          | LowVig.ag |
| mybookieag      | MyBookie.ag |

### US2 (us2)

| Bookmaker key   | Bookmaker  |
|-----------------|------------|
| ballybet        | Bally Bet |
| betanysports    | BetAnything |
| betparx         | betPARX |
| espnbet         | theScore Bet (formerly ESPN Bet) |
| fliff           | Fliff |
| hardrockbet     | Hard Rock Bet |
| rebet           | ReBet (paid) |

### US DFS (us_dfs) – player props

| Bookmaker key   | Bookmaker  |
|-----------------|------------|
| betr_us_dfs     | Betr Picks |
| pick6           | DraftKings Pick6 |
| prizepicks      | PrizePicks |
| underdog        | Underdog Fantasy |

### US Exchanges (us_ex)

| Bookmaker key   | Bookmaker  |
|-----------------|------------|
| betopenly       | BetOpenly |
| kalshi          | Kalshi |
| novig           | Novig |
| polymarket      | Polymarket |
| prophetx        | ProphetX |

### UK (uk)

| Bookmaker key   | Bookmaker  |
|-----------------|------------|
| sport888        | 888sport |
| betfair_ex_uk   | Betfair Exchange |
| betfair_sb_uk   | Betfair Sportsbook |
| betvictor       | Bet Victor |
| betway          | Betway |
| boylesports     | BoyleSports |
| casumo          | Casumo |
| coral           | Coral |
| grosvenor       | Grosvenor |
| ladbrokes_uk    | Ladbrokes |
| leovegas        | LeoVegas |
| livescorebet    | LiveScore Bet |
| matchbook       | Matchbook |
| paddypower      | Paddy Power |
| skybet          | Sky Bet |
| smarkets        | Smarkets |
| unibet_uk       | Unibet |
| virginbet       | Virgin Bet |
| williamhill     | William Hill (UK) |

### EU (eu)

| Bookmaker key   | Bookmaker  |
|-----------------|------------|
| onexbet         | 1xBet |
| sport888        | 888sport |
| betclic_fr      | Betclic (FR) |
| betanysports    | BetAnySports |
| betfair_ex_eu   | Betfair Exchange |
| betonlineag     | BetOnline.ag |
| betsson         | Betsson |
| codere_it       | Codere (IT) |
| betvictor       | Bet Victor |
| coolbet         | Coolbet |
| everygame       | Everygame |
| gtbets          | GTbets |
| leovegas_se     | LeoVegas (SE) |
| marathonbet     | Marathon Bet |
| matchbook       | Matchbook |
| mybookieag      | MyBookie.ag |
| nordicbet       | NordicBet |
| parionssport_fr | Parions Sport (FR) |
| pinnacle        | Pinnacle |
| pmu_fr          | PMU (FR) |
| suprabets       | Suprabets |
| tipico_de       | Tipico (DE) |
| unibet_fr       | Unibet (FR) |
| unibet_it       | Unibet (IT) |
| unibet_nl       | Unibet (NL) |
| unibet_se       | Unibet (SE) |
| williamhill     | William Hill |
| winamax_de      | Winamax (DE) |
| winamax_fr      | Winamax (FR) |

### AU (au)

| Bookmaker key   | Bookmaker  |
|-----------------|------------|
| betfair_ex_au   | Betfair Exchange |
| betr_au         | Betr |
| betright        | Bet Right |
| bet365_au       | Bet365 AU (paid) |
| dabble_au       | Dabble AU (paid) |
| ladbrokes_au    | Ladbrokes |
| neds            | Neds |
| playup          | PlayUp |
| pointsbetau     | PointsBet (AU) |
| sportsbet       | SportsBet |
| tab             | TAB |
| tabtouch        | TABtouch |
| unibet          | Unibet |

---

## Cost (historical)

Cost per request = **10 × (number of regions) × (number of markets)**.  
Example: `regions=us,us2,uk,eu,au` and `markets=h2h,spreads,totals` → 10 × 5 × 3 = **150 credits** per snapshot.

## How to see who returns NCAA baseball historical odds

Run one historical call per region and inspect `bookmaker_lines`:

```bash
# UK
python3 scripts/pull_odds.py --mode historical --date 2026-02-18T18:00:00Z --regions uk --markets h2h --out data/raw/odds/test_uk.jsonl

# EU
python3 scripts/pull_odds.py --mode historical --date 2026-02-18T18:00:00Z --regions eu --markets h2h --out data/raw/odds/test_eu.jsonl

# AU
python3 scripts/pull_odds.py --mode historical --date 2026-02-18T18:00:00Z --regions au --markets h2h --out data/raw/odds/test_au.jsonl
```

Then check which `bookmaker_key` values appear in the output. That is the definitive list of bookmakers in that region that returned historical odds for NCAA baseball for that snapshot.
