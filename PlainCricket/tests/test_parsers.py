"""Regression tests for the Cricbuzz parsing logic.

The HTML fixtures here are synthetic but mirror the structures the parsers
target (the same ids, classes and text layout as Cricbuzz's markup). If
Cricbuzz changes markup, refresh these fixtures from saved real pages and
adjust the parsers until the suite passes again.
"""

from bs4 import BeautifulSoup

from index import (
    assign_scores_to_teams,
    classify_status,
    clean_status_text,
    format_match_times,
    format_team_scores,
    parse_batsmen,
    parse_bowlers,
    parse_did_not_bat,
    parse_extras,
    parse_innings_headers,
    parse_match_list,
    parse_score_text,
    parse_venue_and_date,
    _same_team,
    _title_has_team,
)


def soup_of(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestCleanStatusText:
    def test_expands_ball_shorthand(self):
        assert clean_status_text("PAK need 23 runs in 12b") == "PAK need 23 runs in 12 balls"

    def test_leaves_other_text_alone(self):
        assert clean_status_text("Stumps: Day 2") == "Stumps: Day 2"

    def test_passes_through_falsy(self):
        assert clean_status_text("") == ""
        assert clean_status_text(None) is None


class TestParseScoreText:
    def test_in_progress_score(self):
        assert parse_score_text("310-4 (50 Ov)") == ("310-4", "(50 Ov)")

    def test_all_out_score_has_no_wickets_part(self):
        assert parse_score_text("297 (49.2 Ov)") == ("297", "(49.2 Ov)")

    def test_declared_innings(self):
        assert parse_score_text("657-7 d (158 Ov)") == ("657-7 d", "(158 Ov)")

    def test_slash_separator(self):
        assert parse_score_text("310/4 (50 Ov)") == ("310/4", "(50 Ov)")

    def test_garbage_returns_empty(self):
        assert parse_score_text("Innings Break") == ("", "")
        assert parse_score_text("") == ("", "")


class TestFormatMatchTimes:
    def test_winter_uses_est(self):
        est, ist = format_match_times("2026-01-31T16:00:00.000Z")
        assert "EST" in est
        assert "IST" in ist

    def test_summer_uses_edt(self):
        est, _ = format_match_times("2026-06-15T16:00:00.000Z")
        assert "EDT" in est

    def test_invalid_returns_empty(self):
        assert format_match_times("not a time") == ("", "")


class TestClassifyStatus:
    def test_complete(self):
        assert classify_status("West Indies won by 5 wickets") == "complete"
        assert classify_status("Match drawn") == "complete"
        assert classify_status("Match abandoned due to rain") == "complete"

    def test_upcoming(self):
        assert classify_status("Preview") == "upcoming"
        assert classify_status("Match starts at 7:00 PM") == "upcoming"

    def test_live(self):
        assert classify_status("Need 59 runs in 44 balls") == "live"
        assert classify_status("Innings Break") == "live"
        assert classify_status("Day 2: Session 1") == "live"

    def test_empty_defaults_to_live(self):
        assert classify_status("") == "live"


class TestTeamMatching:
    def test_same_team_full_vs_abbreviation(self):
        assert _same_team("IND", "India")
        assert _same_team("India", "India")

    def test_different_teams(self):
        assert not _same_team("India", "Pakistan")

    def test_india_does_not_match_west_indies_in_title(self):
        title = "WEST INDIES VS ENGLAND, 1ST T20I - PREVIEW"
        assert not _title_has_team(title, "India")
        assert _title_has_team(title, "West Indies")
        assert _title_has_team(title, "England")

    def test_abbreviation_in_title(self):
        assert _title_has_team("IND VS PAK, 3RD ODI", "India")
        assert _title_has_team("IND VS PAK, 3RD ODI", "Pakistan")


# ---------------------------------------------------------------------------
# Score-to-team assignment (list view)
# ---------------------------------------------------------------------------

class TestAssignScoresToTeams:
    def test_scores_follow_team_names_not_innings_order(self):
        """Team listed second in the title batted first - scores must not swap."""
        match = {"team1": "India", "team2": "Pakistan", "score1": "", "score2": ""}
        innings = [
            {"team": "Pakistan", "score": "245", "overs": "(48.3 Ov)"},
            {"team": "India", "score": "120-2", "overs": "(20 Ov)"},
        ]
        assign_scores_to_teams(match, innings)
        assert match["score1"] == "120-2 (20 Ov)"
        assert match["score2"] == "245 (48.3 Ov)"

    def test_single_live_innings_goes_to_batting_team(self):
        match = {"team1": "India", "team2": "Pakistan", "score1": "", "score2": ""}
        innings = [{"team": "Pakistan", "score": "91-2", "overs": "(24 Ov)"}]
        assign_scores_to_teams(match, innings)
        assert match["score1"] == ""
        assert match["score2"] == "91-2 (24 Ov)"

    def test_test_match_innings_are_joined(self):
        match = {"team1": "Australia", "team2": "England", "score1": "", "score2": ""}
        innings = [
            {"team": "England", "score": "325", "overs": "(101 Ov)"},
            {"team": "Australia", "score": "405-7 d", "overs": "(110 Ov)"},
            {"team": "England", "score": "91-2", "overs": "(24 Ov)"},
        ]
        assign_scores_to_teams(match, innings)
        assert match["score1"] == "405-7 d (110 Ov)"
        assert match["score2"] == "325 & 91-2 (24 Ov)"

    def test_fallback_fills_missing_team_names(self):
        match = {"team1": "", "team2": "", "score1": "", "score2": ""}
        innings = [
            {"team": "Nepal", "score": "201-9", "overs": "(50 Ov)"},
            {"team": "UAE", "score": "150-4", "overs": "(30 Ov)"},
        ]
        assign_scores_to_teams(match, innings)
        assert match["team1"] == "Nepal"
        assert match["team2"] == "UAE"
        assert match["score1"] == "201-9 (50 Ov)"
        assert match["score2"] == "150-4 (30 Ov)"

    def test_no_scores_is_a_noop(self):
        match = {"team1": "India", "team2": "Pakistan", "score1": "", "score2": ""}
        assign_scores_to_teams(match, [])
        assert match["score1"] == ""
        assert match["score2"] == ""


class TestFormatTeamScores:
    def test_empty(self):
        assert format_team_scores([]) == ""

    def test_single(self):
        assert format_team_scores([{"score": "245-3", "overs": "(41.1 Ov)"}]) == "245-3 (41.1 Ov)"

    def test_multiple_keeps_overs_on_last_only(self):
        innings = [
            {"score": "245", "overs": "(50 Ov)"},
            {"score": "91-2", "overs": "(24 Ov)"},
        ]
        assert format_team_scores(innings) == "245 & 91-2 (24 Ov)"


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

INNINGS_HEADER_HTML = """
<div id="team-2-innings-1">
  <div class="hidden tb:block font-bold">India</div>
  <div class="tb:hidden font-bold">IND</div>
  <div class="flex gap-4"><div>310-4 (50 Ov)</div></div>
</div>
<div id="team-2-innings-1">duplicate desktop copy
  <div class="hidden tb:block font-bold">India</div>
  <div class="flex gap-4"><div>310-4 (50 Ov)</div></div>
</div>
<div id="team-5-innings-2">
  <div class="tb:hidden font-bold">PAK</div>
  <div class="flex gap-4"><div>297 (49.2 Ov)</div></div>
</div>
"""


class TestParseInningsHeaders:
    def test_parses_and_deduplicates(self):
        innings = parse_innings_headers(soup_of(INNINGS_HEADER_HTML))
        assert len(innings) == 2
        assert innings[0] == {"id": "team-2-innings-1", "team": "India",
                              "score": "310-4", "overs": "(50 Ov)"}
        # Falls back to the short name when the full-name div is absent.
        assert innings[1]["team"] == "PAK"
        assert innings[1]["score"] == "297"


SCARD_HTML = """
<div id="scard-team-2-innings-1">
  <div class="scorecard-bat-grid">
    <div class="font-bold">Batter</div><div>R</div><div>B</div><div>4s</div><div>6s</div>
  </div>
  <div class="scorecard-bat-grid">
    <div><a>Rohit Sharma</a><div class="text-cbTxtSec">c Smith b Starc</div></div>
    <div>45</div><div>38</div><div>6</div><div>1</div>
  </div>
  <div class="scorecard-bat-grid">
    <div><a>Virat Kohli</a><div class="text-cbTxtSec">batting</div></div>
    <div>72</div><div>60</div><div>8</div><div>0</div>
  </div>
  <div>
    <div>Extras</div>
    <div>12</div>
    <div>(b 4, lb 2, w 5, nb 1)</div>
  </div>
  <div>
    <div class="font-bold">Did not Bat</div>
    <div><a>Kuldeep Yadav</a>, <a>Mohammed Siraj</a></div>
  </div>
  <div class="scorecard-bowl-grid">
    <div class="font-bold">Bowler</div><div>O</div><div>M</div><div>R</div><div>W</div>
  </div>
  <div class="scorecard-bowl-grid">
    <a>Mitchell Starc</a><div>10</div><div>1</div><div>52</div><div>2</div>
  </div>
</div>
"""


class TestScorecardBodyParsing:
    def test_parse_batsmen(self):
        scard = soup_of(SCARD_HTML).find("div", id="scard-team-2-innings-1")
        batsmen = parse_batsmen(scard)
        assert batsmen == [
            {"name": "Rohit Sharma", "runs": "45", "balls": "38", "isNotOut": False},
            {"name": "Virat Kohli", "runs": "72", "balls": "60", "isNotOut": True},
        ]

    def test_parse_bowlers(self):
        scard = soup_of(SCARD_HTML).find("div", id="scard-team-2-innings-1")
        bowlers = parse_bowlers(scard)
        assert bowlers == [
            {"name": "Mitchell Starc", "overs": "10", "runs": "52", "wickets": "2"},
        ]

    def test_parse_extras(self):
        scard = soup_of(SCARD_HTML).find("div", id="scard-team-2-innings-1")
        assert parse_extras(scard) == "12 (b 4, lb 2, w 5, nb 1)"

    def test_parse_did_not_bat(self):
        scard = soup_of(SCARD_HTML).find("div", id="scard-team-2-innings-1")
        names, label = parse_did_not_bat(scard)
        assert names == ["Kuldeep Yadav", "Mohammed Siraj"]
        assert label == "Did not bat"

    def test_yet_to_bat_label_is_preserved(self):
        html = """
        <div id="scard-team-9-innings-1">
          <div>
            <div class="font-bold">Yet to Bat</div>
            <div><a>Player One</a>, <a>Player Two</a></div>
          </div>
        </div>
        """
        scard = soup_of(html).find("div", id="scard-team-9-innings-1")
        names, label = parse_did_not_bat(scard)
        assert names == ["Player One", "Player Two"]
        assert label == "Yet to bat"

    def test_missing_sections_return_empty(self):
        scard = soup_of('<div id="scard-team-1-innings-1"><div>nothing here</div></div>') \
            .find("div", id="scard-team-1-innings-1")
        assert parse_batsmen(scard) == []
        assert parse_bowlers(scard) == []
        assert parse_extras(scard) == ""
        assert parse_did_not_bat(scard) == ([], "Did not bat")


class TestParseVenueAndDate:
    def test_finds_venue_and_date(self):
        lines = ["Some nav", "Venue:", "Eden Gardens, Kolkata", "Date:", "Jun 11, 2026", "tail"]
        assert parse_venue_and_date(lines) == ("Eden Gardens, Kolkata", "Jun 11, 2026")

    def test_missing_returns_empty(self):
        assert parse_venue_and_date(["nothing", "relevant"]) == ("", "")


LIVE_PAGE_HTML = """
<html>
<head>
<script type="application/ld+json">
{"@type": "WebPage", "mainEntity": {"itemListElement": [
  {"@type": "SportsEvent", "name": "India vs Pakistan",
   "competitor": [{"name": "India"}, {"name": "Pakistan"}],
   "eventStatus": "Preview", "location": "Colombo, ",
   "startDate": "2026-06-15T14:00:00.000Z"}
]}}
</script>
</head>
<body>
<a href="/live-cricket-scores/12345/ind-vs-pak" class="block"
   title="India vs Pakistan, 3rd ODI - Preview">IND vs PAK 3rd ODI</a>
<a href="/live-cricket-scores/12345/ind-vs-pak" class="block"
   title="India vs Pakistan, 3rd ODI - Preview">duplicate link</a>
<a href="/live-cricket-scores/67890/wi-vs-eng" class="block"
   title="West Indies vs England, 1st T20I - West Indies won by 5 wickets">WI vs ENG - West Indies won by 5 wickets</a>
</body>
</html>
"""


class TestParseMatchList:
    def test_parses_matches_with_json_ld_enrichment(self):
        matches = parse_match_list(soup_of(LIVE_PAGE_HTML))
        assert len(matches) == 2

        ind_pak = matches[0]
        assert ind_pak["id"] == "12345"
        assert ind_pak["team1"] == "India"
        assert ind_pak["team2"] == "Pakistan"
        assert ind_pak["status"] == "Preview"
        assert ind_pak["state"] == "upcoming"
        assert ind_pak["venue"] == "Colombo"
        assert ind_pak["startTime"] == "2026-06-15T14:00:00.000Z"
        assert "EDT" in ind_pak["timeEST"]
        assert "IST" in ind_pak["timeLocal"]

    def test_json_ld_does_not_cross_match_west_indies(self):
        """The India JSON-LD entry must not attach to the West Indies match."""
        matches = parse_match_list(soup_of(LIVE_PAGE_HTML))
        wi_eng = matches[1]
        assert wi_eng["id"] == "67890"
        assert wi_eng["team1"] == "West Indies"
        assert wi_eng["team2"] == "England"
        assert wi_eng["venue"] == ""  # no JSON-LD entry for this match
        assert wi_eng["state"] == "complete"
