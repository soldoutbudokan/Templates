// Shared utilities for The Tilt frontend. Loaded via <script src="common.js">.
// Exposes globals: initNav, initThemeToggle, formatTilt, cssVar, loadTeamIndex,
//                  teamLink, seasonLink.

(function () {
    const NAV_ITEMS = [
        { href: 'index.html', key: 'home', label: 'Home' },
        { href: 'rankings.html', key: 'rankings', label: 'Rankings' },
        { href: 'season.html', key: 'season', label: 'Seasons' },
        { href: 'team.html', key: 'team', label: 'Teams' },
        { href: 'leaders.html', key: 'leaders', label: 'Leaders' },
        { href: 'awards.html', key: 'awards', label: 'Awards' },
        { href: 'goats.html', key: 'goats', label: '\u{1F410} GOATs' },
        { href: 'notes.html', key: 'notes', label: 'Notes' },
        { href: 'about.html', key: 'about', label: 'About' },
    ];

    // Blog post index — single source of truth used by index.html (homepage
    // teaser) and notes.html (full listing). When publishing a post: append
    // here, create public/notes/<id>.md, then add a row in dependencies.md.
    const BLOG_NOTES = [
        {
            id: 'ensemble',
            title: 'Why We Ensemble',
            summary: 'A single LightGBM retrain on a +2-match dataset shifted ABD\'s ball-0 prediction by 4.85pp on a fully-deterministic feature vector and pushed his career rank from #3 to #9. The career signal between #1 and #20 (~10 TILT units) is smaller than the noise floor of one retrain (max 5.5, p95 1.34) — about 14× louder than seven days of daily refreshes. Fix: K=100 ensemble averaged at inference, a 10% holdout frozen via a persisted match list (a fixed seed alone reshuffles as data grows — #193), plus a RETRAIN=1 env-var guardrail to block silent retrains.',
            tags: ['methodology', 'model'],
            date: '2026-05-07',
        },
        {
            id: 'last-ball-snap',
            title: 'Snapping the Final Ball — and Why We Reverted It',
            summary: 'The match-terminal snap looked like a clean fix to wp_after maxing out below 1.0 on chase-winning balls, but it concentrated the entire model-vs-truth gap onto a single bowler per match — Trent Boult\'s 3/26 in a DLS-shortened 2018 chase ballooned to a +0.97 single-game tilt, of which +0.89 was pure snap windfall. The fix: tune the underlying model to push wp toward 0/1 naturally at the boundary, and feed it the correct ball allocation for DLS-revised innings.',
            tags: ['methodology', 'model'],
            date: '2026-05-06',
        },
        {
            id: 'all-rounders',
            title: 'Why TILT Underrates All-Rounders',
            summary: 'By raw per-match impact, zero all-rounders crack the top 50; only three make the top 100. Three veterans — Maxwell, Yusuf Pathan, and Pollard — sneak into the top 50 on the consistency floor, still under the expected share. The gap is structural: 43.4% of all-rounder matches are mixed-sign, producing a 16.0% cancellation drag. Their GOAT-tier rate (4.06%) leads all roles; the floor is the problem.',
            tags: ['methodology', 'roles'],
            date: '2026-05-04',
        },
        {
            id: 'innings-boundary',
            title: 'Fixing the Innings-Boundary Jump',
            summary: 'The chart used to leap by 8.4pp at the median across the innings break — a signed −5.1pp bias favouring the chasing side. A two-step calibration (per-side isotonic + per-match midpoint) drives the cliff to mathematically zero, touches only 1.4% of balls, and reshuffles the top 10 in the directions you would expect. B Kumar drops three bowling spots; ABD stays #1.',
            tags: ['methodology', 'model'],
            date: '2026-04-26',
        },
        {
            id: 'venue-importance',
            title: 'The Importance of Venue',
            summary: 'Relocate RCB\'s 2016 home matches to Chepauk and re-score: ABD\'s TILT rises +0.80pp, Kohli\'s rises +2.50pp — same direction, Kohli roughly three times more. The model reads Chinnaswamy as so batsman-friendly that almost any harder venue lifts both. A model-sensitivity probe of TILT\'s venue feature across 6 player cohorts.',
            tags: ['methodology', 'case study'],
            date: '2026-04-22',
        },
        {
            id: 'kohli-2016-paradox',
            title: 'The 2016 Kohli Dilemma: Why His Greatest Season Survives the TILT Test',
            summary: '973 runs at 152 strike rate. Four centuries. And a TILT of +5.93% per match. The season that should have triggered every one of TILT\'s structural penalties comes through them anyway, more efficient per ball than his celebrated 2019 and across twice the workload.',
            tags: ['batting', 'case study'],
            date: '2026-04-15',
        },
        {
            id: 'innings-bias',
            title: 'The Second Innings Problem',
            summary: '100% of the top-50 batting GOATs are from the 2nd innings. Win probability swings are 1.65x larger when chasing, rising to 2.35x in the death overs. How this affects single-match rankings (and why careers, rho 0.99, barely move) and what we do about it.',
            tags: ['methodology', 'model'],
            date: '2026-04-15',
        },
    ];

    function initNav(activeKey) {
        const linksHtml = NAV_ITEMS
            .map((n) => `<a href="${n.href}" class="${n.key === activeKey ? 'active' : ''}">${n.label}</a>`)
            .join('');
        const navHtml = `
      <div class="top-nav-inner">
        <div class="logo"><a href="index.html"><img src="assets/crest-icon.svg" alt="" class="logo-mark" width="28" height="28"><span class="logo-word">THE TILT</span></a></div>
        <div class="global-search" id="globalSearch">
          <input type="search" id="navSearch"
                 placeholder="Search players and teams…"
                 autocomplete="off" spellcheck="false"
                 aria-label="search the tilt" aria-autocomplete="list"
                 aria-controls="navSearchResults" aria-expanded="false">
          <div class="global-search-panel" id="navSearchResults" role="listbox" hidden></div>
        </div>
        <div class="nav-links">
          ${linksHtml}
          <div class="theme-switch">
            <span class="theme-icon" id="sunIcon">&#9788;</span>
            <button class="theme-toggle" id="themeBtn" aria-label="toggle theme"></button>
            <span class="theme-icon" id="moonIcon">&#9789;</span>
          </div>
        </div>
      </div>`;
        const nav = document.querySelector('nav.top-nav');
        if (nav) nav.innerHTML = navHtml;
        initThemeToggle();
        initGlobalSearch();
    }

    function initThemeToggle() {
        const t = document.documentElement.getAttribute('data-theme') || 'dark';
        const sun = document.getElementById('sunIcon');
        const moon = document.getElementById('moonIcon');
        const btn = document.getElementById('themeBtn');
        if (!sun || !moon || !btn) return;
        document.getElementById(t === 'dark' ? 'moonIcon' : 'sunIcon').classList.add('active');
        btn.addEventListener('click', () => {
            const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            localStorage.setItem('theme', next);
            sun.classList.toggle('active');
            moon.classList.toggle('active');
        });
    }

    function formatTilt(val) {
        if (val == null || Number.isNaN(val)) return '<span class="tilt-neutral">—</span>';
        const sign = val >= 0 ? '+' : '';
        const pct = (val * 100).toFixed(2);
        const cls = val > 0.001 ? 'tilt-positive' : val < -0.001 ? 'tilt-negative' : 'tilt-neutral';
        return `<span class="${cls}">${sign}${pct}%</span>`;
    }

    // ICC codes with a vendored SVG flag in /flags/. ICC and ISO 3166-1
    // alpha-2 codes diverge for England (file is named `en.svg` but holds
    // the GB-ENG subdivision flag) and West Indies (no national flag —
    // hand-drawn placeholder). The vendored SVGs render identically on
    // every browser/OS, unlike the previous Unicode-emoji approach which
    // fell back to a black flag for EN on older Windows / Android.
    const _SVG_FLAG_CODES = new Set([
        'IN', 'AU', 'NZ', 'ZA', 'LK', 'PK', 'BD', 'AF',
        'IE', 'NL', 'ZW', 'NP', 'US', 'EN', 'WI',
    ]);

    function flagSpan(country) {
        if (!country) return '';
        const code = String(country).toUpperCase();
        if (_SVG_FLAG_CODES.has(code)) {
            return `<img class="flag" src="flags/${code.toLowerCase()}.svg" alt="${code}" title="${code}">`;
        }
        return `<span class="flag flag-text" title="${code}">${code}</span>`;
    }

    function cssVar(name) {
        return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }

    // Balls -> cricket overs notation, e.g. 124 -> "20.4" (NOT 124/6=20.67).
    function ballsToOvers(balls) {
        const b = Number(balls) || 0;
        return Math.floor(b / 6) + '.' + (b % 6);
    }

    // Shared stat metadata for season.html + leaders.html. Single source of truth
    // so the chip labels and ordering can't drift between the two pages (#138).
    const STAT_LABELS = {
        runs: 'Runs',
        wickets: 'Wickets',
        batting_tilt: 'Batting TILT',
        bowling_tilt: 'Bowling TILT',
        total_tilt: 'Total TILT',
        sr: 'Strike Rate',
        economy: 'Economy',
        fifties: 'Fifties',
        hundreds: 'Hundreds',
        fours: 'Fours',
        sixes: 'Sixes',
    };
    const STAT_ORDER = [
        'total_tilt', 'batting_tilt', 'bowling_tilt',
        'runs', 'wickets', 'sr', 'economy',
        'fifties', 'hundreds', 'fours', 'sixes',
    ];

    let _teamIndexCache = null;
    let _teamIndexPromise = null;
    async function loadTeamIndex() {
        if (_teamIndexCache) return _teamIndexCache;
        if (_teamIndexPromise) return _teamIndexPromise;
        _teamIndexPromise = (async () => {
            try {
                const res = await fetch('data/team_index.json');
                if (!res.ok) throw new Error('team_index fetch failed');
                const arr = await res.json();
                const map = new Map();
                arr.forEach((t) => {
                    map.set(t.name, t);
                    (t.aliases || []).forEach((a) => map.set(a, t));
                });
                _teamIndexCache = map;
                return map;
            } catch (e) {
                // Don't cache the failure: clear the in-flight promise so a later
                // call retries the fetch instead of returning an empty Map for the
                // rest of the session (which silently breaks every team link) (#137).
                _teamIndexPromise = null;
                return new Map();
            }
        })();
        return _teamIndexPromise;
    }

    // Each season_labels rule is a true (optionally one-sided) range:
    // from_year <= year <= through_year, with a missing bound treated as open
    // (issue #199 — the old short-circuit on through_year made a rule with
    // both bounds match every year <= through_year). A rule with neither
    // bound matches nothing. Mirrors season_team_label() in
    // pipeline/parse_matches.py — keep in sync.
    function teamLabelForSeason(team, season) {
        if (!team || !team.season_labels || season == null) return team ? team.name : '';
        const sStr = String(season);
        const year = parseInt(sStr.split('/')[0], 10);
        if (Number.isNaN(year)) return team.name;
        for (const rule of team.season_labels) {
            const through = rule.through_year != null ? Number(rule.through_year) : null;
            const from = rule.from_year != null ? Number(rule.from_year) : null;
            if (through == null && from == null) continue;
            if (through != null && year > through) continue;
            if (from != null && year < from) continue;
            return rule.label;
        }
        return team.name;
    }

    function teamLink(name, season, teamIndex, opts) {
        opts = opts || {};
        if (!name) return '';
        const t = teamIndex && teamIndex.get(name);
        const label = opts.label || (t ? teamLabelForSeason(t, season) : name);
        if (!t) return label;
        const params = new URLSearchParams({ team: t.slug });
        if (season != null && season !== '') params.set('season', String(season));
        const cls = ` class="${opts.cls || 'team-link'}"`;
        return `<a href="team.html?${params.toString()}"${cls}>${label}</a>`;
    }

    function seasonLink(season, label) {
        if (season == null || season === '') return '';
        return `<a href="season.html?season=${encodeURIComponent(String(season))}">${label != null ? label : season}</a>`;
    }

    // ---------- Global search ----------
    // Lazy-loaded, vanilla substring + prefix scoring over a tiny pre-built
    // index (`data/search_index.json`). Players + teams only — see
    // `pipeline/export_json.export_search_index`. Single closure per page.
    let _searchIdx = null;
    let _searchLoading = null;
    let _searchBound = false;

    function _loadSearchIndex() {
        if (_searchIdx) return Promise.resolve(_searchIdx);
        if (_searchLoading) return _searchLoading;
        _searchLoading = fetch('data/search_index.json')
            .then((r) => (r.ok ? r.json() : []))
            .then((j) => { _searchIdx = j; return j; })
            .catch(() => { _searchIdx = []; return []; });
        return _searchLoading;
    }

    // Score one record against a lowercased query (tokenized on whitespace).
    // All tokens must hit the corpus `x` somewhere (AND). Per token, we pick
    // the *best* signal anywhere in the corpus — not the first indexOf hit:
    //   exact token   (bounded by |...| or string ends)  → 150
    //   token start   (begins a corpus token after `|`)  → 100
    //   substring     (anywhere)                         → 10/(1+i)
    // This matters when the same word appears both inside a phrase and as
    // its own token (e.g. "malinga" inside "lasith malinga" AND alone).
    // Ties break on `b` (boost) then `r` (rank).
    function _tokenScore(x, tok) {
        if (x === tok || x.startsWith(tok + '|') || x.endsWith('|' + tok) || x.indexOf('|' + tok + '|') >= 0) return 150;
        if (x.startsWith(tok) || x.indexOf('|' + tok) >= 0) return 100;
        const i = x.indexOf(tok);
        if (i < 0) return -Infinity;
        return 10 / (1 + i);
    }

    function _scoreRecord(rec, qLower, qTokens) {
        const x = rec.x;
        const lLower = rec.l.toLowerCase();
        let score = 0;
        for (const tok of qTokens) {
            const s = _tokenScore(x, tok);
            if (s === -Infinity) return -Infinity;
            score += s;
        }
        if (lLower === qLower) score += 50;
        else if (lLower.startsWith(qLower)) score += 25;
        score += rec.t === 'p' ? 5 : 4;
        return score;
    }

    function _searchAll(q) {
        if (!_searchIdx || !q) return [];
        const qLower = q.toLowerCase().trim();
        const qTokens = qLower.split(/\s+/).filter(Boolean);
        if (qTokens.length === 0) return [];
        const hits = [];
        for (const rec of _searchIdx) {
            const s = _scoreRecord(rec, qLower, qTokens);
            if (s > -Infinity) hits.push({ rec, s });
        }
        hits.sort((a, b) => {
            if (b.s !== a.s) return b.s - a.s;
            if ((b.rec.b || 0) !== (a.rec.b || 0)) return (b.rec.b || 0) - (a.rec.b || 0);
            return (a.rec.r || 9999) - (b.rec.r || 9999);
        });
        return hits;
    }

    function _escapeHtml(s) {
        return String(s).replace(/[&<>"']/g, (c) =>
            ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    }

    function _highlight(label, qTokens) {
        const safe = _escapeHtml(label);
        if (!qTokens.length) return safe;
        // Match the longest token first so overlapping shorter ones don't break the regex.
        const sorted = qTokens.slice().sort((a, b) => b.length - a.length);
        const pattern = new RegExp('(' + sorted.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|') + ')', 'ig');
        return safe.replace(pattern, '<mark>$1</mark>');
    }

    function _entityHref(rec) {
        if (rec.t === 'p') return `player.html?player=${encodeURIComponent(rec.s)}`;
        if (rec.t === 'team') return `team.html?team=${encodeURIComponent(rec.s)}`;
        return '#';
    }

    // Render the dropdown contents. `state.flat` is the keyboard-navigable
    // ordered list of visible row data; we rebuild it on every render.
    // Groups are ordered by their top hit's score so a strong team match
    // (e.g. "DD" → Delhi Capitals) doesn't get buried under low-scoring
    // player substring hits.
    function _renderPanel(panel, hits, qTokens, state) {
        const LABELS = { p: 'Players', team: 'Teams' };
        const groups = { p: [], team: [] };
        for (const h of hits) {
            if (groups[h.rec.t]) groups[h.rec.t].push(h);
        }
        const order = Object.keys(groups)
            .filter((k) => groups[k].length)
            .sort((a, b) => (groups[b][0].s - groups[a][0].s))
            .map((k) => ({ key: k, label: LABELS[k] }));
        const CAP = 5;
        const flat = [];
        let html = '';
        let totalShown = 0;

        for (const g of order) {
            const all = groups[g.key];
            if (!all.length) continue;
            const expanded = !!state.expanded[g.key];
            const shown = expanded ? all : all.slice(0, CAP);
            html += `<div class="gs-group-label">${g.label}</div>`;
            for (const h of shown) {
                const idx = flat.length;
                flat.push({ rec: h.rec });
                const href = _entityHref(h.rec);
                const lab = _highlight(h.rec.l, qTokens);
                const sub = _escapeHtml(h.rec.sub || '');
                const flagPrefix = h.rec.t === 'p' && h.rec.c ? flagSpan(h.rec.c) : '';
                html += `<a class="gs-row" data-idx="${idx}" href="${href}" role="option">`
                    + `<span class="gs-label">${flagPrefix}${lab}</span>`
                    + `<span class="gs-sub">${sub}</span>`
                    + `</a>`;
            }
            if (!expanded && all.length > CAP) {
                html += `<div class="gs-show-more" data-group="${g.key}">Show all ${all.length} ${g.label.toLowerCase()}</div>`;
            }
            totalShown += shown.length;
        }

        if (totalShown === 0) {
            html = `<div class="gs-empty">No matches.</div>`;
        }

        panel.innerHTML = html;
        state.flat = flat;
        if (state.active >= flat.length) state.active = flat.length - 1;
        _paintActive(panel, state);
    }

    function _paintActive(panel, state) {
        const rows = panel.querySelectorAll('.gs-row');
        rows.forEach((el) => el.classList.remove('active'));
        if (state.active >= 0 && rows[state.active]) {
            rows[state.active].classList.add('active');
            rows[state.active].scrollIntoView({ block: 'nearest' });
        }
    }

    function _openPanel(input, panel) {
        panel.hidden = false;
        input.setAttribute('aria-expanded', 'true');
    }

    function _closePanel(input, panel) {
        panel.hidden = true;
        input.setAttribute('aria-expanded', 'false');
    }

    function initGlobalSearch() {
        const input = document.getElementById('navSearch');
        const panel = document.getElementById('navSearchResults');
        if (!input || !panel) return;

        const state = { active: -1, flat: [], expanded: {}, lastQ: '' };

        function refresh(q) {
            state.lastQ = q;
            if (!q || q.length === 0) {
                _closePanel(input, panel);
                return;
            }
            if (q.length === 1) {
                panel.innerHTML = `<div class="gs-hint">Keep typing | 2+ characters.</div>`;
                state.flat = [];
                state.active = -1;
                _openPanel(input, panel);
                return;
            }
            _loadSearchIndex().then(() => {
                if (state.lastQ !== q) return; // a newer query already arrived
                const hits = _searchAll(q);
                state.active = hits.length ? 0 : -1;
                state.expanded = {};
                _renderPanel(panel, hits, q.toLowerCase().trim().split(/\s+/).filter(Boolean), state);
                _openPanel(input, panel);
            });
        }

        let debounceTimer = null;
        input.addEventListener('input', (e) => {
            const q = e.target.value;
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => refresh(q), 80);
        });

        input.addEventListener('focus', () => {
            _loadSearchIndex();
            const q = input.value;
            if (q && q.length >= 2) refresh(q);
        });

        input.addEventListener('keydown', (e) => {
            if (panel.hidden) return;
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                if (state.flat.length === 0) return;
                state.active = (state.active + 1) % state.flat.length;
                _paintActive(panel, state);
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                if (state.flat.length === 0) return;
                state.active = state.active <= 0 ? state.flat.length - 1 : state.active - 1;
                _paintActive(panel, state);
            } else if (e.key === 'Enter') {
                if (state.flat.length === 0) return;
                e.preventDefault();
                const i = state.active >= 0 ? state.active : 0;
                const target = state.flat[i];
                if (target) window.location.href = _entityHref(target.rec);
            } else if (e.key === 'Escape') {
                input.value = '';
                _closePanel(input, panel);
            }
        });

        panel.addEventListener('mousedown', (e) => {
            const more = e.target.closest('.gs-show-more');
            if (!more) return;
            e.preventDefault(); // keep input focused
            const g = more.getAttribute('data-group');
            state.expanded[g] = true;
            const hits = _searchAll(state.lastQ);
            const tokens = state.lastQ.toLowerCase().trim().split(/\s+/).filter(Boolean);
            _renderPanel(panel, hits, tokens, state);
        });

        panel.addEventListener('mousemove', (e) => {
            const row = e.target.closest('.gs-row');
            if (!row) return;
            const i = parseInt(row.getAttribute('data-idx'), 10);
            if (Number.isFinite(i) && i !== state.active) {
                state.active = i;
                _paintActive(panel, state);
            }
        });

        document.addEventListener('mousedown', (e) => {
            if (panel.hidden) return;
            if (!e.target.closest('#globalSearch')) _closePanel(input, panel);
        });

        // basketball-reference convention: "/" focuses the global search
        // unless the user is already typing somewhere.
        if (!_searchBound) {
            _searchBound = true;
            document.addEventListener('keydown', (e) => {
                if (e.key !== '/') return;
                const t = e.target;
                if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
                e.preventDefault();
                const el = document.getElementById('navSearch');
                if (el) el.focus();
            });
        }
    }

    // Expose
    window.initNav = initNav;
    window.initThemeToggle = initThemeToggle;
    window.formatTilt = formatTilt;
    window.cssVar = cssVar;
    window.ballsToOvers = ballsToOvers;
    window.loadTeamIndex = loadTeamIndex;
    window.teamLink = teamLink;
    window.teamLabelForSeason = teamLabelForSeason;  // used by season.html bracket labels (#190)
    window.seasonLink = seasonLink;
    window.flagSpan = flagSpan;
    window.BLOG_NOTES = BLOG_NOTES;
    window.STAT_LABELS = STAT_LABELS;
    window.STAT_ORDER = STAT_ORDER;
})();
