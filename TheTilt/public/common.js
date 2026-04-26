// Shared utilities for The Tilt frontend. Loaded via <script src="common.js">.
// Exposes globals: initNav, initThemeToggle, formatTilt, cssVar, loadTeamIndex,
//                  teamLink, seasonLink.

(function () {
    const NAV_ITEMS = [
        { href: 'index.html', key: 'leaderboard', label: 'TILT' },
        { href: 'season.html', key: 'season', label: 'Seasons' },
        { href: 'team.html', key: 'team', label: 'Teams' },
        { href: 'leaders.html', key: 'leaders', label: 'Leaders' },
        { href: 'goats.html', key: 'goats', label: '\u{1F410} GOATs' },
        { href: 'notes.html', key: 'notes', label: 'Notes' },
        { href: 'about.html', key: 'about', label: 'About' },
    ];

    function initNav(activeKey) {
        const linksHtml = NAV_ITEMS
            .map((n) => `<a href="${n.href}" class="${n.key === activeKey ? 'active' : ''}">${n.label}</a>`)
            .join('');
        const navHtml = `
      <div class="top-nav-inner">
        <div class="logo"><a href="index.html">THE TILT</a></div>
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

    function cssVar(name) {
        return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }

    let _teamIndexCache = null;
    let _teamIndexPromise = null;
    async function loadTeamIndex() {
        if (_teamIndexCache) return _teamIndexCache;
        if (_teamIndexPromise) return _teamIndexPromise;
        _teamIndexPromise = (async () => {
            try {
                const res = await fetch('data/team_index.json');
                if (!res.ok) return new Map();
                const arr = await res.json();
                const map = new Map();
                arr.forEach((t) => {
                    map.set(t.name, t);
                    (t.aliases || []).forEach((a) => map.set(a, t));
                });
                _teamIndexCache = map;
                return map;
            } catch (e) {
                return new Map();
            }
        })();
        return _teamIndexPromise;
    }

    function teamLabelForSeason(team, season) {
        if (!team || !team.season_labels || season == null) return team ? team.name : '';
        const sStr = String(season);
        const year = parseInt(sStr.split('/')[0], 10);
        if (Number.isNaN(year)) return team.name;
        for (const rule of team.season_labels) {
            if (rule.through_year != null && year <= Number(rule.through_year)) return rule.label;
            if (rule.from_year != null && year >= Number(rule.from_year)) return rule.label;
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
        const cls = opts.cls ? ` class="${opts.cls}"` : '';
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
                html += `<a class="gs-row" data-idx="${idx}" href="${href}" role="option">`
                    + `<span class="gs-label">${lab}</span>`
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
                panel.innerHTML = `<div class="gs-hint">Keep typing — 2+ characters.</div>`;
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
    window.loadTeamIndex = loadTeamIndex;
    window.teamLink = teamLink;
    window.seasonLink = seasonLink;
})();
