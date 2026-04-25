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

    // Expose
    window.initNav = initNav;
    window.initThemeToggle = initThemeToggle;
    window.formatTilt = formatTilt;
    window.cssVar = cssVar;
    window.loadTeamIndex = loadTeamIndex;
    window.teamLink = teamLink;
    window.seasonLink = seasonLink;
})();
