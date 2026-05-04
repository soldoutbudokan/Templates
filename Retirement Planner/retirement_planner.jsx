import React, { useState, useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Area,
  AreaChart
} from 'recharts';

// Editorial almanac palette
const C = {
  bg: '#f0eadb',
  bgDeep: '#e8dfc9',
  paper: '#faf6ec',
  ink: '#1c1814',
  inkSoft: '#4a4136',
  rule: '#8b7d5e',
  ruleSoft: '#c9bd9a',
  accent: '#7a1f2b',
  accentLight: '#a83847',
  navy: '#1f3a5c',
  ocre: '#b08020',
  green: '#3d6b3d'
};

const fmt = (n) => {
  if (n == null || isNaN(n)) return 'n/a';
  return '$' + Math.round(n).toLocaleString();
};
const fmtK = (n) => '$' + Math.round(n / 1000).toLocaleString() + 'k';
const fmtM = (n) => '$' + (n / 1000000).toFixed(2) + 'M';

// Ontario 2026 combined federal+provincial tax (approximate marginal brackets)
function afterTax(gross) {
  if (gross <= 0) return 0;
  let tax = 0;
  const brackets = [
    [15000, 0],
    [53359, 0.2005],
    [90599, 0.2965],
    [106717, 0.3148],
    [111733, 0.3389],
    [150000, 0.3791],
    [173205, 0.4341],
    [220000, 0.4497],
    [246752, 0.4829],
    [Infinity, 0.5353]
  ];
  let prev = 0;
  for (const [top, rate] of brackets) {
    if (gross <= top) {
      tax += (gross - prev) * rate;
      break;
    }
    tax += (top - prev) * rate;
    prev = top;
  }
  return gross - tax;
}

function NumberInput({ label, value, onChange, suffix, step = 1, min, max, hint }) {
  return (
    <label className="flex flex-col gap-1">
      <span style={{ color: C.inkSoft, fontSize: 11, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
        {label}
      </span>
      <div className="flex items-baseline gap-2">
        <input
          type="number"
          value={value}
          step={step}
          min={min}
          max={max}
          onChange={(e) => onChange(Number(e.target.value))}
          style={{
            background: C.paper,
            border: `1px solid ${C.ruleSoft}`,
            color: C.ink,
            padding: '6px 10px',
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: 14,
            width: '100%',
            outline: 'none'
          }}
        />
        {suffix && (
          <span style={{ color: C.inkSoft, fontSize: 12, fontFamily: 'JetBrains Mono, monospace' }}>
            {suffix}
          </span>
        )}
      </div>
      {hint && (
        <span style={{ color: C.inkSoft, fontSize: 10, fontStyle: 'italic' }}>{hint}</span>
      )}
    </label>
  );
}

function SliderRow({ label, value, onChange, min, max, step = 1, suffix }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between">
        <span style={{ color: C.inkSoft, fontSize: 11, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
          {label}
        </span>
        <span style={{ color: C.ink, fontFamily: 'JetBrains Mono, monospace', fontSize: 13 }}>
          {value}{suffix || ''}
        </span>
      </div>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ accentColor: C.accent, width: '100%' }}
      />
    </div>
  );
}

export default function RetirementPlanner() {
  // Personal
  const [currentAge, setCurrentAge] = useState(29);
  const [netWorth, setNetWorth] = useState(400000);
  const [lifespan, setLifespan] = useState(95);

  // Income trajectory
  const [currentIncome, setCurrentIncome] = useState(190000);
  const [peakIncome, setPeakIncome] = useState(215000);
  const [yearsToPeak, setYearsToPeak] = useState(5);
  const [yearsAtPeak, setYearsAtPeak] = useState(5);
  const [postIncome, setPostIncome] = useState(130000);

  // Saving and growth
  const [savingsRate, setSavingsRate] = useState(28);
  const [realReturn, setRealReturn] = useState(7);

  // Retirement
  const [retirementAge, setRetirementAge] = useState(60);

  // Government benefits
  const [cppStart, setCppStart] = useState(70);
  const [cppAt65, setCppAt65] = useState(22000);
  const [oasAt65, setOasAt65] = useState(9000);
  const [careerStart, setCareerStart] = useState(22);

  const incomeAtAge = (age) => {
    const y = age - currentAge;
    if (y < 0) return 0;
    if (y < yearsToPeak) {
      return currentIncome + (peakIncome - currentIncome) * (y / yearsToPeak);
    }
    if (y < yearsToPeak + yearsAtPeak) return peakIncome;
    return postIncome;
  };

  const sim = useMemo(() => {
    const r = realReturn / 100;
    const data = [];
    let portfolio = netWorth;

    // Accumulation
    for (let age = currentAge; age <= retirementAge; age++) {
      const gross = incomeAtAge(age);
      const net = afterTax(gross);
      const save = age < retirementAge ? net * (savingsRate / 100) : 0;
      data.push({
        age,
        portfolio: Math.round(portfolio),
        gross: Math.round(gross),
        save: Math.round(save),
        phase: 'accumulate'
      });
      portfolio = portfolio * (1 + r) + save;
    }

    const portfolioAtRetirement = portfolio;
    const yearsRetired = lifespan - retirementAge;

    // CPP contribution scaling: 39 years of max contributions hits the ceiling
    // (47 year contributory period from 18 to 65, less ~17% general dropout).
    // Working fewer years drags the average down proportionally.
    const workingYears = Math.max(0, retirementAge - careerStart);
    const cppContribFactor = Math.min(1, workingYears / 39);

    // CPP factor (early reduction 7.2% per year before 65, late boost 8.4% per year after 65)
    const cppFactor =
      cppStart < 65
        ? Math.max(0.5, 1 - (65 - cppStart) * 0.072)
        : Math.min(1.42, 1 + (cppStart - 65) * 0.084);
    const cppAnnual = cppAt65 * cppContribFactor * cppFactor;

    // OAS only available 65+, late boost 0.6%/month = 7.2%/year, capped at 70
    const oasStartAge = Math.max(65, Math.min(70, retirementAge < 65 ? 65 : retirementAge));
    const oasFactor = 1 + Math.max(0, oasStartAge - 65) * 0.072;
    const oasAnnual = oasAt65 * oasFactor;

    // Sustainable level draw that depletes portfolio at lifespan, accounting for
    // CPP and OAS arriving partway through retirement.
    // Solve for portfolio draw P such that portfolio runs out at lifespan, given
    // future CPP and OAS reduce required portfolio draw.
    // Simpler approach: target a constant total real income I such that
    // portfolio + PV(CPP from cppStart) + PV(OAS from oasStartAge) = PV(I over yearsRetired)
    const pvAnnuity = (n, rate) => (1 - Math.pow(1 + rate, -n)) / rate;
    const pvDeferred = (rate, defer, years) => {
      if (years <= 0) return 0;
      return pvAnnuity(years, rate) / Math.pow(1 + rate, defer);
    };

    const cppDefer = Math.max(0, cppStart - retirementAge);
    const cppYears = Math.max(0, lifespan - cppStart);
    const oasDefer = Math.max(0, oasStartAge - retirementAge);
    const oasYears = Math.max(0, lifespan - oasStartAge);

    const pvCpp = cppAnnual * pvDeferred(r, cppDefer, cppYears);
    const pvOas = oasAnnual * pvDeferred(r, oasDefer, oasYears);
    const pvTotal = portfolioAtRetirement + pvCpp + pvOas;

    const sustainableTotal = pvTotal / pvAnnuity(yearsRetired, r);

    // OAS clawback estimate (2026: starts ~$93k, recovered at 15%)
    const clawbackThreshold = 93000;
    const oasClawback =
      sustainableTotal > clawbackThreshold
        ? Math.min(oasAnnual, (sustainableTotal - clawbackThreshold) * 0.15)
        : 0;
    const sustainableNetGov = sustainableTotal - oasClawback;

    // Drawdown simulation
    let p = portfolio;
    for (let age = retirementAge + 1; age <= lifespan; age++) {
      const cpp = age >= cppStart ? cppAnnual : 0;
      const oas = age >= oasStartAge ? oasAnnual : 0;
      const portfolioDraw = sustainableTotal - cpp - oas;
      p = p * (1 + r) - portfolioDraw;
      data.push({
        age,
        portfolio: Math.round(Math.max(0, p)),
        cpp: Math.round(cpp),
        oas: Math.round(oas),
        phase: 'retire'
      });
    }

    const portfolioDrawAtStart = sustainableTotal - 0 - 0; // before CPP/OAS

    return {
      data,
      portfolioAtRetirement,
      yearsRetired,
      sustainableTotal,
      sustainableNetGov,
      cppAnnual,
      oasAnnual,
      cppStartAge: cppStart,
      oasStartAge,
      oasClawback,
      portfolioDrawAtStart,
      finalPortfolio: data[data.length - 1].portfolio,
      workingYears,
      cppContribFactor
    };
  }, [
    currentAge,
    netWorth,
    currentIncome,
    peakIncome,
    yearsToPeak,
    yearsAtPeak,
    postIncome,
    savingsRate,
    realReturn,
    retirementAge,
    lifespan,
    cppStart,
    cppAt65,
    oasAt65,
    careerStart
  ]);

  const verdict = (() => {
    const i = sim.sustainableNetGov;
    if (i >= 250000) return { label: 'Lavish', color: C.accent };
    if (i >= 175000) return { label: 'Comfortable', color: C.green };
    if (i >= 110000) return { label: 'Solid', color: C.navy };
    if (i >= 75000) return { label: 'Adequate', color: C.ocre };
    return { label: 'Tight', color: C.accent };
  })();

  return (
    <div
      style={{
        background: C.bg,
        color: C.ink,
        minHeight: '100vh',
        padding: '32px 24px 64px',
        fontFamily: '"Source Serif 4", "Source Serif Pro", Georgia, serif'
      }}
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&family=Source+Serif+4:ital,wght@0,300;0,400;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500;700&display=swap');
        body { margin: 0; }
        input[type="range"] {
          height: 4px;
        }
        input[type="number"]::-webkit-outer-spin-button,
        input[type="number"]::-webkit-inner-spin-button {
          -webkit-appearance: none;
          margin: 0;
        }
        input[type="number"] { -moz-appearance: textfield; }
      `}</style>

      <div style={{ maxWidth: 1200, margin: '0 auto' }}>
        {/* Masthead */}
        <header
          style={{
            borderBottom: `2px solid ${C.ink}`,
            borderTop: `4px double ${C.ink}`,
            padding: '20px 0 16px',
            marginBottom: 28,
            textAlign: 'center'
          }}
        >
          <div
            style={{
              fontSize: 11,
              letterSpacing: '0.4em',
              textTransform: 'uppercase',
              color: C.inkSoft,
              marginBottom: 8
            }}
          >
            A Retirement Almanac and Calculator
          </div>
          <h1
            style={{
              fontFamily: '"Playfair Display", Georgia, serif',
              fontWeight: 900,
              fontSize: 'clamp(36px, 6vw, 64px)',
              margin: 0,
              letterSpacing: '-0.02em',
              fontStyle: 'italic'
            }}
          >
            Years of Plenty,<br />Years of Bread
          </h1>
          <div
            style={{
              fontSize: 12,
              color: C.inkSoft,
              marginTop: 12,
              fontStyle: 'italic'
            }}
          >
            All figures stated in 2026 dollars. Returns are real, net of inflation.
          </div>
        </header>

        {/* HERO: retirement age + income */}
        <section
          style={{
            background: C.paper,
            border: `1px solid ${C.rule}`,
            padding: 28,
            marginBottom: 32,
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: 32
          }}
          className="grid-hero"
        >
          <div>
            <div
              style={{
                fontSize: 11,
                letterSpacing: '0.18em',
                textTransform: 'uppercase',
                color: C.inkSoft,
                marginBottom: 12
              }}
            >
              The age you stop working
            </div>
            <div
              style={{
                fontFamily: '"Playfair Display", Georgia, serif',
                fontSize: 96,
                fontWeight: 700,
                lineHeight: 1,
                color: C.ink,
                marginBottom: 8
              }}
            >
              {retirementAge}
            </div>
            <input
              type="range"
              min={45}
              max={75}
              value={retirementAge}
              onChange={(e) => setRetirementAge(Number(e.target.value))}
              style={{ width: '100%', accentColor: C.accent }}
            />
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                fontSize: 10,
                color: C.inkSoft,
                marginTop: 4,
                fontFamily: 'JetBrains Mono, monospace'
              }}
            >
              <span>45</span>
              <span>60</span>
              <span>75</span>
            </div>
          </div>

          <div
            style={{
              borderLeft: `1px solid ${C.ruleSoft}`,
              paddingLeft: 28,
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between'
            }}
          >
            <div>
              <div
                style={{
                  fontSize: 11,
                  letterSpacing: '0.18em',
                  textTransform: 'uppercase',
                  color: C.inkSoft,
                  marginBottom: 4
                }}
              >
                Annual income, in retirement
              </div>
              <div
                style={{
                  fontFamily: '"Playfair Display", Georgia, serif',
                  fontSize: 56,
                  fontWeight: 700,
                  lineHeight: 1,
                  color: verdict.color
                }}
              >
                {fmt(sim.sustainableNetGov)}
              </div>
              <div
                style={{
                  fontSize: 12,
                  color: C.inkSoft,
                  marginTop: 4,
                  fontStyle: 'italic'
                }}
              >
                pre-tax, level for {sim.yearsRetired} years &nbsp;·&nbsp; {verdict.label.toLowerCase()}
              </div>
            </div>
            <div
              style={{
                marginTop: 16,
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: 12,
                fontSize: 12
              }}
            >
              <div>
                <div style={{ color: C.inkSoft, fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                  Portfolio at retirement
                </div>
                <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 18, color: C.ink }}>
                  {fmtM(sim.portfolioAtRetirement)}
                </div>
              </div>
              <div>
                <div style={{ color: C.inkSoft, fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                  Years of accumulation
                </div>
                <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 18, color: C.ink }}>
                  {retirementAge - currentAge}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Chart */}
        <section
          style={{
            background: C.paper,
            border: `1px solid ${C.rule}`,
            padding: 24,
            marginBottom: 32
          }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'baseline',
              marginBottom: 16,
              borderBottom: `1px solid ${C.ruleSoft}`,
              paddingBottom: 8
            }}
          >
            <h2
              style={{
                fontFamily: '"Playfair Display", Georgia, serif',
                fontStyle: 'italic',
                fontWeight: 400,
                fontSize: 24,
                margin: 0
              }}
            >
              The portfolio, age by age
            </h2>
            <span
              style={{
                fontSize: 10,
                letterSpacing: '0.15em',
                textTransform: 'uppercase',
                color: C.inkSoft
              }}
            >
              Real dollars · {C.accent === C.accent ? 'oxblood marks retirement' : ''}
            </span>
          </div>
          <div style={{ width: '100%', height: 320 }}>
            <ResponsiveContainer>
              <AreaChart data={sim.data} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
                <defs>
                  <linearGradient id="portfolioFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={C.navy} stopOpacity={0.35} />
                    <stop offset="100%" stopColor={C.navy} stopOpacity={0.04} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke={C.ruleSoft} strokeDasharray="2 4" vertical={false} />
                <XAxis
                  dataKey="age"
                  stroke={C.inkSoft}
                  tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, fill: C.inkSoft }}
                  tickLine={false}
                  axisLine={{ stroke: C.rule }}
                />
                <YAxis
                  stroke={C.inkSoft}
                  tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, fill: C.inkSoft }}
                  tickFormatter={(v) => '$' + Math.round(v / 1000000) + 'M'}
                  tickLine={false}
                  axisLine={{ stroke: C.rule }}
                />
                <Tooltip
                  contentStyle={{
                    background: C.paper,
                    border: `1px solid ${C.ink}`,
                    fontFamily: 'JetBrains Mono, monospace',
                    fontSize: 12
                  }}
                  formatter={(v) => fmt(v)}
                  labelFormatter={(age) => `Age ${age}`}
                />
                <Area
                  type="monotone"
                  dataKey="portfolio"
                  stroke={C.navy}
                  strokeWidth={2}
                  fill="url(#portfolioFill)"
                />
                <ReferenceLine
                  x={retirementAge}
                  stroke={C.accent}
                  strokeWidth={2}
                  strokeDasharray="4 3"
                  label={{
                    value: 'retire',
                    position: 'top',
                    fill: C.accent,
                    fontSize: 11,
                    fontFamily: 'JetBrains Mono, monospace'
                  }}
                />
                <ReferenceLine
                  x={sim.cppStartAge}
                  stroke={C.ocre}
                  strokeDasharray="2 4"
                  label={{
                    value: 'CPP',
                    position: 'top',
                    fill: C.ocre,
                    fontSize: 10,
                    fontFamily: 'JetBrains Mono, monospace'
                  }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* Income breakdown */}
        <section
          style={{
            background: C.paper,
            border: `1px solid ${C.rule}`,
            padding: 24,
            marginBottom: 32
          }}
        >
          <h2
            style={{
              fontFamily: '"Playfair Display", Georgia, serif',
              fontStyle: 'italic',
              fontWeight: 400,
              fontSize: 24,
              margin: '0 0 4px',
              borderBottom: `1px solid ${C.ruleSoft}`,
              paddingBottom: 8
            }}
          >
            Where the income comes from
          </h2>
          <table
            style={{
              width: '100%',
              borderCollapse: 'collapse',
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: 13,
              marginTop: 8
            }}
          >
            <thead>
              <tr style={{ color: C.inkSoft, fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                <th style={{ textAlign: 'left', padding: '8px 0', borderBottom: `1px solid ${C.ruleSoft}` }}>
                  Source
                </th>
                <th style={{ textAlign: 'right', padding: '8px 0', borderBottom: `1px solid ${C.ruleSoft}` }}>
                  Annual
                </th>
                <th style={{ textAlign: 'right', padding: '8px 0', borderBottom: `1px solid ${C.ruleSoft}` }}>
                  Begins at age
                </th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ padding: '10px 0' }}>Portfolio draw (level)</td>
                <td style={{ padding: '10px 0', textAlign: 'right' }}>{fmt(sim.sustainableTotal - sim.cppAnnual - sim.oasAnnual)}</td>
                <td style={{ padding: '10px 0', textAlign: 'right', color: C.inkSoft }}>{retirementAge}</td>
              </tr>
              <tr>
                <td style={{ padding: '10px 0' }}>
                  Canada Pension Plan
                  {sim.cppContribFactor < 1 && (
                    <span style={{ color: C.inkSoft, fontSize: 11, fontStyle: 'italic', marginLeft: 6 }}>
                      ({Math.round(sim.cppContribFactor * 100)}% of max, {sim.workingYears} contribution years)
                    </span>
                  )}
                </td>
                <td style={{ padding: '10px 0', textAlign: 'right' }}>{fmt(sim.cppAnnual)}</td>
                <td style={{ padding: '10px 0', textAlign: 'right', color: C.inkSoft }}>{sim.cppStartAge}</td>
              </tr>
              <tr>
                <td style={{ padding: '10px 0' }}>Old Age Security</td>
                <td style={{ padding: '10px 0', textAlign: 'right' }}>{fmt(sim.oasAnnual)}</td>
                <td style={{ padding: '10px 0', textAlign: 'right', color: C.inkSoft }}>{sim.oasStartAge}</td>
              </tr>
              {sim.oasClawback > 0 && (
                <tr style={{ color: C.accent }}>
                  <td style={{ padding: '10px 0', fontStyle: 'italic' }}>OAS clawback (estimate)</td>
                  <td style={{ padding: '10px 0', textAlign: 'right' }}>−{fmt(sim.oasClawback)}</td>
                  <td style={{ padding: '10px 0', textAlign: 'right' }}></td>
                </tr>
              )}
              <tr style={{ borderTop: `2px solid ${C.ink}` }}>
                <td style={{ padding: '12px 0', fontWeight: 700 }}>Net annual income, level</td>
                <td style={{ padding: '12px 0', textAlign: 'right', fontWeight: 700, fontSize: 16 }}>
                  {fmt(sim.sustainableNetGov)}
                </td>
                <td style={{ padding: '12px 0' }}></td>
              </tr>
            </tbody>
          </table>
          <p
            style={{
              fontSize: 12,
              color: C.inkSoft,
              fontStyle: 'italic',
              marginTop: 16,
              borderTop: `1px solid ${C.ruleSoft}`,
              paddingTop: 12
            }}
          >
            The model finds a level real income for life, treating CPP and OAS as deferred annuities and
            drawing the portfolio to fill the rest. Tax on RRSP withdrawals is not subtracted; held in TFSA
            and non-registered accounts the bite is smaller. The clawback figure is rough.
          </p>
        </section>

        {/* INPUTS */}
        <section
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: 20
          }}
          className="grid-inputs"
        >
          <div
            style={{
              background: C.paper,
              border: `1px solid ${C.rule}`,
              padding: 20
            }}
          >
            <h3
              style={{
                fontFamily: '"Playfair Display", Georgia, serif',
                fontStyle: 'italic',
                fontSize: 20,
                margin: '0 0 16px',
                borderBottom: `1px solid ${C.ruleSoft}`,
                paddingBottom: 8
              }}
            >
              You
            </h3>
            <div className="flex flex-col gap-4">
              <NumberInput label="Current age" value={currentAge} onChange={setCurrentAge} min={18} max={70} />
              <NumberInput label="Net worth today" value={netWorth} onChange={setNetWorth} suffix="CAD" step={10000} />
              <SliderRow label="Live until" value={lifespan} onChange={setLifespan} min={75} max={105} />
            </div>
          </div>

          <div
            style={{
              background: C.paper,
              border: `1px solid ${C.rule}`,
              padding: 20
            }}
          >
            <h3
              style={{
                fontFamily: '"Playfair Display", Georgia, serif',
                fontStyle: 'italic',
                fontSize: 20,
                margin: '0 0 16px',
                borderBottom: `1px solid ${C.ruleSoft}`,
                paddingBottom: 8
              }}
            >
              The career
            </h3>
            <div className="flex flex-col gap-4">
              <NumberInput label="Income now" value={currentIncome} onChange={setCurrentIncome} suffix="CAD" step={5000} />
              <NumberInput label="Peak income" value={peakIncome} onChange={setPeakIncome} suffix="CAD" step={5000} />
              <SliderRow label="Years until peak" value={yearsToPeak} onChange={setYearsToPeak} min={0} max={20} suffix=" yrs" />
              <SliderRow label="Years held at peak" value={yearsAtPeak} onChange={setYearsAtPeak} min={0} max={30} suffix=" yrs" />
              <NumberInput label="Income after step-down" value={postIncome} onChange={setPostIncome} suffix="CAD" step={5000} />
              <SliderRow label="Saving rate (after-tax)" value={savingsRate} onChange={setSavingsRate} min={0} max={70} suffix="%" />
            </div>
          </div>

          <div
            style={{
              background: C.paper,
              border: `1px solid ${C.rule}`,
              padding: 20
            }}
          >
            <h3
              style={{
                fontFamily: '"Playfair Display", Georgia, serif',
                fontStyle: 'italic',
                fontSize: 20,
                margin: '0 0 16px',
                borderBottom: `1px solid ${C.ruleSoft}`,
                paddingBottom: 8
              }}
            >
              Markets and the state
            </h3>
            <div className="flex flex-col gap-4">
              <SliderRow label="Real return" value={realReturn} onChange={setRealReturn} min={2} max={10} step={0.5} suffix="%" />
              <SliderRow label="Career start age" value={careerStart} onChange={setCareerStart} min={16} max={30} hint="When you started CPP contributions" />
              <SliderRow label="CPP claim age" value={cppStart} onChange={setCppStart} min={60} max={70} />
              <NumberInput label="CPP at age 65 (full career)" value={cppAt65} onChange={setCppAt65} suffix="CAD" step={1000} hint="Max contributor with 39+ years. Scales down if you retire early." />
              <NumberInput label="OAS at age 65" value={oasAt65} onChange={setOasAt65} suffix="CAD" step={500} />
            </div>
          </div>
        </section>

        {/* Footnote */}
        <footer
          style={{
            marginTop: 40,
            paddingTop: 20,
            borderTop: `1px solid ${C.rule}`,
            fontSize: 11,
            color: C.inkSoft,
            fontStyle: 'italic',
            lineHeight: 1.6
          }}
        >
          <p style={{ margin: '0 0 8px' }}>
            <strong style={{ fontStyle: 'normal' }}>Caveats.</strong> Real returns of 7% are the long-run
            average for stocks. A balanced portfolio with bonds runs closer to 5%. The first decade after
            you retire matters most: a bad sequence of returns can wreck a plan that looks healthy on paper.
            Income tax in retirement is not modelled and depends on whether the money sits in TFSA, RRSP, or
            non-registered accounts.
          </p>
          <p style={{ margin: 0 }}>
            <strong style={{ fontStyle: 'normal' }}>Method.</strong> Income trajectory is linear from
            current to peak, then flat at peak, then a step down. Tax uses Ontario 2026 combined marginal
            brackets. Saving is a fixed share of after-tax income. CPP scales with contribution years
            (39 years at the max earnings ceiling gets you the full benefit, fewer years prorate it).
            CPP and OAS are then valued as deferred life annuities and pooled with the portfolio to find
            a level lifetime real income.
          </p>
        </footer>
      </div>

      <style>{`
        @media (max-width: 800px) {
          .grid-hero { grid-template-columns: 1fr !important; }
          .grid-inputs { grid-template-columns: 1fr !important; }
          .grid-hero > div:nth-child(2) {
            border-left: none !important;
            padding-left: 0 !important;
            border-top: 1px solid ${C.ruleSoft};
            padding-top: 20px !important;
          }
        }
      `}</style>
    </div>
  );
}
