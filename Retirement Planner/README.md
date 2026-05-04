# Retirement Planner

A spreadsheet for thinking about when you can stop working. All figures are in 2026 dollars (real terms, net of inflation), which means a number you read today buys what it would today, regardless of what year in the projection it appears.

## What it does

You set your age, your savings, your income trajectory, and the assumptions you want to use about returns and government benefits. The workbook projects your portfolio year by year, works out the level real income you can draw in retirement, and shows the breakdown across portfolio, CPP, and OAS.

The math is honest about what it knows: the real return you specify, the tax brackets for Ontario in 2026, the CPP and OAS rules as they stand. It is silent about what no model can know: the actual sequence of market returns, future tax law, your real lifespan, and what CPP and OAS will look like decades from now.

## The four tabs

### Inputs
This is the only tab you need to edit. Blue cells are inputs. Black cells are formulas. Green cells link to other tabs. Change any blue cell and the rest of the workbook updates.

The page reads top to bottom: personal facts, career and saving, market and retirement assumptions, government benefit settings, derived values (calculated from your inputs), and the Ontario tax brackets at the bottom.

### Projection
A year-by-year table from your current age forward, one row per year. Fourteen columns walk from gross income through tax, savings, portfolio growth, contributions, CPP, OAS, portfolio draw, and the ending portfolio. The work years show your savings flowing in. The retirement years show the level income flowing out. The portfolio runs down to near zero at lifespan.

### Summary
Two headline numbers at the top: portfolio at retirement, and the level annual income that retirement supports. Below, the math is laid bare. Years in retirement. The annuity factor, which is the present value of one dollar a year paid over your retirement period at the real return. The present values of CPP and OAS at retirement. The gross sustainable income, the clawback estimate, and the net.

### Notes
Method and caveats in plain prose. Worth reading once.

## How the income figure is built

The model treats CPP and OAS as future income streams, calculates what they are worth today, and pools that present value with the portfolio you will have at retirement. That total, divided by the annuity factor for the retirement period, gives the level real income you can spend each year for life. The portfolio funds whatever portion CPP and OAS do not.

CPP is scaled two ways. First by contribution years: 39 years at the maximum earnings ceiling earns the full benefit. Fewer years scale it down. Retire at 52 after 30 working years and you keep about 77% of the max. Second by claim age: claim at 60 cuts the benefit, claim at 70 boosts it 42%. Both factors stack.

OAS is a residence benefit, not a contribution one, so retiring early does not cut it. It can start as early as 65 and as late as 70, with up to 36% boost for waiting.

## Things to change first

The real return is the single most important assumption. The default of 7% is the long-run average for stocks. A balanced portfolio with bonds runs closer to 5%. Drop it to 5% and the sustainable income falls by roughly a third.

Retirement age. Drag it down and you see the portfolio at retirement shrink, the years to fund grow, and the CPP scale down because of fewer contribution years. The combined effect on sustainable income is sharp.

Saving rate. The model treats this as the share of after-tax income you save each year. If you save $40k on a $130k take-home, that is 31%.

## Caveats worth keeping in mind

**Sequence-of-returns risk is the killer this model hides.** A bad first decade after you stop working can wreck a plan that looks fine on paper. The level annuity assumption smooths over that. Anyone who retired in late 1999 or late 2007 can tell you what this risk looks like in real life.

**Income tax in retirement is not modeled.** The level income shown is pre-tax. Where the money sits matters: RRSP withdrawals are fully taxable, TFSA withdrawals are not and do not count toward the OAS clawback. The same dollar in a different account can leave you with different spending power.

**Tax brackets are an approximation of Ontario 2026.** They are inputs. If rates change, edit them.

**OAS clawback.** The model adds the present value of OAS to your wealth and shows the clawback as an estimate. For a high earner the clawback eats most of OAS, so some of that wealth never reaches you. This shows up as a small residual portfolio at lifespan. To model the full-clawback case cleanly, set OAS at 65 to zero on the Inputs tab. Sustainable income drops by a few thousand a year and the portfolio depletes to zero exactly.

## What these numbers mean

The sustainable income figure is what the model says you could spend each year, level, in today's dollars, for life, given the assumptions. It is not a recommendation, not a forecast, and not a guarantee. It is a useful number to compare against your current spending.

If your life today costs $80k and the model says you could sustain $400k in retirement, you have a wide margin. If the model says $90k against $80k spending, you have a thin one. Push the assumptions toward the conservative side and rerun.

## File contents

`retirement_planner.xlsx` contains the workbook. Open it in Excel, Numbers, or LibreOffice. All formulas are visible. Click any cell to see what it depends on.
