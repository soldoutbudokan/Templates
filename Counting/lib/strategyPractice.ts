import { Action, ActionOptions, formatAction, getCorrectAction, getHandTotal, isPair } from './basicStrategy';
import { Card, Value } from './deck';
import { seededRandom } from './training';

export type StrategyCategory = 'hard' | 'soft' | 'pairs';
export type StrategyFocus = StrategyCategory | 'mixed';
export type StrategyMode = 'practice' | 'test';
export const STRATEGY_CATEGORIES: StrategyCategory[] = ['hard', 'soft', 'pairs'];
export const STRATEGY_DEALERS: Value[] = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A'];
export const STRATEGY_PROFILE = '6 decks · dealer hits soft 17 · double any first two cards · double after split · late surrender · US peek';
export const STRATEGY_SOURCE = 'https://wizardofodds.com/games/blackjack/strategy/4-decks/';
export const CATEGORY_LABEL: Record<StrategyFocus, string> = { hard: 'Hard totals', soft: 'Soft totals', pairs: 'Pairs', mixed: 'Mixed' };

export interface StrategyRow { label: string; values: Value[] }
export interface StrategyCase {
  id: string;
  category: StrategyCategory;
  row: string;
  player: Card[];
  dealer: Card;
  options: ActionOptions;
}
export interface StrategyAnswer {
  caseId: string;
  action: Action;
  expected: Action;
  correct: boolean;
  responseMs: number;
}
export interface StrategyResult {
  mode: StrategyMode;
  focus: StrategyFocus;
  cases: StrategyCase[];
  answers: StrategyAnswer[];
  completed: boolean;
  interrupted: boolean;
}

const HARD_VALUES: Value[][] = [
  ['2', '3'], ['2', '4'], ['2', '5'], ['2', '6'], ['4', '5'], ['4', '6'], ['5', '6'],
  ['10', '2'], ['10', '3'], ['10', '4'], ['10', '5'], ['10', '6'], ['10', '7'], ['10', '8'], ['10', '9'], ['10', '6', '4'],
];

export function strategyRows(category: StrategyCategory): StrategyRow[] {
  if (category === 'hard') return HARD_VALUES.map((values, index) => ({ label: `Hard ${index + 5}`, values: [...values] }));
  if (category === 'soft') return ['2', '3', '4', '5', '6', '7', '8', '9'].map(value => ({ label: `A,${value}`, values: ['A', value as Value] }));
  return STRATEGY_DEALERS.map(value => ({ label: `${value},${value}`, values: [value, value] }));
}

export function makeStrategyCase(category: StrategyCategory, row: StrategyRow, dealer: Value): StrategyCase {
  const id = `${category}:${row.label}:${dealer}`;
  return {
    id, category, row: row.label,
    player: row.values.map((value, index) => ({ value, suit: index % 2 ? '♥' : '♠', id: `${id}:player:${index}` })),
    dealer: { value: dealer, suit: '♣', id: `${id}:dealer` },
    options: { canDouble: row.values.length === 2, canSplit: row.values.length === 2, canSurrender: row.values.length === 2 },
  };
}

function shuffle<T>(items: T[], random: () => number): T[] {
  const result = [...items];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

/** Samples chart rows evenly within each category, without repeating a cell.
 * Mixed tests distribute questions across categories, rather than deal a natural shoe.
 */
export function createStrategyCases(seed: number, focus: StrategyFocus, length: 20 | 30): StrategyCase[] {
  if (!Number.isInteger(seed) || !['hard', 'soft', 'pairs', 'mixed'].includes(focus) || ![20, 30].includes(length)) {
    throw new RangeError('Choose a strategy focus and 20 or 30 decisions.');
  }
  const random = seededRandom(seed);
  const categories = focus === 'mixed' ? shuffle(STRATEGY_CATEGORIES, random) : [focus];
  const queues = Object.fromEntries(categories.map(category => {
    const rows = shuffle(strategyRows(category), random);
    const dealers = rows.map(() => shuffle(STRATEGY_DEALERS, random));
    const queue: StrategyCase[] = [];
    for (let pass = 0; pass < 10; pass++) {
      rows.forEach((row, index) => queue.push(makeStrategyCase(category, row, dealers[index][pass])));
    }
    return [category, queue];
  })) as Record<StrategyCategory, StrategyCase[]>;
  const cases: StrategyCase[] = [];
  for (let i = 0; i < length; i++) cases.push(queues[categories[i % categories.length]].shift()!);
  return shuffle(cases, random);
}

export function legalStrategyActions(situation: StrategyCase): Action[] {
  const actions: Action[] = ['hit', 'stand'];
  if (situation.player.length === 2 && situation.options.canDouble !== false) actions.push('double');
  if (isPair(situation.player) && situation.options.canSplit !== false) actions.push('split');
  if (situation.player.length === 2 && situation.options.canSurrender !== false) actions.push('surrender');
  return actions;
}

export function gradeStrategyAnswer(situation: StrategyCase, action: Action, responseMs: number): StrategyAnswer {
  if (!legalStrategyActions(situation).includes(action)) throw new RangeError('This action is unavailable for this hand.');
  const expected = getCorrectAction(situation.player, situation.dealer, situation.options);
  return { caseId: situation.id, action, expected, correct: action === expected, responseMs: Math.max(0, Number.isFinite(responseMs) ? responseMs : 0) };
}

export function strategyCriterionMet(result: StrategyResult): boolean {
  return result.mode === 'test' && result.completed && !result.interrupted && [20, 30].includes(result.cases.length)
    && result.answers.length === result.cases.length && result.answers.every((answer, index) =>
      answer.caseId === result.cases[index].id && legalStrategyActions(result.cases[index]).includes(answer.action)
      && answer.action === getCorrectAction(result.cases[index].player, result.cases[index].dealer, result.cases[index].options));
}

export function strategyBreakdown(result: Pick<StrategyResult, 'cases' | 'answers'>) {
  return STRATEGY_CATEGORIES.map(category => {
    const answers = result.answers.filter((_, index) => result.cases[index]?.category === category);
    return { category, attempted: answers.length, correct: answers.filter(answer => answer.correct).length };
  });
}

export function describeSituation(situation: StrategyCase): string {
  return `${situation.player.map(card => card.value).join(', ')} vs dealer ${situation.dealer.value}`;
}

export function explainStrategyCase(situation: StrategyCase): string {
  const action = getCorrectAction(situation.player, situation.dealer, situation.options);
  const { total, soft } = getHandTotal(situation.player);
  const dealer = situation.dealer.value;
  const description = `${soft ? 'Soft' : 'Hard'} ${total} against ${dealer}`;
  if (action === 'surrender') {
    if (isPair(situation.player)) return dealer === 'A'
      ? 'For this H17 game, surrender 8,8 against an ace. Without surrender, split if allowed; otherwise hit.'
      : `Splitting is unavailable, so treat 8,8 as hard 16. Against ${dealer}, surrender if allowed; otherwise hit.`;
    return `${description}: surrender the initial two-card hand after the dealer has checked for blackjack. ${total === 17 ? 'Without surrender, stand.' : 'Without surrender, hit.'}`;
  }
  if (action === 'split') {
    if (situation.player[0].value === 'A') return 'Split aces. Keeping both aces together uses one as 1; splitting starts two hands with an ace.';
    if (situation.player[0].value === '8') return 'Split 8,8 against this upcard instead of playing hard 16. In this ruleset, surrender 8,8 against an ace when surrender is available.';
    if (situation.player[0].value === '9') return 'Split 9,9 against 2–6, 8, or 9. Stand against 7, 10, or an ace.';
    return `Split this pair against ${dealer} under the double-after-split rules. Each new hand can be doubled when its two cards call for it.`;
  }
  if (action === 'double') {
    const fallback = soft && total >= 18 ? 'stand' : 'hit';
    if (total === 11 && dealer === 'A') return 'Double hard 11 against an ace in this H17 game, after the negative dealer peek. Hit if doubling is unavailable.';
    return `${description}: double these first two cards. If doubling is unavailable, ${fallback}.`;
  }
  if (situation.player.length > 2) return `${description}: ${action}. After a hit, doubling and late surrender are unavailable; follow the hit-or-stand fallback.`;
  if (isPair(situation.player) && situation.player[0].value === '5') return `Treat 5,5 as hard 10. Against ${dealer}, ${action}; do not split fives.`;
  if (isPair(situation.player) && total === 20) return 'Stand on a pair of ten-value cards. Basic strategy keeps the strong total of 20.';
  if (soft && total === 18) return `Soft 18: hit against 9, 10, or an ace; stand against 7 or 8; double against 2–6 when allowed. Here, ${action}.`;
  if (!soft && total === 12) return `Hard 12: stand against 4–6 and hit against other upcards. Here, ${action}.`;
  if (!soft && total >= 13 && total <= 16) return `${description}: ${action}. After checking surrender and splits, stand on hard 13–16 against 2–6; hit against 7–A.`;
  return `${description}: ${formatAction(action).toLowerCase()} under this basic-strategy profile. ${soft && total <= 17 ? 'An ace can fall from 11 to 1, so a soft total is played differently from the same hard total.' : 'Use the declared rules and hand category, without a count-based deviation.'}`;
}
