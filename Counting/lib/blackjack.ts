import { Action, getCorrectAction, getHandTotal, isPair } from './basicStrategy';
import { estimateDecks, toTrueCount } from './countPolicy';
import { Card, Deck, getCardValue } from './deck';
import { seededRandom } from './training';

export type TableAction = Action | 'insure' | 'decline-insurance';
export type TablePhase = 'dealing' | 'insurance' | 'player-turn' | 'automatic' | 'dealer-turn' | 'round-result' | 'count-checkpoint' | 'complete';
export interface TableConfig {
  mode: 'practice' | 'test';
  roundLimit: 10 | 0;
  otherPlayers: 0 | 2;
  speedMs: number;
}
export interface TableDecision {
  round: number;
  seatIndex: number;
  handIndex: number;
  kind: 'play' | 'insurance';
  playerCards: Card[];
  dealerUpcard: Card;
  action: TableAction;
  expectedAction: TableAction;
  legalActions: TableAction[];
  correct: boolean;
  runningCount: number;
  trueCount: number;
}
export interface TableCheckpoint {
  round: number;
  runningGuess: number | null;
  trueGuess: number | null;
  runningTarget: number;
  trueTarget: number;
  decksRemaining: number;
  runningCorrect: boolean;
  trueCorrect: boolean;
  arithmeticCorrect: boolean | null;
}
export interface ExposureEvent {
  index: number;
  round: number;
  seatIndex: number | null;
  handIndex: number | null;
  card: Card;
  runningCount: number;
  kind: 'deal' | 'hit' | 'double' | 'split' | 'hole' | 'dealer-hit';
}
export interface TableHandView {
  cards: Card[];
  total: number;
  soft: boolean;
  status: string;
  units: number;
  outcome: string | null;
}
export interface TableView {
  dealer: { cards: (Card | null)[]; total: number | null };
  seats: { name: string; isUser: boolean; hands: TableHandView[] }[];
  activeSeat: number | null;
  activeHand: number | null;
  insurance: { taken: boolean; resolved: boolean; netUnits: number | null } | null;
}
interface Hand {
  cards: Card[];
  status: 'playing' | 'stood' | 'bust' | 'surrendered' | 'blackjack';
  units: number;
  split: boolean;
  splitAces: boolean;
  outcome: string | null;
}
interface Seat { name: string; isUser: boolean; hands: Hand[] }
type Stage = 'initial' | 'insurance' | 'peek' | 'players' | 'hand-start' | 'dealer-reveal' | 'dealer-play' | 'settle' | 'round-result' | 'checkpoint' | 'complete';
interface Engine {
  /** Private referee state. Render TableSessionState.table, never these cards. */
  shoe: readonly Card[];
  cursor: number;
  runningCount: number;
  exposedIds: string[];
  seats: Seat[];
  dealer: Card[];
  holeRevealed: boolean;
  initialStep: number;
  stage: Stage;
  activeSeat: number;
  activeHand: number;
  insurance: { taken: boolean; resolved: boolean; netUnits: number | null } | null;
  message: string;
}
export interface TableSessionState {
  id: string;
  seed: number;
  config: TableConfig;
  phase: TablePhase;
  round: number;
  remaining: number;
  table: TableView;
  statusText: string;
  decisions: TableDecision[];
  checkpoints: TableCheckpoint[];
  exposures: ExposureEvent[];
  completed: boolean;
  ended: boolean;
  interrupted: boolean;
  _engine: Engine;
}
export interface TableSessionResult {
  version: 1;
  id: string;
  seed: number;
  config: TableConfig;
  rounds: number;
  completed: boolean;
  interrupted: boolean;
  decisions: TableDecision[];
  checkpoints: TableCheckpoint[];
  exposures: ExposureEvent[];
  endedAt: string;
}

function hand(card?: Card): Hand {
  return { cards: card ? [card] : [], status: 'playing', units: 1, split: false, splitAces: false, outcome: null };
}
function clone(state: TableSessionState): TableSessionState {
  return { ...state, config: { ...state.config }, decisions: [...state.decisions], checkpoints: [...state.checkpoints], exposures: [...state.exposures],
    _engine: { ...state._engine, exposedIds: [...state._engine.exposedIds], dealer: [...state._engine.dealer],
      insurance: state._engine.insurance ? { ...state._engine.insurance } : null,
      seats: state._engine.seats.map(seat => ({ ...seat, hands: seat.hands.map(h => ({ ...h, cards: [...h.cards] })) })) } };
}
function refresh(state: TableSessionState): TableSessionState {
  const e = state._engine;
  const phase: TablePhase = e.stage === 'initial' ? 'dealing' : e.stage === 'insurance' ? 'insurance'
    : e.stage === 'players' && e.seats[e.activeSeat]?.isUser ? 'player-turn'
    : e.stage === 'dealer-reveal' || e.stage === 'dealer-play' ? 'dealer-turn'
    : e.stage === 'round-result' ? 'round-result' : e.stage === 'checkpoint' ? 'count-checkpoint'
    : e.stage === 'complete' ? 'complete' : 'automatic';
  const cleared = e.stage === 'checkpoint' || e.stage === 'complete';
  const active = ['players', 'hand-start'].includes(e.stage);
  state.phase = phase;
  state.remaining = e.shoe.length - e.cursor;
  state.table = cleared ? { dealer: { cards: [], total: null }, seats: [], activeSeat: null, activeHand: null, insurance: null } : {
    dealer: { cards: e.dealer.map((card, i) => i === 1 && !e.holeRevealed ? null : card),
      total: e.holeRevealed ? getHandTotal(e.dealer).total : null },
    seats: e.seats.map(seat => ({ name: seat.name, isUser: seat.isUser,
      hands: seat.hands.map(h => ({ cards: [...h.cards], ...getHandTotal(h.cards), status: h.status, units: h.units, outcome: h.outcome })) })),
    activeSeat: active ? e.activeSeat : null, activeHand: active ? e.activeHand : null,
    insurance: e.insurance ? { ...e.insurance } : null,
  };
  state.statusText = e.message;
  return state;
}
function expose(state: TableSessionState, card: Card, kind: ExposureEvent['kind'], seatIndex: number | null, handIndex: number | null): void {
  const e = state._engine;
  if (e.exposedIds.includes(card.id)) return;
  e.exposedIds.push(card.id);
  e.runningCount += getCardValue(card);
  state.exposures.push({ index: state.exposures.length, round: state.round, seatIndex, handIndex, card, runningCount: e.runningCount, kind });
}
function draw(state: TableSessionState, kind: ExposureEvent['kind'], seatIndex: number | null, handIndex: number | null, visible = true): Card {
  const e = state._engine;
  if (e.cursor >= e.shoe.length) throw new RangeError('The shoe is exhausted.');
  const card = e.shoe[e.cursor++];
  if (visible) expose(state, card, kind, seatIndex, handIndex);
  return card;
}
function noCrossShuffle(state: TableSessionState): TableSessionState {
  state.ended = true;
  state.completed = false;
  state.interrupted = true;
  state._engine.stage = 'complete';
  state._engine.message = 'The shoe ran out during this round. No new shoe was mixed in. This is a partial session.';
  return refresh(state);
}
function beginRound(state: TableSessionState): void {
  const e = state._engine;
  state.round++;
  e.seats = state.config.otherPlayers === 2
    ? [{ name: 'Seat 1', isUser: false, hands: [hand()] }, { name: 'You', isUser: true, hands: [hand()] }, { name: 'Seat 3', isUser: false, hands: [hand()] }]
    : [{ name: 'You', isUser: true, hands: [hand()] }];
  e.dealer = [];
  e.holeRevealed = false;
  e.initialStep = 0;
  e.activeSeat = 0;
  e.activeHand = 0;
  e.insurance = null;
  e.stage = 'initial';
  e.message = 'Keep the running count as cards appear.';
}
export function createTableSession(config: TableConfig, seed: number): TableSessionState {
  if (!['practice', 'test'].includes(config.mode) || ![0, 10].includes(config.roundLimit)
    || ![0, 2].includes(config.otherPlayers) || !Number.isFinite(config.speedMs) || config.speedMs < 100 || config.speedMs > 5000
    || !Number.isSafeInteger(seed)) throw new RangeError('Invalid table-session settings.');
  // Deck owns physical-card construction and shuffle. Its immutable draw order is then held by this engine.
  const deck = new Deck(6, 0.75, seededRandom(seed));
  const state: TableSessionState = {
    id: `table-${seed}-${config.mode}-${config.roundLimit}-${config.otherPlayers}-${config.speedMs}`, seed, config: { ...config },
    phase: 'dealing', round: 0, remaining: 312,
    table: { dealer: { cards: [], total: null }, seats: [], activeSeat: null, activeHand: null, insurance: null }, statusText: '',
    decisions: [], checkpoints: [], exposures: [], completed: false, ended: false, interrupted: false,
    _engine: { shoe: deck.draw(312), cursor: 0, runningCount: 0, exposedIds: [], seats: [], dealer: [], holeRevealed: false,
      initialStep: 0, stage: 'initial', activeSeat: 0, activeHand: 0, insurance: null, message: '' },
  };
  beginRound(state);
  return refresh(state);
}
function activeHand(state: TableSessionState): Hand { return state._engine.seats[state._engine.activeSeat].hands[state._engine.activeHand]; }
function actionsFor(state: TableSessionState): Action[] {
  const e = state._engine;
  const h = activeHand(state);
  if (h.status !== 'playing' || h.cards.length < 2 || getHandTotal(h.cards).total >= 21 || h.splitAces) return [];
  const result: Action[] = ['hit', 'stand'];
  if (h.cards.length === 2) result.push('double');
  if (isPair(h.cards) && e.seats[e.activeSeat].hands.length < 4) result.push('split');
  if (h.cards.length === 2 && !h.split) result.push('surrender');
  return result;
}
export function legalActions(state: TableSessionState): TableAction[] {
  if (state.ended) return [];
  if (state.phase === 'insurance') return ['insure', 'decline-insurance'];
  return state.phase === 'player-turn' ? actionsFor(state) : [];
}
function expected(state: TableSessionState): Action {
  const legal = actionsFor(state);
  const action = getCorrectAction(activeHand(state).cards, state._engine.dealer[0], {
    canDouble: legal.includes('double'), canSplit: legal.includes('split'), canSurrender: legal.includes('surrender'),
  });
  if (!legal.includes(action)) throw new Error('Strategy returned an unavailable action.');
  return action;
}
/** Move to the next live hand without taking another playing action or exposing another card. */
function selectNext(state: TableSessionState): void {
  const e = state._engine;
  while (e.activeSeat < e.seats.length) {
    const seat = e.seats[e.activeSeat];
    while (e.activeHand < seat.hands.length) {
      const h = seat.hands[e.activeHand];
      if (h.status === 'playing') {
        e.stage = h.cards.length === 1 ? 'hand-start' : 'players';
        e.message = seat.isUser ? 'Choose your play. Keep the count in mind.' : `${seat.name} is playing.`;
        return;
      }
      e.activeHand++;
    }
    e.activeSeat++;
    e.activeHand = 0;
  }
  e.stage = 'dealer-reveal';
  e.message = 'The dealer reveals the hole card.';
}
function performAction(state: TableSessionState, action: Action): void {
  const e = state._engine;
  const seat = e.seats[e.activeSeat];
  const h = activeHand(state);
  if (action === 'hit' || action === 'double') {
    if (action === 'double') h.units = 2;
    h.cards.push(draw(state, action, e.activeSeat, e.activeHand));
    const total = getHandTotal(h.cards).total;
    if (total > 21) h.status = 'bust';
    else if (action === 'double' || total === 21) h.status = 'stood';
    e.message = `${seat.name} ${action === 'double' ? 'doubles' : 'hits'}.`;
  } else if (action === 'stand') { h.status = 'stood'; e.message = `${seat.name} stands.`; }
  else if (action === 'surrender') { h.status = 'surrendered'; e.message = `${seat.name} surrenders.`; }
  else {
    const aces = h.cards[0].value === 'A';
    const first = { ...hand(h.cards[0]), split: true, splitAces: aces };
    const second = { ...hand(h.cards[1]), split: true, splitAces: aces };
    seat.hands.splice(e.activeHand, 1, first, second);
    e.stage = 'hand-start';
    e.message = `${seat.name} splits. Each split hand receives its next card in turn.`;
    return;
  }
  selectNext(state);
}
function recordDecision(state: TableSessionState, action: TableAction, expectedAction: TableAction, legal: TableAction[], kind: 'play' | 'insurance'): void {
  const e = state._engine;
  const seatIndex = kind === 'insurance' ? e.seats.findIndex(seat => seat.isUser) : e.activeSeat;
  const handIndex = kind === 'insurance' ? 0 : e.activeHand;
  state.decisions.push({ round: state.round, seatIndex, handIndex, kind,
    playerCards: [...e.seats[seatIndex].hands[handIndex].cards], dealerUpcard: e.dealer[0],
    action, expectedAction, legalActions: [...legal], correct: action === expectedAction,
    runningCount: e.runningCount, trueCount: toTrueCount(e.runningCount, estimateDecks(e.shoe.length - e.cursor)) });
}
export function applyTableAction(previous: TableSessionState, action: TableAction): TableSessionState {
  const legal = legalActions(previous);
  if (!legal.includes(action)) throw new RangeError('That action is not available now.');
  const state = clone(previous);
  const e = state._engine;
  if (e.cursor >= e.shoe.length) return noCrossShuffle(state);
  if (state.phase === 'insurance') {
    recordDecision(state, action, 'decline-insurance', legal, 'insurance');
    e.insurance = { taken: action === 'insure', resolved: false, netUnits: null };
    e.stage = 'peek';
    e.message = 'The dealer checks for blackjack.';
  } else {
    recordDecision(state, action, expected(state), legal, 'play');
    try { performAction(state, action as Action); }
    catch (error) { if (error instanceof RangeError && e.cursor >= e.shoe.length) return noCrossShuffle(state); throw error; }
  }
  return refresh(state);
}
function settle(state: TableSessionState): void {
  const e = state._engine;
  const dealerTotal = getHandTotal(e.dealer).total;
  const dealerNatural = e.dealer.length === 2 && dealerTotal === 21;
  for (const seat of e.seats) for (const h of seat.hands) {
    const total = getHandTotal(h.cards).total;
    h.outcome = h.status === 'surrendered' ? 'Surrender' : h.status === 'bust' ? 'Loss'
      : dealerNatural ? h.status === 'blackjack' ? 'Push' : 'Loss'
      : h.status === 'blackjack' ? 'Blackjack · 3:2'
      : dealerTotal > 21 || total > dealerTotal ? 'Win' : total === dealerTotal ? 'Push' : 'Loss';
  }
  e.stage = 'round-result';
  e.message = 'Round complete. Keep your count; the table will clear before the checkpoint.';
}
/** Each call exposes at most one card or performs one automatic decision. Never shuffles. */
export function advanceTable(previous: TableSessionState): TableSessionState {
  if (previous.ended || ['player-turn', 'insurance', 'count-checkpoint'].includes(previous.phase)) return previous;
  const state = clone(previous);
  const e = state._engine;
  try {
    if (e.stage === 'initial') {
      const width = e.seats.length + 1;
      const seatIndex = e.initialStep % width;
      const pass = Math.floor(e.initialStep / width);
      if (seatIndex === e.seats.length) e.dealer.push(draw(state, 'deal', null, null, pass === 0));
      else {
        const h = e.seats[seatIndex].hands[0];
        h.cards.push(draw(state, 'deal', seatIndex, 0));
        if (h.cards.length === 2 && getHandTotal(h.cards).total === 21) h.status = 'blackjack';
      }
      e.initialStep++;
      if (e.initialStep === width * 2) {
        e.stage = e.dealer[0].value === 'A' ? 'insurance' : 'peek';
        e.message = e.stage === 'insurance' ? 'Dealer shows an ace. Take insurance or decline?' : 'Initial deal complete.';
      }
    } else if (e.stage === 'peek') {
      const blackjack = getHandTotal(e.dealer).total === 21;
      if (e.insurance) { e.insurance.resolved = true; e.insurance.netUnits = e.insurance.taken ? blackjack ? 1 : -0.5 : 0; }
      if (blackjack) {
        e.holeRevealed = true;
        expose(state, e.dealer[1], 'hole', null, null);
        e.stage = 'settle';
        e.message = 'Dealer blackjack.';
      } else {
        e.activeSeat = 0; e.activeHand = 0;
        selectNext(state);
      }
    } else if (e.stage === 'hand-start') {
      const h = activeHand(state);
      h.cards.push(draw(state, 'split', e.activeSeat, e.activeHand));
      if (h.splitAces || getHandTotal(h.cards).total === 21) h.status = 'stood';
      selectNext(state);
    } else if (e.stage === 'players') {
      performAction(state, expected(state));
    } else if (e.stage === 'dealer-reveal') {
      e.holeRevealed = true;
      expose(state, e.dealer[1], 'hole', null, null);
      e.stage = 'dealer-play';
      e.message = 'The dealer plays: hit soft 17, stand on hard 17.';
    } else if (e.stage === 'dealer-play') {
      const contest = e.seats.some(seat => seat.hands.some(h => h.status === 'stood'));
      const info = getHandTotal(e.dealer);
      if (contest && (info.total < 17 || info.total === 17 && info.soft)) {
        e.dealer.push(draw(state, 'dealer-hit', null, null));
      } else e.stage = 'settle';
    } else if (e.stage === 'settle') settle(state);
    else if (e.stage === 'round-result') {
      e.stage = 'checkpoint';
      e.message = 'Table cleared. Enter the cumulative running count and true count.';
    }
  } catch (error) { if (error instanceof RangeError && e.cursor >= e.shoe.length) return noCrossShuffle(state); throw error; }
  return refresh(state);
}
export function submitTableCount(previous: TableSessionState, runningGuess: number | null, trueGuess: number | null): TableSessionState {
  if (previous.phase !== 'count-checkpoint' || previous.ended) throw new RangeError('There is no unanswered count checkpoint.');
  for (const value of [runningGuess, trueGuess]) if (value !== null && (!Number.isSafeInteger(value) || Math.abs(value) > 1000)) throw new RangeError('Enter a signed whole number.');
  const state = clone(previous);
  const e = state._engine;
  const remaining = e.shoe.length - e.cursor;
  if (remaining <= 0) return noCrossShuffle(state);
  const decksRemaining = estimateDecks(remaining);
  const trueTarget = toTrueCount(e.runningCount, decksRemaining);
  state.checkpoints.push({ round: state.round, runningGuess, trueGuess, runningTarget: e.runningCount, trueTarget, decksRemaining,
    runningCorrect: runningGuess === e.runningCount, trueCorrect: trueGuess === trueTarget,
    arithmeticCorrect: runningGuess !== null && trueGuess !== null ? toTrueCount(runningGuess, decksRemaining) === trueGuess : null });
  if (state.config.roundLimit === 0 ? e.cursor >= 234 : state.round >= state.config.roundLimit) {
    state.completed = true; state.ended = true; e.stage = 'complete';
    e.message = 'Session complete. Review strategy and counting separately.';
  } else beginRound(state);
  return refresh(state);
}
export function markTableInterrupted(previous: TableSessionState): TableSessionState {
  const state = clone(previous);
  state.interrupted = true;
  return refresh(state);
}
/** Ending early preserves observations without claiming completion. This function never changes the live state. */
export function finishTableSession(state: TableSessionState): TableSessionResult {
  return { version: 1, id: state.id, seed: state.seed, config: { ...state.config }, rounds: state.checkpoints.length,
    completed: state.completed, interrupted: state.interrupted, decisions: [...state.decisions], checkpoints: [...state.checkpoints],
    exposures: [...state.exposures], endedAt: new Date().toISOString() };
}
