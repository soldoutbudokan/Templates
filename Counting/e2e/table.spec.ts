import { expect, test, Page } from '@playwright/test';
import { getCorrectAction } from '../lib/basicStrategy';
import { legalActions } from '../lib/blackjack';
import { estimateDecks, toTrueCount, signed } from '../lib/countPolicy';
import {
  advanceChallenge, applyChallengeAction, chooseChallengeSeed, continueChallenge, createChallenge,
  practiceBetUnits, submitChallengeBet, submitChallengeCount, ChallengePresetId, ChallengeState,
} from '../lib/challenge';

const actionLabels = { hit: /^Hit(?: H)?$/, stand: /^Stand(?: S)?$/, double: /^Double(?: D)?$/, split: /^Split(?: P)?$/, surrender: /^Surrender(?: R)?$/, insure: /^Take insurance(?: I)?$/, 'decline-insurance': /^Decline insurance(?: N)?$/ };
const actionKeys = { hit: 'h', stand: 's', double: 'd', split: 'p', surrender: 'r', insure: 'i', 'decline-insurance': 'n' };

test.beforeEach(async ({ page }) => {
  // Deterministic fresh shoe and short visual deal delays; actual React UI and persistence run unchanged.
  await page.addInitScript(() => {
    Object.defineProperty(crypto, 'getRandomValues', { value: (array: Uint32Array) => { array[0] = 42; return array; } });
    const timeout = window.setTimeout.bind(window);
    window.setTimeout = ((callback: TimerHandler, delay?: number, ...args: unknown[]) =>
      timeout(callback, [650, 900, 1400].includes(delay ?? 0) ? 20 : delay, ...args)) as typeof window.setTimeout;
  });
});

async function noOverflow(page: Page) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
}

async function playSixHands(page: Page, preset: ChallengePresetId) {
  let state: ChallengeState = createChallenge(preset, chooseChallengeSeed(42));
  let captured = false;
  for (let step = 0; step < 500 && !state.completed; step++) {
    const running = state.table.exposures.at(-1)?.runningCount ?? 0;
    const trueCount = toTrueCount(running, estimateDecks(state.table.remaining));
    if (state.phase === 'bet') {
      const units = practiceBetUnits(trueCount);
      await page.getByRole('heading', { name: 'Put your count to work.' }).waitFor();
      await noOverflow(page);
      // Bet feedback cannot expose the answer before the input.
      await expect(page.locator('.challenge-bet-review')).toHaveCount(0);
      await page.waitForTimeout(280);
      await page.getByRole('button', { name: `${units} ${units === 1 ? 'unit' : 'units'}`, exact: true }).click();
      state = submitChallengeBet(state, units);
    } else if (state.phase === 'count-checkpoint') {
      await page.getByLabel('Running count', { exact: true }).waitFor();
      await noOverflow(page);
      await page.getByLabel('Running count', { exact: true }).fill(String(running));
      await page.getByLabel('True count', { exact: true }).fill(String(trueCount));
      await page.waitForTimeout(280);
      await page.getByRole('button', { name: 'Check my count' }).click();
      state = submitChallengeCount(state, running, trueCount);
    } else if (state.phase === 'feedback') {
      await expect(page.getByRole('heading', { name: `Carry ${signed(running)} forward.` })).toBeVisible();
      await page.waitForTimeout(280);
      await page.getByRole('button', { name: state.table.round === 6 ? 'See my results' : 'Carry this count' }).click();
      state = continueChallenge(state);
    } else if ((state.phase === 'player-turn' || state.phase === 'insurance') && preset !== 'rhythm') {
      const view = state.table.table;
      const legal = legalActions(state.table);
      const action = state.phase === 'insurance' ? 'decline-insurance' : getCorrectAction(
        view.seats[view.activeSeat!].hands[view.activeHand!].cards, view.dealer.cards[0]!,
        { canDouble: legal.includes('double'), canSplit: legal.includes('split'), canSurrender: legal.includes('surrender') },
      );
      await page.getByRole('button', { name: actionLabels[action], exact: true }).waitFor();
      const firstAction = !captured;
      if (firstAction) {
        await noOverflow(page);
        await page.screenshot({ path: `test-results/${preset}-table.png`, fullPage: true }); captured = true;
      }
      await page.waitForTimeout(280);
      const button = page.getByRole('button', { name: actionLabels[action], exact: true });
      if (firstAction) {
        await button.focus();
        await page.keyboard.press(actionKeys[action]);
      } else await button.click();
      state = applyChallengeAction(state, action);
    } else state = advanceChallenge(state);
  }
  expect(state.completed).toBe(true);
  await expect(page.getByRole('button', { name: 'Fresh shoe · play again' })).toBeVisible();
  await expect(page.locator('.challenge-verdict')).toContainText('100%');
  await noOverflow(page);
}

test('desktop: complete six hands, save a record, reload and inspect the replay', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('/');
  await page.locator('.challenge-start').waitFor();
  await page.screenshot({ path: 'test-results/desktop-home.png', fullPage: true });
  await page.locator('.challenge-start').click();
  await playSixHands(page, 'rhythm');
  await page.screenshot({ path: 'test-results/desktop-results.png', fullPage: true });
  await page.reload();
  await expect(page.locator('.challenge-records')).toContainText('100%');
  await page.getByRole('button', { name: 'Progress', exact: true }).click();
  await page.getByRole('button', { name: /Find your rhythm/ }).click();
  await page.getByText('Replay the exposed cards', { exact: true }).click();
  await page.getByRole('button', { name: 'Next card', exact: false }).click();
  await expect(page.locator('.challenge-replay-cards [role="img"]')).toHaveCount(1);
  expect(errors).toEqual([]);
});

test('phone: manual play, delayed checks and all six hands fit at 360px', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  await page.setViewportSize({ width: 360, height: 800 });
  await page.goto('/');
  await page.getByRole('button', { name: /02 .*Read the table/ }).click();
  await noOverflow(page);
  await page.screenshot({ path: 'test-results/phone-home.png', fullPage: true });
  await page.locator('.challenge-start').click();
  await playSixHands(page, 'table');
  await page.screenshot({ path: 'test-results/phone-results.png', fullPage: true });
  expect(errors).toEqual([]);
});

test('pause covers cards; lost-count practice survives reload without a record', async ({ page }) => {
  await page.goto('/');
  await page.locator('.challenge-start').click();
  await page.getByRole('button', { name: 'Pause', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Your table is waiting.' })).toBeVisible();
  await expect(page.locator('.challenge-felt')).toHaveCount(0);
  await page.getByRole('button', { name: 'Resume table' }).click();
  await page.getByRole('button', { name: 'I lost the count' }).click();
  await expect(page.getByText('Reset your count and keep going', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'End and review' }).click();
  await expect(page.getByText('You paused this run.', { exact: false })).toBeVisible();
  await page.reload();
  await expect(page.locator('.challenge-records').getByText('100%', { exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Progress', exact: true }).click();
  await expect(page.getByRole('button', { name: /Find your rhythm.*Paused practice/ })).toBeVisible();
});
