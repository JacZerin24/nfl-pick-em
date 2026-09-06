const DATA_URL = 'data/dashboard.json';
const CT_ZONE = 'America/Chicago';

let dashboard = null;
let activeWeek = null;
let activeFilter = 'all';

const $ = (id) => document.getElementById(id);
const pct = (value, digits = 1) => value == null ? '—' : `${(Number(value) * 100).toFixed(digits)}%`;
const signed = (value) => value > 0 ? `+${value}` : `${value}`;

function parseDate(value) {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function formatCT(value, options = {}) {
  const date = parseDate(value);
  if (!date) return '—';
  return new Intl.DateTimeFormat('en-US', {
    timeZone: CT_ZONE,
    weekday: options.weekday ? 'short' : undefined,
    month: options.date ? 'short' : undefined,
    day: options.date ? 'numeric' : undefined,
    hour: 'numeric',
    minute: '2-digit',
    timeZoneName: options.zone ? 'short' : undefined,
  }).format(date);
}

function formatSnapshot(value) {
  const date = parseDate(value);
  if (!date) return '—';
  return new Intl.DateTimeFormat('en-US', {
    timeZone: CT_ZONE,
    month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', timeZoneName: 'short'
  }).format(date);
}

function isFrozen(game) {
  const kickoff = parseDate(game.kickoff_utc);
  return kickoff ? Date.now() >= kickoff.getTime() : Boolean(game.frozen);
}

function kickoffText(game) {
  const kickoff = parseDate(game.kickoff_utc);
  if (!kickoff) return 'Kickoff TBD';
  return formatCT(game.kickoff_utc, { weekday: true, date: true, zone: true });
}

function decisionInfo(type) {
  if (type === 'TRUE_UPSET_CONSENSUS') return { label: 'Upset consensus', cls: 'upset' };
  if ((type || '').startsWith('CLOSE_')) return { label: 'Close decision', cls: 'close' };
  return { label: 'Follow market', cls: 'market' };
}

function marketLine(game) {
  if (game.spread_line == null || !game.market_pick) return '—';
  const line = Math.abs(Number(game.spread_line));
  return `${game.market_pick} -${line % 1 === 0 ? line.toFixed(0) : line.toFixed(1)}`;
}

function selectedWeek() {
  return dashboard?.weeks?.find(w => Number(w.week) === Number(activeWeek));
}

function matchesFilter(game) {
  const close = Number(game.market_fav_prob) < 0.525 || (game.decision_type || '').startsWith('CLOSE_');
  const radar = game.decision_type === 'TRUE_UPSET_CONSENSUS' || Number(game.p_dog_matchup_logistic) > 0.5 || Number(game.p_dog_variance_catboost) > 0.5;
  if (activeFilter === 'close') return close;
  if (activeFilter === 'upset') return radar;
  if (activeFilter === 'frozen') return isFrozen(game);
  return true;
}

function populateHeader() {
  const s = dashboard.scoreboard || {};
  $('modelStatus').textContent = dashboard.model_version
    ? `${dashboard.model_version} • weights frozen through 2025`
    : 'Prospective model';

  const holdout = dashboard.validated_holdout || {};
  $('holdoutAccuracy').textContent = pct(holdout.accuracy, 2);
  $('holdoutDelta').textContent = `${signed(holdout.net_wins || 0)} picks vs market`;

  $('modelRecord').textContent = `${s.model_wins || 0}–${s.model_losses || 0}`;
  $('marketRecord').textContent = `${s.market_wins || 0}–${s.market_losses || 0}`;
  $('modelAccuracy').textContent = s.model_accuracy == null ? 'No games graded' : `${pct(s.model_accuracy)} accuracy`;
  $('marketAccuracy').textContent = s.market_accuracy == null ? 'No games graded' : `${pct(s.market_accuracy)} accuracy`;
  $('netEdge').textContent = signed(s.net_vs_market || 0);
  $('gradedGames').textContent = s.games || 0;
  $('lastUpdate').textContent = dashboard.latest_snapshot_utc ? formatSnapshot(dashboard.latest_snapshot_utc) : '—';
}

function populateWeeks() {
  const select = $('weekSelect');
  select.replaceChildren();
  const weeks = dashboard.weeks || [];
  weeks.forEach((week) => {
    const option = document.createElement('option');
    option.value = week.week;
    option.textContent = `Week ${week.week}`;
    select.appendChild(option);
  });
  activeWeek = dashboard.current_week ?? weeks.at(-1)?.week ?? null;
  if (activeWeek != null) select.value = activeWeek;
  select.addEventListener('change', () => {
    activeWeek = Number(select.value);
    renderGames();
  });
}

function fillGameCard(game) {
  const node = $('gameTemplate').content.firstElementChild.cloneNode(true);
  const decision = decisionInfo(game.decision_type);
  const frozen = isFrozen(game);
  const close = Number(game.market_fav_prob) < 0.525 || decision.cls === 'close';

  if (decision.cls === 'upset') node.classList.add('true-upset');
  else if (close) node.classList.add('close-call');

  node.querySelector('.kickoff').textContent = kickoffText(game);
  const lock = node.querySelector('.lock-badge');
  lock.textContent = frozen ? 'FROZEN' : 'UPDATING';
  if (!frozen) lock.classList.add('live');

  node.querySelector('.away .team-code').textContent = game.away_team || '—';
  node.querySelector('.home .team-code').textContent = game.home_team || '—';
  node.querySelector('.pick-team').textContent = game.final_pick || '—';

  const badge = node.querySelector('.decision-badge');
  badge.textContent = decision.label;
  if (decision.cls !== 'market') badge.classList.add(decision.cls);

  node.querySelector('.market-text').textContent = `${game.market_pick || '—'} ${pct(game.market_fav_prob)}`;
  const marketBar = node.querySelector('.market-bar span');
  marketBar.style.width = `${Math.max(0, Math.min(100, Number(game.market_fav_prob || 0) * 100))}%`;

  const matchup = node.querySelector('.matchup-dog');
  matchup.textContent = pct(game.p_dog_matchup_logistic);
  if (Number(game.p_dog_matchup_logistic) > 0.5) matchup.classList.add('hot');
  const variance = node.querySelector('.variance-dog');
  variance.textContent = pct(game.p_dog_variance_catboost);
  if (Number(game.p_dog_variance_catboost) > 0.5) variance.classList.add('hot');

  node.querySelector('.spread').textContent = marketLine(game);
  node.querySelector('.total').textContent = game.total_line == null ? '—' : Number(game.total_line).toFixed(1);
  node.querySelector('.residual-home').textContent = pct(game.p_home_residual);
  node.querySelector('.elo-home').textContent = pct(game.p_home_elo);
  node.querySelector('.logistic-home').textContent = pct(game.p_home_logistic);
  node.querySelector('.catboost-home').textContent = pct(game.p_home_catboost);
  node.querySelector('.lead-time').textContent = game.lead_minutes == null ? '—' : `${Math.round(game.lead_minutes)} min`;
  node.querySelector('.snapshot-time').textContent = formatSnapshot(game.snapshot_utc);

  if (game.result) {
    const strip = node.querySelector('.result-strip');
    strip.hidden = false;
    const correct = game.result.pick_correct === true;
    const tie = game.result.is_tie === true;
    strip.classList.add(correct ? 'win' : 'loss');
    const score = `${game.away_team} ${game.result.away_score} • ${game.home_team} ${game.result.home_score}`;
    strip.textContent = tie ? `TIE • ${score}` : `${correct ? '✓ WIN' : '✕ LOSS'} • ${score}`;
  }

  return node;
}

function renderGames() {
  const container = $('games');
  container.replaceChildren();
  const week = selectedWeek();
  const games = (week?.games || []).filter(matchesFilter);

  if (!games.length) {
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    empty.textContent = activeFilter === 'all' ? 'No archived games are available for this week yet.' : 'No games match this filter.';
    container.appendChild(empty);
    return;
  }

  games.forEach(game => container.appendChild(fillGameCard(game)));
}

function renderDecisionTable() {
  const byDecision = dashboard.scoreboard?.by_decision || {};
  const entries = Object.entries(byDecision);
  if (!entries.length) {
    $('decisionTable').innerHTML = '<div class="empty-state">No completed games have been graded yet. This table will populate automatically after Week 1 begins.</div>';
    return;
  }

  const labels = {
    FOLLOW_MARKET: 'Follow market',
    CLOSE_MARKET_ALIGNED: 'Close • aligned',
    CLOSE_RESIDUAL_FLIP: 'Close • residual flip',
    TRUE_UPSET_CONSENSUS: 'True upset consensus',
  };
  const rows = entries.map(([key, value]) => `
    <tr>
      <td>${labels[key] || key.replaceAll('_', ' ')}</td>
      <td>${value.games}</td>
      <td>${value.wins}–${value.losses}</td>
      <td>${pct(value.accuracy)}</td>
      <td>${signed(value.net_vs_market)}</td>
    </tr>`).join('');

  $('decisionTable').innerHTML = `
    <table>
      <thead><tr><th>Decision</th><th>Games</th><th>Record</th><th>Accuracy</th><th>Net</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function wireFilters() {
  document.querySelectorAll('.filter').forEach(button => {
    button.addEventListener('click', () => {
      document.querySelectorAll('.filter').forEach(b => b.classList.remove('active'));
      button.classList.add('active');
      activeFilter = button.dataset.filter;
      renderGames();
    });
  });
}

async function init() {
  try {
    const response = await fetch(`${DATA_URL}?v=${Date.now()}`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`Dashboard data returned ${response.status}`);
    dashboard = await response.json();
    populateHeader();
    populateWeeks();
    wireFilters();
    renderGames();
    renderDecisionTable();
    setInterval(renderGames, 60_000);
  } catch (error) {
    console.error(error);
    $('games').innerHTML = '<div class="empty-state">Dashboard data is unavailable. Check the latest GitHub Pages deployment.</div>';
    $('modelStatus').textContent = 'Dashboard data unavailable';
  }
}

document.addEventListener('DOMContentLoaded', init);
