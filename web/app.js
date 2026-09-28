(function () {
  'use strict';
  function boot() {
    const data = window.MATCHUP_DATA;
    const $ = id => document.getElementById(id);
    const home = $('home-team');
    const away = $('away-team');
    const pct = value => (value * 100).toFixed(1) + '%';
    const date = value => new Intl.DateTimeFormat('en-GB', {day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC'}).format(new Date(value + 'T00:00:00Z'));
    const el = (tag, className, text) => {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined) node.textContent = text;
      return node;
    };
    if (!data || !window.Matchup) {
      $('form-error').hidden = false;
      $('form-error').textContent = 'The model data could not be loaded. Keep the web folder together, or open matchday.html.';
      document.querySelector('.predict-button').disabled = true;
      return;
    }
    data.teams.forEach((team, index) => {
      home.add(new Option(team.name, index));
      away.add(new Option(team.name, index));
    });
    home.value = data.teams.findIndex(team => team.name === 'Manchester City');
    away.value = data.teams.findIndex(team => team.name === 'Arsenal');

    function renderBars(match) {
      $('score-bars').replaceChildren();
      [match.home, match.away].forEach((index, side) => {
        const row = el('div', 'score-row');
        const label = el('div', 'score-label');
        const team = el('span', '', data.teams[index].name);
        team.append(el('small', '', side === 0 ? 'HOME' : 'AWAY'));
        label.append(team, el('strong', '', pct(match.scores[side])));
        const track = el('div', 'bar-track');
        const fill = el('div', 'bar-fill' + (side ? ' away' : ''));
        fill.style.width = pct(match.scores[side]);
        track.append(fill);
        row.append(label, track);
        $('score-bars').append(row);
      });
    }

    function renderReasons(match) {
      const focus = match.winner ?? match.lead ?? match.home;
      const team = data.teams[focus];
      const offset = focus === match.home ? 2 : 5;
      const base = match.scores[focus === match.home ? 0 : 1];
      const drivers = [
        {label: 'Attacking form', icon: '↗', value: match.scores[offset], detail: `${team.name} averages ${team.stats.gf.toFixed(2)} goals, ${team.stats.sh.toFixed(2)} shots and ${team.stats.sot.toFixed(2)} shots on target in its last three recorded matches.`, comparison: 'typical training-period attacking form'},
        {label: 'Defensive form', icon: '◇', value: match.scores[offset + 1], detail: `${team.name} concedes ${team.stats.ga.toFixed(2)} goals per match across its last three recorded games.`, comparison: 'typical training-period goals conceded'},
        {label: 'Home or away', icon: '⌂', value: match.scores[offset + 2], detail: `${team.name} is the ${focus === match.home ? 'home' : 'away'} side here. Swapping its venue alone changes the model’s estimate.`, comparison: `playing ${focus === match.home ? 'away' : 'at home'} with the same opponent and form`}
      ].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 2);
      $('reasons').replaceChildren();
      drivers.forEach(driver => {
        const card = el('article', 'reason-card');
        card.append(el('span', 'reason-icon', driver.icon), el('h3', '', driver.label));
        card.append(el('p', 'reason-detail', driver.detail));
        const amount = Math.abs(driver.value * 100).toFixed(1);
        const change = amount === '0.0' ? 'No material shift' : `${driver.value > 0 ? '+' : '−'}${amount} score points`;
        card.append(el('p', 'reason-effect', change));
        card.append(el('p', 'reason-context', `${team.name}: ${pct(base)} vs ${pct(base - driver.value)} with ${driver.comparison}.`));
        $('reasons').append(card);
      });
    }

    function renderForm(match) {
      const selected = [data.teams[match.home], data.teams[match.away]];
      const table = el('table', 'comparison-table');
      const head = el('thead');
      const heading = el('tr');
      heading.append(el('th', '', 'Recent form'));
      selected.forEach((team, index) => {
        const th = el('th', '', team.name);
        th.scope = 'col';
        th.append(el('small', '', index === 0 ? 'HOME' : 'AWAY'));
        heading.append(th);
      });
      head.append(heading);
      table.append(head);
      const body = el('tbody');
      const formRow = el('tr');
      const formLabel = el('th', '', 'Last three results');
      formLabel.scope = 'row';
      formLabel.append(el('small', '', 'Oldest → newest'));
      formRow.append(formLabel);
      selected.forEach(team => {
        const td = el('td');
        const form = el('div', 'form-badges');
        team.form.forEach(result => {
          const badge = el('span', 'form-badge ' + result.toLowerCase(), result);
          badge.title = {W: 'Win', D: 'Draw', L: 'Loss'}[result];
          form.append(badge);
        });
        td.append(form);
        formRow.append(td);
      });
      body.append(formRow);
      [['Goals scored', 'gf', false], ['Goals conceded', 'ga', true], ['Shots on target', 'sot', false], ['Total shots', 'sh', false]].forEach(([label, key, lower]) => {
        const row = el('tr');
        const th = el('th', '', label);
        th.scope = 'row';
        row.append(th);
        selected.forEach((team, side) => {
          const other = selected[1 - side].stats[key];
          const better = lower ? team.stats[key] < other : team.stats[key] > other;
          row.append(el('td', better ? 'stat-edge' : '', team.stats[key].toFixed(2)));
        });
        body.append(row);
      });
      const coverage = el('tr', 'coverage-row');
      coverage.append(el('th', '', 'Last recorded match'));
      selected.forEach(team => coverage.append(el('td', '', date(team.lastDate))));
      body.append(coverage);
      table.append(body);
      $('form-comparison').replaceChildren(table);
      const stale = selected.filter(team => window.Matchup.isStale(team, data.dataThrough));
      $('coverage-warning').hidden = stale.length === 0;
      $('coverage-warning').textContent = stale.map(team => `${team.name}’s records end on ${date(team.lastDate)}.`).join(' ') + ' These older snapshots are not comparable current form.';
      $('recent-matches').replaceChildren();
      selected.forEach(team => {
        const column = el('div');
        column.append(el('h3', '', team.name));
        team.recent.forEach(game => {
          const row = el('div', 'recent-row');
          const info = el('div');
          info.append(el('span', '', game.opponent), el('small', '', `${date(game.date)} · ${game.venue}`));
          row.append(info, el('strong', '', `${game.gf} – ${game.ga}`));
          column.append(row);
        });
        $('recent-matches').append(column);
      });
    }

    function predict() {
      let match;
      try { match = window.Matchup.getMatchup(data, Number(home.value), Number(away.value)); }
      catch (error) {
        $('form-error').textContent = error.message;
        $('form-error').hidden = false;
        return;
      }
      $('form-error').hidden = true;
      $('result').classList.remove('pending');
      if (match.winner !== null) {
        $('result-title').textContent = `${data.teams[match.winner].name} has the edge.`;
        $('verdict-badge').textContent = match.winner === match.home ? 'Home side favoured' : 'Away side favoured';
        $('result-description').textContent = 'The model predicts a win for this side and a non-win for its opponent. It is a lean, not a guaranteed result.';
        $('result-status').textContent = 'One side above the 50% threshold';
      } else {
        $('result-title').textContent = 'No clear winner.';
        $('verdict-badge').textContent = match.conflict ? 'Conflicting signals' : 'Uncertain matchup';
        $('result-description').textContent = match.conflict ? 'Both independent team scores exceed 50%, so they disagree about the outcome. The model cannot make a consistent pick.' : 'Neither team clears the win threshold. A draw or either team winning remains possible; this model cannot predict a draw separately.';
        $('result-status').textContent = match.lead === null ? 'Equal model scores' : `Higher score: ${data.teams[match.lead].name}`;
      }
      renderBars(match);
      renderReasons(match);
      renderForm(match);
    }
    $('match-form').addEventListener('submit', event => {event.preventDefault(); predict();});
    function markPending() {
      $('form-error').hidden = true;
      $('result').classList.add('pending');
      $('result-status').textContent = 'Selection changed — press Predict matchup';
    }
    home.addEventListener('change', markPending);
    away.addEventListener('change', markPending);
    $('swap').addEventListener('click', () => {
      const old = home.value;
      home.value = away.value;
      away.value = old;
      predict();
    });
    $('evaluation-copy').textContent = `Trained on matches before 1 January 2022, then tested on ${data.metrics.testRows} later team-match records. The rolling model achieved ${pct(data.metrics.accuracy)} accuracy, ${pct(data.metrics.precision)} win precision and ${pct(data.metrics.recall)} win recall. Win precision is the share of predicted wins that really won; recall is the share of actual wins found.`;
    predict();
  }
  // Supports both external deferred scripts and the bundled, offline HTML edition.
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
