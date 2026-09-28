const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const {getMatchup, isStale} = require('../web/predict.js');
const context = {window: {}};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../web/data.js'), 'utf8'), context);
const data = context.window.MATCHUP_DATA;

test('all 506 distinct ordered matchups exist and have valid scores', () => {
  assert.equal(data.teams.length, 23);
  assert.equal(Object.keys(data.pairings).length, 506);
  data.teams.forEach((_, home) => data.teams.forEach((__, away) => {
    if (home === away) return;
    const match = getMatchup(data, home, away);
    assert.ok(match.scores.slice(0, 2).every(score => score >= 0 && score <= 1));
    assert.ok(match.scores.slice(2).every(delta => Math.abs(delta) <= 1));
    if (match.winner !== null) {
      const side = match.winner === home ? 0 : 1;
      assert.ok(match.scores[side] > .5 && match.scores[1 - side] <= .5);
    }
  }));
});
test('same teams and invalid team IDs are rejected', () => {
  assert.throws(() => getMatchup(data, 0, 0), /different teams/);
  for (const bad of [-1, 23, NaN, 0.5, '0']) assert.throws(() => getMatchup(data, bad, 1));
});
test('uncertainty, conflicts and exact 50% do not fabricate winners or draws', () => {
  const example = scores => ({teams: [{}, {}], pairings: {'0:1': [...scores, 0, 0, 0, 0, 0, 0]}});
  assert.equal(getMatchup(example([.4, .3]), 0, 1).winner, null);
  assert.equal(getMatchup(example([.8, .6]), 0, 1).conflict, true);
  assert.equal(getMatchup(example([.8, .6]), 0, 1).winner, null);
  assert.equal(getMatchup(example([.5, .5]), 0, 1).winner, null);
  assert.equal(getMatchup(example([.6, .4]), 0, 1).winner, 0);
  assert.equal(getMatchup(example([.4, .6]), 0, 1).winner, 1);
});
test('venue swap uses a separate model evaluation', () => {
  const city = data.teams.findIndex(team => team.name === 'Manchester City');
  const arsenal = data.teams.findIndex(team => team.name === 'Arsenal');
  const first = getMatchup(data, city, arsenal);
  const swapped = getMatchup(data, arsenal, city);
  assert.notEqual(first.scores[0], swapped.scores[1]);
});
test('old Liverpool snapshot is explicitly identified', () => {
  const liverpool = data.teams.find(team => team.name === 'Liverpool');
  assert.equal(liverpool.lastDate, '2021-05-23');
  assert.ok(isStale(liverpool, data.dataThrough));
  assert.ok(!isStale(data.teams.find(team => team.name === 'Manchester City'), data.dataThrough));
});
