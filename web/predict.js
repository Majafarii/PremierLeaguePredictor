/* Shared, dependency-free decision rules for the browser and Node tests. */
(function (root) {
  'use strict';
  function getMatchup(data, home, away) {
    if (!Number.isInteger(home) || !Number.isInteger(away) || !data.teams[home] || !data.teams[away]) {
      throw new Error('Choose two teams from the list.');
    }
    if (home === away) throw new Error('Choose two different teams to make a matchup.');
    const scores = data.pairings[`${home}:${away}`];
    if (!scores || scores.length !== 8 || !scores.every(Number.isFinite)) throw new Error('This matchup is unavailable.');
    const homeWin = scores[0] > 0.5;
    const awayWin = scores[1] > 0.5;
    const winner = homeWin !== awayWin ? (homeWin ? home : away) : null;
    return {home, away, scores, winner, conflict: homeWin && awayWin,
      lead: scores[0] === scores[1] ? null : (scores[0] > scores[1] ? home : away)};
  }
  function isStale(team, through) {
    return (Date.parse(through + 'T00:00:00Z') - Date.parse(team.lastDate + 'T00:00:00Z')) / 86400000 > 60;
  }
  const api = {getMatchup, isStale};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Matchup = api;
})(typeof window !== 'undefined' ? window : globalThis);
