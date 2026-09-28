import unittest
import numpy as np
import pandas as pd
from build_web import ROOT, COLS, FEATURES, ALIASES, build_payload, prepare_model


class WebsiteModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = build_payload()
        cls.matches, cls.train, cls.model, cls.metrics = prepare_model()

    def test_holdout_reproduces_existing_rolling_model(self):
        recorded = pd.read_csv(ROOT / 'analysis/model_results.csv').set_index('model').loc['With rolling features']
        for key in ['accuracy', 'precision', 'recall']:
            self.assertAlmostEqual(self.metrics[key], recorded[key], places=12)
        self.assertEqual(self.metrics['testRows'], recorded.test_rows)

    def test_snapshots_use_latest_three_completed_games(self):
        for team in self.payload['teams']:
            recent = self.matches[self.matches.team == team['name']].sort_values('date').tail(3)
            np.testing.assert_allclose([team['stats'][col] for col in COLS], recent[COLS].mean(), atol=0.00005)
            self.assertEqual(team['lastDate'], recent.date.max().strftime('%Y-%m-%d'))
            self.assertEqual(team['form'], recent.result.tolist())

    def test_both_viewpoints_and_sensitivity_are_real_model_outputs(self):
        teams = self.payload['teams']
        home = next(i for i, team in enumerate(teams) if team['name'] == 'Manchester City')
        away = next(i for i, team in enumerate(teams) if team['name'] == 'Arsenal')
        saved = self.payload['pairings'][f'{home}:{away}']
        for own, other, venue, offset in [(home, away, 'Home', 0), (away, home, 'Away', 1)]:
            recent = self.matches[self.matches.team == teams[own]['name']].sort_values('date').tail(3)
            venue_code = self.matches.loc[self.matches.venue == venue, 'venue_code'].iloc[0]
            opponent_code = self.matches.loc[self.matches.opponent == ALIASES.get(teams[other]['name'], teams[other]['name']), 'opp_code'].iloc[0]
            original = pd.DataFrame([[venue_code, opponent_code, 15, 5, *recent[COLS].mean()]], columns=FEATURES)
            score = self.model.predict_proba(original)[0, 1]
            self.assertAlmostEqual(saved[offset], score, places=6)
            typical = original.copy()
            for col in ['gf_rolling', 'sh_rolling', 'sot_rolling']:
                typical[col] = self.train[col].median()
            self.assertAlmostEqual(saved[2 if offset == 0 else 5], score - self.model.predict_proba(typical)[0, 1], places=6)


if __name__ == '__main__':
    unittest.main()
