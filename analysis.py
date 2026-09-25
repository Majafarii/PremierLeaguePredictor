# %% [markdown]
# # Premier League: performance analysis and win prediction
# What distinguishes winning performances, and how well can pre-match information
# predict a win? The target is **1 = win, 0 = draw or loss**. This analysis preserves
# the original Random Forest settings and features. Run from the repository root.

# %%
from pathlib import Path
import json
import platform
import hashlib
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    plt.switch_backend('Agg')

ARTIFACTS = Path('analysis')
FIGURES = ARTIFACTS / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titleweight': 'bold'})
BLUE, ORANGE, GREEN = '#2563A6', '#D97928', '#25816C'

def show_table(frame):
    try:
        from IPython.display import display
        display(frame.round(3))
    except ImportError:
        print(frame.round(3).to_string())

def save_figure(fig, name):
    fig.tight_layout()
    # Compact PNG encoding preserves dimensions while keeping repository files small.
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=160, bbox_inches='tight')
    buffer.seek(0)
    Image.open(buffer).convert('RGB').quantize(colors=128).save(FIGURES / name, optimize=True)
    plt.show()
    plt.close(fig)

# %% [markdown]
# ## 1. Data coverage and quality
# Each row is a **team's perspective on a match**, not a distinct fixture. A fixture
# can appear twice. Rates use team-match rows. `season` is the season's ending year.
# The CSV has no verifiable source URL or scrape date; provenance remains undocumented.

# %%
matches = pd.read_csv('matches.csv', index_col=0).reset_index(drop=True)
matches['date'] = pd.to_datetime(matches['date'])
matches['target'] = (matches['result'] == 'W').astype(int)
assert matches['result'].isin(['W', 'D', 'L']).all()
assert not matches.duplicated(['team', 'date']).any(), 'Duplicate team-match records'
coverage = matches.groupby('season').agg(
    team_match_rows=('team', 'size'), teams=('team', 'nunique'),
    first_date=('date', 'min'), last_date=('date', 'max'))
show_table(coverage)
show_table(matches.groupby(['season', 'team']).size().unstack(0).rename_axis(columns='season'))
missing = matches.isna().sum().rename('missing_rows').to_frame()
show_table(missing[missing['missing_rows'] > 0])
coverage.to_csv(ARTIFACTS / 'coverage.csv')

# %% [markdown]
# There are 1,389 team-match rows across 23 teams. 2020–21 has 760 rows (20 × 38).
# 2021–22 stops on 25 April 2022 with 629 rows across 19 teams; Liverpool has no team
# rows in that season. Use percentages and per-match averages for unequal exposure.
# Pooled comparisons still mix different seasons and schedules.
# Attendance is missing in 696 rows, notes in all rows, and shot distance in one row.
# All EDA fields are complete. We retain missing distance and drop only rolling rows
# without three complete prior observations. No dataset rows are fabricated.

# %% [markdown]
# ## 2. Which teams win most, and score/concede most?
# Counts accompany win rates. The season-level table supports comparisons within a
# season; pooled rankings are descriptive and are not a current league table.

# %%
team_summary = matches.groupby('team').agg(
    matches=('target', 'size'), wins=('target', 'sum'), win_rate=('target', 'mean'),
    goals_for=('gf', 'mean'), goals_against=('ga', 'mean'),
    xg_per_match=('xg', 'mean'), goals_total=('gf', 'sum'), xg_total=('xg', 'sum'))
team_summary['goals_minus_xg_per_match'] = team_summary['goals_for'] - team_summary['xg_per_match']
team_summary = team_summary.sort_values('win_rate', ascending=False)
season_summary = matches.groupby(['season', 'team']).agg(
    matches=('target', 'size'), win_rate=('target', 'mean'),
    goals_for=('gf', 'mean'), goals_against=('ga', 'mean'))
show_table(team_summary)
show_table(season_summary)
leader = team_summary.iloc[0]
print(f'{team_summary.index[0]} has the highest observed win rate: {leader.win_rate:.1%} '
      f'({int(leader.wins)} wins / {int(leader.matches)} matches).')
team_summary.to_csv(ARTIFACTS / 'team_summary.csv')
season_summary.to_csv(ARTIFACTS / 'team_by_season.csv')
fig, ax = plt.subplots(figsize=(10, 8))
ranked = team_summary.sort_values('win_rate')
ax.barh(ranked.index, ranked['win_rate'] * 100, color=BLUE)
for i, row in enumerate(ranked.itertuples()):
    ax.text(row.win_rate * 100 + 1, i, f'{row.win_rate:.1%} (n={row.matches})', va='center', fontsize=8)
ax.set(xlabel='Win percentage (%)', xlim=(0, 90), title='Team win rates in the observed sample')
fig.suptitle('Pooled 2020–21 and partial 2021–22 • unequal team coverage', fontsize=10)
save_figure(fig, '01_team_win_percentage.png')
fig, ax = plt.subplots(figsize=(10, 8))
ranked = team_summary.sort_values('goals_for')
y = np.arange(len(ranked))
ax.barh(y + .18, ranked['goals_for'], height=.35, label='Goals scored', color=BLUE)
ax.barh(y - .18, ranked['goals_against'], height=.35, label='Goals conceded', color=ORANGE)
ax.set(yticks=y, yticklabels=ranked.index, xlabel='Goals per team-match',
       title='Attacking output and defensive exposure by team')
ax.legend(loc='lower right')
save_figure(fig, '02_goals_for_against.png')

# %% [markdown]
# ## 3. How large is the observed home advantage?
# This describes association, not the causal effect of crowds or venue. Opponent
# strength, season and incomplete coverage can affect the gap.

# %%
venue_summary = matches.groupby('venue').agg(matches=('target', 'size'), win_rate=('target', 'mean'))
venue_by_season = matches.groupby(['season', 'venue']).agg(matches=('target', 'size'), win_rate=('target', 'mean'))
show_table(venue_summary)
show_table(venue_by_season)
print(f'Observed home-minus-away gap: '
      f'{100 * (venue_summary.loc["Home", "win_rate"] - venue_summary.loc["Away", "win_rate"]):.1f} percentage points.')
venue_summary.to_csv(ARTIFACTS / 'venue_summary.csv')
fig, ax = plt.subplots(figsize=(7, 4))
venue_plot = venue_summary.reindex(['Home', 'Away'])
bars = ax.bar(venue_plot.index, venue_plot['win_rate'] * 100, color=[BLUE, ORANGE], width=.55)
ax.bar_label(bars, labels=[f'{r.win_rate:.1%}\nn={r.matches}' for r in venue_plot.itertuples()], padding=5)
ax.set(ylabel='Win percentage (%)', ylim=(0, 65), title='Home vs away win percentage')
save_figure(fig, '03_home_advantage.png')

# %% [markdown]
# ## 4. How do shots on target relate to wins?
# These are same-match, **post-match statistics**: descriptive, not pre-match inputs.
# Group high counts to avoid presenting very small groups as stable estimates.

# %%
sot_bins = pd.cut(matches['sot'], bins=[-1, 1, 3, 5, 7, np.inf], labels=['0–1', '2–3', '4–5', '6–7', '8+'])
sot_summary = matches.groupby(sot_bins, observed=True).agg(matches=('target', 'size'), win_rate=('target', 'mean'))
show_table(sot_summary)
print(f'Win rates range from {sot_summary.iloc[0].win_rate:.1%} with 0–1 shots on target '
      f'to {sot_summary.iloc[-1].win_rate:.1%} with 8+.')
sot_summary.to_csv(ARTIFACTS / 'shots_on_target_summary.csv')
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(sot_summary.index.astype(str), sot_summary['win_rate'] * 100, color=BLUE)
ax.bar_label(bars, labels=[f'{r.win_rate:.1%}\nn={r.matches}' for r in sot_summary.itertuples()], padding=4)
ax.set(xlabel='Shots on target in the same match', ylabel='Win percentage (%)', ylim=(0, 100),
       title='Shots on target and winning performances')
save_figure(fig, '04_shots_on_target.png')

# %% [markdown]
# ## 5. How does xG relate to goals and wins? Who over/underperforms?
# xG estimates chance quality. Transparency shows overlap in team-match observations.
# The dashed line is goals = xG, not a fitted model. Over/underperformance means
# **goals minus xG per match**, not points above expectation or proof of lasting
# finishing skill. xG and goals are post-match and excluded from pre-match predictors.

# %%
xg_bins = pd.cut(matches['xg'], bins=[-np.inf, .5, 1, 1.5, 2, np.inf],
                 labels=['≤0.5', '(0.5,1]', '(1,1.5]', '(1.5,2]', '>2'])
xg_summary = matches.groupby(xg_bins, observed=True).agg(matches=('target', 'size'), win_rate=('target', 'mean'))
xg_goal_correlation = matches['xg'].corr(matches['gf'])
print(f'Pearson correlation, same-match xG and goals: {xg_goal_correlation:.3f}')
show_table(xg_summary)
show_table(team_summary[['matches', 'goals_for', 'xg_per_match', 'goals_minus_xg_per_match']]
           .sort_values('goals_minus_xg_per_match', ascending=False))
xg_summary.to_csv(ARTIFACTS / 'xg_summary.csv')
fig, axes = plt.subplots(1, 3, figsize=(18, 8), gridspec_kw={'width_ratios': [1, 1, 1.5]})
axes[0].scatter(matches['xg'], matches['gf'], alpha=.16, s=18, color=BLUE)
limit = matches['xg'].max()
axes[0].plot([0, limit], [0, limit], '--', color='gray', label='Goals = xG')
axes[0].set(xlabel='Same-match expected goals (xG)', ylabel='Goals scored', title=f'xG vs goals (r={xg_goal_correlation:.2f})')
axes[0].legend()
bars = axes[1].bar(xg_summary.index.astype(str), xg_summary['win_rate'] * 100, color=BLUE)
axes[1].bar_label(bars, labels=[f'{r.win_rate:.0%}\nn={r.matches}' for r in xg_summary.itertuples()], fontsize=8, padding=3)
axes[1].set(xlabel='Same-match xG band', ylabel='Win percentage (%)', ylim=(0, 100), title='xG vs wins')
axes[1].tick_params(axis='x', rotation=35)
ranked = team_summary.sort_values('goals_minus_xg_per_match')
axes[2].barh(ranked.index, ranked['goals_minus_xg_per_match'],
             color=[GREEN if v >= 0 else ORANGE for v in ranked['goals_minus_xg_per_match']])
axes[2].axvline(0, color='gray', linewidth=1)
axes[2].set(xlabel='Goals minus xG per team-match', title='Finishing above / below xG')
save_figure(fig, '05_expected_goals.png')

# %% [markdown]
# ## 6. Baseline and chronological evaluation
# Preserve category encodings, match hour, weekday and Random Forest settings:
# 50 trees, min_samples_split=10, random_state=1. Opponent codes have an arbitrary
# numeric ordering, an existing limitation. No tuning or random split is performed.
# Train **before 1 January 2022**, test **on or after 1 January 2022**. The original
# `>` test filter omitted New Year's Day; `>=` closes that gap.
# Accuracy: fraction of all rows correct. Win precision: fraction of predicted wins
# that actually won. Recall: fraction of actual wins recovered.

# %%
matches['venue_code'] = matches['venue'].astype('category').cat.codes
matches['opp_code'] = matches['opponent'].astype('category').cat.codes
matches['hour'] = matches['time'].str.replace(':.+', '', regex=True).astype(int)
matches['day_code'] = matches['date'].dt.dayofweek
predictors = ['venue_code', 'opp_code', 'hour', 'day_code']
CUTOFF = pd.Timestamp('2022-01-01')

def make_predictions(data, feature_names, name):
    train = data[data['date'] < CUTOFF]
    test = data[data['date'] >= CUTOFF]
    assert len(train) and len(test) and train['date'].max() < test['date'].min()
    model = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
    model.fit(train[feature_names], train['target'])
    predicted = model.predict(test[feature_names])
    row = {'model': name, 'train_rows': len(train), 'test_rows': len(test),
           'accuracy': accuracy_score(test['target'], predicted),
           'precision': precision_score(test['target'], predicted, zero_division=0),
           'recall': recall_score(test['target'], predicted, zero_division=0),
           'predicted_wins': int(predicted.sum())}
    combined = test[['date', 'team', 'opponent', 'result', 'target']].copy()
    combined = combined.rename(columns={'target': 'actual'})
    combined['predicted'] = predicted
    print(name, 'confusion matrix (rows actual 0/1, columns predicted 0/1):')
    print(confusion_matrix(test['target'], predicted, labels=[0, 1]))
    return combined, row

baseline_predictions, baseline_metrics = make_predictions(matches, predictors, 'Baseline')
show_table(pd.DataFrame([baseline_metrics]))

# %% [markdown]
# ## 7. Three-match rolling averages: previous matches only
# Sort each team's rows by date. `rolling(3, closed='left').mean()` at match t uses
# t−3, t−2, t−1, excluding t. Three complete prior observations are required.
# Features: goals for/against, shots, shots on target, shot distance, free kicks,
# penalties scored and penalties attempted. Windows span seasons as originally coded;
# they count available team matches, not calendar days or missing fixtures.
# The model is fit once before the cutoff. Later test matches may use statistics
# from earlier completed test matches: **sequential pre-match evaluation**, not a
# season-long forecast made on 1 January. Current/future match results are excluded.

# %%
cols = ['gf', 'ga', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt']
new_cols = [f'{c}_rolling' for c in cols]

def rolling_averages(group, cols, new_cols):
    group = group.sort_values('date').copy()
    group[new_cols] = group[cols].rolling(3, closed='left').mean()
    return group.dropna(subset=new_cols)

# Concatenation retains team labels across pandas versions.
matches_rolling = pd.concat([
    rolling_averages(group, cols, new_cols) for _, group in matches.groupby('team')
])
# Validate every retained window against an independent lag-then-roll expression.
for _, group in matches.groupby('team'):
    ordered = group.sort_values('date')
    expected = ordered[cols].shift(1).rolling(3).mean().dropna()
    actual = matches_rolling.loc[expected.index, new_cols]
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy())

rolling_predictions, rolling_metrics = make_predictions(matches_rolling, predictors + new_cols, 'With rolling features')
control_predictions, control_metrics = make_predictions(matches_rolling, predictors, 'Baseline on rolling-eligible rows')
assert rolling_predictions.index.equals(control_predictions.index)
results = pd.DataFrame([baseline_metrics, rolling_metrics, control_metrics]).set_index('model')
show_table(results)
results.to_csv(ARTIFACTS / 'model_results.csv')
baseline_predictions.to_csv(ARTIFACTS / 'baseline_predictions.csv', index_label='row_id')
rolling_predictions.to_csv(ARTIFACTS / 'rolling_predictions.csv', index_label='row_id')
test = matches[matches['date'] >= CUTOFF]
nonwin_accuracy = (test['target'] == 0).mean()
print(f'Always predict non-win accuracy: {nonwin_accuracy:.2%}; precision: 0 (no predicted wins).')
print(f'Adding rolling features changes accuracy by '
      f'{100 * (rolling_metrics["accuracy"] - baseline_metrics["accuracy"]):+.2f} percentage points '
      f'and precision by {100 * (rolling_metrics["precision"] - baseline_metrics["precision"]):+.2f} points.')
print(f'Rolling win recall is {rolling_metrics["recall"]:.2%}; many actual wins remain undetected.')
print(f'Rolling features remove {len(matches) - len(matches_rolling)} rows in total.')
print(f'Cutoff-day rows retained: {(matches["date"] == CUTOFF).sum()}')
run_metadata = {
    'python': platform.python_version(), 'pandas': pd.__version__,
    'numpy': np.__version__, 'scikit_learn': sklearn.__version__, 'matplotlib': matplotlib.__version__,
    'data_sha256': hashlib.sha256(Path('matches.csv').read_bytes()).hexdigest(),
    'cutoff': str(CUTOFF.date()), 'test_includes_cutoff': True,
    'always_nonwin_accuracy': float(nonwin_accuracy), 'xg_goals_correlation': float(xg_goal_correlation),
    'dropped_rolling_rows': len(matches) - len(matches_rolling),
}
(ARTIFACTS / 'run_metadata.json').write_text(json.dumps(run_metadata, indent=2) + '\n')

# %% [markdown]
# ## 8. Optional opposing-team agreement check
# Preserve the original exploration: count actual outcomes when one team's predicted
# win coincides with its opponent's predicted non-win. This subset is not the headline
# metric. Incomplete opponent coverage excludes fixtures. Join on both reciprocal
# team names and date to ensure the rows describe the same fixture.

# %%
map_values = {'Brighton and Hove Albion': 'Brighton', 'Manchester United': 'Manchester Utd',
              'Newcastle United': 'Newcastle Utd', 'Tottenham Hotspur': 'Tottenham',
              'West Ham United': 'West Ham', 'Wolverhampton Wanderers': 'Wolves'}
combined = rolling_predictions.copy()
combined['new_team'] = combined['team'].replace(map_values)
merged = combined.merge(combined, left_on=['date', 'new_team', 'opponent'],
                        right_on=['date', 'opponent', 'new_team'], validate='one_to_one')
agreement = merged[(merged['predicted_x'] == 1) & (merged['predicted_y'] == 0)]
print(f'Opposing-team agreement subset: {len(agreement)} predicted wins')
print(agreement['actual_x'].value_counts().rename(index={0: 'Non-win', 1: 'Win'}))

# %% [markdown]
# ## Interpretation and next steps
# Better precision can come with fewer predicted wins; compare recall and the
# always-non-win benchmark. Team perspectives on a fixture are related observations.
# A single holdout and incomplete historical sample do not establish future results.
# Next steps: complete coverage and provenance, multiple chronological windows,
# calibration, and training-only opponent encoding. These are separate experiments.
