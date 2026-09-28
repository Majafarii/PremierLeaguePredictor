"""Build an offline matchup demo using the evaluated historical Random Forest.

Run `python build_web.py` from any directory. No live data or backend is needed.
The model is trained before 2022-01-01, exactly as in analysis.py. A hypothetical
next fixture uses each team's last three completed observations in matches.csv.
"""
from pathlib import Path
import hashlib
import json
import re
import base64

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent
COLS = ['gf', 'ga', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt']
ROLLING = [f'{col}_rolling' for col in COLS]
FEATURES = ['venue_code', 'opp_code', 'hour', 'day_code'] + ROLLING
ALIASES = {'Brighton and Hove Albion': 'Brighton', 'Manchester United': 'Manchester Utd',
           'Newcastle United': 'Newcastle Utd', 'Tottenham Hotspur': 'Tottenham',
           'West Ham United': 'West Ham', 'Wolverhampton Wanderers': 'Wolves',
           'Sheffield United': 'Sheffield Utd', 'West Bromwich Albion': 'West Brom'}
CUTOFF = pd.Timestamp('2022-01-01')


def prepare_model():
    matches = pd.read_csv(ROOT / 'matches.csv', index_col=0).reset_index(drop=True)
    matches['date'] = pd.to_datetime(matches['date'])
    matches['target'] = (matches['result'] == 'W').astype(int)
    for column, output in [('venue', 'venue_code'), ('opponent', 'opp_code')]:
        matches[output] = matches[column].astype('category').cat.codes
    matches['hour'] = matches['time'].str.split(':').str[0].astype(int)
    matches['day_code'] = matches['date'].dt.dayofweek
    groups = []
    for _, group in matches.groupby('team'):
        group = group.sort_values('date').copy()
        group[ROLLING] = group[COLS].rolling(3, closed='left').mean()
        groups.append(group.dropna(subset=ROLLING))
    rolling = pd.concat(groups)
    train = rolling[rolling.date < CUTOFF]
    test = rolling[rolling.date >= CUTOFF]
    model = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
    model.fit(train[FEATURES], train.target)
    predictions = model.predict(test[FEATURES])
    metrics = {'accuracy': accuracy_score(test.target, predictions),
               'precision': precision_score(test.target, predictions),
               'recall': recall_score(test.target, predictions), 'testRows': len(test)}
    return matches, train, model, metrics


def build_payload():
    matches, train, model, metrics = prepare_model()
    opponent_codes = matches.groupby('opponent').opp_code.first().to_dict()
    venue_codes = matches.groupby('venue').venue_code.first().to_dict()
    teams, vectors = [], []
    for name, group in matches.groupby('team'):
        recent = group.sort_values('date').tail(3)
        if len(recent) != 3 or recent[COLS].isna().any().any():
            raise ValueError(f'{name} lacks three complete latest matches')
        opponent_name = ALIASES.get(name, name)
        if opponent_name not in opponent_codes:
            raise ValueError(f'No opponent encoding for {name}')
        averages = recent[COLS].mean()
        teams.append({'name': name, 'opponentCode': int(opponent_codes[opponent_name]),
                      'lastDate': recent.date.max().strftime('%Y-%m-%d'),
                      'firstDate': recent.date.min().strftime('%Y-%m-%d'),
                      'form': recent.result.tolist(),
                      'stats': {col: round(float(averages[col]), 4) for col in COLS},
                      'recent': [{'date': row.date.strftime('%Y-%m-%d'), 'opponent': row.opponent,
                                  'venue': row.venue, 'result': row.result,
                                  'gf': int(row.gf), 'ga': int(row.ga)}
                                 for row in recent.itertuples()]})
        vectors.append(averages.to_numpy())

    # One batch contains each viewpoint plus three controlled model sensitivities.
    # Attack/defense use TRAINING medians; venue toggles home/away, all else fixed.
    feature_rows = []
    keys = []
    medians = train[FEATURES].median()
    for home in range(len(teams)):
        for away in range(len(teams)):
            if home == away:
                continue
            keys.append(f'{home}:{away}')
            for own, other, venue in [(home, away, 'Home'), (away, home, 'Away')]:
                original = pd.Series([venue_codes[venue], teams[other]['opponentCode'], 15, 5,
                                      *vectors[own]], index=FEATURES)
                feature_rows.append(original)
                attack = original.copy()
                for feature in ['gf_rolling', 'sh_rolling', 'sot_rolling']:
                    attack[feature] = medians[feature]
                feature_rows.append(attack)
                defense = original.copy()
                defense['ga_rolling'] = medians['ga_rolling']
                feature_rows.append(defense)
                venue_change = original.copy()
                venue_change['venue_code'] = venue_codes['Away' if venue == 'Home' else 'Home']
                feature_rows.append(venue_change)
    scores = model.predict_proba(pd.DataFrame(feature_rows, columns=FEATURES))[:, 1]
    pairings = {}
    for position, key in enumerate(keys):
        home, ha, hd, hv, away, aa, ad, av = scores[position * 8:position * 8 + 8]
        pairings[key] = [round(float(value), 6) for value in
                         [home, away, home - ha, home - hd, home - hv,
                          away - aa, away - ad, away - av]]
    assert len(pairings) == len(teams) * (len(teams) - 1)
    assert all(0 <= pair[0] <= 1 and 0 <= pair[1] <= 1 for pair in pairings.values())
    return {'teams': teams, 'pairings': pairings, 'metrics': metrics,
            'dataThrough': matches.date.max().strftime('%Y-%m-%d'),
            'trainBefore': CUTOFF.strftime('%Y-%m-%d'),
            'assumption': 'Saturday, 15:00',
            'dataHash': hashlib.sha256((ROOT / 'matches.csv').read_bytes()).hexdigest()}


def main():
    payload = build_payload()
    web = ROOT / 'web'
    web.mkdir(exist_ok=True)
    (web / 'data.js').write_text('window.MATCHUP_DATA = ' + json.dumps(payload, separators=(',', ':')) + ';\n', encoding='utf-8')
    # Portable one-file edition: opening this HTML needs neither Python nor a server.
    html = (web / 'index.html').read_text(encoding='utf-8')
    for sheet in ['styles.css', 'theme.css', 'typography.css']:
        style = '<style>' + (web / sheet).read_text(encoding='utf-8') + '</style>'
        html = re.sub(r'<link rel="stylesheet" href="' + re.escape(sheet) + r'(?:\?[^"]*)?">', lambda _: style, html)
    font = base64.b64encode((web / 'fonts/albert-sans-latin.woff2').read_bytes()).decode()
    html = html.replace("url('fonts/albert-sans-latin.woff2')", "url('data:font/woff2;base64," + font + "')")
    html += '\n<!-- Bundled Albert Sans font license:\n' + (web / 'fonts/OFL.txt').read_text(encoding='utf-8') + '\n-->\n'
    for name in ['data.js', 'predict.js', 'app.js']:
        source = (web / name).read_text(encoding='utf-8').replace('</script', '<\\/script')
        html = html.replace(f'<script src="{name}" defer></script>', '<script defer>' + source + '</script>')
    (web / 'matchday.html').write_text(html, encoding='utf-8')
    print(f'Built {len(payload["teams"])} teams and {len(payload["pairings"])} ordered matchups.')
    print('Open web/index.html or the portable web/matchday.html in a browser.')


if __name__ == '__main__':
    main()
