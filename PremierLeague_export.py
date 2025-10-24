# --- Cell 0 ---
import pandas as pd

# --- Cell 1 ---
matches = pd.read_csv("matches.csv", index_col = 0)

# --- Cell 2 ---
matches["date"] = pd.to_datetime(matches["date"])

# --- Cell 3 ---
matches["venue_code"] = matches["venue"].astype("category").cat.codes

# --- Cell 4 ---
matches["opp_code"] = matches["opponent"].astype("category").cat.codes

# --- Cell 5 ---
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")

# --- Cell 6 ---
matches["day_code"] = matches["date"].dt.dayofweek

# --- Cell 7 ---
matches["target"] = (matches["result"] == "W").astype("int") 

# --- Cell 8 ---
from sklearn.ensemble import RandomForestClassifier

# --- Cell 9 ---
rf = RandomForestClassifier(n_estimators = 50, min_samples_split=10, random_state=1)

# --- Cell 10 ---
train = matches[matches["date"]<'2022-01-01']

# --- Cell 11 ---
test = matches[matches["date"]> '2022-01-01']

# --- Cell 12 ---
predictors = ["venue_code", "opp_code", "hour", "day_code"]

# --- Cell 13 ---
rf.fit(train[predictors], train["target"])

# --- Cell 14 ---
preds = rf.predict(test[predictors])

# --- Cell 15 ---
from sklearn.metrics import accuracy_score

# --- Cell 16 ---
acc = accuracy_score(test["target"], preds)

# --- Cell 17 ---
acc

# --- Cell 18 ---
combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))

# --- Cell 19 ---
pd.crosstab(index=combined["actual"], columns=combined["prediction"])

# --- Cell 20 ---
from sklearn.metrics import precision_score

# --- Cell 21 ---
precision_score(test["target"], preds)

# --- Cell 22 ---
grouped_matches = matches.groupby("team")

# --- Cell 23 ---
group = grouped_matches.get_group("Manchester City").sort_values("date")

# --- Cell 24 ---
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

# --- Cell 25 ---
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

rolling_averages(group, cols, new_cols)

# --- Cell 26 ---
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))

# --- Cell 27 ---
matches_rolling

# --- Cell 28 ---
matches_rolling = matches_rolling.droplevel('team')

# --- Cell 29 ---
matches_rolling

# --- Cell 30 ---
matches_rolling.index = range(matches_rolling.shape[0])

# --- Cell 31 ---
def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    error = precision_score(test["target"], preds)
    return combined, error

# --- Cell 32 ---
combined, error = make_predictions(matches_rolling, predictors + new_cols)

# --- Cell 33 ---
error

# --- Cell 34 ---
combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)

# --- Cell 35 ---

combined.head(10)

# --- Cell 36 ---

class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {"Brighton and Hove Albion": "Brighton", "Manchester United": "Manchester Utd", "Newcastle United": "Newcastle Utd", "Tottenham Hotspur": "Tottenham", "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves"} 
mapping = MissingDict(**map_values)

# --- Cell 37 ---

combined["new_team"] = combined["team"].map(mapping)

# --- Cell 38 ---
merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])

# --- Cell 39 ---
merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] ==0)]["actual_x"].value_counts()

# --- Cell 40 ---

