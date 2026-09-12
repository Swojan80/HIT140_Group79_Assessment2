import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# ============================================================
# HIT140 Assessment 2 - Checkpoint 2
# Swastik Bista - Yellow Cards Analysis
# ============================================================

OUTPUT = Path("swastik_yellow_cards_checkpoint2")
OUTPUT.mkdir(exist_ok=True)

EVENTS_URL = (
    "https://raw.githubusercontent.com/Alamyy/Worldcup26/refs/heads/main/"
    "data/csv/player_events.csv"
)
MATCHES_URL = (
    "https://raw.githubusercontent.com/Alamyy/Worldcup26/refs/heads/main/"
    "data/csv/matches.csv"
)

# ------------------------------------------------------------
# 1. LOAD THE DATA
# ------------------------------------------------------------
events = pd.read_csv(EVENTS_URL)
matches = pd.read_csv(MATCHES_URL)

# ------------------------------------------------------------
# 2. DATA WRANGLING
# Count yellow-card events for each team in each match.
# ------------------------------------------------------------
cards = events.loc[
    events["event_type"].eq("yellow_card"),
    ["match_id", "team", "team_id"]
].copy()

card_counts = (
    cards.groupby(["match_id", "team", "team_id"])
    .size()
    .reset_index(name="yellow_cards")
)

# ------------------------------------------------------------
# 3. DATA PREPARATION
# Create one observation for EACH TEAM in EACH MATCH.
# This is important because a team can receive zero cards.
# ------------------------------------------------------------
home = matches[
    ["match_id", "match_number", "date", "group",
     "home_team", "home_team_id", "home_score", "away_score"]
].copy()

home.columns = [
    "match_id", "match_number", "date", "stage",
    "team", "team_id", "team_score", "opponent_score"
]

away = matches[
    ["match_id", "match_number", "date", "group",
     "away_team", "away_team_id", "away_score", "home_score"]
].copy()

away.columns = [
    "match_id", "match_number", "date", "stage",
    "team", "team_id", "team_score", "opponent_score"
]

team_rows = pd.concat([home, away], ignore_index=True)

team_rows["stage"] = np.where(
    team_rows["stage"].astype(str).str.startswith("Group"),
    "Group-stage",
    "Knockout-stage"
)

team_rows["yellow_cards"] = 0

df = team_rows.merge(
    card_counts,
    on=["match_id", "team", "team_id"],
    how="left",
    suffixes=("", "_from_events")
)

df["yellow_cards"] = (
    df["yellow_cards_from_events"].fillna(0).astype(int)
)

df = df.drop(columns=["yellow_cards_from_events"])
df["match_number"] = df["match_number"].astype(int)
df = df.sort_values(["match_number", "team"]).reset_index(drop=True)

# ------------------------------------------------------------
# 4. QUALITY CHECKS
# FIFA World Cup 2026 has 104 matches = 208 team-match rows.
# ------------------------------------------------------------
assert len(matches) == 104, (
    f"Expected 104 matches, found {len(matches)}"
)
assert len(df) == 208, (
    f"Expected 208 team-match observations, found {len(df)}"
)
assert df["yellow_cards"].ge(0).all()
assert df["stage"].isin(
    ["Group-stage", "Knockout-stage"]
).all()

# Save the actual analysis-ready dataset.
df.to_csv(
    OUTPUT / "swastik_yellow_cards_dataset.csv",
    index=False
)

# ------------------------------------------------------------
# 5. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------
summary = (
    df.groupby("stage")["yellow_cards"]
    .agg(
        n="count",
        mean="mean",
        median="median",
        std="std",
        minimum="min",
        maximum="max"
    )
    .reset_index()
)

summary.to_csv(
    OUTPUT / "descriptive_statistics.csv",
    index=False
)

# ------------------------------------------------------------
# 6. HYPOTHESES
#
# H0: There is no difference in mean yellow cards.
# H1: There is a difference in mean yellow cards.
#
# Two-sided Welch two-sample t-test is used because the
# assignment requires a two-sample t-test and Welch's version
# does not require equal population variances.
# ------------------------------------------------------------
group = df.loc[
    df["stage"].eq("Group-stage"),
    "yellow_cards"
]

knockout = df.loc[
    df["stage"].eq("Knockout-stage"),
    "yellow_cards"
]

t_stat, p_value = stats.ttest_ind(
    group,
    knockout,
    equal_var=False
)

# ------------------------------------------------------------
# 7. 95% CONFIDENCE INTERVAL
# Difference is defined as:
# Group-stage mean - Knockout-stage mean
# ------------------------------------------------------------
n1, n2 = len(group), len(knockout)
m1, m2 = group.mean(), knockout.mean()
s1, s2 = group.std(ddof=1), knockout.std(ddof=1)

difference = m1 - m2

standard_error = np.sqrt(
    (s1 ** 2 / n1) + (s2 ** 2 / n2)
)

welch_df = (
    ((s1 ** 2 / n1) + (s2 ** 2 / n2)) ** 2
    /
    (
        ((s1 ** 2 / n1) ** 2 / (n1 - 1))
        +
        ((s2 ** 2 / n2) ** 2 / (n2 - 1))
    )
)

critical_value = stats.t.ppf(
    0.975,
    welch_df
)

ci_low = difference - critical_value * standard_error
ci_high = difference + critical_value * standard_error

decision = (
    "Reject H0"
    if p_value < 0.05
    else "Fail to reject H0"
)

test_result = pd.DataFrame([{
    "test": "Welch two-sample t-test",
    "group_1": "Group-stage",
    "group_2": "Knockout-stage",
    "n_group_stage": n1,
    "n_knockout_stage": n2,
    "mean_group_stage": m1,
    "mean_knockout_stage": m2,
    "difference_group_minus_knockout": difference,
    "t_statistic": t_stat,
    "p_value": p_value,
    "welch_degrees_of_freedom": welch_df,
    "ci95_low": ci_low,
    "ci95_high": ci_high,
    "alpha": 0.05,
    "decision": decision
}])

test_result.to_csv(
    OUTPUT / "t_test_and_confidence_interval.csv",
    index=False
)

# ------------------------------------------------------------
# 8. CHART 1 - MEAN COMPARISON
# ------------------------------------------------------------
plt.figure(figsize=(7, 5))

means = (
    df.groupby("stage")["yellow_cards"]
    .mean()
    .reindex(["Group-stage", "Knockout-stage"])
)

plt.bar(means.index, means.values)
plt.ylabel("Average yellow cards per team-match")
plt.xlabel("Match stage")
plt.title("Average Yellow Cards by Match Stage")
plt.tight_layout()

plt.savefig(
    OUTPUT / "chart_mean_yellow_cards_by_stage.png",
    dpi=200
)

plt.close()

# ------------------------------------------------------------
# 9. CHART 2 - DISTRIBUTION
# ------------------------------------------------------------
plt.figure(figsize=(7, 5))

df.boxplot(
    column="yellow_cards",
    by="stage"
)

plt.suptitle("")
plt.title("Yellow Cards Distribution by Match Stage")
plt.xlabel("Match stage")
plt.ylabel("Yellow cards per team-match")
plt.tight_layout()

plt.savefig(
    OUTPUT / "chart_yellow_cards_distribution.png",
    dpi=200
)

plt.close()

# ------------------------------------------------------------
# 10. PRINT RESULTS
# ------------------------------------------------------------
print("\nDATASET CREATED")
print("Rows:", len(df))
print("Columns:", list(df.columns))

print("\nDESCRIPTIVE STATISTICS")
print(summary.to_string(index=False))

print("\nT-TEST AND 95% CONFIDENCE INTERVAL")
print(test_result.to_string(index=False))

print("\nFiles saved in:")
print(OUTPUT.resolve())
