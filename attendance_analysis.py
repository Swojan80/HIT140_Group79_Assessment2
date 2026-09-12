# ------------------------------------------
# FIFA World Cup 2026 Match Attendance Analysis
# Name: Rupesh Timalsina
# Student ID: S400656
# ------------------------------------------

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# ------------------------------------------
# LOAD DATA

df = pd.read_csv(
    "Rupesh_2026_World_Cup_Attendance_Raw.csv",
    encoding="latin1"
)

print("\n--- ORIGINAL DATA ---")
print("Total rows:", len(df))
print("\nColumns:")
print(df.columns.tolist())


# ------------------------------------------
# DATA WRANGLING AND CLEANING

# Remove rows where Round is missing
df = df.dropna(subset=["Round"])

# Convert Attendance to numeric
df["Attendance"] = (
    df["Attendance"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df["Attendance"] = pd.to_numeric(df["Attendance"], errors="coerce")

# Keep only rows with valid attendance
df = df.dropna(subset=["Attendance"])

print("\n--- CLEANED DATA ---")
print("Total valid matches:", len(df))
print("Missing attendance:", df["Attendance"].isna().sum())


# ------------------------------------------
# CREATE STAGE GROUPS

def classify_stage(round_name):

    if round_name == "Group stage":
        return "Group Stage"

    else:
        return "Knockout Stage"


df["Stage_Group"] = df["Round"].apply(classify_stage)

print("\n--- MATCHES BY STAGE ---")
print(df["Stage_Group"].value_counts())


# ------------------------------------------
# 4. DATA PREPARATION AND SAMPLING

population = df.copy()

# Stratified random sample
sample = (
    population
    .groupby("Stage_Group", group_keys=False)
    .sample(frac=0.7, random_state=42)
)

print("\n--- SAMPLING ---")
print("Population size:", len(population))
print("Sample size:", len(sample))
print(sample["Stage_Group"].value_counts())


# ------------------------------------------
# 5. DESCRIPTIVE STATISTICS

group_stage = sample[sample["Stage_Group"] == "Group Stage"]["Attendance"]
knockout = sample[sample["Stage_Group"] == "Knockout Stage"]["Attendance"]

descriptive = pd.DataFrame({
    "Group Stage": [
        len(group_stage),
        group_stage.mean(),
        group_stage.median(),
        group_stage.std(),
        group_stage.min(),
        group_stage.max()
    ],
    "Knockout Stage": [
        len(knockout),
        knockout.mean(),
        knockout.median(),
        knockout.std(),
        knockout.min(),
        knockout.max()
    ]
},
index=["Count", "Mean", "Median", "Standard Deviation", "Minimum", "Maximum"])

print("\n--- DESCRIPTIVE STATISTICS ---")
print(descriptive.round(2))


# ------------------------------------------
# 6. 95% CONFIDENCE INTERVAL

def confidence_interval(data, confidence=0.95):

    n = len(data)
    mean = np.mean(data)
    standard_error = stats.sem(data)

    margin_error = standard_error * stats.t.ppf(
        (1 + confidence) / 2,
        n - 1
    )

    return mean - margin_error, mean + margin_error


group_ci = confidence_interval(group_stage)
knockout_ci = confidence_interval(knockout)

print("\n--- 95% CONFIDENCE INTERVALS ---")

print("Group Stage:",
      round(group_ci[0], 2),
      "to",
      round(group_ci[1], 2))

print("Knockout Stage:",
      round(knockout_ci[0], 2),
      "to",
      round(knockout_ci[1], 2))


# ------------------------------------------
# 7. WELCH'S TWO-SAMPLE T-TEST

t_stat, p_value = stats.ttest_ind(
    group_stage,
    knockout,
    equal_var=False
)

print("\n--- WELCH'S TWO-SAMPLE T-TEST ---")
print("T-statistic:", round(t_stat, 4))
print("P-value:", round(p_value, 4))

alpha = 0.05

if p_value < alpha:
    print("Result: Reject the null hypothesis.")
    print("There is a statistically significant difference in average attendance.")
else:
    print("Result: Fail to reject the null hypothesis.")
    print("There is no statistically significant difference in average attendance.")


# ------------------------------------------
# 8. BOXPLOT

plt.figure(figsize=(8, 5))

plt.boxplot(
    [group_stage, knockout],
    tick_labels=["Group Stage", "Knockout Stage"]
)

plt.title("Match Attendance by Tournament Stage")
plt.ylabel("Attendance")

plt.tight_layout()
plt.savefig("attendance_boxplot.png")
plt.show()


# ------------------------------------------
# 9. MEAN ATTENDANCE BAR CHART

means = [
    group_stage.mean(),
    knockout.mean()
]

plt.figure(figsize=(7, 5))

plt.bar(
    ["Group Stage", "Knockout Stage"],
    means
)

plt.title("Average Match Attendance by Tournament Stage")
plt.ylabel("Average Attendance")

plt.tight_layout()
plt.savefig("average_attendance.png")
plt.show()


# ------------------------------------------
# 10. SAVE CLEANED DATA AND SAMPLE

df.to_csv(
    "Rupesh_Attendance_Cleaned.csv",
    index=False
)

sample.to_csv(
    "Rupesh_Attendance_Sample.csv",
    index=False
)

print("\n--- FILES SAVED ---")
print("Rupesh_Attendance_Cleaned.csv")
print("Rupesh_Attendance_Sample.csv")
print("attendance_boxplot.png")
print("average_attendance.png")