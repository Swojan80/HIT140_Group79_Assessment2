# IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statistics import mean, median
from scipy.stats import tstd, ttest_ind
from statsmodels.stats.weightstats import _zconfint_generic
from math import sqrt


# Dataset 1:

possession_raw = pd.read_csv("WC2026_GroupStage_Possession.csv")


# Dataset 2:

progression_raw = pd.read_csv("WC2026_Progression.csv")


# DATA WRANGLING AND CLEANING

possession_clean = possession_raw.dropna(subset=["Team", "Round", "Possession"])

progression_clean = progression_raw.dropna(subset=["Team", "Advanced"])

print("Cleaned Possession Data:")
print(possession_clean)

print("\nCleaned Progression Data:")
print(progression_clean)


# KEEP GROUP STAGE ONLY

group_stage = possession_clean[possession_clean["Round"] == "Group stage"]


# CALCULATE AVERAGE POSSESSION FOR EACH TEAM

team_average_possession = group_stage.groupby("Team")["Possession"].apply(mean)

team_average_possession.name = "AvgPossession"


# COMBINE POSSESSION AND PROGRESSION DATA

analysis_data = progression_clean.join(team_average_possession, on="Team")

analysis_data = analysis_data.dropna(subset=["AvgPossession"])


# FEATURE ENGINEERING

analysis_data["PossessionBand"] = pd.cut(analysis_data["AvgPossession"], bins=[0, 45, 55, 100], labels=["Low", "Medium", "High"], include_lowest=True)

analysis_data["ProgressionStatus"] = analysis_data["Advanced"].apply(lambda x: "Advanced" if x == 1 else "Eliminated")


# CREATE POSSESSION CONSISTENCY FEATURE

team_possession_sd = group_stage.groupby("Team")["Possession"].apply(tstd)

team_possession_sd.name = "PossessionSD"

analysis_data = analysis_data.join(team_possession_sd, on="Team")


# DISPLAY ANALYSIS DATA

print("\nCombined Analysis Data:")
print(analysis_data)


# DIVIDE TEAMS INTO TWO GROUPS

advanced = analysis_data[analysis_data["Advanced"] == 1][["AvgPossession"]]

eliminated = analysis_data[analysis_data["Advanced"] == 0][["AvgPossession"]]


# DESCRIPTIVE STATISTICS

advanced_stats = advanced.describe()

eliminated_stats = eliminated.describe()

print("\nAdvanced Teams Statistics:")
print(advanced_stats)

print("\nEliminated Teams Statistics:")
print(eliminated_stats)


# ADVANCED TEAMS

advanced_mean = mean(advanced["AvgPossession"])

advanced_median = median(advanced["AvgPossession"])

advanced_std = tstd(advanced["AvgPossession"])

advanced_min = np.min(advanced["AvgPossession"])

advanced_max = np.max(advanced["AvgPossession"])

advanced_q1 = np.percentile(advanced["AvgPossession"], 25)

advanced_q3 = np.percentile(advanced["AvgPossession"], 75)

print("\nADVANCED TEAMS")
print("Mean:", advanced_mean)
print("Median:", advanced_median)
print("Standard Deviation:", advanced_std)
print("Minimum:", advanced_min)
print("Q1:", advanced_q1)
print("Q3:", advanced_q3)
print("Maximum:", advanced_max)


# ELIMINATED TEAMS

eliminated_mean = mean(eliminated["AvgPossession"])

eliminated_median = median(eliminated["AvgPossession"])

eliminated_std = tstd(eliminated["AvgPossession"])

eliminated_min = np.min(eliminated["AvgPossession"])

eliminated_max = np.max(eliminated["AvgPossession"])

eliminated_q1 = np.percentile(eliminated["AvgPossession"], 25)

eliminated_q3 = np.percentile(eliminated["AvgPossession"], 75)

print("\nELIMINATED TEAMS")
print("Mean:", eliminated_mean)
print("Median:", eliminated_median)
print("Standard Deviation:", eliminated_std)
print("Minimum:", eliminated_min)
print("Q1:", eliminated_q1)
print("Q3:", eliminated_q3)
print("Maximum:", eliminated_max)


# MEAN DIFFERENCE

mean_difference = advanced_mean - eliminated_mean

print("\nMean Difference:")
print(mean_difference)


# HISTOGRAM FOR ADVANCED TEAMS

plt.hist(advanced["AvgPossession"], bins=8)
plt.title("Average Possession of Teams Advanced to Knockout Stage")
plt.xlabel("Average Ball Possession (%)")
plt.ylabel("Number of Teams")
plt.show()


# HISTOGRAM FOR ELIMINATED TEAMS

plt.hist(eliminated["AvgPossession"], bins=8)
plt.title("Average Possession of Teams Eliminated in Group Stage")
plt.xlabel("Average Ball Possession (%)")
plt.ylabel("Number of Teams")
plt.show()


# SAMPLE SIZE

advanced_n = advanced_stats["AvgPossession"]["count"]

eliminated_n = eliminated_stats["AvgPossession"]["count"]

print("\nSample Sizes")
print("Advanced Teams:", advanced_n)
print("Eliminated Teams:", eliminated_n)


# STANDARD ERROR

advanced_se = advanced_std / sqrt(advanced_n)

eliminated_se = eliminated_std / sqrt(eliminated_n)

print("\nStandard Errors")
print("Advanced Teams:", advanced_se)
print("Eliminated Teams:", eliminated_se)


# CONFIDENCE INTERVALS

advanced_ci = _zconfint_generic(advanced_mean, advanced_se, 0.05, "two-sided")

eliminated_ci = _zconfint_generic(eliminated_mean, eliminated_se, 0.05, "two-sided")

difference_se = sqrt((advanced_std ** 2 / advanced_n) + (eliminated_std ** 2 / eliminated_n))

difference_ci = _zconfint_generic(mean_difference, difference_se, 0.05, "two-sided")

print("\nConfidence Intervals")
print("The 95% confidence interval for advanced teams is from", round(advanced_ci[0], 2), "to", round(advanced_ci[1], 2))
print("The 95% confidence interval for eliminated teams is from", round(eliminated_ci[0], 2), "to", round(eliminated_ci[1], 2))
print("The mean difference in possession is", round(mean_difference, 2), "percentage points")
print("The 95% confidence interval for the mean difference is from", round(difference_ci[0], 2), "to", round(difference_ci[1], 2))


# TWO-SAMPLE T-TEST

t_test = ttest_ind(advanced["AvgPossession"], eliminated["AvgPossession"], equal_var=False)

t_statistic = t_test.statistic

p_value = t_test.pvalue

statistically_significant = p_value < 0.05

print("\nT-Test Results")
print("T-Statistic:", round(t_statistic, 3))
print("P-Value:", round(p_value, 3))
print("Statistically Significant:", statistically_significant)

# MAIN BOXPLOT AND SWARMPLOT

plt.figure(figsize=(8, 6))
sns.boxplot(data=analysis_data, x="ProgressionStatus", y="AvgPossession")
sns.swarmplot(data=analysis_data, x="ProgressionStatus", y="AvgPossession")

plt.title("Group-Stage Possession by Tournament Progression")
plt.xlabel("Tournament Progression")
plt.ylabel("Average Ball Possession (%)")

plt.show()


# SCATTERPLOT FOR POSSESSION CONSISTENCY

sns.scatterplot(data=analysis_data, x="AvgPossession", y="PossessionSD", hue="ProgressionStatus")
plt.title("Average Possession and Possession Variability")
plt.xlabel("Average Ball Possession (%)")
plt.ylabel("Possession Standard Deviation")
plt.show()


# TEST POSSESSION CONSISTENCY

advanced_consistency = analysis_data[analysis_data["Advanced"] == 1][["PossessionSD"]]

eliminated_consistency = analysis_data[analysis_data["Advanced"] == 0][["PossessionSD"]]

advanced_consistency_mean = mean(advanced_consistency["PossessionSD"])

eliminated_consistency_mean = mean(eliminated_consistency["PossessionSD"])

consistency_test = ttest_ind(advanced_consistency["PossessionSD"], eliminated_consistency["PossessionSD"], equal_var=False)

consistency_p_value = consistency_test.pvalue

print("\nPossession Consistency Analysis")
print("Advanced Teams Possession SD Mean:", round(advanced_consistency_mean, 2))
print("Eliminated Teams Possession SD Mean:", round(eliminated_consistency_mean, 2))
print("Consistency P-Value:", round(consistency_p_value, 3))


# FINAL RESULT

print("\nFinal Conclusion")
print("Advanced teams had a higher average group-stage possession than eliminated teams.")
print("Advanced Mean Possession:", round(advanced_mean, 2))
print("Eliminated Mean Possession:", round(eliminated_mean, 2))
print("Mean Difference:", round(mean_difference, 2))
print("P-Value:", round(p_value, 3))
print("Statistically Significant:", statistically_significant)