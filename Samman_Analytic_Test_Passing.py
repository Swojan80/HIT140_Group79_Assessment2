# IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statistics import mean, median
from scipy.stats import tstd, ttest_ind
from statsmodels.stats.weightstats import _zconfint_generic
from math import sqrt


# Dataset 1:

possession_raw = pd.read_csv("WC2026_GroupStage_Possession.csv")


# Dataset 2:

progression_raw = pd.read_csv("WC2026_Progression.csv")


# DATA WRANGLING AND CLEANING

# Remove rows with missing Team, Round or Possession values.

possession_clean = possession_raw.dropna(subset=["Team", "Round", "Possession"])


# Remove rows with missing Team or Advanced values.

progression_clean = progression_raw.dropna(subset=["Team", "Advanced"])

print("Cleaned Possession Data:")
print(possession_clean)

print("\nCleaned Progression Data:")
print(progression_clean)

# Keep only the group stage matches.

group_stage = possession_clean[possession_clean["Round"] == "Group stage"]

# CALCULATE AVERAGE GROUP-STAGE POSSESSION FOR EACH TEAM

team_average_possession = group_stage.groupby("Team")["Possession"].apply(mean)


# Give the average possession column a name.

team_average_possession.name = "AvgPossession"


# COMBINE POSSESSION AND PROGRESSION DATA

analysis_data = progression_clean.join(team_average_possession, on="Team")


# Remove any missing average possession values after joining.

analysis_data = analysis_data.dropna(subset=["AvgPossession"])


# FEATURE ENGINEERING

# Create possession categories using average possession percentage.

# Low = 0% to 45%
# Medium = greater than 45% to 55%
# High = greater than 55%

analysis_data["PossessionBand"] = pd.cut(analysis_data["AvgPossession"], bins=[0, 45, 55, 100], labels=["Low", "Medium", "High"], include_lowest=True)


# Displaying the dataset with the new feature.

print("\nCombined Analysis Data:")
print(analysis_data)



# DIVIDE TEAMS INTO THE TWO GROUPS

# Teams that ADVANCED to the knockout stage.

advanced = analysis_data[analysis_data["Advanced"] == 1][["AvgPossession"]]


# Teams that were ELIMINATED in the group stage.

eliminated = analysis_data[analysis_data["Advanced"] == 0][["AvgPossession"]]


# DESCRIPTIVE STATISTICS

# Summary statistics for advanced teams.

advanced_stats = advanced.describe()


# Summary statistics for eliminated teams.

eliminated_stats = eliminated.describe()

# Display summary statistics.
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


# Display descriptive statistics for advanced teams.

# SHOW ADVANCED TEAM VALUES

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


# Display descriptive statistics for eliminated teams.

print("\nELIMINATED TEAMS")
print("Mean:", eliminated_mean)
print("Median:", eliminated_median)
print("Standard Deviation:", eliminated_std)
print("Minimum:", eliminated_min)
print("Q1:", eliminated_q1)
print("Q3:", eliminated_q3)
print("Maximum:", eliminated_max)


# CALCULATE DIFFERENCE BETWEEN THE TWO MEANS

# Positive value means advanced teams had higher average possession.
#
# Negative value means eliminated teams had higher average possession.

mean_difference = advanced_mean - eliminated_mean


# Display the mean difference.

print("\nMean Difference:")
print(mean_difference)

# HISTOGRAM FOR ADVANCED TEAMS

# This histogram shows the distribution of average possession
# percentages for teams that advanced to the knockout stage.

plt.hist(advanced["AvgPossession"], bins=8)
plt.title("Average Possession of Teams Advanced to Knockout Stage")
plt.xlabel("Average Ball Possession (%)")
plt.ylabel("Number of Teams")
plt.show()


# HISTOGRAM FOR ELIMINATED TEAMS

# This histogram shows the distribution of average possession
# percentages for teams eliminated in the group stage.

plt.hist(eliminated["AvgPossession"], bins=8)
plt.title("Average Possession of Teams Eliminated in Group Stage")
plt.xlabel("Average Ball Possession (%)")
plt.ylabel("Number of Teams")
plt.show()


# SAMPLE SIZE

# Get the number of teams in each group using describe().

advanced_n = advanced_stats["AvgPossession"]["count"]

eliminated_n = eliminated_stats["AvgPossession"]["count"]


# Display the sample sizes.

advanced_n, eliminated_n


# STANDARD ERROR

# Standard Error = Standard Deviation / Square Root of Sample Size

advanced_se = advanced_std / sqrt(advanced_n)

eliminated_se = eliminated_std / sqrt(eliminated_n)


# Display standard errors.

print("\nStandard Errors")
print("Advanced Teams:", advanced_se)
print("Eliminated Teams:", eliminated_se)


# 95% CONFIDENCE INTERVAL FOR ADVANCED TEAMS

advanced_ci = _zconfint_generic(advanced_mean, advanced_se, 0.05, "two-sided")


# Display confidence interval.

print("The 95% confidence interval for advanced teams is from",advanced_ci[0], "to", advanced_ci[1])


# 95% CONFIDENCE INTERVAL FOR ELIMINATED TEAMS

eliminated_ci = _zconfint_generic(eliminated_mean, eliminated_se, 0.05, "two-sided")


# Display confidence interval.

print("The 95% confidence interval for eliminated teams is from",eliminated_ci[0], "to", eliminated_ci[1])



# CONFIDENCE INTERVAL FOR DIFFERENCE BETWEEN THE TWO MEANS

# Calculate the standard error of the difference
# between the two independent group means.

difference_se = sqrt((advanced_std ** 2 / advanced_n) + (eliminated_std ** 2 / eliminated_n))


# Calculate the 95% confidence interval for the mean difference.

difference_ci = _zconfint_generic(mean_difference, difference_se, 0.05, "two-sided")


# Display mean difference and confidence interval.

print("The mean difference in possession is", mean_difference,
      "percentage points")

print("The 95% confidence interval for the mean difference is from",
      difference_ci[0], "to", difference_ci[1])

# TWO-SAMPLE T-TEST
"""
Null Hypothesis:
There is no significant difference in average possession
between teams that advanced and teams that were eliminated.
Alternative Hypothesis:
There is a significant difference in average possession
between teams that advanced and teams that were eliminated.
equal_var=False performs Welch's independent two-sample t-test.
"""

t_test = ttest_ind(advanced["AvgPossession"], eliminated["AvgPossession"], equal_var=False)


# EXTRACT T-STATISTIC AND P-VALUE

t_statistic = t_test.statistic

p_value = t_test.pvalue


# Display t-statistic and p-value.

print("\nT-Test Results")
print("T-Statistic:", t_statistic)
print("P-Value:", p_value)

# DETERMINE STATISTICAL SIGNIFICANCE

# Significance level:
# alpha = 0.05
#
# If p-value < 0.05:
# The difference is statistically significant.
#
# If p-value >= 0.05:
# The difference is not statistically significant.

statistically_significant = p_value < 0.05

#Printing Final Data
print("Advanced Mean Possession:", advanced_mean)
print("Eliminated Mean Possession:", eliminated_mean)
print("Mean Difference:", mean_difference)
print("P-Value:", p_value)
print("Statistically Significant:", statistically_significant)