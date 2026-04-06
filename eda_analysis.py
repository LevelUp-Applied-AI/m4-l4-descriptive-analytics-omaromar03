"""Lab 4 — Descriptive Analytics: Student Performance EDA

Conduct exploratory data analysis on the student performance dataset.
Produce distribution plots, correlation analysis, hypothesis tests,
and a written findings report.

Usage:
    python eda_analysis.py


"""
import pandas as pd
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_and_profile():
    """Load the dataset and generate a data profile report.

    Args:
        filepath: path to the CSV file (e.g., 'data/student_performance.csv')

    Returns:
        DataFrame: the loaded dataset

    Side effects:
        Saves a text profile to output/data_profile.txt containing:
        - Shape (rows, columns)
        - Data types for each column
        - Missing value counts per column
        - Descriptive statistics for numeric columns
    """
    # TODO: Load the dataset and report its shape, data types, missing values,
    #       and descriptive statistics to output/data_profile.txt
    
    df = pd.read_csv("data/student_performance.csv")

    print("Shape:", df.shape)
    print("\nInfo:")
    print(df.info())
    print("\nDescribe:")
    print(df.describe())
    print("\nMissing values:")
    print(df.isnull().sum())

    return df

def clean_data(df):
    df = df.dropna(subset=["study_hours_weekly"])

    median_commute = df["commute_minutes"].median()
    df["commute_minutes"] = df["commute_minutes"].fillna(median_commute) 
    df["scholarship"] = df["scholarship"].fillna("None")
    return df
def save_data_profile(df):
    os.makedirs("output", exist_ok=True)
    
    with open("output/data_profile.txt", "w") as f:
        f.write(f"Shape: {df.shape}\n\n")

        f.write("Data Types:\n")
        f.write(str(df.dtypes))
        f.write("\n\n")

        f.write("Missing Values:\n")
        f.write(str(df.isnull().sum()))
        f.write("\n\n")
        
        f.write("Handling Decisions:\n")
        f.write("- study_hours_weekly: dropped rows (low missing ~5%)\n")
        f.write("- commute_minutes: filled with median (~10% missing)\n")
        f.write("- scholarship: filled with 'None' (represents no scholarship)\n")
def plot_distributions(df):
    os.makedirs("output", exist_ok=True)

    # Histogram 1: GPA
    plt.figure(figsize=(8, 5))
    sns.histplot(df["gpa"], kde=True)
    plt.title("Distribution of GPA")
    plt.xlabel("GPA")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("output/gpa_distribution.png")
    plt.close()

    # Histogram 2: Study Hours Weekly
    plt.figure(figsize=(8, 5))
    sns.histplot(df["study_hours_weekly"], kde=True)
    plt.title("Distribution of Weekly Study Hours")
    plt.xlabel("Study Hours Weekly")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("output/study_hours_distribution.png")
    plt.close()

    # Histogram 3: Attendance Percentage
    plt.figure(figsize=(8, 5))
    sns.histplot(df["attendance_pct"], kde=True)
    plt.title("Distribution of Attendance Percentage")
    plt.xlabel("Attendance Percentage")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("output/attendance_distribution.png")
    plt.close()

    # Box plot: GPA by Department
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="department", y="gpa")
    plt.title("GPA by Department")
    plt.xlabel("Department")
    plt.ylabel("GPA")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("output/gpa_by_department.png")
    plt.close()

    # Bar chart: Scholarship distribution
    plt.figure(figsize=(10, 6))
    df["scholarship"].value_counts().plot(kind="bar")
    plt.title("Scholarship Distribution")
    plt.xlabel("Scholarship Type")
    plt.ylabel("Count")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("output/scholarship_distribution.png")
    plt.close()
    


def plot_correlations(df):
    os.makedirs("output", exist_ok=True)

    numeric_df = df.select_dtypes(include=["number"])
    corr_matrix = numeric_df.corr(method="pearson")

    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("output/correlation_heatmap.png")
    plt.close()

    # Find top 2 correlated pairs excluding self-correlation
    corr_pairs = corr_matrix.abs().unstack()
    corr_pairs = corr_pairs[corr_pairs < 1.0]
    corr_pairs = corr_pairs.sort_values(ascending=False)

    top_pairs = []
    seen = set()

    for (col1, col2), value in corr_pairs.items():
        pair = tuple(sorted((col1, col2)))
        if pair not in seen:
            seen.add(pair)
            top_pairs.append((col1, col2, value))
        if len(top_pairs) == 2:
            break

    # Scatter plots for top 2 pairs
    for i, (col1, col2, value) in enumerate(top_pairs, start=1):
        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=df, x=col1, y=col2)
        plt.title(f"Scatter Plot: {col1} vs {col2} (corr={value:.2f})")
        plt.tight_layout()
        plt.savefig(f"output/scatter_{i}_{col1}_vs_{col2}.png")
        plt.close()

    return corr_matrix, top_pairs
    


def run_hypothesis_tests(df):
    results = {}

    # =========================
    # 1) T-test (Internship vs GPA)
    # =========================
    group_yes = df[df["has_internship"] == "Yes"]["gpa"]
    group_no = df[df["has_internship"] == "No"]["gpa"]

    t_stat, p_value = stats.ttest_ind(group_yes, group_no)

    print("\nT-Test: Internship vs GPA")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Result: Statistically significant difference in GPA")
    else:
        print("Result: No significant difference")

    results["internship_ttest"] = (t_stat, p_value)

    # =========================
    # 2) Chi-Square (Scholarship vs Department)
    # =========================
    contingency = pd.crosstab(df["department"], df["scholarship"])

    chi2, p, dof, expected = stats.chi2_contingency(contingency)

    print("\nChi-Square: Scholarship vs Department")
    print(f"chi2-statistic: {chi2:.4f}")
    print(f"p-value: {p:.4f}")
    print(f"degrees of freedom: {dof}")

    if p < 0.05:
        print("Result: Significant association between scholarship and department")
    else:
        print("Result: No significant association")

    results["scholarship_chi2"] = (chi2, p)

    return results
    
def main():
    df = load_and_profile()
    df = clean_data(df)
    save_data_profile(df)
    plot_distributions(df)
    plot_correlations(df)
    run_hypothesis_tests(df)
if __name__ == "__main__":
    main()
    # TODO: Load and profile the dataset
    # TODO: Generate distribution plots
    # TODO: Analyze correlations
    # TODO: Run hypothesis tests
    # TODO: Write a FINDINGS.md summarizing your analysis


