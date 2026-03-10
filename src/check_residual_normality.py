import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


def check_residual_normality(y_ground, y_pred):
    """
    Checks the normality of residuals using Shapiro-Wilk,
    D'Agostino's K^2, and visual plots.
    """
    # 1. Calculate Residuals
    residuals = y_ground - y_pred

    # 2. Statistical Tests
    # Shapiro-Wilk (Best for small samples N < 5000)
    shapiro_stat, shapiro_p = stats.shapiro(residuals)

    # D'Agostino’s K^2 (Good for larger samples)
    k2_stat, k2_p = stats.normaltest(residuals)

    # 3. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram + KDE
    sns.histplot(residuals, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title("Histogram of Residuals")

    # Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Normal Q-Q Plot")

    plt.tight_layout()
    plt.show()

    # Print Results
    print(f"--- Normality Test Results ---")
    print(f"Shapiro-Wilk Test: Stat={shapiro_stat:.4f}, p-value={shapiro_p:.4e}")
    print(f"D'Agostino's K^2:  Stat={k2_stat:.4f}, p-value={k2_p:.4e}")

    # Interpretation
    alpha = 0.05
    if shapiro_p > alpha:
        print("\nConclusion: Residuals look Gaussian (fail to reject H0)")
    else:
        print("\nConclusion: Residuals do NOT look Gaussian (reject H0)")

# Usage:
# check_residual_normality(y_test, y_predictions)
