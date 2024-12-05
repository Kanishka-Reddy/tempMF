# experiments/analysis_experiment.py
import torch
from analysis import JacobianAnalyzer
import matplotlib.pyplot as plt
from analysis import compute_convergence_timescale


def run_analysis_experiment(model_critical, model_standard, device):
    # Initialize analyzers
    analyzer_critical = JacobianAnalyzer(model_critical, device)
    analyzer_standard = JacobianAnalyzer(model_standard, device)

    # Compute spectral properties
    mJJT_1_critical, sigma_JJT_critical, sv_critical = analyzer_critical.analyze_spectral_properties()
    mJJT_1_standard, sigma_JJT_standard, sv_standard = analyzer_standard.analyze_spectral_properties()

    print("\nSpectral Analysis Results:")
    print("\nCritical Initialization:")
    print(f"Mean of squared singular values: {mJJT_1_critical:.4f}")
    print(f"Variance of squared singular values: {sigma_JJT_critical:.4f}")

    print("\nStandard Initialization:")
    print(f"Mean of squared singular values: {mJJT_1_standard:.4f}")
    print(f"Variance of squared singular values: {sigma_JJT_standard:.4f}")

    # Plot distributions
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    analyzer_critical.plot_singular_value_distribution(sv_critical)
    plt.title("Critical Initialization")

    plt.subplot(1, 2, 2)
    analyzer_standard.plot_singular_value_distribution(sv_standard)
    plt.title("Standard Initialization")

    plt.tight_layout()
    plt.show()

    # Compute convergence timescales
    xi_critical, corr_critical = compute_convergence_timescale(model_critical)
    xi_standard, corr_standard = compute_convergence_timescale(model_standard)

    print("\nConvergence Timescale Analysis:")
    print(f"Critical initialization ξ: {xi_critical:.2f}")
    print(f"Standard initialization ξ: {xi_standard:.2f}")

    # Plot correlation decay
    plt.figure(figsize=(8, 6))
    plt.plot(corr_critical, label='Critical')
    plt.plot(corr_standard, label='Standard')
    plt.xlabel('Timestep')
    plt.ylabel('State Correlation')
    plt.title('Correlation Decay Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()