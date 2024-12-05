# main.py
import torch
from peepholeLSTM import PeepholeLSTM
import matplotlib.pyplot as plt
from data import run_experiment
from analysis_experiment import run_analysis_experiment

def plot_results(results, sequence_length):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(results['train_accs'], label='Train')
    plt.plot(results['test_accs'], label='Test')
    plt.title(f'Accuracy (Sequence Length={sequence_length})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(results['train_losses'], label='Train')
    plt.plot(results['test_losses'], label='Test')
    plt.title(f'Loss (Sequence Length={sequence_length})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Experiment parameters
    input_size = 784  # MNIST image size
    hidden_size = 128
    sequence_lengths = [100, 200, 400, 800]  # As in paper
    batch_size = 32
    epochs = 1 # 50

    for sequence_length in sequence_lengths:
        print(f"\nRunning experiment with sequence length: {sequence_length}")

        # Run with critical initialization
        model_critical = PeepholeLSTM(input_size, hidden_size,
                                      critical_init=True).to(device)
        results_critical = run_experiment(model_critical, sequence_length,
                                          batch_size, epochs, device)

        # Run with standard initialization
        model_standard = PeepholeLSTM(input_size, hidden_size,
                                      critical_init=False).to(device)
        results_standard = run_experiment(model_standard, sequence_length,
                                          batch_size, epochs, device)

        # Plot results
        plot_results(results_critical, f"{sequence_length} (Critical)")
        plot_results(results_standard, f"{sequence_length} (Standard)")

        # Run analysis experiments
        print("\nRunning Jacobian analysis...")
        run_analysis_experiment(model_critical, model_standard, device)


if __name__ == "__main__":
    main()