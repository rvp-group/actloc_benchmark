import argparse
import numpy as np


def load_error_data(filepath):
    """
    Load translation and rotation errors from a file.
    Each line should have: <image_name> <translation_error> <rotation_error>
    """
    errors = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            _, t_err, r_err = parts
            errors.append((float(t_err), float(r_err)))
    return np.array(errors)  # shape (N, 2)


def evaluate_thresholds(errors, thresholds):
    """
    Count how many errors are within each (translation, rotation) threshold.
    """
    total = errors.shape[0]
    results = []

    for t_thresh, r_thresh in thresholds:
        count = np.sum((errors[:, 0] <= t_thresh) & (errors[:, 1] <= r_thresh))
        percentage = (count / total) * 100
        results.append((t_thresh, r_thresh, count, percentage))

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate pose errors in visual localization.")
    parser.add_argument('--error-file', type=str, required=True,
                        help="Path to file containing translation and rotation errors.")
    args = parser.parse_args()
    thresholds = [(0.05, 0.4), (0.25, 2.0), (0.5, 5.0), (5.0, 10.0)]

    errors = load_error_data(args.error_file)
    results = evaluate_thresholds(errors, thresholds)

    print("\nEvaluation Summary:")
    for t_thresh, r_thresh, count, percent in results:
        print(f"≤ {t_thresh:.2f}m / {r_thresh:.1f}°: {count}/{len(errors)} predictions ({percent:.2f}%)")


if __name__ == "__main__":
    main()
