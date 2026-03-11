import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split

# =========================
# Configuration
# =========================
INPUT_CSV = "../clips2.csv"
OUTPUT_TRAIN = "../train.csv"
OUTPUT_VAL = "../val.csv"
OUTPUT_TEST = "../test.csv"

TRAIN_RATIO = 0.9
VAL_RATIO = 0.05
TEST_RATIO = 0.05

RANDOM_SEED = 42

# =========================
# Helper function to extract person ID
# =========================
def extract_person_id(path):
    """
    Extracts person ID from path.
    Example:
    /.../V00_S0627_I00000503_P0780/V00_S0627_I00000503_P0780_001_....mp4
    -> V00_S0627_I00000503_P0780
    """
    # Get the directory name just before the filename
    return os.path.basename(os.path.dirname(path))


# =========================
# Main
# =========================
def main():
    # Load CSV
    df = pd.read_csv(INPUT_CSV)

    if "path_to_clip" not in df.columns or "transcript" not in df.columns:
        raise ValueError("CSV must contain 'path_to_clip' and 'transcript' columns")

    # Extract person IDs
    df["person_id"] = df["path_to_clip"].apply(extract_person_id)

    # Get unique person IDs
    unique_persons = df["person_id"].unique()
    print(f"Total unique persons: {len(unique_persons)}")

    # First split: train vs temp (val+test)
    train_persons, temp_persons = train_test_split(
        unique_persons,
        test_size=(1 - TRAIN_RATIO),
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # Second split: val vs test
    val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    val_persons, test_persons = train_test_split(
        temp_persons,
        test_size=(1 - val_ratio_adjusted),
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # Create splits
    train_df = df[df["person_id"].isin(train_persons)].drop(columns=["person_id"])
    val_df = df[df["person_id"].isin(val_persons)].drop(columns=["person_id"])
    test_df = df[df["person_id"].isin(test_persons)].drop(columns=["person_id"])

    # Save CSV files
    train_df.to_csv(OUTPUT_TRAIN, index=False)
    val_df.to_csv(OUTPUT_VAL, index=False)
    test_df.to_csv(OUTPUT_TEST, index=False)

    print("Split completed!")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    # Verify no overlap
    assert set(train_persons).isdisjoint(val_persons)
    assert set(train_persons).isdisjoint(test_persons)
    assert set(val_persons).isdisjoint(test_persons)
    print("No person overlap between splits ✓")


if __name__ == "__main__":
    main()