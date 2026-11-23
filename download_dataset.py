from pathlib import Path
import kagglehub


def main():
    """Download the Kaggle dataset into the project."""

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    path = kagglehub.dataset_download(
        "whenamancodes/fraud-detection",
        output_dir=str(data_dir)
    )

    print(f"Dataset downloaded to: {path}")

    for file in data_dir.rglob("*.csv"):
        print(f"CSV file found: {file}")


if __name__ == "__main__":
    main()