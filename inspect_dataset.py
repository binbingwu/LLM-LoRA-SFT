from datasets import load_dataset


def main():
    dataset_name = "AI-MO/NuminaMath-CoT"
    dataset = load_dataset(dataset_name)

    print(dataset)
    print()
    print("Train example:")
    print(dataset["train"][0])


if __name__ == "__main__":
    main()

