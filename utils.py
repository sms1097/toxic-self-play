from datasets import load_dataset, DatasetDict, Dataset


def load_anthropic_reverse():
    ds = load_dataset("Anthropic/hh-rlhf")
    train_dict = ds["train"].to_dict()
    train_dict["chosen"], train_dict["rejected"] = (
        train_dict["rejected"],
        train_dict["chosen"],
    )
    test_dict = ds["test"].to_dict()
    test_dict["chosen"], test_dict["rejected"] = (
        test_dict["rejected"],
        test_dict["chosen"],
    )
    ds = DatasetDict(
        {"train": Dataset.from_dict(train_dict), "test": Dataset.from_dict(test_dict)}
    )
    return ds


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert (
        search_term_idx != -1
    ), f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def split_prompt_and_responses(sample):
    prompt = extract_anthropic_prompt(sample["chosen"])
    return {
        "prompt": prompt,
        "chosen": sample["chosen"][len(prompt) :],
        "rejected": sample["rejected"][len(prompt) :],
    }


def build_dataset(
    sanity_check: bool = False, silent: bool = False, cache_dir: str = None
) -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    dataset = load_anthropic_reverse()
    if sanity_check:
        dataset["train"] = dataset["train"].select(range(min(len(dataset), 1000)))
        dataset["test"] = dataset["test"].select(range(min(len(dataset), 1000)))

    dataset["train"] = dataset["train"].map(split_prompt_and_responses)
    dataset["test"] = dataset["test"].map(split_prompt_and_responses)
    return dataset
