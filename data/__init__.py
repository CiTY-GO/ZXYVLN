"""CoT-assisted Q-SFT Data Loader."""
from data.processor_qsft_cot import UMMCOTPreprocessorQSFT, QSFTRewardCalculator

def load_dataset_qsft_cot(datasets, split_dataset_ratio=0., seed=42, num_proc=1, shuffle=False, **kwargs):
    """Load dataset with CoT-assisted Q-SFT preprocessor."""
    # Placeholder - actual implementation depends on FantasyVLN data structure
    from swift.llm.dataset.loader import DatasetLoader
    return DatasetLoader.load(datasets, split_dataset_ratio=split_dataset_ratio, seed=seed, num_proc=num_proc, shuffle=shuffle, **kwargs)
