from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomTextDataset(Dataset):
    """Generic dataset class to handle different datasets based on their structure."""
    
    def __init__(self, dataset_name_or_path, split_name=None, **kwargs):
        """
        Initializes the dataset.
        
        Args:
            dataset_name_or_path (str): Path or name of the dataset.
            split_name (str, optional): Name of the split to load (e.g., 'train', 'validation').
                                        If None, attempts to find a default split.
            **kwargs: Additional arguments to pass to load_dataset.
        """
        logger.info(f"Loading dataset from {dataset_name_or_path}")
        self.dataset = load_dataset(dataset_name_or_path, **kwargs)
        
        # Determine the split to use
        if split_name and split_name in self.dataset:
            self.dataset_split = self.dataset[split_name]
        else:
            # Default to 'train' if it exists, otherwise take the first available split
            available_splits = list(self.dataset.keys())
            if 'train' in available_splits:
                self.dataset_split = self.dataset['train']
                logger.info("Using 'train' split as default.")
            elif available_splits:
                self.dataset_split = self.dataset[available_splits[0]]
                logger.info(f"Using first available split: '{available_splits[0]}'.")
            else:
                raise ValueError(f"No splits found in dataset {dataset_name_or_path}")
        
        # Determine the prompt generation function based on the dataset path or name
        # This is a simple heuristic; in practice, you might need a more robust mapping
        if 'alpaca' in dataset_name_or_path.lower():
            self.prompt_func = self._get_alpaca_prompt
        elif 'commonsense_qa' in dataset_name_or_path.lower() or 'csqa' in dataset_name_or_path.lower():
             # Note: The actual Hugging Face dataset might be named 'tau/commonsense_qa'
             # Adjust the check as needed based on how you load it.
             # The original CSQA dataset structure is followed by some versions.
            self.prompt_func = self._get_csqa_prompt
        elif 'gsm8k' in dataset_name_or_path.lower():
            self.prompt_func = self._get_gsm8k_prompt
        elif 'hellaswag' in dataset_name_or_path.lower():
            self.prompt_func = self._get_hellaswag_prompt
        elif 'piqa' in dataset_name_or_path.lower():
            self.prompt_func = self._get_piqa_prompt
        elif 'siqa' in dataset_name_or_path.lower():
            self.prompt_func = self._get_siqa_prompt
        elif 'sst2' in dataset_name_or_path.lower() or 'sst-2' in dataset_name_or_path.lower():
            self.prompt_func = self._get_sst2_prompt
        elif 'sum' in dataset_name_or_path.lower(): # Assuming 'sum' is the identifier
            self.prompt_func = self._get_sum_prompt
        else:
            # Default to a generic function that tries common fields
            self.prompt_func = self._get_generic_prompt
        
        logger.info(f"Using prompt generation function for dataset: {dataset_name_or_path}")

    def _get_alpaca_prompt(self, idx):
        """Generates a prompt for the Alpaca dataset using the 'instruction' field."""
        item = self.dataset_split[idx]
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        # Combine instruction and input if input exists
        if input_text:
            prompt = f"{instruction}\n{input_text}"
        else:
            prompt = instruction
        return prompt

    def _get_csqa_prompt(self, idx):
        """Generates a prompt for the CommonsenseQA dataset using the 'question' field."""
        item = self.dataset_split[idx]
        question = item.get('question', '')
        # Optionally, you could format the choices into the prompt as well
        # For now, just using the question as the prompt as requested
        return question

    def _get_gsm8k_prompt(self, idx):
        """Generates a prompt for the GSM8K dataset using the 'question' field."""
        item = self.dataset_split[idx]
        question = item.get('question', '')
        return question

    def _get_hellaswag_prompt(self, idx):
        """Generates a prompt for the HellaSwag dataset using 'ctx_a' and 'ctx_b'."""
        item = self.dataset_split[idx]
        ctx_a = item.get('ctx_a', '')
        ctx_b = item.get('ctx_b', '')
        # Combine ctx_a and ctx_b to form the prompt
        prompt = f"{ctx_a} {ctx_b}"
        return prompt

    def _get_piqa_prompt(self, idx):
        """Generates a prompt for the PIQA dataset using 'goal' and one of the solutions."""
        item = self.dataset_split[idx]
        goal = item.get('goal', '')
        sol1 = item.get('sol1', '')
        sol2 = item.get('sol2', '')
        # PIQA often requires comparing sol1 and sol2. For a simple prompt,
        # we can combine the goal with the first solution (sol1) as an example.
        # Or, we could format it differently depending on the specific task requirement.
        # Here, we just use the goal as the prompt, similar to how other datasets might be used,
        # or combine goal with one solution. Let's combine goal with sol1 for a basic prompt.
        # You might want to modify this depending on how you plan to use the prompt downstream.
        prompt = f"{goal} {sol1}"
        return prompt

    def _get_siqa_prompt(self, idx):
        """Generates a prompt for the SIQA dataset using 'context' and 'question'."""
        item = self.dataset_split[idx]
        context = item.get('context', '')
        question = item.get('question', '')
        # Combine context and question to form the prompt
        prompt = f"{context} {question}"
        return prompt

    def _get_sst2_prompt(self, idx):
        """Generates a prompt for the SST2 dataset using the 'sentence' field."""
        item = self.dataset_split[idx]
        sentence = item.get('sentence', '')
        return sentence

    def _get_sum_prompt(self, idx):
        """Generates a prompt for the 'sum' dataset using the 'turns' field."""
        item = self.dataset_split[idx]
        # 'turns' seems to be a list of text strings
        # Join them into a single string to form the prompt
        turns_list = item.get('turns', [])
        if isinstance(turns_list, list):
            prompt = " ".join(turns_list) # Join list items with a space
        else:
            # Fallback if 'turns' is not a list
            prompt = str(turns_list)
        return prompt

    def _get_generic_prompt(self, idx):
        """Generates a prompt using common fields, prioritizing 'text', 'content', 'question', 'instruction'."""
        item = self.dataset_split[idx]
        # Try common field names for the prompt text
        prompt = (item.get('text') or 
                  item.get('content') or 
                  item.get('question') or 
                  item.get('instruction') or 
                  item.get('prompt') or 
                  str(item)) # Fallback to string representation of the item
        return prompt

    def __len__(self):
        return len(self.dataset_split)

    def __getitem__(self, idx):
        prompt = self.prompt_func(idx)
        return prompt

    def get_raw_item(self, idx):
        """Returns the raw item from the underlying dataset split."""
        return self.dataset_split[idx]
