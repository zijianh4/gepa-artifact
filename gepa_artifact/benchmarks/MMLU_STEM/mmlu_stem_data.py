from ..benchmark import Benchmark
import dspy
from datasets import load_dataset
import random

class MMLUSTEMBench(Benchmark):
    def init_dataset(self):
        # Load the MMLU-STEM dataset from HuggingFace
        # Note: This dataset only has test split, so we'll split it ourselves
        raw_dataset = load_dataset("TIGER-Lab/MMLU-STEM")['test']
        
        # Convert to dspy examples
        all_data = []
        for x in raw_dataset:
            # Construct the question with choices
            question_text = x['question']
            choices = x['choices']
            
            # Format choices as A, B, C, D
            formatted_choices = []
            for i, choice in enumerate(choices):
                choice_letter = chr(ord('A') + i)  # A, B, C, D
                formatted_choices.append(f"{choice_letter}. {choice}")
            
            # Combine question and choices
            full_question = question_text + "\n" + "\n".join(formatted_choices)
            
            # Get the correct answer (answer is 0, 1, 2, or 3, we need A, B, C, D)
            correct_choice_index = x['answer']
            correct_answer = chr(ord('A') + correct_choice_index)
            
            all_data.append(
                dspy.Example({
                    "problem": full_question,
                    "answer": correct_answer,
                    "choices": formatted_choices,
                    "raw_answer_index": correct_choice_index
                }).with_inputs("problem")
            )
        
        # Shuffle the data with a fixed seed for reproducibility
        random.Random(42).shuffle(all_data)
        
        # Split into train (500), validation (500), and test (300)
        # Take the first 1300 samples to ensure we have enough data
        total_samples = min(1300, len(all_data))
        data_subset = all_data[:total_samples]
        
        self.train_set = data_subset[:500]
        self.val_set = data_subset[500:1000]
        self.test_set = data_subset[1000:1300]  # Take 300 for test
        
        # Combine all datasets
        self.dataset = self.train_set + self.val_set + self.test_set
