from ..benchmark import Benchmark
import dspy
import re
from datasets import load_dataset

class MATHBench(Benchmark):
    def extract_answer_from_solution(self, solution):
        """Extract the final answer from the solution text."""
        if not solution:
            return ""
        
        # Look for \boxed{} pattern which contains the answer
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        match = re.search(boxed_pattern, solution)
        if match:
            return match.group(1).strip()
        
        # If no \boxed{} found, try to extract from the end of the solution
        # This is a fallback for cases where the answer might not be boxed
        lines = solution.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('\\') and not line.startswith('%'):
                # Look for common answer patterns
                answer_patterns = [
                    r'answer[:\s]*([^\s,\.]+)',
                    r'result[:\s]*([^\s,\.]+)',
                    r'therefore[:\s]*([^\s,\.]+)',
                    r'final[:\s]*([^\s,\.]+)',
                    r'([^\s,\.]+)\s*$'  # Last non-whitespace sequence
                ]
                for pattern in answer_patterns:
                    match = re.search(pattern, line.lower())
                    if match:
                        return match.group(1).strip()
        
        return ""
    
    def init_dataset(self):
        # Load training data from xDAN2099/lighteval-MATH
        train_split = load_dataset("xDAN2099/lighteval-MATH")['train']
        
        # Convert to dspy examples and extract answers from solutions
        train_data = []
        for x in train_split:
            answer = self.extract_answer_from_solution(x.get('solution', ''))
            if answer:  # Only include examples where we can extract an answer
                train_data.append(
                    dspy.Example({
                        "problem": x['problem'],
                        'solution': x.get('solution', ''),
                        'answer': answer,
                    }).with_inputs("problem")
                )
        
        # Shuffle and split into train and validation sets
        import random
        random.Random(0).shuffle(train_data)
        
        # Take 500 samples for training and 500 for validation
        self.train_set = train_data[:500]
        self.val_set = train_data[500:1000]
        
        # Load test data from HuggingFaceH4/MATH-500
        test_split = load_dataset("HuggingFaceH4/MATH-500")['test']
        test_data = []
        for x in test_split:
            answer = self.extract_answer_from_solution(x.get('solution', ''))
            if answer:  # Only include examples where we can extract an answer
                test_data.append(
                    dspy.Example({
                        "problem": x['problem'],
                        'solution': x.get('solution', ''),
                        'answer': answer,
                    }).with_inputs("problem")
                )
        
        self.test_set = test_data
        
        # Combine all datasets
        self.dataset = self.train_set + self.val_set + self.test_set
