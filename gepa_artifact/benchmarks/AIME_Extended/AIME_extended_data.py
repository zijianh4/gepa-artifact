from ..benchmark import Benchmark
import dspy

from datasets import load_dataset

class AIMEExtendedBench(Benchmark):
    def init_dataset(self):
        # Load training data from di-zhang-fdu/AIME_1983_2024
        train_split = load_dataset("di-zhang-fdu/AIME_1983_2024")['train']
        train_split = [
            dspy.Example({
                "problem": x['Question'],
                'answer': x['Answer'],
            }).with_inputs("problem")
            for x in train_split
        ]
        import random
        random.Random(0).shuffle(train_split)
        tot_num = len(train_split)

        # Load test data from MathArena/aime_2025
        test_split = load_dataset("MathArena/aime_2025")['train']
        test_split = [
            dspy.Example({
                "problem": x['problem'],
                'answer': x['answer'],
            }).with_inputs("problem")
            for x in test_split
        ]

        # 50/50 split for training/validation
        self.train_set = train_split[:int(0.5 * tot_num)]
        self.val_set = train_split[int(0.5 * tot_num):]
        self.test_set = test_split * 5

        self.dataset = self.train_set + self.val_set + self.test_set
