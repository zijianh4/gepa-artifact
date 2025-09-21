import dspy
import re

from .mmlu_stem_data import MMLUSTEMBench
from .mmlu_stem_program import program_cot

from ..benchmark import BenchmarkMeta

def metric(example, prediction, trace=None):
    correct_answer = str(example['answer']).strip().upper()
    try:
        llm_answer = str(prediction.answer).strip()
    except (ValueError, AttributeError):
        return 0
    
    # Look for \boxed{} pattern which contains the answer
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    match = re.search(boxed_pattern, llm_answer)
    if match:
        llm_answer = match.group(1).strip().upper()
    else:
        # No \boxed{} format found - treat as no answer and wrong
        return 0
    
    return int(correct_answer == llm_answer)

def metric_with_feedback(example, prediction, trace=None):
    correct_answer = str(example['answer']).strip().upper()
    
    try:
        llm_answer = str(prediction.answer).strip()
    except (ValueError, AttributeError):
        feedback_text = f"The final answer must be in the format \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, or \\boxed{{D}}. You responded with '{prediction.answer}', which couldn't be processed. Please ensure your answer is in the correct \\boxed{{}} format."
        feedback_text += f" The correct answer is '{correct_answer}'."
        return dspy.Prediction(score=0, feedback=feedback_text)

    # Look for \boxed{} pattern which contains the answer
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    match = re.search(boxed_pattern, llm_answer)
    if match:
        llm_answer = match.group(1).strip().upper()
    else:
        # No \boxed{} format found - treat as no answer and wrong
        feedback_text = f"The final answer must be in the format \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, or \\boxed{{D}}. You responded with '{prediction.answer}', which doesn't contain the required \\boxed{{}} format. Please ensure your answer is in the \\boxed{{}} format."
        feedback_text += f" The correct answer is '{correct_answer}'."
        return dspy.Prediction(score=0, feedback=feedback_text)

    score = int(correct_answer == llm_answer)

    feedback_text = ""
    if score == 1:
        feedback_text = f"Your answer is correct. The correct answer is '{correct_answer}'."
    else:
        feedback_text = f"Your answer is incorrect. You answered '{llm_answer}', but the correct answer is '{correct_answer}'."

    return dspy.Prediction(score=score, feedback=feedback_text)

benchmark = [
    BenchmarkMeta(
        MMLUSTEMBench,
        [
            program_cot,
        ],
        metric=metric,
        metric_with_feedback=metric_with_feedback,
    )
]

__all__ = ['MMLUSTEMBench', 'benchmark']
