import dspy
import re

from .mmlu_stem_data import MMLUSTEMBench
from .mmlu_stem_program import program_cot

from ..benchmark import BenchmarkMeta

def metric(example, prediction, trace=None):
    correct_answer = str(example['answer']).strip().upper()
    try:
        llm_answer = str(prediction.answer).strip().upper()
    except (ValueError, AttributeError):
        return 0
    
    # Extract just the letter (A, B, C, D) from the answer
    llm_letter = re.search(r'[ABCD]', llm_answer)
    if llm_letter:
        llm_answer = llm_letter.group()
    
    return int(correct_answer == llm_answer)

def metric_with_feedback(example, prediction, trace=None):
    correct_answer = str(example['answer']).strip().upper()
    
    try:
        llm_answer = str(prediction.answer).strip().upper()
    except (ValueError, AttributeError):
        feedback_text = f"The final answer must be one of the choices A, B, C, or D. You responded with '{prediction.answer}', which couldn't be processed. Please ensure your answer is one of the valid choices."
        feedback_text += f" The correct answer is '{correct_answer}'."
        return dspy.Prediction(score=0, feedback=feedback_text)

    # Extract just the letter (A, B, C, D) from the answer
    llm_letter = re.search(r'[ABCD]', llm_answer)
    if llm_letter:
        llm_answer = llm_letter.group()
    else:
        feedback_text = f"The final answer must be one of the choices A, B, C, or D. You responded with '{prediction.answer}', which doesn't contain a valid choice letter. Please ensure your answer is one of the valid choices."
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
