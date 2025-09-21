import dspy
import re

from .MATH_data import MATHBench
from .MATH_program import program_cot

from ..benchmark import BenchmarkMeta

def metric(example, prediction, trace=None):
    correct_answer = str(example['answer']).strip()
    try:
        llm_answer = str(prediction.answer).strip()
    except (ValueError, AttributeError):
        return 0
    
    # Try to normalize answers for comparison
    correct_answer = correct_answer.lower()
    llm_answer = llm_answer.lower()
    
    # Remove common prefixes/suffixes
    for prefix in ['the answer is', 'answer:', 'result:', 'therefore']:
        if llm_answer.startswith(prefix):
            llm_answer = llm_answer[len(prefix):].strip()
    
    # For mathematical expressions, we need to preserve mathematical symbols
    # Only remove extra whitespace and normalize spacing around operators
    correct_answer = re.sub(r'\s+', ' ', correct_answer).strip()
    llm_answer = re.sub(r'\s+', ' ', llm_answer).strip()
    
    return int(correct_answer == llm_answer)

def metric_with_feedback(example, prediction, trace=None):
    correct_answer = str(example['answer']).strip()
    written_solution = example.get('solution', '')
    
    try:
        llm_answer = str(prediction.answer).strip()
    except (ValueError, AttributeError):
        feedback_text = f"The final answer must be a valid response. You responded with '{prediction.answer}', which couldn't be processed. Please ensure your answer is clear and complete."
        feedback_text += f" The correct answer is '{correct_answer}'."
        if written_solution:
            feedback_text += f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems."
        return dspy.Prediction(score=0, feedback=feedback_text)

    # Try to normalize answers for comparison
    correct_answer_norm = correct_answer.lower()
    llm_answer_norm = llm_answer.lower()
    
    # Remove common prefixes/suffixes
    for prefix in ['the answer is', 'answer:', 'result:', 'therefore']:
        if llm_answer_norm.startswith(prefix):
            llm_answer_norm = llm_answer_norm[len(prefix):].strip()
    
    # For mathematical expressions, we need to preserve mathematical symbols
    # Only remove extra whitespace and normalize spacing around operators
    correct_answer_norm = re.sub(r'\s+', ' ', correct_answer_norm).strip()
    llm_answer_norm = re.sub(r'\s+', ' ', llm_answer_norm).strip()
    
    score = int(correct_answer_norm == llm_answer_norm)

    feedback_text = ""
    if score == 1:
        feedback_text = f"Your answer is correct. The correct answer is '{correct_answer}'."
    else:
        feedback_text = f"Your answer is incorrect. You answered '{llm_answer}', but the correct answer is '{correct_answer}'."
    
    if written_solution:
        feedback_text += f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems."

    return dspy.Prediction(score=score, feedback=feedback_text)

benchmark = [
    BenchmarkMeta(
        MATHBench,
        [
            program_cot,
        ],
        metric=metric,
        metric_with_feedback=metric_with_feedback,
    )
]
