import dspy

from .. import dspy_program

class GenerateResponse(dspy.Signature):
    """Solve the multiple choice question and provide the answer as one of A, B, C, or D in \\boxed{} format."""
    problem = dspy.InputField()
    answer = dspy.OutputField()

program_cot = dspy_program.CoT(GenerateResponse)
