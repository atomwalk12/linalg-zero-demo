import random

from linalg_zero.generator.models import DifficultyCategory, QuestionTemplate, Task


def get_static_templates(
    question_type: Task, difficulty: DifficultyCategory, previous_step: int | None
) -> list[QuestionTemplate]:
    verb = random.choice(["Find", "Calculate", "Compute", "Determine", "Evaluate"])
    step_vars = {0: "B", 1: "C", 2: "D", 3: "E", 4: "F", 5: "G", 6: "H", 7: "I", 8: "J", 9: "K"}
    new_var = step_vars[previous_step] if previous_step is not None else "A"

    templates = []
    if question_type == Task.ONE_DETERMINANT:
        templates.extend([
            QuestionTemplate(
                template_string=f"{verb} the determinant of matrix {new_var}, where {new_var} = {{matrix}}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.ONE_DETERMINANT,
            ),
            QuestionTemplate(
                template_string=f"Given matrix {new_var} = {{matrix}}, find det({new_var}).",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.ONE_DETERMINANT,
            ),
            QuestionTemplate(
                template_string=f"For {new_var} = {{matrix}}, compute det({new_var}).",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.ONE_DETERMINANT,
            ),
        ])
    elif question_type == Task.ONE_FROBENIUS_NORM:
        templates.extend([
            QuestionTemplate(
                template_string=f"{verb} the Frobenius norm of matrix {new_var} = {{matrix}}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.ONE_FROBENIUS_NORM,
            ),
            QuestionTemplate(
                template_string=f"Given matrix {new_var} = {{matrix}}, find ||{new_var}||_F.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.ONE_FROBENIUS_NORM,
            ),
            QuestionTemplate(
                template_string=f"What is ||{new_var}||_F for {new_var} = {{matrix}}?",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.ONE_FROBENIUS_NORM,
            ),
        ])
    elif question_type == Task.ONE_RANK:
        templates.extend([
            QuestionTemplate(
                template_string=f"{verb} the rank of matrix {new_var} = {{matrix}}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.ONE_RANK,
            ),
            QuestionTemplate(
                template_string=f"What is the rank of matrix {new_var} = {{matrix}}?",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.ONE_RANK,
            ),
            QuestionTemplate(
                template_string=f"Find rank({new_var}) for {new_var} = {{matrix}}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.ONE_RANK,
            ),
        ])
    elif question_type == Task.ONE_TRANSPOSE:
        templates.extend([
            QuestionTemplate(
                template_string=f"{verb} the transpose of matrix {new_var} = {{matrix}}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.ONE_TRANSPOSE,
            ),
            QuestionTemplate(
                template_string=f"Find {new_var}^T for {new_var} = {{matrix}}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.ONE_TRANSPOSE,
            ),
            QuestionTemplate(
                template_string=f"What is the transpose of {new_var} = {{matrix}}?",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.ONE_TRANSPOSE,
            ),
        ])
    elif question_type == Task.ONE_TRACE:
        templates.extend([
            QuestionTemplate(
                template_string=f"{verb} the trace of matrix {new_var} = {{matrix}}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.ONE_TRACE,
            ),
            QuestionTemplate(
                template_string=f"Find tr({new_var}) for {new_var} = {{matrix}}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.ONE_TRACE,
            ),
            QuestionTemplate(
                template_string=f"What is the trace of {new_var} = {{matrix}}?",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.ONE_TRACE,
            ),
        ])
    elif question_type == Task.ONE_COFACTOR:
        templates.extend([
            QuestionTemplate(
                template_string=f"{verb} the cofactor matrix of {new_var} = {{matrix}}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.ONE_COFACTOR,
            ),
            QuestionTemplate(
                template_string=f"Find the cofactor matrix for {new_var} = {{matrix}}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.ONE_COFACTOR,
            ),
            QuestionTemplate(
                template_string=f"What is the matrix of cofactors for {new_var} = {{matrix}}?",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.ONE_COFACTOR,
            ),
        ])
    else:
        raise ValueError(f"Question type {question_type} not supported")
    return templates
