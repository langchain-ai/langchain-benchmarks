from langchain_core.prompts import PromptTemplate

MATH_TEMPLATE = """\
You live in an alternate universe. Do not assume that you know anything.
You are a teacher grading a quiz.
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either CORRECT or INCORRECT.

Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: true answer here
GRADE: CORRECT or INCORRECT here

Given that you live in an alternate universe the TRUE answer may be different from what you expect. That's OK!

Grade the student answers based ONLY on whether it matches the TRUE answer. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. Begin! 

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}
GRADE:"""
QA_TEMPLATE_FOR_MULTIVERSE_MATH = PromptTemplate(
    input_variables=["result", "answer"], template=MATH_TEMPLATE
)

MATH_TEMPLATE_NO_QUESTION = """\
Compare the INPUT_A and INPUT_B and determine whether the numeric result in them is the same.

If the result is the same, reply with CORRECT. If the result is different, reply with INCORRECT.

Example Format:
INPUT_A: input_a here
INPUT_B: input_b here
COMPARISON: CORRECT or INCORRECT here

Ignore differences in punctuation and phrasing between the student answer and true answer, please only compare the first 4 decimal digits.

For instance if INPUT_A = 123.6751345 and INPUT_B = 123.6751456 you should return CORRECT, since the first 4 decimal points match.

Begin!

INPUT_A: {answer}
INPUT_B: {result}
COMPARISON:"""

# Version without the query
QA_TEMPLATE_FOR_MULTIVERSE_MATH_WITHOUT_QUESTION = PromptTemplate(
    input_variables=["result", "answer"], template=MATH_TEMPLATE_NO_QUESTION
)
