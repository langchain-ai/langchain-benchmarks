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
