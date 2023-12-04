from enum import Enum
from typing import List, Optional

from langchain.pydantic_v1 import BaseModel, Field


class QuestionCategory(str, Enum):
    IMPLEMENTATION_ISSUES = "Implementation Issues"  # about existing implementation
    FEATURE_REQUESTS = "Feature Requests"
    CONCEPT_EXPLANATIONS = "Concept Explanations"
    CODE_OPTIMIZATION = "Code Optimization"
    SECURITY_AND_PRIVACY_CONCERNS = "Security and Privacy Concerns"
    MODEL_TRAINING_AND_FINE_TUNING = "Model Training and Fine-tuning"
    DATA_HANDLING_AND_MANIPULATION = "Data Handling and Manipulation"
    USER_INTERACTION_FLOW = "User Interaction Flow"
    TECHNICAL_INTEGRATION = "Technical Integration"
    ERROR_HANDLING_AND_LOGGING = "Error Handling and Logging"
    CUSTOMIZATION_AND_CONFIGURATION = "Customization and Configuration"
    EXTERNAL_API_AND_DATA_SOURCE_INTEGRATION = (
        "External API and Data Source Integration"
    )
    LANGUAGE_AND_LOCALIZATION = "Language and Localization"
    STREAMING_AND_REAL_TIME_PROCESSING = "Streaming and Real-time Processing"
    TOOL_DEVELOPMENT = "Tool Development"
    FUNCTION_CALLING = "Function Calling"
    LLM_INTEGRATIONS = "LLM Integrations"
    GENERAL_AGENT_QUESTIONS = "General Agent Question"
    GENERAL_CHIT_CHAT = "General Chit Chat"
    MEMORY = "Memory"
    DEBUGGING_HELP = "Debugging Help"
    APPLICATION_DESIGN = "Application Design"
    PROMPT_TEMPLATES = "Prompt Templates"
    COST_TRACKING = "Cost Tracking"
    OTHER = "Other"


class Sentiment(str, Enum):
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"
    POSITIVE = "Positive"


class ProgrammingLanguage(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    UNKNOWN = "unknown"
    OTHER = "other"


class QuestionCategorization(BaseModel):
    question_category: QuestionCategory
    category_if_other: Optional[str] = Field(
        default=None, description="question category if the category above is 'other'"
    )
    is_off_topic: bool = Field(
        description="If the input is general chit chat or does not pertain to technical inqueries about LangChain or building/debugging applications with LLMs/AI, it is off topic. For context, LangChain is a library and framework designed"
        " to assist in building applications with LLMs. Questions may also be about similar packages like LangServe, LangSmith, OpenAI, Anthropic, vectorstores, agents, etc."
    )
    toxicity: int = Field(
        ge=0, lt=6, description="Whether or not the input question is toxic"
    )
    sentiment: Sentiment
    programming_language: ProgrammingLanguage


#  resolve the issue, provide guidance, or ask for more information
class ResponseType(str, Enum):
    RESOLVE_ISSUE = "resolve issue"
    PROVIDE_GUIDANCE = "provide guidance"
    REQUEST_INFORMATION = "request information"
    GIVE_UP = "give up"
    NONE = "none"
    OTHER = "other"


class ResponseCategorization(BaseModel):
    response_type: ResponseType
    response_type_if_other: Optional[str] = None
    confidence_level: int = Field(
        ge=0, lt=6, description="The confidence of the assistant in its answer."
    )
    followup_actions: Optional[List[str]] = Field(
        description="Actions the assistant recommended the user take."
    )


class GenerateTicket(BaseModel):
    """Generate a ticket containing all the extracted information."""

    issue_summary: str = Field(
        description="short (<10 word) summary of the issue or question"
    )
    question: QuestionCategorization = Field(
        description="Information inferred from the the question."
    )
    response: ResponseCategorization = Field(
        description="Information inferred from the the response."
    )
