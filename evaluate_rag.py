
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase

from src.rag_chain import create_rag_chain_with_context


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 2. Load vector store and RAG chain
# --------------------------------------------------

vector_store = Chroma(
    collection_name="enterprise_policy_docs",
    embedding_function=OpenAIEmbeddings(
        model="text-embedding-3-small"
    ),
    persist_directory=str(PROJECT_ROOT / "chroma_db"),
)

rag_chain = create_rag_chain_with_context(vector_store)


# --------------------------------------------------
# 3. Evaluation questions
# --------------------------------------------------

answerable_questions = [
    "What is the maximum hotel reimbursement per night?",
    "What is the daily meal expense limit?",
    "How many annual leave days do full-time employees receive?",
    "How far in advance must annual leave be requested?",
    "When is a medical certificate required for sick leave?",
    "How many unused annual leave days can be carried forward?",
    "What are the minimum password requirements?",
    "Where is multi-factor authentication mandatory?",
    "When must employees submit expense claims after a trip?",
    "When are itemized receipts required for expenses?",
]

unanswerable_questions = [
    "What is the company's work from home policy?",
    "How many paid parental leave days do employees receive?",
]


# --------------------------------------------------
# 4. Evaluate answerable questions
# --------------------------------------------------

test_cases = []

print("\n========== ANSWERABLE QUESTIONS ==========")

for index, question in enumerate(answerable_questions, start=1):
    logger.info(
        "Evaluating answerable question %s/%s",
        index,
        len(answerable_questions),
    )

    result = rag_chain.invoke(question)

    answer = result["answer"]
    context = result["context"]

    sources = sorted({
        doc.metadata.get("source", "unknown")
        for doc in result["source_documents"]
    })

    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    print(f"Retrieved sources: {sources}")

    test_cases.append(
        LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=[context],
        )
    )


# --------------------------------------------------
# 5. Run DeepEval metrics
# --------------------------------------------------

print("\n========== DEEPEVAL METRICS ==========")

metrics = [
    FaithfulnessMetric(
        threshold=0.7,
        model="gpt-4o-mini",
        async_mode=False,
    ),
    AnswerRelevancyMetric(
        threshold=0.7,
        model="gpt-4o-mini",
        async_mode=False,
    ),
    ContextualRelevancyMetric(
        threshold=0.7,
        model="gpt-4o-mini",
        async_mode=False,
    ),
]

logger.info(
    "Starting DeepEval for %s test cases",
    len(test_cases),
)

evaluate(
    test_cases=test_cases,
    metrics=metrics,
)

logger.info("DeepEval evaluation completed")


# --------------------------------------------------
# 6. Test unanswerable questions
# --------------------------------------------------

print("\n========== UNANSWERABLE QUESTIONS ==========")

for question in unanswerable_questions:
    result = rag_chain.invoke(question)
    answer = result["answer"]

    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")

    # Basic check for the expected refusal phrase.
    if "don't know based on the available policy documents" in answer.lower():
        print("Result: PASS - Expected refusal detected")
    else:
        print(
            "Result: REVIEW - Expected refusal phrase not detected"
        )

print("\n========== EVALUATION COMPLETED ==========")
