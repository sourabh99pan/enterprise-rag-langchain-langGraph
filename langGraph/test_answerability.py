
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from src.vector_store import load_vector_store

load_dotenv()

vector_store = load_vector_store()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

question = input("Enter your policy question: ")

# Retrieve chunks using the original question for this experiment.
results = vector_store.similarity_search_with_score(
    question,
    k=3,
)

threshold = 1.0

relevant_docs = [
    (doc, score)
    for doc, score in results
    if score <= threshold
]

context = "\n\n".join(
    f"Source: {doc.metadata.get('source', 'unknown')}\n"
    f"Content: {doc.page_content}"
    for doc, score in relevant_docs
)

print("\nRetrieved context:")
print(context or "No relevant documents found.")

# Ask the LLM whether the context supports an answer.
check_prompt = f"""
You are checking whether company policy evidence supports an answer.

Question:
{question}

Retrieved context:
{context if context else "No context was retrieved."}

Does the context contain enough information to answer the question?

Reply with exactly one word:
YES
or
NO

Reply YES only if the context supports answering the actual question.
If the information is absent or insufficient, reply NO.
"""

check_response = llm.invoke(
    [
        SystemMessage(
            content="You are a strict evidence sufficiency checker."
        ),
        HumanMessage(content=check_prompt),
    ]
)

decision = check_response.content.strip().upper()

print(f"\nAnswerability decision: {decision}")

if decision == "YES":
    answer_prompt = f"""
Answer the question using only the policy context below.
Do not add unsupported information.
Mention the source filename.

Question:
{question}

Policy context:
{context}
"""

    answer = llm.invoke(answer_prompt)
    print("\nAssistant answer:")
    print(answer.content)

else:
    print(
        "\nAssistant answer: "
        "I don't know based on the available policy documents."
    )
