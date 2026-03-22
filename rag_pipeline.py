from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# -----------------------------
# Graph State
# -----------------------------
class State(TypedDict):
    question: str

    # Retrieval
    retrieval_query: str
    rewrite_tries: int
    need_retrieval: bool

    # Documents
    docs: List[Document]
    relevant_docs: List[Document]
    context: str

    # Answer generation
    answer: str

    # Verification
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str]
    retries: int

    # Usefulness
    isuse: Literal["useful", "not_useful"]
    use_reason: str


# -----------------------------
# Pydantic models for structured outputs
# -----------------------------
class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        ...,
        description="True if Ayurvedic knowledge base is needed, False for general knowledge"
    )

class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        ...,
        description="True if document discusses the topic/entity in the question"
    )

class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str] = Field(default_factory=list, description="Up to 3 direct quotes from context")

class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str = Field(..., description="Short reason in 1 line")

class RewriteDecision(BaseModel):
    retrieval_query: str = Field(
        ...,
        description="Rewritten query optimized for Ayurvedic knowledge retrieval"
    )


# -----------------------------
# Build the Self-RAG graph
# -----------------------------
def build_graph(retriever, llm):
    retrieve_parser = PydanticOutputParser(pydantic_object=RetrieveDecision)
    relevance_parser = PydanticOutputParser(pydantic_object=RelevanceDecision)
    issup_parser = PydanticOutputParser(pydantic_object=IsSUPDecision)
    isuse_parser = PydanticOutputParser(pydantic_object=IsUSEDecision)
    rewrite_parser = PydanticOutputParser(pydantic_object=RewriteDecision)

    # --- Prompts ---
    decide_retrieval_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an Ayurvedic AI assistant. Decide if retrieval from Ayurvedic texts is needed.\n\n"
            "Guidelines:\n"
            "- should_retrieve=True for questions about:\n"
            "  * Specific herbs, plants, or medicinal substances\n"
            "  * Doshas (Vata, Pitta, Kapha)\n"
            "  * Ayurvedic treatments, therapies, or remedies\n"
            "  * Classical Ayurvedic texts or principles\n"
            "  * Ayurvedic diet, lifestyle, or seasonal regimens\n"
            "  * Specific diseases and their Ayurvedic management\n\n"
            "- should_retrieve=False for:\n"
            "  * General wellness advice not specific to Ayurveda\n"
            "  * Basic definitions of common terms\n"
            "  * General health questions\n\n"
            "- If unsure, choose True to ensure accurate Ayurvedic information.\n\n"
            + retrieve_parser.get_format_instructions()
        ),
        ("human", "Question: {question}"),
    ])

    direct_generation_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are Ojas.ai, a friendly and knowledgeable Ayurvedic assistant.\n\n"
            "Answer using general Ayurvedic knowledge in clear, simple English.\n"
            "- Write in complete, readable sentences\n"
            "- Never use symbols like <, >, +, #\n"
            "- Mention herb names and Sanskrit terms naturally\n"
            "- Keep your answer concise and helpful\n"
            "- End with: Please consult an Ayurvedic practitioner before starting any treatment."
        ),
        ("human", "{question}"),
    ])

    is_relevant_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are judging if an Ayurvedic document is relevant to the question.\n\n"
            "A document is relevant if it discusses:\n"
            "- The specific herb, plant, or substance mentioned\n"
            "- The dosha, treatment, or condition asked about\n"
            "- Related Ayurvedic principles or concepts\n\n"
            "Examples:\n"
            "- Questions about Ashwagandha are relevant to documents discussing adaptogens, stress, or that specific herb\n"
            "- Questions about Vata dosha are relevant to documents about Vata, air element, or nervous system\n"
            "- Questions about digestive fire are relevant to documents about Agni, digestion, or metabolic processes\n\n"
            "When unsure, return is_relevant=true.\n\n"
            + relevance_parser.get_format_instructions()
        ),
        ("human", "Question:\n{question}\n\nDocument:\n{document}"),
    ])

    rag_generation_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are Ojas.ai, an expert Ayurvedic AI assistant. You answer questions in clear, friendly, readable English.\n\n"
            "You will receive raw CONTEXT extracted from scanned Ayurvedic texts. This raw text may contain:\n"
            "- Symbols like <, >, +, -, #, * used as PDF formatting artifacts\n"
            "- Reference codes like -Si12#15.7, (Chopra et al., 1958) etc.\n"
            "- Incomplete sentences or fragmented lines from PDF extraction\n\n"
            "Your job:\n"
            "1. Extract the relevant Ayurvedic knowledge from this raw text\n"
            "2. Write a clean, well-structured answer in plain English\n"
            "3. Preserve herb names, Sanskrit terms, and Ayurvedic concepts accurately\n"
            "4. Remove all symbols, reference codes, and formatting artifacts from your answer\n"
            "5. Do NOT copy raw text directly - always rewrite in clean sentences\n"
            "6. Do NOT mention that you received context or raw text\n"
            "7. Do NOT hallucinate - only use what is in the context\n"
            "8. If context is insufficient say: Based on available Ayurvedic texts and give what you have\n\n"
            "Format your answer as:\n"
            "- A brief intro sentence answering the question directly\n"
            "- Key herbs or remedies with their benefits in simple terms\n"
            "- Any important precautions or recommendations\n"
            "- End with: Please consult an Ayurvedic practitioner before starting any treatment."
        ),
        ("human", "Question:\n{question}\n\nContext:\n{context}"),
    ])

    issup_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You verify if the ANSWER about Ayurveda is supported by the CONTEXT.\n\n"
            "How to decide issup:\n"
            "- fully_supported:\n"
            "  Every claim about herbs, doshas, treatments, or principles is explicitly in CONTEXT.\n"
            "  No interpretive or qualitative words unless present in CONTEXT.\n\n"
            "- partially_supported:\n"
            "  Core Ayurvedic facts are supported, BUT answer adds interpretation or\n"
            "  qualitative descriptions not in CONTEXT (e.g., 'powerful herb', 'best for', etc.).\n\n"
            "- no_support:\n"
            "  Key claims are not supported by CONTEXT.\n\n"
            "Rules:\n"
            "- Be strict about Ayurvedic accuracy\n"
            "- Evidence: include up to 3 direct quotes from CONTEXT\n"
            "- Do not use outside knowledge\n\n"
            + issup_parser.get_format_instructions()
        ),
        ("human", "Question:\n{question}\n\nAnswer:\n{answer}\n\nContext:\n{context}"),
    ])

    revise_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an Ayurvedic answer reviser. Your job is to rewrite the answer in clean, readable English.\n\n"
            "Rules:\n"
            "- Use ONLY information from the CONTEXT\n"
            "- Remove all symbols like <, >, +, -, #, * from your answer\n"
            "- Remove all reference codes like (Chopra et al.) or -Si12#15.7\n"
            "- Rewrite in clear, complete sentences - never copy raw text\n"
            "- Preserve herb names and Sanskrit terms accurately\n"
            "- Do NOT hallucinate or add information not in CONTEXT\n"
            "- End with: Please consult an Ayurvedic practitioner before starting any treatment."
        ),
        ("human", "Question:\n{question}\n\nCurrent Answer:\n{answer}\n\nCONTEXT:\n{context}"),
    ])

    isuse_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You judge if the ANSWER is useful for the Ayurvedic QUESTION.\n\n"
            "Rules:\n"
            "- useful: Answer directly addresses the Ayurvedic question with specific information\n"
            "- not_useful: Answer is generic, off-topic, or doesn't provide requested Ayurvedic knowledge\n"
            "- Do NOT re-check grounding (IsSUP already did that)\n"
            "- Only check: 'Did we answer the Ayurvedic question?'\n\n"
            + isuse_parser.get_format_instructions()
        ),
        ("human", "Question:\n{question}\n\nAnswer:\n{answer}"),
    ])

    rewrite_for_retrieval_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Rewrite the QUESTION for optimal retrieval from Ayurvedic texts.\n\n"
            "Rules:\n"
            "- Keep it concise (8-20 words)\n"
            "- Include Sanskrit terms if relevant (e.g., Vata, Pitta, Kapha, Agni, rasayana)\n"
            "- Add Ayurvedic keywords that appear in classical texts\n"
            "- Preserve herb/plant names exactly\n"
            "- Remove filler words\n\n"
            "Examples:\n"
            "Q: 'What are the benefits of Ashwagandha?'\n"
            "A: 'Ashwagandha benefits properties uses rasayana adaptogen stress'\n\n"
            "Q: 'How to balance Vata dosha?'\n"
            "A: 'Vata dosha balance pacify treatment diet lifestyle oil massage warm'\n\n"
            + rewrite_parser.get_format_instructions()
        ),
        ("human", "QUESTION:\n{question}\n\nPrevious retrieval query:\n{retrieval_query}\n\nAnswer:\n{answer}"),
    ])

    MAX_RETRIES = 2
    MAX_REWRITE_TRIES = 1

    # --- Node functions ---
    def decide_retrieval(state: State):
        try:
            response = llm.invoke(decide_retrieval_prompt.format_messages(question=state["question"]))
            decision = retrieve_parser.parse(response.content)
            return {"need_retrieval": decision.should_retrieve}
        except:
            return {"need_retrieval": True}

    def route_after_decide(state: State) -> Literal["generate_direct", "retrieve"]:
        return "retrieve" if state["need_retrieval"] else "generate_direct"

    def generate_direct(state: State):
        out = llm.invoke(direct_generation_prompt.format_messages(question=state["question"]))
        return {"answer": out.content}

    def retrieve(state: State):
        q = state.get("retrieval_query") or state["question"]
        return {"docs": retriever.invoke(q)}

    def is_relevant(state: State):
        relevant_docs: List[Document] = []
        for doc in state.get("docs", []):
            try:
                response = llm.invoke(
                    is_relevant_prompt.format_messages(
                        question=state["question"],
                        document=doc.page_content,
                    )
                )
                decision = relevance_parser.parse(response.content)
                if decision.is_relevant:
                    relevant_docs.append(doc)
            except:
                relevant_docs.append(doc)
        return {"relevant_docs": relevant_docs}

    def route_after_relevance(state: State) -> Literal["generate_from_context", "no_answer_found"]:
        if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
            return "generate_from_context"
        return "no_answer_found"

    def generate_from_context(state: State):
        context = "\n\n---\n\n".join([d.page_content for d in state.get("relevant_docs", [])]).strip()
        if not context:
            return {"answer": "No answer found in Ayurvedic texts.", "context": ""}
        out = llm.invoke(
            rag_generation_prompt.format_messages(question=state["question"], context=context)
        )
        return {"answer": out.content, "context": context}

    def no_answer_found(state: State):
        return {
            "answer": "I could not find relevant information in the Ayurvedic knowledge base. Please consult an Ayurvedic practitioner for specific guidance.",
            "context": ""
        }

    def is_sup(state: State):
        try:
            response = llm.invoke(
                issup_prompt.format_messages(
                    question=state["question"],
                    answer=state.get("answer", ""),
                    context=state.get("context", ""),
                )
            )
            decision = issup_parser.parse(response.content)
            return {"issup": decision.issup, "evidence": decision.evidence}
        except:
            return {"issup": "partially_supported", "evidence": []}

    def route_after_issup(state: State) -> Literal["accept_answer", "revise_answer"]:
        if state.get("issup") == "fully_supported":
            return "accept_answer"
        if state.get("retries", 0) >= MAX_RETRIES:
            return "accept_answer"
        return "revise_answer"

    def accept_answer(state: State):
        return {}

    def revise_answer(state: State):
        out = llm.invoke(
            revise_prompt.format_messages(
                question=state["question"],
                answer=state.get("answer", ""),
                context=state.get("context", ""),
            )
        )
        return {
            "answer": out.content,
            "retries": state.get("retries", 0) + 1,
        }

    def is_use(state: State):
        try:
            response = llm.invoke(
                isuse_prompt.format_messages(
                    question=state["question"],
                    answer=state.get("answer", ""),
                )
            )
            decision = isuse_parser.parse(response.content)
            return {"isuse": decision.isuse, "use_reason": decision.reason}
        except:
            return {"isuse": "useful", "use_reason": "Parsing failed, assuming useful"}

    def route_after_isuse(state: State) -> Literal["END", "rewrite_question", "no_answer_found"]:
        if state.get("isuse") == "useful":
            return "END"
        if state.get("rewrite_tries", 0) >= MAX_REWRITE_TRIES:
            return "no_answer_found"
        return "rewrite_question"

    def rewrite_question(state: State):
        try:
            response = llm.invoke(
                rewrite_for_retrieval_prompt.format_messages(
                    question=state["question"],
                    retrieval_query=state.get("retrieval_query", ""),
                    answer=state.get("answer", ""),
                )
            )
            decision = rewrite_parser.parse(response.content)
            retrieval_query = decision.retrieval_query
        except:
            retrieval_query = state["question"]
        return {
            "retrieval_query": retrieval_query,
            "rewrite_tries": state.get("rewrite_tries", 0) + 1,
            "docs": [],
            "relevant_docs": [],
            "context": "",
        }

    # --- Build graph ---
    g = StateGraph(State)

    g.add_node("decide_retrieval", decide_retrieval)
    g.add_node("generate_direct", generate_direct)
    g.add_node("retrieve", retrieve)
    g.add_node("is_relevant", is_relevant)
    g.add_node("generate_from_context", generate_from_context)
    g.add_node("no_answer_found", no_answer_found)
    g.add_node("is_sup", is_sup)
    g.add_node("accept_answer", accept_answer)
    g.add_node("revise_answer", revise_answer)
    g.add_node("is_use", is_use)
    g.add_node("rewrite_question", rewrite_question)

    g.add_edge(START, "decide_retrieval")
    g.add_conditional_edges(
        "decide_retrieval",
        route_after_decide,
        {"generate_direct": "generate_direct", "retrieve": "retrieve"},
    )
    g.add_edge("generate_direct", END)
    g.add_edge("retrieve", "is_relevant")
    g.add_conditional_edges(
        "is_relevant",
        route_after_relevance,
        {"generate_from_context": "generate_from_context", "no_answer_found": "no_answer_found"},
    )
    g.add_edge("no_answer_found", END)
    g.add_edge("generate_from_context", "is_sup")
    g.add_conditional_edges(
        "is_sup",
        route_after_issup,
        {"accept_answer": "is_use", "revise_answer": "revise_answer"},
    )
    g.add_edge("revise_answer", "is_sup")
    g.add_conditional_edges(
        "is_use",
        route_after_isuse,
        {"END": END, "rewrite_question": "rewrite_question", "no_answer_found": "no_answer_found"},
    )
    g.add_edge("rewrite_question", "retrieve")

    return g.compile()