"""
Small factory so every node asks for "the generator LLM" or "the grader LLM"
without caring whether we're on OpenAI, Groq, or a local HuggingFace pipeline.
"""
from src.config import settings


def _openai_chat(model_name: str, temperature: float = 0.0):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=settings.openai_api_key,
    )


def _groq_chat(model_name: str, temperature: float = 0.0):
    # Groq exposes an OpenAI-compatible chat endpoint, so we reuse ChatOpenAI
    # and just point it at Groq's base_url with a Groq API key + model name.
    # max_retries makes LangChain automatically retry with exponential
    # backoff on 429 rate-limit responses instead of crashing the whole
    # run -- Groq's free tier has a fairly low tokens-per-minute cap, so
    # transient rate limits are expected and should be waited out, not
    # treated as fatal errors.
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
        max_retries=6,
        timeout=60,
    )


def _local_chat(temperature: float = 0.0):
    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline

    pipe = pipeline(
        "text-generation",
        model=settings.local_llm_model,
        max_new_tokens=512,
        do_sample=temperature > 0,
        temperature=max(temperature, 0.01),
    )
    return HuggingFacePipeline(pipeline=pipe)


def _chat_for(model_name: str, temperature: float):
    if settings.llm_provider == "local":
        return _local_chat(temperature=temperature)
    if settings.llm_provider == "groq":
        return _groq_chat(model_name, temperature=temperature)
    return _openai_chat(model_name, temperature=temperature)


def get_generator_llm(temperature: float = 0.2):
    """The model that writes the final answer -- allowed a bit of temperature."""
    return _chat_for(settings.generator_model, temperature)


def get_grader_llm(temperature: float = 0.0):
    """The model used for relevance/hallucination/answer grading -- kept at
    temperature 0 because we want consistent, deterministic-as-possible
    yes/no judgments, and can be a cheaper model than the generator."""
    return _chat_for(settings.grader_model, temperature)