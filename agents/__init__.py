import logging
from typing import Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .recorder import Recorder
from .swarm import Swarm
from .templates.llm_agents import (
    DirectLocalLLM,
    FastLLM,
    GuidedLLM,
    LLM,
    LocalLLM,
    ReasoningLLM,
)
from .templates.random_agent import Random

logger = logging.getLogger(__name__)

load_dotenv()

__all__ = [
    "Swarm",
    "Random",
    "LLM",
    "FastLLM",
    "ReasoningLLM",
    "GuidedLLM",
    "LocalLLM",
    "DirectLocalLLM",
    "Agent",
    "Recorder",
    "Playback",
    "AVAILABLE_AGENTS",
]


def _import_optional_agents() -> None:
    optional_imports = [
        (
            ".templates.langgraph_functional_agent",
            ["LangGraphFunc", "LangGraphTextOnly"],
        ),
        (".templates.langgraph_random_agent", ["LangGraphRandom"]),
        (".templates.langgraph_thinking", ["LangGraphThinking"]),
        (".templates.multimodal", ["MultiModalLLM"]),
        (".templates.openclaw_agent", ["OpenClaw"]),
        (".templates.reasoning_agent", ["ReasoningAgent"]),
        (".templates.smolagents", ["SmolCodingAgent", "SmolVisionAgent"]),
    ]

    for module_name, names in optional_imports:
        try:
            module = __import__(module_name, globals(), locals(), names, 1)
        except Exception as exc:
            logger.info("Skipping optional agents from %s: %s", module_name, exc)
            continue

        for name in names:
            obj = getattr(module, name, None)
            if obj is None:
                continue
            globals()[name] = obj
            __all__.append(name)


_import_optional_agents()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cast(Type[Agent], cls)
    for cls in Agent.__subclasses__()
    if cls.__name__ != "Playback"
}

for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback

if "ReasoningAgent" in globals():
    AVAILABLE_AGENTS["reasoningagent"] = cast(
        Type[Agent], globals()["ReasoningAgent"]
    )
