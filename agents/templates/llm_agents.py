from __future__ import annotations

import json
import logging
import os
import re
import textwrap
from typing import Any, ClassVar, Optional

import openai
from arcengine import FrameData, GameAction, GameState
from openai import OpenAI as OpenAIClient

from ..agent import Agent

logger = logging.getLogger()
OFFLINE_MODE = os.environ.get("OPERATION_MODE", "").strip().lower() == "local"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8000/v1"


class LLM(Agent):
    """An agent that uses a base LLM model to play games."""

    MAX_ACTIONS: int = 80
    DO_OBSERVATION: bool = True
    REASONING_EFFORT: Optional[str] = None
    MODEL_REQUIRES_TOOLS: bool = False

    MESSAGE_LIMIT: int = 10
    MODEL: str = "gpt-4o-mini"
    messages: list[dict[str, Any]]
    token_counter: int

    _latest_tool_call_id: str = "call_12345"
    _JSON_BLOB = re.compile(r"\{.*\}", re.DOTALL)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.messages = []
        self.token_counter = 0
        self.base_url = (
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("LOCAL_LLM_BASE_URL")
            or (DEFAULT_LOCAL_BASE_URL if OFFLINE_MODE else None)
        )
        self.api_key = (
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("LOCAL_LLM_API_KEY")
            or ("local-dev" if self.base_url else "")
        )
        self.runtime_model = (
            os.environ.get("LOCAL_LLM_MODEL") if OFFLINE_MODE else None
        ) or os.environ.get("OPENAI_MODEL_OVERRIDE") or self.MODEL
        self.api_style = self._resolve_api_style()
        self._supports_reasoning_effort = self._env_flag(
            "LLM_SUPPORTS_REASONING_EFFORT", default=not self.uses_local_backend
        )
        self._max_frame_chars = int(os.environ.get("LLM_MAX_FRAME_CHARS", "4000"))

    @property
    def name(self) -> str:
        obs = "with-observe" if self.DO_OBSERVATION else "no-observe"
        sanitized_model_name = self.runtime_model.replace("/", "-").replace(":", "-")
        name = f"{super().name}.{sanitized_model_name}.{obs}"
        if self.REASONING_EFFORT:
            name += f".{self.REASONING_EFFORT}"
        if self.uses_local_backend:
            name += ".local"
        return name

    @property
    def uses_local_backend(self) -> bool:
        return bool(self.base_url)

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return any(
            [
                latest_frame.state is GameState.WIN,
                # uncomment below to only let the agent play one time
                # latest_frame.state is GameState.GAME_OVER,
            ]
        )

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose which action the Agent should take, fill in any arguments, and return it."""

        logging.getLogger("openai").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)

        client = self.build_client()

        if self.api_style == "json":
            return self.choose_action_json(client, frames, latest_frame)

        functions = self.build_functions()
        tools = self.build_tools()

        # if latest_frame.state in [GameState.NOT_PLAYED]:
        if len(self.messages) == 0:
            # have to manually trigger the first reset to kick off agent
            user_prompt = self.build_user_prompt(latest_frame)
            message0 = {"role": "user", "content": user_prompt}
            self.push_message(message0)
            if self.MODEL_REQUIRES_TOOLS:
                message1 = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": self._latest_tool_call_id,
                            "type": "function",
                            "function": {
                                "name": GameAction.RESET.name,
                                "arguments": json.dumps({}),
                            },
                        }
                    ],
                }
            else:
                message1 = {
                    "role": "assistant",
                    "function_call": {"name": "RESET", "arguments": json.dumps({})},  # type: ignore
                }
            self.push_message(message1)
            action = GameAction.RESET
            return action

        # let the agent comment observations before choosing action
        # on the first turn, this will be in response to RESET action
        function_name = latest_frame.action_input.id.name
        function_response = self.build_func_resp_prompt(latest_frame)
        if self.MODEL_REQUIRES_TOOLS:
            message2 = {
                "role": "tool",
                "tool_call_id": self._latest_tool_call_id,
                "content": str(function_response),
            }
        else:
            message2 = {
                "role": "function",
                "name": function_name,
                "content": str(function_response),
            }
        self.push_message(message2)

        if self.DO_OBSERVATION:
            logger.info("Sending to Assistant for observation...")
            try:
                create_kwargs = {
                    "model": self.runtime_model,
                    "messages": self.messages,
                }
                if (
                    self.REASONING_EFFORT is not None
                    and self._supports_reasoning_effort
                ):
                    create_kwargs["reasoning_effort"] = self.REASONING_EFFORT
                response = client.chat.completions.create(**create_kwargs)
            except openai.BadRequestError as e:
                logger.info(f"Message dump: {self.messages}")
                raise e
            self.track_tokens(
                response.usage.total_tokens, response.choices[0].message.content
            )
            message3 = {
                "role": "assistant",
                "content": response.choices[0].message.content,
            }
            logger.info(f"Assistant: {response.choices[0].message.content}")
            self.push_message(message3)

        # now ask for the next action
        user_prompt = self.build_user_prompt(latest_frame)
        message4 = {"role": "user", "content": user_prompt}
        self.push_message(message4)

        name = GameAction.ACTION5.name  # default action if LLM doesnt call one
        arguments = None
        message5 = None

        if self.MODEL_REQUIRES_TOOLS:
            logger.info("Sending to Assistant for action...")
            try:
                create_kwargs = {
                    "model": self.runtime_model,
                    "messages": self.messages,
                    "tools": tools,
                    "tool_choice": "required",
                }
                if (
                    self.REASONING_EFFORT is not None
                    and self._supports_reasoning_effort
                ):
                    create_kwargs["reasoning_effort"] = self.REASONING_EFFORT
                response = client.chat.completions.create(**create_kwargs)
            except openai.BadRequestError as e:
                logger.info(f"Message dump: {self.messages}")
                raise e
            self.track_tokens(response.usage.total_tokens)
            message5 = response.choices[0].message
            logger.debug(f"... got response {message5}")
            tool_call = message5.tool_calls[0]
            self._latest_tool_call_id = tool_call.id
            logger.debug(
                f"Assistant: {tool_call.function.name} ({tool_call.id}) {tool_call.function.arguments}"
            )
            name = tool_call.function.name
            arguments = tool_call.function.arguments

            # sometimes the model will call multiple tools which isnt allowed
            extra_tools = message5.tool_calls[1:]
            for tc in extra_tools:
                logger.info(
                    "Error: assistant called more than one action, only using the first."
                )
                message_extra = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "Error: assistant can only call one action (tool) at a time. default to only the first chosen action.",
                }
                self.push_message(message_extra)
        else:
            logger.info("Sending to Assistant for action...")
            try:
                create_kwargs = {
                    "model": self.runtime_model,
                    "messages": self.messages,
                    "functions": functions,
                    "function_call": "auto",
                }
                if (
                    self.REASONING_EFFORT is not None
                    and self._supports_reasoning_effort
                ):
                    create_kwargs["reasoning_effort"] = self.REASONING_EFFORT
                response = client.chat.completions.create(**create_kwargs)
            except openai.BadRequestError as e:
                logger.info(f"Message dump: {self.messages}")
                raise e
            self.track_tokens(response.usage.total_tokens)
            message5 = response.choices[0].message
            function_call = message5.function_call
            logger.debug(f"Assistant: {function_call.name} {function_call.arguments}")
            name = function_call.name
            arguments = function_call.arguments

        if message5:
            self.push_message(message5)
        action_id = name
        if arguments:
            try:
                data = json.loads(arguments) or {}
            except Exception as e:
                data = {}
                logger.warning(f"JSON parsing error on LLM function response: {e}")
        else:
            data = {}

        action = GameAction.from_name(action_id)
        action.set_data(data)
        return action

    def build_client(self) -> OpenAIClient:
        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        return OpenAIClient(**client_kwargs)

    def choose_action_json(
        self, client: OpenAIClient, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            return GameAction.RESET

        if self.DO_OBSERVATION:
            observation_prompt = self.build_json_observation_prompt(latest_frame)
            self.push_message({"role": "user", "content": observation_prompt})
            logger.info("Sending to Assistant for observation...")
            observation_response = client.chat.completions.create(
                model=self.runtime_model,
                messages=self.messages,
            )
            observation_text = observation_response.choices[0].message.content or ""
            self.track_tokens(observation_response.usage.total_tokens, observation_text)
            self.push_message({"role": "assistant", "content": observation_text})

        prompt = self.build_json_action_prompt(latest_frame)
        self.push_message({"role": "user", "content": prompt})
        logger.info("Sending to Assistant for JSON action...")
        response = client.chat.completions.create(
            model=self.runtime_model,
            messages=self.messages,
        )
        message = response.choices[0].message
        content = message.content or ""
        self.track_tokens(response.usage.total_tokens, content)
        self.push_message({"role": "assistant", "content": content})

        blob = self.parse_action_blob(content)
        action = self.action_from_blob(blob)
        reasoning = self.reasoning_from_blob(blob)
        if reasoning is not None:
            action.reasoning = reasoning
        return action

    def track_tokens(self, tokens: int, message: str = "") -> None:
        self.token_counter += tokens
        if hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record(
                {
                    "tokens": tokens,
                    "total_tokens": self.token_counter,
                    "assistant": message,
                }
            )
        logger.info(f"Received {tokens} tokens, new total {self.token_counter}")
        # handle tool to debug messages:
        # with open("messages.json", "w") as f:
        #     json.dump(
        #         [
        #             msg if isinstance(msg, dict) else msg.model_dump()
        #             for msg in self.messages
        #         ],
        #         f,
        #         indent=2,
        #     )

    def push_message(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """Push a message onto stack, store up to MESSAGE_LIMIT with FIFO."""
        self.messages.append(message)
        if len(self.messages) > self.MESSAGE_LIMIT:
            self.messages = self.messages[-self.MESSAGE_LIMIT :]
        if self.MODEL_REQUIRES_TOOLS:
            # cant clip the message list between tool
            # and tool_call else llm will error
            while (
                self.messages[0].get("role")
                if isinstance(self.messages[0], dict)
                else getattr(self.messages[0], "role", None)
            ) == "tool":
                self.messages.pop(0)
        return self.messages

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _resolve_api_style(self) -> str:
        style = os.environ.get("LLM_API_STYLE", "auto").strip().lower()
        if style not in {"auto", "tools", "functions", "json"}:
            logger.warning(f"Unknown LLM_API_STYLE={style!r}; falling back to auto")
            style = "auto"

        if style == "auto":
            if self.uses_local_backend:
                return "json"
        if self.MODEL_REQUIRES_TOOLS and not self.uses_local_backend:
            return "tools"
        return "functions"
        return style

    def parse_action_blob(self, content: str) -> Optional[dict[str, Any]]:
        text = content.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        match = self._JSON_BLOB.search(text)
        if match is None:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def action_from_blob(self, blob: Optional[dict[str, Any]]) -> GameAction:
        if not isinstance(blob, dict) or "action" not in blob:
            logger.warning(
                "LLM reply did not parse to action JSON; falling back to ACTION5."
            )
            return GameAction.ACTION5

        raw = str(blob.get("action", "")).upper().strip()
        try:
            action = GameAction.from_name(raw)
        except (KeyError, ValueError, AttributeError):
            logger.warning(f"Unknown action {raw!r}; falling back to ACTION5")
            return GameAction.ACTION5

        if action.is_complex():
            try:
                action.set_data(
                    {"x": int(blob.get("x", 32)), "y": int(blob.get("y", 32))}
                )
            except (TypeError, ValueError):
                action.set_data({"x": 32, "y": 32})
        else:
            action.set_data({})
        return action

    def reasoning_from_blob(self, blob: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not isinstance(blob, dict):
            return None
        reasoning = blob.get("reasoning")
        if reasoning is None:
            return None
        if isinstance(reasoning, dict):
            return reasoning
        return {"text": str(reasoning)}

    def build_functions(self) -> list[dict[str, Any]]:
        """Build JSON function description of game actions for LLM."""
        empty_params: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
        functions: list[dict[str, Any]] = [
            {
                "name": GameAction.RESET.name,
                "description": "Start or restart a game. Must be called first when NOT_PLAYED or after GAME_OVER to play again.",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION1.name,
                "description": "Send this simple input action (1, W, Up).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION2.name,
                "description": "Send this simple input action (2, S, Down).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION3.name,
                "description": "Send this simple input action (3, A, Left).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION4.name,
                "description": "Send this simple input action (4, D, Right).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION5.name,
                "description": "Send this simple input action (5, Enter, Spacebar, Delete).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION6.name,
                "description": "Send this complex input action (6, Click, Point).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "string",
                            "description": "Coordinate X which must be Int<0,63>",
                        },
                        "y": {
                            "type": "string",
                            "description": "Coordinate Y which must be Int<0,63>",
                        },
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False,
                },
            },
        ]
        return functions

    def build_tools(self) -> list[dict[str, Any]]:
        """Support models that expect tool_call format."""
        functions = self.build_functions()
        tools: list[dict[str, Any]] = []
        for f in functions:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f["name"],
                        "description": f["description"],
                        "parameters": f.get("parameters", {}),
                        "strict": True,
                    },
                }
            )
        return tools

    def build_func_resp_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(
            """
# State:
{state}

# Score:
{score}

# Frame:
{latest_frame}

# TURN:
Reply with a few sentences of plain-text strategy observation about the frame to inform your next action.
        """.format(
                latest_frame=self.pretty_print_3d(latest_frame.frame),
                score=latest_frame.levels_completed,
                state=latest_frame.state.name,
            )
        )

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        """Build the user prompt for the LLM. Override this method to customize the prompt."""
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

# TURN:
Call exactly one action.
        """.format()
        )

    def build_json_observation_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(
            """
        # CONTEXT:
        You are analyzing the latest state of a turn-based grid game.

        # State:
        {state}

        # Score:
        {score}

        # Frame:
        {latest_frame}

        # TURN:
        Briefly summarize what changed, what seems important, and what to test next.
        Keep it under 6 short bullet points.
        """.format(
                latest_frame=self.pretty_print_3d(latest_frame.frame),
                score=latest_frame.levels_completed,
                state=latest_frame.state.name,
            )
        )

    def build_json_action_prompt(self, latest_frame: FrameData) -> str:
        available_actions = ", ".join(a.name for a in latest_frame.available_actions)
        return textwrap.dedent(
            """
        # CONTEXT:
        You are an agent playing a dynamic game. Your objective is to WIN and avoid GAME_OVER.
        Choose exactly one next action.

        # State:
        {state}

        # Score:
        {score}

        # Available Actions:
        {available_actions}

        # Frame:
        {latest_frame}

        # RESPONSE FORMAT:
        Return exactly one JSON object and nothing else.
        For simple actions:
        {{"action":"ACTION1","reasoning":"short reason"}}
        For click actions:
        {{"action":"ACTION6","x":12,"y":34,"reasoning":"short reason"}}
        """.format(
                latest_frame=self.pretty_print_3d(latest_frame.frame),
                score=latest_frame.levels_completed,
                state=latest_frame.state.name,
                available_actions=available_actions or "RESET",
            )
        )

    def pretty_print_3d(self, array_3d: list[list[list[Any]]]) -> str:
        lines = []
        for i, block in enumerate(array_3d):
            lines.append(f"Grid {i}:")
            for row in block:
                lines.append(f"  {row}")
            lines.append("")
        rendered = "\n".join(lines)
        if len(rendered) > self._max_frame_chars:
            return rendered[: self._max_frame_chars] + "\n...<truncated>"
        return rendered

    def cleanup(self, *args: Any, **kwargs: Any) -> None:
        if self._cleanup:
            if hasattr(self, "recorder") and not self.is_playback:
                meta = {
                    "llm_user_prompt": self.build_user_prompt(self.frames[-1]),
                    "llm_tools": self.build_tools()
                    if self.MODEL_REQUIRES_TOOLS
                    else self.build_functions(),
                    "llm_tool_resp_prompt": self.build_func_resp_prompt(
                        self.frames[-1]
                    ),
                }
                self.recorder.record(meta)
        super().cleanup(*args, **kwargs)


class ReasoningLLM(LLM, Agent):
    """An LLM agent that uses o4-mini and captures reasoning metadata in the action.reasoning field."""

    MAX_ACTIONS = 80
    DO_OBSERVATION = True
    MODEL_REQUIRES_TOOLS = True
    MODEL = "o4-mini"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_reasoning_tokens = 0
        self._last_response_content = ""
        self._total_reasoning_tokens = 0

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Override choose_action to capture and store reasoning metadata."""

        action = super().choose_action(frames, latest_frame)

        # Store reasoning metadata in the action.reasoning field
        action.reasoning = {
            "model": self.runtime_model,
            "action_chosen": action.name,
            "reasoning_tokens": self._last_reasoning_tokens,
            "total_reasoning_tokens": self._total_reasoning_tokens,
            "game_context": {
                "score": latest_frame.levels_completed,
                "state": latest_frame.state.name,
                "action_counter": self.action_counter,
                "frame_count": len(frames),
            },
            "response_preview": self._last_response_content[:200] + "..."
            if len(self._last_response_content) > 200
            else self._last_response_content,
        }

        return action

    def track_tokens(self, tokens: int, message: str = "") -> None:
        """Override to capture reasoning token information from reasoning models."""
        super().track_tokens(tokens, message)

        # Store the response content for reasoning context (avoid empty or JSON strings)
        if message and not message.startswith("{"):
            self._last_response_content = message
        self._last_reasoning_tokens = tokens
        self._total_reasoning_tokens += tokens

    def capture_reasoning_from_response(self, response: Any) -> None:
        """Helper method to capture reasoning tokens from OpenAI API response.

        This should be called from the parent class if we have access to the raw response.
        For reasoning models, reasoning tokens are in response.usage.completion_tokens_details.reasoning_tokens
        """
        if hasattr(response, "usage") and hasattr(
            response.usage, "completion_tokens_details"
        ):
            if hasattr(response.usage.completion_tokens_details, "reasoning_tokens"):
                self._last_reasoning_tokens = (
                    response.usage.completion_tokens_details.reasoning_tokens
                )
                self._total_reasoning_tokens += self._last_reasoning_tokens
                logger.debug(
                    f"Captured {self._last_reasoning_tokens} reasoning tokens from {self.runtime_model} response"
                )


class FastLLM(LLM, Agent):
    """Similar to LLM, but skips observations."""

    MAX_ACTIONS = 80
    DO_OBSERVATION = False
    MODEL = "gpt-4o-mini"
    MESSAGE_LIMIT = 6

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

# TURN:
Call exactly one action.
        """.format()
        )


class GuidedLLM(LLM, Agent):
    """Similar to LLM, with explicit human-provided rules in the user prompt to increase success rate."""

    MAX_ACTIONS = 80
    DO_OBSERVATION = True
    MODEL = "o3"
    MODEL_REQUIRES_TOOLS = True
    MESSAGE_LIMIT = 10
    REASONING_EFFORT = "high"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_reasoning_tokens = 0
        self._last_response_content = ""
        self._total_reasoning_tokens = 0

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Override choose_action to capture and store reasoning metadata."""

        action = super().choose_action(frames, latest_frame)

        # Store reasoning metadata in the action.reasoning field
        action.reasoning = {
            "model": self.runtime_model,
            "action_chosen": action.name,
            "reasoning_effort": self.REASONING_EFFORT,
            "reasoning_tokens": self._last_reasoning_tokens,
            "total_reasoning_tokens": self._total_reasoning_tokens,
            "game_context": {
                "score": latest_frame.levels_completed,
                "state": latest_frame.state.name,
                "action_counter": self.action_counter,
                "frame_count": len(frames),
            },
            "agent_type": "guided_llm",
            "game_rules": "locksmith",
            "response_preview": self._last_response_content[:200] + "..."
            if len(self._last_response_content) > 200
            else self._last_response_content,
        }

        return action

    def track_tokens(self, tokens: int, message: str = "") -> None:
        """Override to capture reasoning token information from o3 models."""
        super().track_tokens(tokens, message)

        # Store the response content for reasoning context (avoid empty or JSON strings)
        if message and not message.startswith("{"):
            self._last_response_content = message
        self._last_reasoning_tokens = tokens
        self._total_reasoning_tokens += tokens

    def capture_reasoning_from_response(self, response: Any) -> None:
        """Helper method to capture reasoning tokens from OpenAI API response.

        This should be called from the parent class if we have access to the raw response.
        For o3 models, reasoning tokens are in response.usage.completion_tokens_details.reasoning_tokens
        """
        if hasattr(response, "usage") and hasattr(
            response.usage, "completion_tokens_details"
        ):
            if hasattr(response.usage.completion_tokens_details, "reasoning_tokens"):
                self._last_reasoning_tokens = (
                    response.usage.completion_tokens_details.reasoning_tokens
                )
                self._total_reasoning_tokens += self._last_reasoning_tokens
                logger.debug(
                    f"Captured {self._last_reasoning_tokens} reasoning tokens from o3 response"
                )

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

You are playing a game called LockSmith. Rules and strategy:
* RESET: start over, ACTION1: move up, ACTION2: move down, ACTION3: move left, ACTION4: move right (ACTION5 and ACTION6 do nothing in this game)
* you may may one action per turn
* your goal is find and collect a matching key then touch the exit door
* 6 levels total, score shows which level, complete all levels to win (grid row 62)
* start each level with limited energy. you GAME_OVER if you run out (grid row 61)
* the player is a 4x4 square: [[X,X,X,X],[0,0,0,X],[4,4,4,X],[4,4,4,X]] where X is transparent to the background
* the grid represents a birds-eye view of the level
* walls are made of INT<10>, you cannot move through a wall
* walkable floor area is INT<8>
* you can refill energy by touching energy pills (a 2x2 of INT<6>)
* current key is shown in bottom-left of entire grid
* the exit door is a 4x4 square with INT<11> border
* to find a new key shape, touch the key rotator, a 4x4 square denoted by INT<9> and INT<4> in the top-left corner of the square
* to find a new key color, touch the color rotator, a 4x4 square denoted by INT<9> and INT<2> and in the bottom-left corner of the square
* to rotate more than once, move 1 space away from the rotator and back on
* continue rotating the shape and color of the key until the key matches the one inside the exit door (scaled down 2X)
* if the grid does not change after an action, you probably tried to move into a wall

An example of a good strategy observation:
The player 4x4 made of INT<4> and INT<0> is standing below a wall of INT<10>, so I cannot move up anymore and should
move left towards the rotator with INT<11>.

# TURN:
Call exactly one action.
        """.format()
        )


class LocalLLM(LLM, Agent):
    """A low-memory local-LLM preset for offline OpenAI-compatible servers."""

    MAX_ACTIONS = 80
    DO_OBSERVATION = False
    MODEL = os.environ.get("LOCAL_LLM_MODEL", "local-model")
    MESSAGE_LIMIT = 6

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.api_style = "json"


class DirectLocalLLM(LLM, Agent):
    """Offline local model runner that calls Transformers directly."""

    MAX_ACTIONS = 80
    DO_OBSERVATION = False
    MODEL = os.environ.get("DIRECT_LLM_MODEL_PATH", "local-model")
    MESSAGE_LIMIT = 6

    _shared_model: ClassVar[Any] = None
    _shared_tokenizer: ClassVar[Any] = None
    _shared_model_path: ClassVar[Optional[str]] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        model_path = os.environ.get("DIRECT_LLM_MODEL_PATH") or os.environ.get(
            "LOCAL_LLM_MODEL_PATH"
        )
        if not model_path:
            raise ValueError(
                "DIRECT_LLM_MODEL_PATH must be set for DirectLocalLLM."
            )
        self.model_path = model_path
        self.runtime_model = self.model_path
        super().__init__(*args, **kwargs)
        self._max_new_tokens = int(os.environ.get("DIRECT_LLM_MAX_NEW_TOKENS", "160"))
        self._temperature = float(os.environ.get("DIRECT_LLM_TEMPERATURE", "0.0"))
        self._do_sample = self._temperature > 0
        self._trust_remote_code = self._env_flag(
            "DIRECT_LLM_TRUST_REMOTE_CODE", default=False
        )
        self._load_in_4bit = self._env_flag("DIRECT_LLM_LOAD_IN_4BIT", default=False)
        self._device_map = os.environ.get("DIRECT_LLM_DEVICE_MAP", "auto")
        self._dtype = os.environ.get("DIRECT_LLM_DTYPE", "auto")
        self._attn_implementation = os.environ.get("DIRECT_LLM_ATTENTION", "")
        self._ensure_local_model()

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            return GameAction.RESET

        if self.DO_OBSERVATION:
            observation_prompt = self.build_json_observation_prompt(latest_frame)
            self.push_message({"role": "user", "content": observation_prompt})
            observation_text = self._generate_text(self.messages)
            self.track_tokens(0, observation_text)
            self.push_message({"role": "assistant", "content": observation_text})

        prompt = self.build_json_action_prompt(latest_frame)
        self.push_message({"role": "user", "content": prompt})
        content = self._generate_text(self.messages)
        self.track_tokens(0, content)
        self.push_message({"role": "assistant", "content": content})

        blob = self.parse_action_blob(content)
        action = self.action_from_blob(blob)
        reasoning = self.reasoning_from_blob(blob)
        if reasoning is not None:
            action.reasoning = reasoning
        return action

    def _ensure_local_model(self) -> None:
        if (
            DirectLocalLLM._shared_model is not None
            and DirectLocalLLM._shared_tokenizer is not None
            and DirectLocalLLM._shared_model_path == self.model_path
        ):
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self._trust_remote_code,
            "device_map": self._device_map,
        }
        if self._dtype != "auto":
            import torch

            model_kwargs["torch_dtype"] = getattr(torch, self._dtype)
        else:
            model_kwargs["torch_dtype"] = "auto"
        if self._load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        if self._attn_implementation:
            model_kwargs["attn_implementation"] = self._attn_implementation

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=self._trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        DirectLocalLLM._shared_tokenizer = tokenizer
        DirectLocalLLM._shared_model = model
        DirectLocalLLM._shared_model_path = self.model_path

    @property
    def tokenizer(self) -> Any:
        return DirectLocalLLM._shared_tokenizer

    @property
    def model(self) -> Any:
        return DirectLocalLLM._shared_model

    def _generate_text(self, messages: list[dict[str, Any]]) -> str:
        tokenizer = self.tokenizer
        model = self.model
        prompt_text = self._render_messages(messages)

        if hasattr(tokenizer, "apply_chat_template"):
            try:
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    tokenize=True,
                )
            except Exception:
                inputs = tokenizer(prompt_text, return_tensors="pt")
        else:
            inputs = tokenizer(prompt_text, return_tensors="pt")

        device = getattr(model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)

        attention_mask = None
        input_ids = inputs
        if isinstance(inputs, dict):
            attention_mask = inputs.get("attention_mask")
            input_ids = inputs["input_ids"]

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": self._do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if self._do_sample:
            generate_kwargs["temperature"] = self._temperature
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask

        outputs = model.generate(input_ids=input_ids, **generate_kwargs)
        generated_ids = outputs[0][input_ids.shape[-1] :]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return text

    def _render_messages(self, messages: list[dict[str, Any]]) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            parts.append(f"[{role}]\n{content}")
        parts.append("[ASSISTANT]")
        return "\n\n".join(parts)


# Example of a custom LLM agent
class MyCustomLLM(LLM):
    """Template for creating your own custom LLM agent."""

    MAX_ACTIONS = 80
    MODEL = "gpt-4o-mini"
    DO_OBSERVATION = True

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        """Customize this method to provide instructions to the LLM."""
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

# CUSTOM INSTRUCTIONS:
Add your game instructions and strategy here.
For example, explain the game rules, objectives, and optimal strategies.

# TURN:
Call exactly one action.
        """.format()
        )
