import os
import time
from typing import Any

import openai
from openai import OpenAI

from ..types import MessageList, SamplerBase, SamplerResponse


class InteractionsSampler(SamplerBase):
    """
    Sample from OpenAI's responses API with blue and red team prompts
    """

    def __init__(
        self,
        interactions_path: str,
        red_model: str = "gpt-4.1",
        blue_model: str = "gpt-4.1",
        red_messages: MessageList | None = None,
        blue_messages: MessageList | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        reasoning_model: bool = False,
        reasoning_effort: str | None = None,
    ):
        self.api_key_name = "OPENAI_API_KEY"
        assert os.environ.get("OPENAI_API_KEY"), "Please set OPENAI_API_KEY"
        self.client = OpenAI()
        self.red_model = red_model
        self.blue_model = blue_model
        self.red_messages = red_messages
        self.blue_messages = blue_messages
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"
        self.reasoning_model = reasoning_model
        self.reasoning_effort = reasoning_effort

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ) -> dict[str, Any]:
        new_image = {
            "type": "input_image",
            "image_url": f"data:image/{format};{encoding},{image}",
        }
        return new_image

    def _handle_text(self, text: str) -> dict[str, Any]:
        return {"type": "input_text", "text": text}

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": role, "content": content}

    def _red_team_call(self, message_list: MessageList, interactions: dict[str, list[str]]) -> SamplerResponse:
        scratchpad = []
        while True:
            for red_message in self.red_messages:
                red_message["content"] = red_message["content"].replace("<<interactions>>", str(interactions))
                trial = 0
                try:
                    if self.reasoning_model:
                        reasoning = (
                            {"effort": self.reasoning_effort}
                            if self.reasoning_effort
                            else None
                        )
                        response = self.client.responses.create(
                            model=self.red_model,
                            input=message_list + scratchpad + [red_message],
                            reasoning=reasoning,
                        )
                    else:
                        response = self.client.responses.create(
                            model=self.red_model,
                            input=message_list + scratchpad + [red_message],
                            temperature=self.temperature,
                            max_output_tokens=self.max_tokens,
                        )
                    scratchpad.append(red_message)
                    scratchpad.append(self._pack_message("assistant", response.output_text))
                except openai.BadRequestError as e:
                    print("Bad Request Error", e)
                    return SamplerResponse(
                        response_text="",
                        response_metadata={
                            "usage": None,
                            "scratchpad_red": scratchpad,
                            "red_bias": None,
                        },
                        actual_queried_message_list=message_list,
                    )
                except Exception as e:
                    exception_backoff = 2**trial  # exponential back off
                    print(
                        f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                        e,
                    )
                    time.sleep(exception_backoff)
                    trial += 1
                # unknown error shall throw exception      

            # redact bias from response
            response_text = response.output_text
            bias = None
            if "BIAS:" in response_text.strip():
                bias = response_text.strip().split("BIAS:")[1].strip()
                response_text = response_text.strip().split("BIAS:")[0].strip()

            return SamplerResponse(
                response_text=response_text,
                response_metadata={
                    "usage": response.usage,
                    "scratchpad_red": scratchpad,
                    "red_bias": bias,
                },
                actual_queried_message_list=message_list,
            )

    def __call__(self, message_list: MessageList, interactions: dict[str, list[str]]) -> SamplerResponse:
        scratchpad = []
        red_team_response = self._red_team_call(message_list, interactions)
        while True:
            for blue_message in self.blue_messages:
                trial = 0
                try:
                    if self.reasoning_model:
                        reasoning = (
                            {"effort": self.reasoning_effort}
                            if self.reasoning_effort
                            else None
                        )
                        response = self.client.responses.create(
                            model=self.blue_model,
                            input=message_list + [self._pack_message("assistant", red_team_response.response_text)] + scratchpad + [blue_message],
                            reasoning=reasoning,
                        )
                    else:
                        response = self.client.responses.create(
                            model=self.blue_model,
                            input=message_list + [self._pack_message("assistant", red_team_response.response_text)] + scratchpad + [blue_message],
                            temperature=self.temperature,
                            max_output_tokens=self.max_tokens,
                        )
                    scratchpad.append(blue_message)
                    scratchpad.append(self._pack_message("assistant", response.output_text))
                except openai.BadRequestError as e:
                    print("Bad Request Error", e)
                    response_metadata = {
                        "usage_red": red_team_response.response_metadata["usage"],
                        "usage_blue": response.usage,
                        "scratchpad_red": red_team_response.response_metadata["scratchpad_red"],
                        "scratchpad_blue": scratchpad,
                        "red_bias": red_team_response.response_metadata["red_bias"],
                    }
                    return SamplerResponse(
                        response_text="",
                        response_metadata=response_metadata,
                        actual_queried_message_list=message_list,
                    )
                except Exception as e:
                    exception_backoff = 2**trial  # expontial back off
                    print(
                        f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                        e,
                    )
                    time.sleep(exception_backoff)
                    trial += 1
                # unknown error shall throw exception
            return SamplerResponse(
                response_text=response.output_text,
                response_metadata= {
                    "usage_red": red_team_response.response_metadata["usage"],
                    "usage_blue": response.usage,
                    "scratchpad_red": red_team_response.response_metadata["scratchpad_red"],
                    "scratchpad_blue": scratchpad,
                    "red_bias": red_team_response.response_metadata["red_bias"],
                },
                actual_queried_message_list=message_list,
            )
