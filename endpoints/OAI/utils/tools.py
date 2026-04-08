import json
import re
from loguru import logger
from typing import List

from endpoints.OAI.types.tools import ToolCall


TOOL_CALL_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "function": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {
                        # Converted to OAI's string in post process
                        "type": "object"
                    },
                },
                "required": ["name", "arguments"],
            },
        },
        "required": ["function"],
    },
}


class ToolCallProcessor:
    @staticmethod
    def from_json(tool_calls_str: str) -> List[ToolCall]:
        """Postprocess tool call JSON to a parseable class"""

        tool_calls = json.loads(tool_calls_str)
        for idx, tool_call in enumerate(tool_calls):
            tool_call["function"]["arguments"] = json.dumps(
                tool_call["function"]["arguments"]
            )
            tool_call["index"] = idx

        return [ToolCall(**tool_call) for tool_call in tool_calls]


    @staticmethod
    def from_qwen3_xml(
        raw_text: str,
        tool_start: str = "<tool_call>",
        tool_end: str = "</tool_call>",
    ) -> List[ToolCall]:
        """Parse Qwen3-style XML tool calls into ToolCall objects.

        Expected format per call:
        <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
        """

        text = raw_text.strip()
        start_re = re.escape(tool_start)
        end_re = re.escape(tool_end)

        pattern = rf"{start_re}(.*?){end_re}"
        matches = re.findall(pattern, text, re.DOTALL)

        if not matches:
            text = tool_start + raw_text
            matches = re.findall(pattern, text, re.DOTALL)

        tool_calls = []
        for idx, match in enumerate(matches):
            match = match.strip()
            
            # Extract function name
            func_match = re.search(r"<function=(\w+)>", match)
            if not func_match:
                logger.warning(f"Could not parse function name from: {match}")
                continue
            func_name = func_match.group(1)

            # Extract parameters
            args = {}
            param_pattern = r"<parameter=(\w+)>((?:(?!</parameter>).)*)"
            for key, value in re.findall(param_pattern, match, re.DOTALL):
                value = value.strip()
                try:
                    args[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    args[key] = value

            tool_calls.append(
                ToolCall(
                    index=idx,
                    function={"name": func_name, "arguments": json.dumps(args)},
                )
            )

        if not tool_calls:
            logger.warning(f"No tool calls parsed from Qwen3 XML output: {raw_text}")

        return tool_calls

    @staticmethod
    def from_qwen3(
        raw_text: str,
        tool_start: str = "<|tool_start|>",
        tool_end: str = "<|tool_end|>",
    ) -> List[ToolCall]:
        """Parse Qwen3-style tool calls into ToolCall objects.

        Expected format:
        <|tool_start|>[{"name": "func", "arguments": {...}}]<|tool_end|>
        """

        text = raw_text.strip()

        start_re = re.escape(tool_start)
        end_re = re.escape(tool_end)

        pattern = rf"{start_re}\s*(.*?)\s*{end_re}"
        matches = re.findall(pattern, text, re.DOTALL)

        if not matches:
            pattern = rf"{start_re}\s*(.*)"
            matches = re.findall(pattern, text, re.DOTALL)
            if matches and tool_end in text:
                matches[0] = matches[0].rsplit(tool_end, 1)[0]

        tool_calls = []
        for idx, match in enumerate(matches):
            match = match.strip()
            if not match:
                continue

            try:
                parsed = json.loads(match)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse Qwen3 tool call JSON: {match}")
                continue

            if isinstance(parsed, dict):
                parsed = [parsed]
            elif not isinstance(parsed, list):
                logger.warning(f"Unexpected Qwen3 tool call format: {match}")
                continue

            for item in parsed:
                func_name = item.get("name", "")
                arguments = item.get("arguments", {})

                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass

                tool_calls.append(
                    ToolCall(
                        index=len(tool_calls),
                        function={
                            "name": func_name,
                            "arguments": json.dumps(arguments),
                        },
                    )
                )

        if not tool_calls:
            logger.warning(f"No tool calls parsed from Qwen3 output: {raw_text}")

        return tool_calls

    @staticmethod
    def from_native_xml(
        raw_text: str,
        tool_start: str = "<tool_call>",
        tool_end: str = "</tool_call>",
    ) -> List[ToolCall]:
        """Parse native XML-style tool calls (e.g. GLM-4 format) into ToolCall objects.

        Expected format per call:
        <tool_call>func_name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>
        """

        # Wrap raw_text so regex can match consistently
        text = tool_start + raw_text

        # Escape markers for regex
        start_re = re.escape(tool_start)
        end_re = re.escape(tool_end)

        pattern = rf"{start_re}(.*?){end_re}"
        matches = re.findall(pattern, text, re.DOTALL)

        tool_calls = []
        for idx, match in enumerate(matches):
            # Function name is everything before the first <arg_key>
            name_match = re.match(r"([^<]+)", match.strip())
            if not name_match:
                logger.warning(f"Could not parse tool call function name from: {match}")
                continue
            func_name = name_match.group(1).strip()

            # Extract key-value pairs
            args = {}
            kv_pattern = r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>"
            for key, value in re.findall(kv_pattern, match, re.DOTALL):
                key = key.strip()
                value = value.strip()
                try:
                    args[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    args[key] = value

            tool_calls.append(
                ToolCall(
                    index=idx,
                    function={"name": func_name, "arguments": json.dumps(args)},
                )
            )

        if not tool_calls:
            logger.warning(f"No tool calls parsed from native XML output: {raw_text}")

        return tool_calls

    @staticmethod
    def dump(tool_calls: List[ToolCall]) -> List[dict]:
        """
        Convert ToolCall objects to a list of dictionaries.

        Args:
            tool_calls (List[ToolCall]): List of ToolCall objects to convert

        Returns:
            List[dict]: List of dictionaries representing the tool calls
        """

        # Don't use list comprehension here
        # as that will fail rather than warn
        dumped_tool_calls = []
        for tool_call_obj in tool_calls:
            try:
                dumped_tool_calls.append(tool_call_obj.model_dump())
            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning(f"Error processing tool call: {e}")
        return dumped_tool_calls

    @staticmethod
    def to_json(tool_calls: List[ToolCall]) -> str:
        """
        Convert ToolCall objects to JSON string representation.

        Args:
            tool_calls (List[ToolCall]): List of ToolCall objects to convert

        Returns:
            str: JSON representation of the tool calls
        """

        if not tool_calls:
            return ""

        # Use the dump method to get the list of dictionaries
        dumped_tool_calls = ToolCallProcessor.dump(tool_calls)

        # Serialize the dumped array
        return json.dumps(dumped_tool_calls, indent=2)
