import json
from textwrap import dedent

from chat2plot.schema import get_schema_of_chart_config

JSON_TAG = ["<json>", "</json>"]
EXPLANATION_TAG = ["<explain>", "</explain>"]


def system_prompt(model_type: str = "simple") -> str:
    return (
        _task_definition_part(model_type) + "\n" + _data_and_detailed_instruction_part()
    )


def error_correction_prompt() -> str:
    return dedent(
        """
        Your response fails with the following error:
        {error_message}

        Correct the json and return a new explanation and json that fixes the above mentioned error.
        Do not generate the same json again.
    """
    )


def _task_definition_part(model_type: str) -> str:
    if model_type == "simple":
        schema_json = json.dumps(
            get_schema_of_chart_config(inlining_refs=True, remove_title=True), indent=2
        )

        return (
            dedent(
                """
            Your task is to generate chart configuration for the given dataset and user question delimited by <>.
            Responses should be in JSON format compliant to the following JSON Schema.

            """
            )
            + schema_json.replace("{", "{{").replace("}", "}}")
        )

    else:
        return dedent(
            """
            Your task is to generate chart configuration for the given dataset and user question delimited by <>.
            Responses should be in JSON format compliant with the vega-lite specification,
            but `data` field must be excluded.
            """
        )


def _data_and_detailed_instruction_part() -> str:
    return dedent(
        f"""
        Note that the user may want to refine the chart by asking a follow-up question to a previous request,
        or may want to create a new chart in a completely new context.
        In the latter case, be careful not to use the context used for the previous chart.

        This is the result of `print(df.head())`:

        {{dataset}}

        You should do the following step by step, and your response should include both 1 and 2:
        1. Explain whether filters should be applied to the data, which chart_type and columns should be used,
           and what transformations are necessary to fulfill the user's request.
           Answers should be in the same language as the user and be understandable to someone who does not know
           the JSON schema definition.
           This text should be enclosed with {EXPLANATION_TAG[0]} and {EXPLANATION_TAG[1]} tag.
        2. Generate schema-compliant JSON that represents 1.
           This text should be enclosed with {JSON_TAG[0]} and {JSON_TAG[1]} tag.

        """
    )
