import json
import re
from enum import Enum
from typing import Any, Type

import pydantic


class SchemaWithoutTitle:
    @staticmethod
    def schema_extra(schema: dict[str, Any], _: Type[Any]) -> None:
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)


class ChartType(str, Enum):
    PIE = "pie"
    SCATTER = "scatter"
    LINE = "line"
    BAR = "bar"
    AREA = "area"
    HORIZONTAL_BAR = "horizontal-bar"


class AggregationType(str, Enum):
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    COUNT = "COUNT"
    DISTINCT_COUNT = "DISTINCT_COUNT"


class ResponseType(str, Enum):
    SUCCESS = "success"
    NOT_RELATED = "not related"
    UNKNOWN = "unknown"
    FAILED_TO_RENDER = "failed to render"


class SortingCriteria(str, Enum):
    NAME = "name"
    VALUE = "value"


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


class Field(pydantic.BaseModel):
    column: str = pydantic.Field(None, description="column name of the dataset")
    aggregation: AggregationType | None = pydantic.Field(
        None, description="Type of aggregation. will be ignored when it is scatter plot"
    )

    def name(self) -> str:
        return (
            f"{self.aggregation.value}({self.column})"
            if self.aggregation
            else self.column
        )

    @classmethod
    def parse_from_llm(cls, d: dict[str, str]) -> "Field":
        return Field(
            column=d["column"],
            aggregation=AggregationType(d["aggregation"].upper())
            if d.get("aggregation")
            else None,
        )

    class Config(SchemaWithoutTitle):
        pass


class Filter(pydantic.BaseModel):
    lhs: str
    rhs: str
    op: str

    class Config(SchemaWithoutTitle):
        pass

    def escaped(self) -> str:
        lhs = f"`{self.lhs}`" if self.lhs[0] != "`" else self.lhs
        return f"{lhs} {self.op} {self.rhs}"

    @classmethod
    def parse_from_llm(cls, f: str) -> "Filter":
        f = f.strip()
        if f[0] == "(" and f[-1] == ")":
            f = f[1:-1]  # strip parenthesis

        supported_operators = ["==", "!=", ">=", "<=", ">", "<"]
        for op in supported_operators:
            m = re.match(rf"^(.*){op}(.*?)$", f)
            if m:
                lhs = m.group(1).strip()
                rhs = m.group(2).strip()
                return Filter(lhs=lhs, rhs=rhs, op=op)

        raise ValueError(f"Unsupported op or failed to parse: {f}")


class Axis(pydantic.BaseModel):
    field: Field
    min_value: float | None
    max_value: float | None
    label: str | None

    class Config(SchemaWithoutTitle):
        pass

    @classmethod
    def parse_from_llm(cls, d: dict[str, str | float | dict[str, str]]) -> "Axis":
        return Axis(
            field=Field.parse_from_llm(d["field"]),  # type: ignore
            min_value=d.get("min_value"),  # type: ignore
            max_value=d.get("max_value"),  # type: ignore
            label=d.get("label"),  # type: ignore
        )


class PlotConfig(pydantic.BaseModel):
    chart_type: ChartType = pydantic.Field(None, description="the type of the chart")
    x: Axis | None = pydantic.Field(
        None, description="X-axis for the chart, or label column for pie chart"
    )
    y: Axis = pydantic.Field(
        None,
        description="Y-axis or measure value for the chart, or the wedge sizes for pie chart.",
    )
    filters: list[str] = pydantic.Field(
        None,
        description='List of filter conditions, where each filter must be a legal string that can be passed to df.query(), such as "x >= 0".',
    )
    hue: Field | None = pydantic.Field(
        None,
        description="Dimension used as grouping variables that will produce different colors.",
    )
    sort_criteria: SortingCriteria | None = pydantic.Field(
        None, description="The sorting criteria for x-axis"
    )
    sort_order: SortOrder | None = pydantic.Field(
        None, description="Sorting order for x-axis"
    )

    class Config(SchemaWithoutTitle):
        pass

    @property
    def required_columns(self) -> list[str]:
        columns = [self.y.field.column]
        if self.x:
            columns.append(self.x.field.column)
        return columns

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "PlotConfig":
        assert "chart_type" in json_data
        assert "y" in json_data

        def wrap_if_not_list(value: str | list[str]) -> list[str]:
            if not isinstance(value, list):
                return [] if not value else [value]
            else:
                return value

        chart_type = ChartType(json_data["chart_type"])

        return cls(
            chart_type=chart_type,
            x=Axis.parse_from_llm(json_data["x"]) if json_data.get("x") else None,
            y=Axis.parse_from_llm(json_data["y"]),
            filters=wrap_if_not_list(json_data.get("filters", [])),
            hue=Field.parse_from_llm(json_data["hue"])
            if json_data.get("hue")
            else None,
            sort_criteria=SortingCriteria(json_data["sort_criteria"])
            if json_data.get("sort_criteria")
            else None,
            sort_order=SortOrder(json_data["sort_order"])
            if json_data.get("sort_order")
            else None,
        )


class LLMResponse(pydantic.BaseModel):
    response_type: ResponseType
    config: PlotConfig | None


def inlining_refs_in_schema(
    schema: dict[str, Any], MAX_TRIES: int = 100, not_inlining: list[str] | None = None
) -> dict[str, Any]:
    not_inlining = not_inlining or []
    assert not_inlining is not None

    def replace_value_in_dict(item: Any, original_schema: dict[str, Any]) -> Any:
        if isinstance(item, list):
            return [replace_value_in_dict(i, original_schema) for i in item]
        elif isinstance(item, dict):
            if list(item.keys()) == ["$ref"]:
                definitions = item["$ref"][2:].split("/")
                if definitions[-1] in not_inlining:  # type: ignore
                    return item
                res = original_schema.copy()
                for definition in definitions:
                    res = res[definition]
                return res
            else:
                return {
                    key: replace_value_in_dict(i, original_schema)
                    for key, i in item.items()
                }
        else:
            return item

    for i in range(MAX_TRIES):
        if "$ref" not in json.dumps(schema):
            break
        schema = replace_value_in_dict(schema.copy(), schema.copy())

    defs = list(schema["definitions"].keys())
    for key in defs:
        if key not in not_inlining:
            del schema["definitions"][key]

    if not schema["definitions"]:
        del schema["definitions"]

    return schema


def get_schema_of_chart_config(inlining_refs: bool = True) -> dict[str, Any]:
    schema = PlotConfig.schema()
    if inlining_refs:
        schema = inlining_refs_in_schema(schema)

    return schema
