import re
from enum import Enum
from typing import Any, Type

import jsonref
import pydantic

from chat2plot.dictionary_helper import remove_field_recursively, flatten_single_element_allof


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


class TimeUnit(str, Enum):
    YEAR= "year"
    MONTH = "month"
    WEEK = "week"
    QUARTER = "quarter"
    DAY = "day"


class Transform(pydantic.BaseModel):
    aggregation: AggregationType | None = pydantic.Field(
        None,
        description=f"Type of aggregation. It will be ignored when it is scatter plot",
    )
    bin_size: int | None = pydantic.Field(
        None,
        description="Integer value as the number of bins used to discretizes numeric values into a set of bins"
    )
    time_unit: TimeUnit | None = pydantic.Field(
        None,
        description="The time unit used to descretize date/datetime values"
    )

    def transformed_name(self, col: str) -> str:
        dst = col
        if self.time_unit:
            dst = f"UNIT({col}, {self.time_unit.value})"
        if self.bin_size:
            dst = f"BINNING({col}, {self.bin_size})"
        if self.aggregation:
            dst = f"{self.aggregation.value}({col})"
        return dst

    @classmethod
    def parse_from_llm(cls, d: dict[str, str]) -> "Transform":
        return Transform(
            aggregation=AggregationType(d["aggregation"].upper())
            if d.get("aggregation")
            else None,
            bin_size=d.get("bin_size") or None,
            time_unit=d.get("time_unit") or None
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
    column: str = pydantic.Field(None, description="column in datasets used for the axis")
    transform: Transform | None = pydantic.Field(None, description="transformation applied to column")
    min_value: float | None
    max_value: float | None
    label: str | None

    def transformed_name(self):
        return self.transform.transformed_name(self.column) if self.transform else self.column

    class Config(SchemaWithoutTitle):
        pass

    @classmethod
    def parse_from_llm(cls, d: dict[str, str | float | dict[str, str]]) -> "Axis":
        return Axis(
            column=d.get("column") or None,
            transform=Transform.parse_from_llm(d["transform"]) if "transform" in d else None,  # type: ignore
            min_value=d.get("min_value"),  # type: ignore
            max_value=d.get("max_value"),  # type: ignore
            label=d.get("label") or None,  # type: ignore
        )


class PlotConfig(pydantic.BaseModel):
    chart_type: ChartType = pydantic.Field(None, description="the type of the chart")
    filters: list[str] = pydantic.Field(
        None,
        description="List of filter conditions, where each filter must be a legal string that can be passed to df.query(),"
        ' such as "x >= 0". Filters will be calculated before transforming axis.',
    )
    x: Axis | None = pydantic.Field(
        None, description="X-axis for the chart, or label column for pie chart"
    )
    y: Axis = pydantic.Field(
        None,
        description="Y-axis or measure value for the chart, or the wedge sizes for pie chart.",
    )
    hue: str | None = pydantic.Field(
        None,
        description="Column name used as grouping variables that will produce different colors.",
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
        columns = [self.y.column]
        if self.x:
            columns.append(self.x.column)
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
            hue=json_data.get("hue") or None,
            sort_criteria=SortingCriteria(json_data["sort_criteria"])
            if json_data.get("sort_criteria")
            else None,
            sort_order=SortOrder(json_data["sort_order"])
            if json_data.get("sort_order")
            else None,
        )


def get_schema_of_chart_config(inlining_refs: bool = False, remove_title: bool = True) -> dict[str, Any]:
    defs = jsonref.loads(PlotConfig.schema_json()) if inlining_refs else PlotConfig.schema()  # type: ignore

    if remove_title:
        defs = remove_field_recursively(defs, "title")

    defs = flatten_single_element_allof(defs)

    return defs
