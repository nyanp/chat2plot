import copy
import re
from enum import Enum
from typing import Any, Type

import jsonref
import pydantic

from chat2plot.dictionary_helper import (
    flatten_single_element_allof,
    remove_field_recursively,
)


class ChartType(str, Enum):
    PIE = "pie"
    SCATTER = "scatter"
    LINE = "line"
    BAR = "bar"
    AREA = "area"
    SCALAR = "scalar"


class AggregationType(str, Enum):
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    COUNT = "COUNT"
    COUNTROWS = "COUNTROWS"
    DISTINCT_COUNT = "DISTINCT_COUNT"


class ResponseType(str, Enum):
    SUCCESS = "success"
    UNKNOWN = "unknown"
    FAILED_TO_RENDER = "failed to render"


class SortingCriteria(str, Enum):
    NAME = "name"
    VALUE = "value"


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


class TimeUnit(str, Enum):
    YEAR = "year"
    MONTH = "month"
    WEEK = "week"
    QUARTER = "quarter"
    DAY = "day"


class BarMode(str, Enum):
    STACK = "stacked"
    GROUP = "group"


class Filter(pydantic.BaseModel):
    lhs: str
    rhs: str
    op: str

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


class XAxis(pydantic.BaseModel):
    column: str = pydantic.Field(
        description="name of the column in the df used for the x-axis"
    )
    bin_size: int | None = pydantic.Field(
        None,
        description="Integer value as the number of bins used to discretizes numeric values into a set of bins",
    )
    time_unit: TimeUnit | None = pydantic.Field(
        None, description="The time unit used to descretize date/datetime values"
    )
    min_value: float | None
    max_value: float | None
    label: str | None

    def transformed_name(self) -> str:
        dst = self.column
        if self.time_unit:
            dst = f"UNIT({dst}, {self.time_unit.value})"
        if self.bin_size:
            dst = f"BINNING({dst}, {self.bin_size})"
        return dst

    @classmethod
    def parse_from_llm(cls, d: dict[str, str | float | dict[str, str]]) -> "XAxis":
        return XAxis(
            column=d.get("column") or None,  # type: ignore
            min_value=d.get("min_value"),  # type: ignore
            max_value=d.get("max_value"),  # type: ignore
            label=d.get("label") or None,  # type: ignore
            bin_size=d.get("bin_size") or None,  # type: ignore
            time_unit=TimeUnit(d["time_unit"]) if d.get("time_unit") else None,  # type: ignore
        )


class YAxis(pydantic.BaseModel):
    column: str = pydantic.Field(
        description="name of the column in the df used for the y-axis"
    )
    aggregation: AggregationType | None = pydantic.Field(
        None,
        description="Type of aggregation. Required for all chart types but scatter plots.",
    )
    min_value: float | None
    max_value: float | None
    label: str | None

    def transformed_name(self) -> str:
        dst = self.column
        if self.aggregation:
            dst = f"{self.aggregation.value}({dst})"
        return dst

    @classmethod
    def parse_from_llm(
        cls, d: dict[str, str | float | dict[str, str]], needs_aggregation: bool = False
    ) -> "YAxis":
        agg = d.get("aggregation")
        if needs_aggregation and not agg:
            agg = "AVG"

        if not d.get("column") and needs_aggregation:
            agg = "COUNTROWS"
        elif agg == "COUNTROWS":
            agg = "COUNT"

        return YAxis(
            column=d.get("column") or "",  # type: ignore
            aggregation=AggregationType(agg) if agg else None,
            min_value=d.get("min_value"),  # type: ignore
            max_value=d.get("max_value"),  # type: ignore
            label=d.get("label") or None,  # type: ignore
        )


class PlotConfig(pydantic.BaseModel):
    chart_type: ChartType = pydantic.Field(
        description="The type of the chart. Use scatter plots as little as possible unless explicitly specified by the user. Choose 'scalar' if we need only single scalar."
    )
    filters: list[str] = pydantic.Field(
        description="List of filter conditions, where each filter must be a legal string that can be passed to df.query(),"
        ' such as "x >= 0". Filters are calculated before transforming axes.　'
        "When using AVG on the y-axis, do not filter the same column to a specific single value.",
    )
    x: XAxis | None = pydantic.Field(
        None, description="X-axis for the chart, or label column for pie chart"
    )
    y: YAxis = pydantic.Field(
        description="Y-axis or measure value for the chart, or the wedge sizes for pie chart.",
    )
    color: str | None = pydantic.Field(
        None,
        description="Column name used as grouping variables that will produce different colors.",
    )
    bar_mode: BarMode | None = pydantic.Field(
        None,
        description="If 'stacked', bars are stacked. In 'group' mode, bars are placed beside each other.",
    )
    sort_criteria: SortingCriteria | None = pydantic.Field(
        None, description="The sorting criteria for x-axis"
    )
    sort_order: SortOrder | None = pydantic.Field(
        None, description="Sorting order for x-axis"
    )
    horizontal: bool | None = pydantic.Field(
        None, description="If true, the chart is drawn in a horizontal orientation"
    )
    limit: int | None = pydantic.Field(
        None, description="Limit a number of data to top-N items"
    )

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "PlotConfig":
        assert "chart_type" in json_data
        assert "y" in json_data

        json_data = copy.deepcopy(json_data)

        def wrap_if_not_list(value: str | list[str]) -> list[str]:
            if not isinstance(value, list):
                return [] if not value else [value]
            else:
                return value

        if not json_data["chart_type"] or json_data["chart_type"].lower() == "none":
            # treat chart as bar if x-axis does not exist
            chart_type = ChartType.BAR
        else:
            chart_type = ChartType(json_data["chart_type"])

        if not json_data.get("x") and chart_type == ChartType.PIE:
            # LLM sometimes forget to fill x in pie-chart
            json_data["x"] = copy.deepcopy(json_data["y"])

        return cls(
            chart_type=chart_type,
            x=XAxis.parse_from_llm(json_data["x"]) if json_data.get("x") else None,
            y=YAxis.parse_from_llm(
                json_data["y"], needs_aggregation=chart_type != ChartType.SCATTER
            ),
            filters=wrap_if_not_list(json_data.get("filters", [])),
            color=json_data.get("color") or None,
            bar_mode=BarMode(json_data["bar_mode"])
            if json_data.get("bar_mode")
            else None,
            sort_criteria=SortingCriteria(json_data["sort_criteria"])
            if json_data.get("sort_criteria")
            else None,
            sort_order=SortOrder(json_data["sort_order"])
            if json_data.get("sort_order")
            else None,
            horizontal=json_data.get("horizontal"),
            limit=json_data.get("limit"),
        )


def get_schema_of_chart_config(
    target_schema: Type[pydantic.BaseModel],
    inlining_refs: bool = True,
    remove_title: bool = True,
    as_function: bool = False,
) -> dict[str, Any]:
    defs = jsonref.loads(target_schema.schema_json()) if inlining_refs else target_schema.schema()  # type: ignore

    if remove_title:
        defs = remove_field_recursively(defs, "title")

    defs = flatten_single_element_allof(defs)

    defs = copy.deepcopy(defs)

    if as_function:
        return {
            "name": "generate_chart",
            "description": "Generate the chart with given parameters",
            "parameters": defs,
        }

    return defs  # type: ignore
