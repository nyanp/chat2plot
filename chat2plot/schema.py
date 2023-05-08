import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


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


@dataclass(frozen=True)
class Measure:
    column: str
    aggregation_method: AggregationType | None = None

    @classmethod
    def from_text(cls, text: str) -> "Measure":
        for agg in AggregationType:
            m = re.match(rf"{agg.value}\((.*)\)", text)
            if m:
                return Measure(m.group(1), agg)

        return Measure(text)

    def __repr__(self) -> str:
        if self.aggregation_method:
            return f"{self.aggregation_method.value}({self.column})"
        else:
            return self.column


@dataclass(frozen=True)
class Field:
    column: str
    aggregation: AggregationType | None = None

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "Field":
        return Field(
            d["column"],
            AggregationType(d["aggregation"]) if d.get("aggregation") else None,
        )


@dataclass(frozen=True)
class Filter:
    query: str


@dataclass
class PlotConfig:
    chart_type: ChartType
    x: Field | None
    y: Field
    filters: list[Filter]
    hue: Field | None = None
    xmin: float | None = None
    xmax: float | None = None
    ymin: float | None = None
    ymax: float | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    sort_criteria: SortingCriteria | None = None
    sort_order: SortOrder | None = None

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

        filters = [Filter(q) for q in wrap_if_not_list(json_data.get("filters", []))]

        return cls(
            chart_type,
            Field.from_dict(json_data["x"]) if json_data.get("x") else None,
            Field.from_dict(json_data["y"]),
            filters,
            Field.from_dict(json_data["hue"]) if json_data.get("hue") else None,
            json_data.get("xmin"),
            json_data.get("xmax"),
            json_data.get("ymin"),
            json_data.get("ymax"),
            json_data.get("xlabel"),
            json_data.get("ylabel"),
            SortingCriteria(json_data["sort_criteria"])
            if json_data.get("sort_criteria")
            else None,
            SortOrder(json_data["sort_order"]) if json_data.get("sort_order") else None,
        )


@dataclass
class LLMResponse:
    response_type: ResponseType
    config: PlotConfig | None = None
