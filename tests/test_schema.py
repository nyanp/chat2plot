from chat2plot import schema


def test_field():
    f = schema.Field.parse_from_llm({"column": "foo"})
    assert f.column == "foo"
    assert not f.aggregation

    f = schema.Field.parse_from_llm({"column": "a b c", "aggregation": "SUM"})
    assert f.column == "a b c"
    assert f.aggregation == schema.AggregationType.SUM

    # case-insensitive
    f = schema.Field.parse_from_llm({"column": "あああ", "aggregation": "avg"})
    assert f.column == "あああ"
    assert f.aggregation == schema.AggregationType.AVG


def test_filter():
    f = schema.Filter.parse_from_llm("a == 3")
    assert f.lhs == "`a`"
    assert f.op == "=="
    assert f.rhs == "3"

    f = schema.Filter.parse_from_llm("a b c < 4")
    assert f.lhs == "`a b c`"
    assert f.op == "<"
    assert f.rhs == "4"
