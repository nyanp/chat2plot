from chat2plot import schema


def test_field():
    f = schema.Field.from_dict({"column": "foo"})
    assert f.column == "foo"
    assert not f.aggregation

    f = schema.Field.from_dict({"column": "a b c", "aggregation": "SUM"})
    assert f.column == "a b c"
    assert f.aggregation == schema.AggregationType.SUM

    # case-insensitive
    f = schema.Field.from_dict({"column": "あああ", "aggregation": "avg"})
    assert f.column == "あああ"
    assert f.aggregation == schema.AggregationType.AVG


def test_filter():
    f = schema.Filter.from_text("a == 3")
    assert f.lhs == "`a`"
    assert f.op == "=="
    assert f.rhs == "3"

    f = schema.Filter.from_text("a b c < 4")
    assert f.lhs == "`a b c`"
    assert f.op == "<"
    assert f.rhs == "4"
