from chat2plot import schema


def test_filter():
    f = schema.Filter.parse_from_llm("a == 3")
    assert f.lhs == "a"
    assert f.op == "=="
    assert f.rhs == "3"
    assert f.escaped() == "`a` == 3"

    f = schema.Filter.parse_from_llm("a b c < 4")
    assert f.lhs == "a b c"
    assert f.op == "<"
    assert f.rhs == "4"
    assert f.escaped() == "`a b c` < 4"
