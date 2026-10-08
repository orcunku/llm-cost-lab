import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.config import ROOT  # noqa: E402


def _app():
    return AppTest.from_file(str(ROOT / "dashboard" / "streamlit_app.py"), default_timeout=60).run()


def _input(at, label):
    return next(n for n in at.number_input if n.label.startswith(label))


def test_hosted_entry_point_renders_committed_results():
    at = _app()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Engines", "Load test", "Capacity & cost planner", "Report"]
    assert not [w for w in at.warning if "Capacity was measured" in w.value]   # planner defaults to load-tested size


def test_planner_survives_edge_inputs():
    at = _app()
    for label, value in [("Requests per day", 1_000), ("Hosted API input price", 0.0),
                         ("Hosted API output price", 0.0), ("Instance price", 0.001)]:
        _input(at, label).set_value(value).run()
        assert not at.exception, (label, [e.message for e in at.exception])
    assert dict((m.label, m.value) for m in at.metric)["1-instance break-even"] == "n/a"   # free API never breaks even
