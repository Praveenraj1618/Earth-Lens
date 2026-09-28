from streamlit.testing.v1 import AppTest
from pathlib import Path


def run_synthetic_case(case: str, align: bool = False) -> AppTest:
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    app.radio[0].set_value("Built-in synthetic demo").run()
    app.selectbox[0].set_value(case).run()
    if align:
        app.checkbox[0].set_value(True).run()
    app.button[0].click().run()
    assert not app.exception, [error.message for error in app.exception]
    return app


def test_construction_case_reports_controlled_metrics():
    app = run_synthetic_case("New construction")
    values = [metric.value for metric in app.metric]
    assert values[0] == "4.6%"
    assert values[3:6] == ["0.93", "1.00", "0.93"]


def test_no_change_case_has_zero_candidate_pixels():
    app = run_synthetic_case("No visible change")
    assert app.metric[0].value == "0.0%"


def test_lighting_change_exposes_false_alarm():
    app = run_synthetic_case("Lighting shift false alarm")
    assert app.metric[0].value == "100.0%"
    assert app.metric[3].value == "0.00"


def test_translation_alignment_path_runs():
    app = run_synthetic_case("New construction", align=True)
    alignment = app.session_state["analysis"]["alignment"]
    assert alignment["applied"] is True
    assert alignment["score"] is not None
