import pandas as pd

from conftest import signal_matrix
from overfit.cli import main


def test_report_runs(tmp_path, capsys):
    df = pd.DataFrame(signal_matrix(0, t=800, n=20))
    df.insert(0, "date", pd.bdate_range("2020-01-01", periods=800).astype(str))
    path = tmp_path / "r.csv"
    df.to_csv(path, index=False)
    assert main(["report", str(path), "--blocks", "8", "--periods-per-year", "252"]) == 0
    out = capsys.readouterr().out
    for key in ("PBO", "DSR", "Minimum Track Record Length", "significant"):
        assert key in out


def test_report_error_is_reported(tmp_path, capsys):
    path = tmp_path / "r.csv"
    pd.DataFrame(signal_matrix(0, t=100, n=3)).to_csv(path, index=False)
    assert main(["report", str(path), "--blocks", "7"]) == 1
    assert "even" in capsys.readouterr().err
