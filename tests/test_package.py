from __future__ import annotations

def test_package_has_version():
    import kaggle_lab
    assert isinstance(kaggle_lab.__version__, str)

def test_cli_entry_is_callable():
    from kaggle_lab.cli import main
    assert callable(main)
