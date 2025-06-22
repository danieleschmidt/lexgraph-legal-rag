import pathlib


def test_pytest_configuration_exists():
    pyproject = pathlib.Path("pyproject.toml")
    pyproject_has_config = False
    if pyproject.exists():
        pyproject_has_config = "[tool.pytest.ini_options]" in pyproject.read_text()
    pytest_ini = pathlib.Path("pytest.ini")
    assert (
        pyproject_has_config or pytest_ini.exists()
    ), "No pytest configuration found in pyproject.toml or pytest.ini"
