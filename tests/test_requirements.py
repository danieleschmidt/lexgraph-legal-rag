import pathlib


def test_pytest_listed_in_requirements():
    req = pathlib.Path("requirements.txt")
    assert req.exists(), "requirements.txt does not exist"
    assert "pytest" in req.read_text().lower()
