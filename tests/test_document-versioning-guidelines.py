import pathlib


def test_usage_guide_includes_versioning_examples():
    guide = pathlib.Path("API_USAGE_GUIDE.md")
    assert guide.exists(), "API_USAGE_GUIDE.md is missing"
    content = guide.read_text().lower()
    assert "/v1/ping" in content, "Versioned ping endpoint missing in docs"
    assert "/ping" in content, "Unversioned ping example missing"
