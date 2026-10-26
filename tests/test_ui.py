"""Streamlit pages render without errors.

Render-only: no question is asked, so these stay fast and cost nothing. They catch the mistakes
that are easy to make in Streamlit code — bad imports, misused widget APIs, caching errors.
The answering flow itself is covered by the agent tests.
"""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from expert_finder.config import PROJECT_ROOT, get_settings  # noqa: E402

pytestmark = pytest.mark.integration

# AppTest resolves relative paths against the calling file, so build absolute ones.
HOME = str(PROJECT_ROOT / "ui" / "Home.py")
COMPARE = str(PROJECT_ROOT / "ui" / "pages" / "1_Compare.py")
PAGES = [HOME, COMPARE]


@pytest.fixture(scope="module", autouse=True)
def require_stores():
    s = get_settings()
    if not (s.openai_api_key and s.pinecone_api_key and (s.storage_dir / "home_services.db").exists()):
        pytest.skip("needs credentials and a built SQLite store")


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exceptions(page):
    at = AppTest.from_file(page, default_timeout=180).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert at.title, "page should render a title"


def test_chat_page_offers_examples_and_a_mode_switch():
    at = AppTest.from_file(HOME, default_timeout=180).run()
    # AppTest reports the displayed labels, which format_func capitalises.
    assert at.radio[0].options == ["Graph", "Vector"]
    assert at.radio[0].value == "graph"
    assert len(at.button) >= 5          # example prompts plus "New conversation"
    assert at.chat_input, "chat page needs a chat input"


def test_compare_page_has_a_form_for_both_modes():
    at = AppTest.from_file(COMPARE, default_timeout=180).run()
    assert at.button, "compare page needs a submit button"
    assert len(at.selectbox) >= 2       # question and city
