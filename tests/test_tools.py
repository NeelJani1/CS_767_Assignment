"""Unit tests for tools.py."""

import os
import tempfile
from unittest.mock import MagicMock, patch

from tools import (
    get_default_tools,
    safe_search,
    safe_wiki,
    save_to_txt_file,
    save_tool,
    search_tool,
    wiki_tool,
)


class TestTools:
    """Test suite for agent tool execution and helper functions."""

    def test_save_to_txt_file_creates_file_and_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test_history.txt")
            data = "Test conversation log entry."
            result = save_to_txt_file(data, filename=test_file)

            assert os.path.isfile(test_file)
            assert "saved to" in result

            with open(test_file, "r", encoding="utf-8") as f:
                content = f.read()

            assert "---Conversation History---" in content
            assert "Test conversation log entry." in content
            assert "-" * 50 in content

    def test_save_to_txt_file_creates_nested_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_file = os.path.join(tmpdir, "logs", "nested", "history.txt")
            result = save_to_txt_file("Nested test log", filename=nested_file)

            assert os.path.isfile(nested_file)
            assert "saved to" in result

    def test_tool_declarations_and_default_tools(self):
        defaults = get_default_tools()
        assert len(defaults) == 2
        assert search_tool in defaults
        assert wiki_tool in defaults
        assert search_tool.name == "Search_tool"
        assert wiki_tool.name == "Wikipedia"
        assert save_tool.name == "Save_to_txt_file"

    @patch("tools.DuckDuckGoSearchRun")
    def test_safe_search_success(self, mock_ddg_class):
        mock_instance = MagicMock()
        mock_instance.run.return_value = "Auckland is a city in New Zealand."
        mock_ddg_class.return_value = mock_instance

        output = safe_search("Auckland travel")
        assert "Auckland is a city" in output

    @patch("tools.DuckDuckGoSearchRun")
    def test_safe_search_handles_error(self, mock_ddg_class):
        mock_instance = MagicMock()
        mock_instance.run.side_effect = RuntimeError("Network timeout")
        mock_ddg_class.return_value = mock_instance

        output = safe_search("Auckland travel")
        assert "Web search encountered an error" in output

    @patch("tools.WikipediaQueryRun")
    @patch("tools.WikipediaAPIWrapper")
    def test_safe_wiki_success(self, mock_wrapper, mock_wiki_run):
        mock_instance = MagicMock()
        mock_instance.run.return_value = "Hobbiton Movie Set is located in Matamata."
        mock_wiki_run.return_value = mock_instance

        output = safe_wiki("Hobbiton")
        assert "Hobbiton Movie Set" in output

    @patch("tools.WikipediaQueryRun")
    @patch("tools.WikipediaAPIWrapper")
    def test_safe_wiki_handles_error(self, mock_wrapper, mock_wiki_run):
        mock_instance = MagicMock()
        mock_instance.run.side_effect = Exception("Page not found")
        mock_wiki_run.return_value = mock_instance

        output = safe_wiki("NonExistentPage12345")
        assert "Wikipedia search encountered an error" in output
