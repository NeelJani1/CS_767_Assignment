"""Unit tests for ResponseSchema and JSON output sanitization in main.py."""

import pytest
from pydantic import ValidationError

from main import ResponseSchema, clean_json_output


class TestResponseSchema:
    """Test suite for Pydantic schema validation."""

    def test_valid_schema_instantiation(self):
        data = {
            "topic": "Auckland 7-Day Itinerary",
            "summary": "Detailed exploration of Auckland, Waiheke, and Rotorua.",
            "source": ["Wikipedia", "DuckDuckGo"],
            "tools_used": ["Search_tool", "Wikipedia"],
        }
        model = ResponseSchema(**data)
        assert model.topic == "Auckland 7-Day Itinerary"
        assert len(model.source) == 2
        assert len(model.tools_used) == 2

    def test_schema_defaults_for_optional_lists(self):
        data = {
            "topic": "Quick Query",
            "summary": "Brief explanation.",
        }
        model = ResponseSchema(**data)
        assert model.source == []
        assert model.tools_used == []

    def test_missing_required_fields_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ResponseSchema(summary="Missing topic")


class TestCleanJsonOutput:
    """Test suite for agent output sanitization and JSON extraction."""

    def test_pure_json_string(self):
        raw = '{"topic": "Test", "summary": "A test summary", "source": [], "tools_used": []}'
        cleaned = clean_json_output(raw)
        assert cleaned == raw

    def test_markdown_code_fence_json(self):
        raw = '```json\n{"topic": "Trip", "summary": "Go to beach", "source": [], "tools_used": []}\n```'
        cleaned = clean_json_output(raw)
        assert (
            cleaned == '{"topic": "Trip", "summary": "Go to beach", "source": [], "tools_used": []}'
        )

    def test_markdown_code_fence_without_language_tag(self):
        raw = '```\n{"topic": "Trip", "summary": "Visit museums", "source": [], "tools_used": []}\n```'
        cleaned = clean_json_output(raw)
        assert (
            cleaned
            == '{"topic": "Trip", "summary": "Visit museums", "source": [], "tools_used": []}'
        )

    def test_surrounding_conversational_text(self):
        raw = (
            "Here is the itinerary you requested:\n\n"
            '{"topic": "Rotorua", "summary": "Geothermal hot springs", "source": ["Web"], "tools_used": ["Search_tool"]}\n\n'
            "Hope you have a fantastic trip!"
        )
        cleaned = clean_json_output(raw)
        assert (
            cleaned
            == '{"topic": "Rotorua", "summary": "Geothermal hot springs", "source": ["Web"], "tools_used": ["Search_tool"]}'
        )

    def test_chunked_list_input(self):
        raw_list = [
            '{"topic": "Chunked", ',
            '"summary": "Combined message", ',
            '"source": [], ',
            '"tools_used": []}',
        ]
        cleaned = clean_json_output(raw_list)
        assert (
            cleaned
            == '{"topic": "Chunked", "summary": "Combined message", "source": [], "tools_used": []}'
        )

    def test_chunked_dict_list_input(self):
        raw_list = [
            {"text": '{"topic": "DictChunk", '},
            {"text": '"summary": "Parsed text", "source": [], "tools_used": []}'},
        ]
        cleaned = clean_json_output(raw_list)
        assert (
            cleaned
            == '{"topic": "DictChunk", "summary": "Parsed text", "source": [], "tools_used": []}'
        )
