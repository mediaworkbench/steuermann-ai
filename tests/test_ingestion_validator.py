"""Unit tests for language validator."""

import pytest
from unittest.mock import patch, Mock

from universal_agentic_framework.ingestion.validator import LanguageValidator


class TestLanguageValidator:
    """Tests for LanguageValidator."""
    
    def test_init_default(self):
        """Test default initialization."""
        validator = LanguageValidator(target_language="en")
        
        assert validator.target_language == "en"
        assert validator.confidence_threshold == 0.8
    
    def test_init_custom(self):
        """Test custom initialization."""
        validator = LanguageValidator(
            target_language="de",
            confidence_threshold=0.9
        )
        
        assert validator.target_language == "de"
        assert validator.confidence_threshold == 0.9
    
    @patch("universal_agentic_framework.ingestion.validator.detect_langs")
    def test_validate_correct_language(self, mock_detect_langs):
        """Test validation with correct language."""
        # Mock detection
        mock_lang = Mock()
        mock_lang.lang = "en"
        mock_lang.prob = 0.95
        mock_detect_langs.return_value = [mock_lang]
        
        validator = LanguageValidator(target_language="en")
        is_valid, detected_lang, confidence = validator.validate("English text here")
        
        assert is_valid is True
        assert detected_lang == "en"
        assert confidence == 0.95
    
    @patch("universal_agentic_framework.ingestion.validator.detect_langs")
    def test_validate_wrong_language(self, mock_detect_langs):
        """Test validation with wrong language."""
        # Mock detection
        mock_lang = Mock()
        mock_lang.lang = "de"
        mock_lang.prob = 0.95
        mock_detect_langs.return_value = [mock_lang]
        
        validator = LanguageValidator(target_language="en")
        is_valid, detected_lang, confidence = validator.validate("German text hier")
        
        assert is_valid is False
        assert detected_lang == "de"
        assert confidence == 0.95
    
    @patch("universal_agentic_framework.ingestion.validator.detect_langs")
    def test_validate_low_confidence(self, mock_detect_langs):
        """Test validation with low confidence."""
        # Mock detection
        mock_lang = Mock()
        mock_lang.lang = "en"
        mock_lang.prob = 0.5  # Below threshold
        mock_detect_langs.return_value = [mock_lang]
        
        validator = LanguageValidator(target_language="en", confidence_threshold=0.8)
        is_valid, detected_lang, confidence = validator.validate("Mixed language text")
        
        assert is_valid is False
        assert detected_lang == "en"
        assert confidence == 0.5
        
        assert is_valid is False
        assert detected_lang == "en"
        assert confidence == 0.5
    
    @patch("universal_agentic_framework.ingestion.validator.detect_langs")
    def test_get_language_percentage(self, mock_detect_langs):
        """Test getting language percentage."""
        # Mock mixed language detection
        mock_en = Mock()
        mock_en.lang = "en"
        mock_en.prob = 0.7
        
        mock_de = Mock()
        mock_de.lang = "de"
        mock_de.prob = 0.3
        
        mock_detect_langs.return_value = [mock_en, mock_de]
        
        validator = LanguageValidator(target_language="en")
        percentage = validator.get_language_percentage("Mixed text")
        
        # Returns percentage of text matching target language
        assert isinstance(percentage, float)
        assert 0.0 <= percentage <= 1.0
    
    @patch("universal_agentic_framework.ingestion.validator.detect_langs")
    def test_should_accept_valid(self, mock_detect_langs):
        """Test should_accept with valid document."""
        # Mock detection
        mock_lang = Mock()
        mock_lang.lang = "en"
        mock_lang.prob = 0.95
        mock_detect_langs.return_value = [mock_lang]
        
        validator = LanguageValidator(target_language="en")
        should_accept, reason, detected_lang, confidence = validator.should_accept("English text")
        
        assert should_accept is True
        assert "Valid" in reason
        assert detected_lang == "en"
        assert confidence == 0.95
    
    @patch("universal_agentic_framework.ingestion.validator.detect_langs")
    def test_should_accept_wrong_language(self, mock_detect_langs):
        """Test should_accept with wrong language (now accepts and tags)."""
        # Mock detection
        mock_lang = Mock()
        mock_lang.lang = "de"
        mock_lang.prob = 0.95
        mock_detect_langs.return_value = [mock_lang]
        
        validator = LanguageValidator(target_language="en")
        should_accept, reason, detected_lang, confidence = validator.should_accept("German text")
        
        # As of 2026-01-22: all languages accepted and tagged with detected language
        assert should_accept is True
        assert detected_lang == "de"
        assert confidence == 0.95
        assert "de" in reason.lower()
    
    @patch("universal_agentic_framework.ingestion.validator.detect_langs")
    def test_should_accept_low_confidence(self, mock_detect_langs):
        """Test should_accept with low confidence (now accepts and tags)."""
        # Mock detection
        mock_lang = Mock()
        mock_lang.lang = "en"
        mock_lang.prob = 0.6
        mock_detect_langs.return_value = [mock_lang]
        
        validator = LanguageValidator(target_language="en", confidence_threshold=0.8)
        should_accept, reason, detected_lang, confidence = validator.should_accept("Unclear text")
        
        # As of 2026-01-22: all documents accepted and tagged with detected language and confidence
        assert should_accept is True
        assert detected_lang == "en"
        assert confidence == 0.6
        assert "confidence" in reason.lower()
    
    def test_validate_empty_text(self):
        """Test validation with empty text."""
        validator = LanguageValidator(target_language="en")
        
        with pytest.raises(Exception):  # langdetect raises on empty text
            validator.validate("")
    
    @patch("universal_agentic_framework.ingestion.validator.detect_langs")
    def test_validate_short_text(self, mock_detect_langs):
        """Test validation with short text (should raise exception)."""
        # Short text should raise exception before calling detect_langs
        validator = LanguageValidator(target_language="en", confidence_threshold=0.7)
        
        with pytest.raises(Exception):  # LangDetectException
            validator.validate("Short")
