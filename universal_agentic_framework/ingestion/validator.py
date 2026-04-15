"""Language validation for ingested documents."""

from typing import Optional
from langdetect import detect_langs, LangDetectException


class LanguageValidator:
    """Validates document language against expected language."""
    
    def __init__(
        self,
        target_language: str,
        confidence_threshold: float = 0.8
    ):
        """Initialize validator.
        
        Args:
            target_language: Expected language code (e.g., 'en', 'de')
            confidence_threshold: Minimum confidence to accept (0.0-1.0)
        """
        self.target_language = target_language.lower()
        self.confidence_threshold = confidence_threshold
    
    def validate(self, text: str) -> tuple[bool, Optional[str], float]:
        """Validate text language.
        
        Args:
            text: Text to validate
            
        Returns:
            Tuple of (is_valid, detected_language, confidence)
            where confidence is probability from detect_langs (0.0-1.0)
        """
        if not text or len(text.strip()) < 10:
            # Empty or too short - raise exception like langdetect does
            raise LangDetectException(1, "Text too short for language detection")
        
        try:
            # Use detect_langs to get confidence scores
            lang_probs = detect_langs(text)
            
            if not lang_probs:
                raise LangDetectException(1, "No language detected")
            
            # Get the most probable language
            most_probable = lang_probs[0]
            detected_lang = most_probable.lang.lower()
            confidence = most_probable.prob
            
            # Check if it matches target and meets confidence threshold
            is_valid = (detected_lang == self.target_language and 
                       confidence >= self.confidence_threshold)
            
            return is_valid, detected_lang, confidence
            
        except LangDetectException:
            # Detection failed completely
            raise
    
    def get_language_percentage(self, text: str, sample_size: int = 1000) -> float:
        """Estimate percentage of text in target language.
        
        Args:
            text: Text to analyze
            sample_size: Characters per sample
            
        Returns:
            Estimated percentage (0.0-1.0) of text in target language
        """
        if not text or len(text) < sample_size:
            try:
                is_valid, _, _ = self.validate(text)
                return 1.0 if is_valid else 0.0
            except LangDetectException:
                return 0.0
        
        # Sample multiple chunks
        num_samples = min(5, len(text) // sample_size)
        matches = 0
        
        for i in range(num_samples):
            start = i * sample_size
            end = start + sample_size
            sample = text[start:end]
            
            try:
                is_valid, _, _ = self.validate(sample)
                if is_valid:
                    matches += 1
            except LangDetectException:
                # Skip samples that can't be detected
                continue
        
        return matches / num_samples if num_samples > 0 else 0.0
    
    def should_accept(self, text: str) -> tuple[bool, str, str, float]:
        """Determine if document should be accepted and detect its language.
        
        Args:
            text: Document text
            
        Returns:
            Tuple of (should_accept, reason, detected_language, confidence)
        """
        try:
            is_valid, detected, confidence = self.validate(text)
            
            # Always accept, just tag with detected language
            if is_valid:
                return True, f"Valid {self.target_language} document (confidence: {confidence:.2%})", detected, confidence
            
            # If detected language matches target but confidence is low
            if detected == self.target_language:
                return True, f"Accepted {self.target_language} document (low confidence: {confidence:.2%}, but correct language)", detected, confidence
            
            # Different language detected - still accept but tag appropriately
            return True, f"Accepted document with detected language: {detected} (confidence: {confidence:.2%})", detected, confidence
            
        except LangDetectException as e:
            # Failed to detect - accept with unknown language
            return True, f"Language detection failed, accepting anyway: {str(e)}", "unknown", 0.0
