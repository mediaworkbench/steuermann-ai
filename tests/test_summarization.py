"""Tests for conversation summarization."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from universal_agentic_framework.memory.summarization import ConversationSummarizer


class TestConversationSummarizer:
    """Test conversation summarization functionality."""
    
    def test_token_estimation(self):
        """Test token estimation heuristic."""
        summarizer = ConversationSummarizer()
        
        # Rough estimate: 1 token per 4 characters
        assert summarizer.estimate_tokens("hello world") == 2  # 11 chars / 4 = 2
        assert summarizer.estimate_tokens("") == 1  # Minimum 1
        
        # Longer text
        long_text = "a" * 400  # 400 characters
        assert summarizer.estimate_tokens(long_text) == 100  # 400 / 4
    
    def test_conversation_tokens(self):
        """Test total conversation token counting."""
        summarizer = ConversationSummarizer()
        
        messages = [
            {"content": "Hello world"},
            {"content": "How are you?"},
            {"content": "I am fine, thanks for asking."}
        ]
        
        tokens = summarizer.calculate_conversation_tokens(messages)
        
        # Should sum tokens from each message
        assert tokens > 0
        assert tokens == (
            summarizer.estimate_tokens("Hello world") +
            summarizer.estimate_tokens("How are you?") +
            summarizer.estimate_tokens("I am fine, thanks for asking.")
        )
    
    def test_should_summarize(self):
        """Test summarization threshold logic."""
        summarizer = ConversationSummarizer()
        
        # Few messages - should not summarize
        messages = [
            {"content": "msg1"},
            {"content": "msg2"}
        ]
        assert summarizer.should_summarize(messages, max_tokens=1000) is False
        
        # Below token threshold - should not summarize
        messages = [
            {"content": f"message {i}"}
            for i in range(5)
        ]
        assert summarizer.should_summarize(messages, max_tokens=10000) is False
        
        # Above threshold - should summarize
        long_messages = [
            {"content": "a" * 500}
            for i in range(20)
        ]
        assert summarizer.should_summarize(
            long_messages,
            max_tokens=1000,
            min_messages=10
        ) is True
    
    def test_identify_summary_boundary(self):
        """Test finding where to split conversation."""
        summarizer = ConversationSummarizer()
        
        # Create messages with larger content to trigger boundary
        messages = [
            {"content": "a" * 200}  # ~50 tokens
            for i in range(10)
        ]
        
        # Find split point for keeping 300 tokens of recent messages
        # With 50 tokens per message, we need ~6 messages for 300 tokens
        boundary = summarizer.identify_summary_boundary(messages, target_tokens=300)
        
        # Should return an index < len(messages) (some messages will be summarized)
        assert 0 <= boundary < len(messages)
        
        # Recent messages should be fewer than total
        recent_messages = messages[boundary:]
        assert len(recent_messages) <= len(messages)
    
    def test_create_summary_message(self):
        """Test creating a summary message."""
        summarizer = ConversationSummarizer()
        
        summary_text = "User discussed Python programming basics."
        msg = summarizer.create_summary_message(summary_text, 15)
        
        assert msg["role"] == "system"
        assert msg["type"] == "summary"
        assert "SUMMARY" in msg["content"]
        assert "15" in msg["content"]  # Message count mentioned
        assert summary_text in msg["content"]
        assert msg["digest_id"]
        assert msg["digest_message_count"] == 15
        assert "timestamp" in msg

    def test_extract_digest_chain(self):
        """Test extraction of rolling digest metadata from summary messages."""
        summarizer = ConversationSummarizer()

        messages = [
            {
                "role": "system",
                "type": "summary",
                "digest_id": "d1",
                "previous_digest_id": None,
                "digest_message_count": 8,
                "timestamp": "2026-01-01T00:00:00+00:00",
                "content": "[SUMMARY: Previous 8 messages summarized] ...",
            },
            {"role": "user", "content": "latest question"},
            {
                "role": "system",
                "type": "summary",
                "digest_id": "d2",
                "previous_digest_id": "d1",
                "digest_message_count": 6,
                "timestamp": "2026-01-02T00:00:00+00:00",
                "content": "[SUMMARY: Previous 6 messages summarized] ...",
            },
        ]

        chain = summarizer.extract_digest_chain(messages)
        assert len(chain) == 2
        assert chain[0]["digest_id"] == "d2"
        assert chain[0]["previous_digest_id"] == "d1"
        assert chain[1]["digest_id"] == "d1"
    
    def test_calculate_savings(self):
        """Test compression savings calculation."""
        summarizer = ConversationSummarizer()
        
        savings = summarizer.calculate_savings(
            original_count=20,
            compressed_count=10
        )
        
        assert savings["messages_original"] == 20
        assert savings["messages_compressed"] == 10
        assert savings["messages_removed"] == 10
        assert savings["reduction_percent"] == 50.0
        assert savings["estimated_tokens_saved"] == 500  # 10 * 50 tokens per message
    
    @pytest.mark.asyncio
    async def test_generate_summary_no_factory(self):
        """Test summary generation with no LLM factory."""
        summarizer = ConversationSummarizer(llm_factory=None)
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        # Should return None without factory
        summary = await summarizer.generate_summary(messages, "user123")
        assert summary is None
    
    @pytest.mark.asyncio
    async def test_generate_summary_with_factory(self):
        """Test summary generation with LLM factory."""
        # Mock LLM
        mock_llm = AsyncMock()
        mock_llm.apredict.return_value = "Summary: User asked a question and got help."
        
        # Mock factory
        mock_factory = MagicMock()
        mock_factory.create_llm.return_value = mock_llm
        
        summarizer = ConversationSummarizer(llm_factory=mock_factory)
        
        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language..."}
        ]
        
        summary = await summarizer.generate_summary(messages, "user123")
        
        assert summary is not None
        assert "Summary" in summary
        mock_factory.create_llm.assert_called_once_with("user123")
        mock_llm.apredict.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_compress_conversation(self):
        """Test conversation compression."""
        # Mock factory for summarization
        mock_llm = AsyncMock()
        mock_llm.apredict.return_value = "Previous: User asked about AI and received overview."
        
        mock_factory = MagicMock()
        mock_factory.create_llm.return_value = mock_llm
        
        summarizer = ConversationSummarizer(llm_factory=mock_factory)
        
        # Create a long conversation
        messages = []
        for i in range(20):
            if i % 2 == 0:
                messages.append({
                    "role": "user",
                    "content": f"Question {i}: What is topic {i}?"
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": f"Answer {i}: Here is information about topic {i-1}..."
                })
        
        # Compress keeping last 5 messages
        compressed = await summarizer.compress_conversation(
            messages,
            "user123",
            keep_recent_count=5
        )
        
        # Should have summary + recent messages
        assert len(compressed) <= 6  # 1 summary + 5 recent
        
        # First message should be the summary
        assert compressed[0]["type"] == "summary"
        
        # Should have recent messages
        recent_count = sum(
            1 for msg in compressed
            if msg.get("type") != "summary"
        )
        assert recent_count == 5
    
    @pytest.mark.asyncio
    async def test_compress_short_conversation(self):
        """Test that short conversations are not compressed."""
        summarizer = ConversationSummarizer()
        
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"}
        ]
        
        compressed = await summarizer.compress_conversation(
            messages,
            "user123",
            keep_recent_count=5
        )
        
        # Should return unchanged since conversation is short
        assert compressed == messages
