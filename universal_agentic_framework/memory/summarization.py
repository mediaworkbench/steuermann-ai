"""Conversation summarization for memory optimization.

Implements token-aware summarization to reduce context window usage
and maintain conversation history efficiently.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)


class ConversationSummarizer:
    """Manages conversation summarization with token awareness."""
    
    # Token count approximations (rough estimates)
    TOKENS_PER_MESSAGE = 50  # Average message in tokens
    SUMMARY_TARGET_TOKENS = 500  # Target summary size
    
    def __init__(self, llm_factory=None):
        """Initialize summarizer.
        
        Args:
            llm_factory: Optional LLM factory for generating summaries
        """
        self.llm_factory = llm_factory
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.
        
        Uses simple heuristic: ~1 token per 4 characters.
        """
        tokens = len(text) // 4
        return max(1, tokens)
    
    def calculate_conversation_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Calculate approximate token count for conversation.
        
        Args:
            messages: List of message dicts with 'content' key
            
        Returns:
            Estimated token count
        """
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            total += self.estimate_tokens(content)
        return total
    
    def should_summarize(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        min_messages: int = 10
    ) -> bool:
        """Determine if conversation should be summarized.
        
        Args:
            messages: Conversation messages
            max_tokens: Token threshold for summarization
            min_messages: Minimum messages before considering summary
            
        Returns:
            True if summarization recommended
        """
        if len(messages) < min_messages:
            return False
        
        token_count = self.calculate_conversation_tokens(messages)
        return token_count > max_tokens
    
    def identify_summary_boundary(
        self,
        messages: List[Dict[str, str]],
        target_tokens: int = 500
    ) -> int:
        """Find where to split messages for summarization.
        
        Returns index of messages to keep (recent) vs summarize (old).
        
        Args:
            messages:
 Full conversation messages
            target_tokens: Target tokens for recent messages to keep
            
        Returns:
            Index where recent messages start (messages before should be summarized)
        """
        if not messages:
            return 0
        
        recent_tokens = 0
        for i in range(len(messages) - 1, -1, -1):
            msg_tokens = self.estimate_tokens(messages[i].get("content", ""))
            recent_tokens += msg_tokens
            if recent_tokens > target_tokens:
                return max(0, i + 1)
        return 0
    
    async def generate_summary(
        self,
        messages: List[Dict[str, str]],
        user_id: str
    ) -> Optional[str]:
        """Generate summary of conversation messages.
        
        Args:
            messages: Messages to summarize
            user_id: User context for LLM selection
            
        Returns:
            Summary text or None if generation fails
        """
        if not self.llm_factory or not messages:
            return None
        
        try:
            # Build prompt for summarization
            messages_text = "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in messages[-10:]  # Last 10 messages
            ])
            
            prompt = f"""Summarize the following conversation concisely in 2-3 sentences.
Focus on key topics, decisions, and action items.

---
{messages_text}
---

Summary:"""
            
            llm = self.llm_factory.create_llm(user_id)
            summary = await llm.apredict(prompt)
            
            logger.debug(f"Generated summary ({len(summary)} chars) for user {user_id}")
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            return None
    
    def create_summary_message(
        self,
        summary_text: str,
        message_count: int,
        previous_digest_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Create a summary message to insert in conversation.
        
        Args:
            summary_text: Summary content
            message_count: Number of messages summarized
            
        Returns:
            System message containing summary
        """
        from datetime import timezone
        digest_id = str(uuid.uuid4())
        return {
            "role": "system",
            "content": f"[SUMMARY: Previous {message_count} messages summarized] {summary_text}",
            "type": "summary",
            "digest_id": digest_id,
            "previous_digest_id": previous_digest_id,
            "digest_message_count": message_count,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def extract_digest_chain(
        self,
        messages: List[Dict[str, str]],
        max_items: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return newest-first digest metadata extracted from summary messages."""
        digests: List[Dict[str, Any]] = []
        for msg in reversed(messages):
            if msg.get("type") != "summary":
                continue
            digest_id = msg.get("digest_id")
            if not digest_id:
                continue
            digests.append(
                {
                    "digest_id": digest_id,
                    "previous_digest_id": msg.get("previous_digest_id"),
                    "digest_message_count": msg.get("digest_message_count"),
                    "timestamp": msg.get("timestamp"),
                }
            )
            if len(digests) >= max_items:
                break
        return digests
    
    async def compress_conversation(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        keep_recent_count: int = 5
    ) -> List[Dict[str, str]]:
        """Compress conversation by summarizing old messages.
        
        Args:
            messages: Full conversation
            user_id: User context
            keep_recent_count: Number of recent messages to preserve
            
        Returns:
            Compressed message list with summary inserted
        """
        if len(messages) <= keep_recent_count:
            return messages
        
        # Split into old and recent
        split_idx = len(messages) - keep_recent_count
        old_messages = messages[:split_idx]
        recent_messages = messages[split_idx:]

        previous_digest_id = None
        if old_messages and old_messages[0].get("type") == "summary":
            previous_digest_id = old_messages[0].get("digest_id")
        
        # Generate summary of old messages
        summary_text = await self.generate_summary(old_messages, user_id)
        if not summary_text:
            # If summary generation fails, just use recent messages
            logger.warning(f"Summary generation failed for user {user_id}, using recent only")
            return recent_messages
        
        # Create summary message and combine with recent
        summary_msg = self.create_summary_message(
            summary_text,
            len(old_messages),
            previous_digest_id=previous_digest_id,
        )
        return [summary_msg] + recent_messages
    
    def calculate_savings(
        self,
        original_count: int,
        compressed_count: int
    ) -> Dict[str, Any]:
        """Calculate token savings from compression.
        
        Args:
            original_count: Original message count
            compressed_count: Compressed message count
            
        Returns:
            Dictionary with savings metrics
        """
        messages_removed = original_count - compressed_count
        pct_reduction = (messages_removed / original_count * 100) if original_count > 0 else 0
        
        # Rough token savings (50 tokens per message)
        tokens_saved = messages_removed * self.TOKENS_PER_MESSAGE
        
        return {
            "messages_original": original_count,
            "messages_compressed": compressed_count,
            "messages_removed": messages_removed,
            "reduction_percent": pct_reduction,
            "estimated_tokens_saved": tokens_saved
        }
