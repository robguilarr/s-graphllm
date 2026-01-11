"""
LLM Agent for graph reasoning using OpenAI API.
"""

from typing import Optional, List, Dict, Any
import logging
from dataclasses import dataclass
import os

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM agent."""
    model: str = "gpt-4.1-mini"
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.9
    api_key: Optional[str] = None


class LLMAgent:
    """
    LLM Agent for performing reasoning tasks on graphs.
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM agent.
        
        Args:
            config: LLM configuration
        """
        self.config = config
        
        # Initialize OpenAI client
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        
        # Conversation history for context
        self.conversation_history: List[Dict[str, str]] = []
        
        logger.info(f"LLM Agent initialized with model: {config.model}")
    
    def reason(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        include_history: bool = False
    ) -> str:
        """
        Perform reasoning using the LLM.
        
        Args:
            prompt: The reasoning prompt
            system_prompt: Optional system prompt
            include_history: Whether to include conversation history
        
        Returns:
            LLM response
        """
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        else:
            messages.append({
                "role": "system",
                "content": self._get_default_system_prompt()
            })
        
        # Add conversation history if requested
        if include_history:
            messages.extend(self.conversation_history)
        
        # Add current prompt
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p
            )
            
            # Extract response
            answer = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append({
                "role": "user",
                "content": prompt
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": answer
            })
            
            # Keep history size manageable
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            logger.debug(f"LLM response: {answer[:100]}...")
            return answer
        
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise
    
    def batch_reason(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None
    ) -> List[str]:
        """
        Perform batch reasoning on multiple prompts.
        
        Args:
            prompts: List of reasoning prompts
            system_prompt: Optional system prompt
        
        Returns:
            List of LLM responses
        """
        responses = []
        for prompt in prompts:
            response = self.reason(prompt, system_prompt=system_prompt, include_history=False)
            responses.append(response)
        
        return responses
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract entities from text using the LLM.
        
        Args:
            text: Input text
        
        Returns:
            List of extracted entities
        """
        prompt = f"""Extract all named entities (people, places, organizations, etc.) from the following text. Return them as a comma-separated list.

Text: {text}

Entities:"""
        
        response = self.reason(prompt, include_history=False)
        entities = [e.strip() for e in response.split(",")]
        return entities
    
    def extract_relationships(self, text: str) -> List[Dict[str, str]]:
        """
        Extract relationships from text using the LLM.
        
        Args:
            text: Input text
        
        Returns:
            List of relationships as dictionaries
        """
        prompt = f"""Extract all relationships from the following text. For each relationship, provide the source entity, relationship type, and target entity in the format: source -> [relationship] -> target

Text: {text}

Relationships:"""
        
        response = self.reason(prompt, include_history=False)
        
        # Parse relationships
        relationships = []
        for line in response.split("\n"):
            if "->" in line:
                parts = line.split("->")
                if len(parts) == 3:
                    relationships.append({
                        "source": parts[0].strip(),
                        "relationship": parts[1].strip().strip("[]"),
                        "target": parts[2].strip()
                    })
        
        return relationships
    
    def answer_question(
        self,
        question: str,
        context: str
    ) -> str:
        """
        Answer a question based on provided context.
        
        Args:
            question: The question to answer
            context: Context for answering
        
        Returns:
            Answer to the question
        """
        prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say "I don't know".

Context:
{context}

Question: {question}

Answer:"""
        
        response = self.reason(prompt, include_history=False)
        return response
    
    def multi_hop_reasoning(
        self,
        question: str,
        graph_context: str,
        max_hops: int = 3
    ) -> str:
        """
        Perform multi-hop reasoning on a graph.
        
        Args:
            question: The question to answer
            graph_context: Description of the graph
            max_hops: Maximum number of reasoning hops
        
        Returns:
            Answer with reasoning trace
        """
        prompt = f"""Perform multi-hop reasoning to answer the following question. Show your reasoning step by step, considering up to {max_hops} hops through the graph.

Graph Context:
{graph_context}

Question: {question}

Please provide:
1. The reasoning path(s) through the graph
2. Intermediate conclusions at each hop
3. The final answer

Reasoning:"""
        
        response = self.reason(prompt, include_history=False)
        return response
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for graph reasoning."""
        return """You are an expert AI assistant specialized in reasoning over knowledge graphs. 
Your task is to answer questions by analyzing graph structures, identifying relevant entities and relationships, 
and performing multi-hop reasoning when necessary. 

When reasoning:
1. Identify the relevant entities and relationships in the graph
2. Trace paths through the graph to find connections
3. Consider multiple possible paths and relationships
4. Provide clear, step-by-step reasoning
5. State your confidence in the answer

Be precise, logical, and thorough in your reasoning."""
