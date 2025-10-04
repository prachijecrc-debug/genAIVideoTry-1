"""
Prompt generator for creating structured video outlines
"""

import json
from typing import Dict, Any, List
from loguru import logger


class PromptGenerator:
    """Generate structured prompts for video content"""
    
    def __init__(self, config):
        """
        Initialize prompt generator
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.llm_provider = config.get("llm.provider", "ollama")
        self.model = config.get("llm.model", "llama3")
        self.temperature = config.get("llm.temperature", 0.7)
        
        # Style templates
        self.style_templates = {
            "educational": {
                "tone": "informative and clear",
                "structure": ["introduction", "main concepts", "examples", "summary"],
                "speech_pattern": "teacher-like, methodical"
            },
            "exciting": {
                "tone": "energetic and enthusiastic",
                "structure": ["hook", "build-up", "climax", "call-to-action"],
                "speech_pattern": "dynamic, with exclamations"
            },
            "storytelling": {
                "tone": "narrative and engaging",
                "structure": ["setup", "conflict", "resolution", "moral"],
                "speech_pattern": "conversational, with pauses"
            },
            "tutorial": {
                "tone": "step-by-step and practical",
                "structure": ["overview", "steps", "tips", "conclusion"],
                "speech_pattern": "clear, instructional"
            },
            "motivational": {
                "tone": "inspiring and uplifting",
                "structure": ["problem", "solution", "benefits", "action"],
                "speech_pattern": "powerful, emphatic"
            },
            "conversational": {
                "tone": "casual and friendly",
                "structure": ["greeting", "main points", "personal touch", "farewell"],
                "speech_pattern": "natural, with fillers"
            }
        }
        
        logger.info("Prompt generator initialized")
    
    def generate(self, topic: str, style: str = "conversational", 
                duration: int = 30) -> Dict[str, Any]:
        """
        Generate structured prompt for video content
        
        Args:
            topic: Video topic
            style: Content style
            duration: Target duration in seconds
            
        Returns:
            Structured prompt dictionary
        """
        logger.info(f"Generating prompt for: {topic}")
        
        # Get style template
        template = self.style_templates.get(style, self.style_templates["conversational"])
        
        # Calculate content parameters
        words_per_second = 2.5  # Average speaking rate
        target_words = int(duration * words_per_second)
        
        # Build prompt structure
        prompt = {
            "topic": topic,
            "style": style,
            "duration": duration,
            "target_words": target_words,
            "tone": template["tone"],
            "structure": template["structure"],
            "speech_pattern": template["speech_pattern"],
            "sections": self._generate_sections(topic, template, duration),
            "keywords": self._extract_keywords(topic),
            "emotion_profile": self._get_emotion_profile(style)
        }
        
        # Add LLM instructions
        prompt["llm_instructions"] = self._create_llm_instructions(prompt)
        
        return prompt
    
    def _generate_sections(self, topic: str, template: Dict, duration: int) -> List[Dict]:
        """
        Generate content sections based on template
        
        Args:
            topic: Video topic
            template: Style template
            duration: Duration in seconds
            
        Returns:
            List of section dictionaries
        """
        sections = []
        structure = template["structure"]
        section_duration = duration / len(structure)
        
        for i, section_type in enumerate(structure):
            sections.append({
                "type": section_type,
                "order": i + 1,
                "duration": section_duration,
                "words": int(section_duration * 2.5),
                "content_hints": self._get_content_hints(section_type, topic)
            })
        
        return sections
    
    def _get_content_hints(self, section_type: str, topic: str) -> str:
        """
        Get content hints for section type
        
        Args:
            section_type: Type of section
            topic: Video topic
            
        Returns:
            Content hints string
        """
        hints = {
            "introduction": f"Introduce {topic} in an engaging way",
            "hook": f"Start with a surprising fact about {topic}",
            "greeting": f"Welcome viewers warmly and introduce {topic}",
            "overview": f"Give a brief overview of {topic}",
            "setup": f"Set the scene for {topic}",
            "problem": f"Present a challenge related to {topic}",
            "main concepts": f"Explain the key ideas of {topic}",
            "main points": f"Cover the main aspects of {topic}",
            "build-up": f"Build excitement around {topic}",
            "conflict": f"Present the central challenge in {topic}",
            "steps": f"Break down {topic} into clear steps",
            "solution": f"Present solutions for {topic}",
            "examples": f"Give concrete examples of {topic}",
            "climax": f"Reach the peak moment about {topic}",
            "resolution": f"Resolve the main points about {topic}",
            "tips": f"Share practical tips about {topic}",
            "benefits": f"Highlight benefits of understanding {topic}",
            "personal touch": f"Add personal perspective on {topic}",
            "summary": f"Summarize key points about {topic}",
            "call-to-action": f"Encourage viewer action on {topic}",
            "conclusion": f"Wrap up the discussion of {topic}",
            "moral": f"Share the lesson from {topic}",
            "action": f"Inspire action related to {topic}",
            "farewell": f"Close warmly and invite engagement"
        }
        
        return hints.get(section_type, f"Discuss {topic}")
    
    def _extract_keywords(self, topic: str) -> List[str]:
        """
        Extract keywords from topic
        
        Args:
            topic: Video topic
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction
        import re
        words = re.findall(r'\b\w+\b', topic.lower())
        
        # Filter common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                       'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        keywords = [w for w in words if w not in common_words and len(w) > 2]
        
        return keywords[:5]  # Return top 5 keywords
    
    def _get_emotion_profile(self, style: str) -> Dict[str, Any]:
        """
        Get emotion profile for style
        
        Args:
            style: Content style
            
        Returns:
            Emotion profile dictionary
        """
        profiles = {
            "educational": {
                "energy": "moderate",
                "pace": "steady",
                "emotions": ["curious", "thoughtful", "clear"]
            },
            "exciting": {
                "energy": "high",
                "pace": "fast",
                "emotions": ["enthusiastic", "amazed", "energetic"]
            },
            "storytelling": {
                "energy": "varied",
                "pace": "natural",
                "emotions": ["intrigued", "empathetic", "reflective"]
            },
            "tutorial": {
                "energy": "moderate",
                "pace": "measured",
                "emotions": ["confident", "helpful", "encouraging"]
            },
            "motivational": {
                "energy": "high",
                "pace": "building",
                "emotions": ["inspired", "determined", "powerful"]
            },
            "conversational": {
                "energy": "relaxed",
                "pace": "natural",
                "emotions": ["friendly", "casual", "warm"]
            }
        }
        
        return profiles.get(style, profiles["conversational"])
    
    def _create_llm_instructions(self, prompt: Dict) -> str:
        """
        Create instructions for LLM
        
        Args:
            prompt: Prompt dictionary
            
        Returns:
            LLM instruction string
        """
        instructions = f"""Create a {prompt['duration']}-second video script about "{prompt['topic']}" in a {prompt['style']} style.

Requirements:
- Target length: {prompt['target_words']} words
- Tone: {prompt['tone']}
- Speech pattern: {prompt['speech_pattern']}
- Include natural speech elements like "you know", "let's see", "actually"
- Add emotion cues in brackets like [smile], [pause], [excited]
- Add gesture cues like [hand raise], [nod], [point]

Structure:
"""
        
        for section in prompt['sections']:
            instructions += f"\n{section['order']}. {section['type'].title()} ({section['words']} words): {section['content_hints']}"
        
        instructions += f"""

Keywords to emphasize: {', '.join(prompt['keywords'])}

Make it conversational and engaging for Instagram Reels. The script should feel natural and human-like, not robotic."""
        
        return instructions
    
    def refine_prompt(self, prompt: Dict, feedback: str) -> Dict:
        """
        Refine prompt based on feedback
        
        Args:
            prompt: Original prompt
            feedback: User feedback
            
        Returns:
            Refined prompt
        """
        logger.info("Refining prompt based on feedback")
        
        # Add feedback to instructions
        prompt["llm_instructions"] += f"\n\nAdditional requirements based on feedback: {feedback}"
        
        # Adjust parameters based on common feedback patterns
        if "shorter" in feedback.lower():
            prompt["target_words"] = int(prompt["target_words"] * 0.8)
        elif "longer" in feedback.lower():
            prompt["target_words"] = int(prompt["target_words"] * 1.2)
        
        if "more exciting" in feedback.lower():
            prompt["emotion_profile"]["energy"] = "high"
        elif "calmer" in feedback.lower():
            prompt["emotion_profile"]["energy"] = "low"
        
        return prompt
