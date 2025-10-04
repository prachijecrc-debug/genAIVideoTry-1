"""
Script writer for generating conversational dialogue
"""

import re
import json
from typing import Dict, Any, List, Optional
from loguru import logger


class ScriptWriter:
    """Generate conversational scripts from prompts"""
    
    def __init__(self, config):
        """
        Initialize script writer
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.llm_provider = config.get("llm.provider", "ollama")
        self.model = config.get("llm.model", "llama3")
        
        # Dialogue patterns for natural speech
        self.fillers = [
            "you know", "actually", "basically", "like", "I mean",
            "let's see", "so", "well", "right", "okay"
        ]
        
        self.transitions = [
            "Now,", "So,", "Alright,", "Moving on,", "Next up,",
            "Here's the thing:", "But wait,", "And get this:",
            "The cool part is,", "What's interesting is,"
        ]
        
        logger.info("Script writer initialized")
    
    def write(self, prompt: Dict[str, Any], style: str = "conversational",
             duration: int = 30) -> Dict[str, Any]:
        """
        Write script from prompt
        
        Args:
            prompt: Structured prompt dictionary
            style: Writing style
            duration: Target duration
            
        Returns:
            Script dictionary with dialogue and cues
        """
        logger.info(f"Writing script for: {prompt['topic']}")
        
        # Generate dialogue based on prompt
        if self.llm_provider == "ollama":
            dialogue = self._generate_with_ollama(prompt)
        elif self.llm_provider == "openai":
            dialogue = self._generate_with_openai(prompt)
        else:
            dialogue = self._generate_fallback(prompt)
        
        # Process and enhance dialogue
        enhanced_dialogue = self._enhance_dialogue(dialogue, style)
        
        # Extract cues
        emotion_cues = self._extract_emotion_cues(enhanced_dialogue)
        gesture_cues = self._extract_gesture_cues(enhanced_dialogue)
        
        # Clean dialogue for TTS
        clean_dialogue = self._clean_for_tts(enhanced_dialogue)
        
        script = {
            "topic": prompt["topic"],
            "style": style,
            "duration": duration,
            "dialogue": clean_dialogue,
            "dialogue_with_cues": enhanced_dialogue,
            "emotion_cues": emotion_cues,
            "gesture_cues": gesture_cues,
            "sections": self._segment_dialogue(enhanced_dialogue, prompt["sections"]),
            "word_count": len(clean_dialogue.split()),
            "estimated_duration": len(clean_dialogue.split()) / 2.5  # words per second
        }
        
        return script
    
    def _generate_with_ollama(self, prompt: Dict) -> str:
        """
        Generate dialogue using Ollama
        
        Args:
            prompt: Prompt dictionary
            
        Returns:
            Generated dialogue
        """
        try:
            import requests
            
            response = requests.post(
                f"{self.config.get('llm.base_url')}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt["llm_instructions"],
                    "stream": False,
                    "options": {
                        "temperature": self.config.get("llm.temperature", 0.7),
                        "num_predict": self.config.get("llm.max_tokens", 2000)
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.warning(f"Ollama API error: {response.status_code}")
                return self._generate_fallback(prompt)
                
        except Exception as e:
            logger.warning(f"Error with Ollama: {str(e)}")
            return self._generate_fallback(prompt)
    
    def _generate_with_openai(self, prompt: Dict) -> str:
        """
        Generate dialogue using OpenAI
        
        Args:
            prompt: Prompt dictionary
            
        Returns:
            Generated dialogue
        """
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.config.get("llm.api_key"))
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a creative script writer for Instagram Reels."},
                    {"role": "user", "content": prompt["llm_instructions"]}
                ],
                temperature=self.config.get("llm.temperature", 0.7),
                max_tokens=self.config.get("llm.max_tokens", 2000)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"Error with OpenAI: {str(e)}")
            return self._generate_fallback(prompt)
    
    def _generate_fallback(self, prompt: Dict) -> str:
        """
        Generate fallback script when LLM is unavailable
        
        Args:
            prompt: Prompt dictionary
            
        Returns:
            Fallback dialogue
        """
        logger.info("Using fallback script generation")
        
        topic = prompt["topic"]
        style = prompt["style"]
        sections = prompt["sections"]
        
        dialogue = []
        
        # Generate content for each section
        for section in sections:
            section_type = section["type"]
            
            if section_type == "greeting":
                dialogue.append(f"[wave] Hey everyone! So today, we're talking about {topic}. [smile]")
            
            elif section_type == "introduction":
                dialogue.append(f"[nod] Let me introduce you to {topic}. It's actually pretty fascinating when you think about it.")
            
            elif section_type == "hook":
                dialogue.append(f"[excited] Did you know that {topic} is way more interesting than you might think? [pause] Let me show you why.")
            
            elif section_type == "main points" or section_type == "main concepts":
                dialogue.append(f"[gesture] So, here's the thing about {topic}. There are a few key points you need to know.")
                dialogue.append(f"First off, [point] it's important to understand the basics.")
                dialogue.append(f"And then, [hand raise] there's this really cool aspect that most people don't know about.")
            
            elif section_type == "examples":
                dialogue.append(f"[thoughtful] Let me give you a real example. You know how sometimes you see {topic} in everyday life?")
                dialogue.append(f"Well, [smile] it's actually happening all around us!")
            
            elif section_type == "personal touch":
                dialogue.append(f"[casual] Personally, I find {topic} really fascinating because, you know, it affects us all in some way.")
            
            elif section_type == "summary":
                dialogue.append(f"[nod] So to wrap up, {topic} is definitely something worth understanding better.")
            
            elif section_type == "call-to-action":
                dialogue.append(f"[enthusiastic] If you found this interesting, make sure to follow for more content like this!")
                dialogue.append(f"[wave] And drop a comment below with your thoughts on {topic}!")
            
            elif section_type == "farewell":
                dialogue.append(f"[smile] Thanks for watching, everyone! [wave] See you in the next one!")
            
            else:
                dialogue.append(f"[gesture] Now about {topic}, there's something interesting here.")
        
        return " ".join(dialogue)
    
    def _enhance_dialogue(self, dialogue: str, style: str) -> str:
        """
        Enhance dialogue with natural speech patterns
        
        Args:
            dialogue: Raw dialogue
            style: Speaking style
            
        Returns:
            Enhanced dialogue
        """
        # Add natural pauses
        dialogue = re.sub(r'([.!?])', r'\1 [pause]', dialogue)
        
        # Add occasional fillers based on style
        if style in ["conversational", "casual"]:
            sentences = dialogue.split('. ')
            enhanced = []
            for i, sentence in enumerate(sentences):
                if i > 0 and i % 3 == 0:  # Add filler every 3rd sentence
                    filler = self.fillers[i % len(self.fillers)]
                    sentence = f"{filler}, {sentence}"
                enhanced.append(sentence)
            dialogue = '. '.join(enhanced)
        
        # Add transitions
        paragraphs = dialogue.split('\n\n')
        if len(paragraphs) > 1:
            enhanced_paragraphs = [paragraphs[0]]
            for i, para in enumerate(paragraphs[1:], 1):
                transition = self.transitions[i % len(self.transitions)]
                enhanced_paragraphs.append(f"{transition} {para}")
            dialogue = '\n\n'.join(enhanced_paragraphs)
        
        return dialogue
    
    def _extract_emotion_cues(self, dialogue: str) -> List[Dict]:
        """
        Extract emotion cues from dialogue
        
        Args:
            dialogue: Dialogue with cues
            
        Returns:
            List of emotion cue dictionaries
        """
        cues = []
        pattern = r'\[([^\]]+)\]'
        
        emotions = ["smile", "laugh", "excited", "thoughtful", "serious", 
                   "happy", "curious", "surprised", "confident", "warm"]
        
        matches = re.finditer(pattern, dialogue)
        for match in matches:
            cue_text = match.group(1).lower()
            if any(emotion in cue_text for emotion in emotions):
                position = match.start()
                word_position = len(dialogue[:position].split())
                cues.append({
                    "emotion": cue_text,
                    "position": word_position,
                    "timestamp": word_position / 2.5  # Approximate timestamp
                })
        
        return cues
    
    def _extract_gesture_cues(self, dialogue: str) -> List[Dict]:
        """
        Extract gesture cues from dialogue
        
        Args:
            dialogue: Dialogue with cues
            
        Returns:
            List of gesture cue dictionaries
        """
        cues = []
        pattern = r'\[([^\]]+)\]'
        
        gestures = ["wave", "nod", "point", "hand", "gesture", "shrug",
                   "thumbs up", "clap", "raise", "lean"]
        
        matches = re.finditer(pattern, dialogue)
        for match in matches:
            cue_text = match.group(1).lower()
            if any(gesture in cue_text for gesture in gestures):
                position = match.start()
                word_position = len(dialogue[:position].split())
                cues.append({
                    "gesture": cue_text,
                    "position": word_position,
                    "timestamp": word_position / 2.5  # Approximate timestamp
                })
        
        return cues
    
    def _clean_for_tts(self, dialogue: str) -> str:
        """
        Clean dialogue for TTS processing
        
        Args:
            dialogue: Dialogue with cues
            
        Returns:
            Clean dialogue text
        """
        # Remove all cues in brackets
        clean = re.sub(r'\[([^\]]+)\]', '', dialogue)
        
        # Clean up extra spaces
        clean = re.sub(r'\s+', ' ', clean)
        
        # Remove leading/trailing whitespace
        clean = clean.strip()
        
        return clean
    
    def _segment_dialogue(self, dialogue: str, sections: List[Dict]) -> List[Dict]:
        """
        Segment dialogue into sections
        
        Args:
            dialogue: Full dialogue
            sections: Section definitions
            
        Returns:
            List of dialogue segments
        """
        # Split dialogue into roughly equal parts based on sections
        words = dialogue.split()
        total_words = len(words)
        
        segments = []
        start_idx = 0
        
        for section in sections:
            section_words = section["words"]
            end_idx = min(start_idx + section_words, total_words)
            
            segment_text = ' '.join(words[start_idx:end_idx])
            
            segments.append({
                "type": section["type"],
                "text": segment_text,
                "start_word": start_idx,
                "end_word": end_idx,
                "duration": section["duration"]
            })
            
            start_idx = end_idx
        
        return segments
    
    def refine_script(self, script: Dict, feedback: str) -> Dict:
        """
        Refine script based on feedback
        
        Args:
            script: Original script
            feedback: User feedback
            
        Returns:
            Refined script
        """
        logger.info("Refining script based on feedback")
        
        dialogue = script["dialogue_with_cues"]
        
        # Apply feedback-based modifications
        if "more energy" in feedback.lower():
            dialogue = dialogue.replace("[pause]", "[excited]")
            dialogue = dialogue.replace(".", "! ")
        
        if "slower" in feedback.lower():
            dialogue = re.sub(r'([.!?])', r'\1 [pause] [pause]', dialogue)
        
        if "more casual" in feedback.lower():
            # Add more fillers
            sentences = dialogue.split('. ')
            enhanced = []
            for sentence in sentences:
                if len(sentence) > 20:  # Add filler to longer sentences
                    filler = self.fillers[len(sentence) % len(self.fillers)]
                    sentence = f"{filler}, {sentence}"
                enhanced.append(sentence)
            dialogue = '. '.join(enhanced)
        
        # Rebuild script
        script["dialogue_with_cues"] = dialogue
        script["dialogue"] = self._clean_for_tts(dialogue)
        script["emotion_cues"] = self._extract_emotion_cues(dialogue)
        script["gesture_cues"] = self._extract_gesture_cues(dialogue)
        
        return script
