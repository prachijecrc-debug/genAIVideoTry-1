"""
Voice generator for text-to-speech synthesis
"""

import os
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from loguru import logger


class VoiceGenerator:
    """Generate natural-sounding voice from text"""
    
    def __init__(self, config):
        """
        Initialize voice generator
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.engine = config.get("tts.engine", "edge-tts")
        self.sample_rate = config.get("tts.sample_rate", 24000)
        self.temp_dir = Path(config.get("paths.temp", "temp"))
        self.temp_dir.mkdir(exist_ok=True)
        self.output_dir = self.temp_dir / "voice"
        self.output_dir.mkdir(exist_ok=True)
        
        # Voice profiles for Edge TTS
        self.voice_profiles = {
            "default": "en-US-AriaNeural",
            "female": "en-US-JennyNeural",
            "male": "en-US-GuyNeural",
            "energetic": "en-US-JennyMultilingualNeural",
            "calm": "en-US-EricNeural",
            "british_female": "en-GB-SoniaNeural",
            "british_male": "en-GB-RyanNeural",
            "young_female": "en-US-AnaNeural",
            "mature_male": "en-US-ChristopherNeural"
        }
        
        logger.info(f"Voice generator initialized with {self.engine} engine")
    
    def synthesize(self, text: str, voice_profile: str = "default",
                  emotion_cues: Optional[List[Dict]] = None) -> str:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            voice_profile: Voice profile to use
            emotion_cues: Optional emotion cues for expressive speech
            
        Returns:
            Path to generated audio file
        """
        logger.info(f"Synthesizing speech with {voice_profile} profile")
        
        # Select synthesis method
        if self.engine == "edge-tts":
            audio_path = self._synthesize_with_edge_tts(text, voice_profile, emotion_cues)
        elif self.engine == "bark":
            audio_path = self._synthesize_with_bark(text, voice_profile, emotion_cues)
        elif self.engine == "xtts":
            audio_path = self._synthesize_with_xtts(text, voice_profile, emotion_cues)
        else:
            audio_path = self._synthesize_fallback(text, voice_profile)
        
        return audio_path
    
    def _synthesize_with_edge_tts(self, text: str, voice_profile: str,
                                  emotion_cues: Optional[List[Dict]] = None) -> str:
        """
        Synthesize using Edge TTS (Microsoft neural voices)
        
        Args:
            text: Text to synthesize
            voice_profile: Voice profile to use
            emotion_cues: Optional emotion cues
            
        Returns:
            Path to audio file
        """
        try:
            import edge_tts
            
            # Get voice for profile
            voice = self.voice_profiles.get(voice_profile, self.voice_profiles["default"])
            
            # Process text with emotion cues if provided
            if emotion_cues:
                text = self._add_ssml_emotions(text, emotion_cues)
            
            # Output path
            output_path = self.output_dir / f"voice_{os.urandom(4).hex()}.mp3"
            
            # Create async function for Edge TTS
            async def generate():
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(str(output_path))
            
            # Run async function
            asyncio.run(generate())
            
            logger.info(f"Edge TTS synthesis complete: {output_path}")
            return str(output_path)
            
        except ImportError:
            logger.warning("edge-tts not installed, using fallback")
            return self._synthesize_fallback(text, voice_profile)
        except Exception as e:
            logger.error(f"Error with Edge TTS synthesis: {str(e)}")
            return self._synthesize_fallback(text, voice_profile)
    
    def _synthesize_with_bark(self, text: str, voice_profile: str,
                             emotion_cues: Optional[List[Dict]] = None) -> str:
        """
        Synthesize using Bark
        
        Args:
            text: Text to synthesize
            voice_profile: Voice profile to use
            emotion_cues: Optional emotion cues
            
        Returns:
            Path to audio file
        """
        try:
            from bark import generate_audio, SAMPLE_RATE
            from scipy.io.wavfile import write as write_wav
            
            # Bark voice presets
            bark_voices = {
                "default": "v2/en_speaker_0",
                "female": "v2/en_speaker_9",
                "male": "v2/en_speaker_1",
                "energetic": "v2/en_speaker_3",
                "calm": "v2/en_speaker_6"
            }
            
            voice_preset = bark_voices.get(voice_profile, bark_voices["default"])
            
            # Add emotion markers if provided
            if emotion_cues:
                text = self._add_bark_emotions(text, emotion_cues)
            
            # Split text into chunks for better quality
            chunks = self._split_text(text, max_length=400)
            audio_arrays = []
            
            for chunk in chunks:
                # Generate audio for chunk
                audio_array = generate_audio(
                    chunk,
                    history_prompt=voice_preset
                )
                audio_arrays.append(audio_array)
            
            # Concatenate audio chunks
            if len(audio_arrays) > 1:
                audio_array = np.concatenate(audio_arrays)
            else:
                audio_array = audio_arrays[0]
            
            # Save audio
            output_path = self.output_dir / f"voice_{os.urandom(4).hex()}.wav"
            write_wav(str(output_path), SAMPLE_RATE, audio_array)
            
            logger.info(f"Bark synthesis complete: {output_path}")
            return str(output_path)
            
        except ImportError:
            logger.warning("Bark not installed, using Edge TTS")
            return self._synthesize_with_edge_tts(text, voice_profile, emotion_cues)
        except Exception as e:
            logger.error(f"Error with Bark synthesis: {str(e)}")
            return self._synthesize_with_edge_tts(text, voice_profile, emotion_cues)
    
    def _synthesize_with_xtts(self, text: str, voice_profile: str,
                             emotion_cues: Optional[List[Dict]] = None) -> str:
        """
        Synthesize using XTTS (Coqui TTS)
        
        Args:
            text: Text to synthesize
            voice_profile: Voice profile to use
            emotion_cues: Optional emotion cues
            
        Returns:
            Path to audio file
        """
        try:
            from TTS.api import TTS
            
            # Initialize TTS
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
            
            # Output path
            output_path = self.output_dir / f"voice_{os.urandom(4).hex()}.wav"
            
            # Generate speech
            tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker="default",
                language="en"
            )
            
            logger.info(f"XTTS synthesis complete: {output_path}")
            return str(output_path)
            
        except ImportError:
            logger.warning("XTTS not installed, using Edge TTS")
            return self._synthesize_with_edge_tts(text, voice_profile, emotion_cues)
        except Exception as e:
            logger.error(f"Error with XTTS synthesis: {str(e)}")
            return self._synthesize_with_edge_tts(text, voice_profile, emotion_cues)
    
    def _synthesize_fallback(self, text: str, voice_profile: str) -> str:
        """
        Fallback synthesis using basic TTS
        
        Args:
            text: Text to synthesize
            voice_profile: Voice profile to use
            
        Returns:
            Path to audio file
        """
        logger.info("Using fallback TTS synthesis")
        
        try:
            import pyttsx3
            
            # Initialize engine
            engine = pyttsx3.init()
            
            # Set properties
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            
            # Get available voices
            voices = engine.getProperty('voices')
            if voices:
                # Try to select appropriate voice
                if "female" in voice_profile:
                    for voice in voices:
                        if "female" in voice.name.lower():
                            engine.setProperty('voice', voice.id)
                            break
                elif "male" in voice_profile:
                    for voice in voices:
                        if "male" in voice.name.lower():
                            engine.setProperty('voice', voice.id)
                            break
            
            # Save to file
            output_path = self.output_dir / f"voice_{os.urandom(4).hex()}.wav"
            engine.save_to_file(text, str(output_path))
            engine.runAndWait()
            
            logger.info(f"Fallback synthesis complete: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Fallback TTS failed: {str(e)}")
            
            # Create a silent audio file as last resort
            output_path = self.output_dir / f"voice_{os.urandom(4).hex()}.wav"
            self._create_silent_audio(output_path, duration=10)
            return str(output_path)
    
    def _add_ssml_emotions(self, text: str, emotion_cues: List[Dict]) -> str:
        """
        Add SSML emotion markers for Edge TTS
        
        Args:
            text: Original text
            emotion_cues: List of emotion cues
            
        Returns:
            Text with SSML markers
        """
        # Edge TTS supports SSML for prosody control
        ssml_parts = ['<speak>']
        
        words = text.split()
        current_idx = 0
        
        for cue in sorted(emotion_cues, key=lambda x: x.get('position', 0)):
            position = cue.get('position', 0)
            emotion = cue.get('emotion', '').lower()
            
            # Add text before emotion
            if position > current_idx:
                ssml_parts.append(' '.join(words[current_idx:position]))
            
            # Add emotion prosody
            if emotion == 'happy':
                ssml_parts.append('<prosody pitch="+10%" rate="110%">')
            elif emotion == 'sad':
                ssml_parts.append('<prosody pitch="-10%" rate="90%">')
            elif emotion == 'excited':
                ssml_parts.append('<prosody pitch="+15%" rate="120%">')
            elif emotion == 'calm':
                ssml_parts.append('<prosody pitch="-5%" rate="85%">')
            elif emotion == 'emphasis':
                ssml_parts.append('<emphasis level="strong">')
            
            # Add the word at position
            if position < len(words):
                ssml_parts.append(words[position])
                current_idx = position + 1
            
            # Close prosody tag
            if emotion in ['happy', 'sad', 'excited', 'calm']:
                ssml_parts.append('</prosody>')
            elif emotion == 'emphasis':
                ssml_parts.append('</emphasis>')
        
        # Add remaining text
        if current_idx < len(words):
            ssml_parts.append(' '.join(words[current_idx:]))
        
        ssml_parts.append('</speak>')
        
        return ' '.join(ssml_parts)
    
    def _add_bark_emotions(self, text: str, emotion_cues: List[Dict]) -> str:
        """
        Add Bark-specific emotion markers to text
        
        Args:
            text: Original text
            emotion_cues: List of emotion cues
            
        Returns:
            Text with emotion markers
        """
        # Bark emotion mappings
        emotion_map = {
            "happy": "[laughs]",
            "sad": "[sighs]",
            "excited": "[gasps]",
            "thoughtful": "[clears throat]",
            "laugh": "[laughter]",
            "pause": "...",
            "emphasis": "♪"
        }
        
        words = text.split()
        
        for cue in sorted(emotion_cues, key=lambda x: x.get('position', 0), reverse=True):
            position = cue.get("position", 0)
            emotion = cue.get("emotion", "")
            
            # Find matching emotion marker
            for key, marker in emotion_map.items():
                if key in emotion.lower():
                    # Insert marker at position
                    if position < len(words):
                        words.insert(position, marker)
                    break
        
        return " ".join(words)
    
    def _split_text(self, text: str, max_length: int = 400) -> List[str]:
        """
        Split text into chunks for processing
        
        Args:
            text: Text to split
            max_length: Maximum chunk length
            
        Returns:
            List of text chunks
        """
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > max_length and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        return chunks
    
    def _create_silent_audio(self, output_path: Path, duration: float = 10.0):
        """
        Create a silent audio file
        
        Args:
            output_path: Path to save audio
            duration: Duration in seconds
        """
        try:
            import wave
            
            # Generate silent audio
            sample_rate = self.sample_rate
            num_samples = int(duration * sample_rate)
            silence = np.zeros(num_samples, dtype=np.int16)
            
            # Write to WAV file
            with wave.open(str(output_path), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(silence.tobytes())
            
            logger.info(f"Created silent audio: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create silent audio: {str(e)}")
    
    def list_available_voices(self) -> List[str]:
        """
        List available voice profiles
        
        Returns:
            List of voice profile names
        """
        return list(self.voice_profiles.keys())
