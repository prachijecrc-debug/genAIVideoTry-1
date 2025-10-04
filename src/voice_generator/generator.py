"""
Voice generator for text-to-speech synthesis
"""

import os
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
        self.engine = config.get("tts.engine", "bark")
        self.sample_rate = config.get("tts.sample_rate", 24000)
        self.temp_dir = Path(config.get("paths.temp", "temp"))
        self.temp_dir.mkdir(exist_ok=True)
        
        # Voice profiles
        self.voice_profiles = {
            "default": {
                "bark": "v2/en_speaker_0",
                "xtts": "default",
                "speed": 1.0,
                "pitch": 1.0
            },
            "female": {
                "bark": "v2/en_speaker_9",
                "xtts": "female_1",
                "speed": 1.0,
                "pitch": 1.1
            },
            "male": {
                "bark": "v2/en_speaker_1",
                "xtts": "male_1",
                "speed": 0.95,
                "pitch": 0.9
            },
            "energetic": {
                "bark": "v2/en_speaker_3",
                "xtts": "energetic",
                "speed": 1.1,
                "pitch": 1.05
            },
            "calm": {
                "bark": "v2/en_speaker_6",
                "xtts": "calm",
                "speed": 0.9,
                "pitch": 0.95
            }
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
        
        # Get voice settings
        profile = self.voice_profiles.get(voice_profile, self.voice_profiles["default"])
        
        # Select synthesis method
        if self.engine == "bark":
            audio_path = self._synthesize_with_bark(text, profile, emotion_cues)
        elif self.engine == "xtts":
            audio_path = self._synthesize_with_xtts(text, profile, emotion_cues)
        else:
            audio_path = self._synthesize_fallback(text, profile)
        
        # Post-process audio
        audio_path = self._post_process_audio(audio_path, profile)
        
        return audio_path
    
    def _synthesize_with_bark(self, text: str, profile: Dict, 
                             emotion_cues: Optional[List[Dict]] = None) -> str:
        """
        Synthesize using Bark
        
        Args:
            text: Text to synthesize
            profile: Voice profile settings
            emotion_cues: Optional emotion cues
            
        Returns:
            Path to audio file
        """
        try:
            from bark import SAMPLE_RATE, generate_audio, preload_models
            from scipy.io.wavfile import write as write_wav
            
            # Preload models
            preload_models()
            
            # Add emotion markers if provided
            if emotion_cues:
                text = self._add_bark_emotions(text, emotion_cues)
            
            # Generate audio
            voice_preset = profile["bark"]
            
            # Split text into chunks for better quality
            chunks = self._split_text(text, max_length=400)
            audio_arrays = []
            
            for chunk in chunks:
                # Add voice preset to text
                prompted_text = f"[{voice_preset}] {chunk}"
                
                # Generate audio for chunk
                audio_array = generate_audio(
                    prompted_text,
                    history_prompt=voice_preset
                )
                audio_arrays.append(audio_array)
            
            # Concatenate audio chunks
            if len(audio_arrays) > 1:
                audio_array = np.concatenate(audio_arrays)
            else:
                audio_array = audio_arrays[0]
            
            # Save audio
            output_path = self.temp_dir / f"voice_{os.getpid()}.wav"
            write_wav(str(output_path), SAMPLE_RATE, audio_array)
            
            logger.info(f"Bark synthesis complete: {output_path}")
            return str(output_path)
            
        except ImportError:
            logger.warning("Bark not installed, using fallback")
            return self._synthesize_fallback(text, profile)
        except Exception as e:
            logger.error(f"Error with Bark synthesis: {str(e)}")
            return self._synthesize_fallback(text, profile)
    
    def _synthesize_with_xtts(self, text: str, profile: Dict,
                             emotion_cues: Optional[List[Dict]] = None) -> str:
        """
        Synthesize using XTTS (Coqui TTS)
        
        Args:
            text: Text to synthesize
            profile: Voice profile settings
            emotion_cues: Optional emotion cues
            
        Returns:
            Path to audio file
        """
        try:
            from TTS.api import TTS
            
            # Initialize TTS
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
            
            # Output path
            output_path = self.temp_dir / f"voice_{os.getpid()}.wav"
            
            # Generate speech
            tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker="default",
                language="en",
                speed=profile.get("speed", 1.0)
            )
            
            logger.info(f"XTTS synthesis complete: {output_path}")
            return str(output_path)
            
        except ImportError:
            logger.warning("XTTS not installed, using fallback")
            return self._synthesize_fallback(text, profile)
        except Exception as e:
            logger.error(f"Error with XTTS synthesis: {str(e)}")
            return self._synthesize_fallback(text, profile)
    
    def _synthesize_fallback(self, text: str, profile: Dict) -> str:
        """
        Fallback synthesis using basic TTS
        
        Args:
            text: Text to synthesize
            profile: Voice profile settings
            
        Returns:
            Path to audio file
        """
        logger.info("Using fallback TTS synthesis")
        
        try:
            import pyttsx3
            
            # Initialize engine
            engine = pyttsx3.init()
            
            # Set properties
            engine.setProperty('rate', 150 * profile.get("speed", 1.0))
            engine.setProperty('volume', 1.0)
            
            # Get available voices
            voices = engine.getProperty('voices')
            if voices:
                # Try to select appropriate voice
                if "female" in str(profile):
                    for voice in voices:
                        if "female" in voice.name.lower():
                            engine.setProperty('voice', voice.id)
                            break
                elif "male" in str(profile):
                    for voice in voices:
                        if "male" in voice.name.lower():
                            engine.setProperty('voice', voice.id)
                            break
            
            # Save to file
            output_path = self.temp_dir / f"voice_{os.getpid()}.wav"
            engine.save_to_file(text, str(output_path))
            engine.runAndWait()
            
            logger.info(f"Fallback synthesis complete: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Fallback TTS failed: {str(e)}")
            
            # Create a silent audio file as last resort
            output_path = self.temp_dir / f"voice_{os.getpid()}.wav"
            self._create_silent_audio(output_path, duration=10)
            return str(output_path)
    
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
        
        for cue in emotion_cues:
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
    
    def _post_process_audio(self, audio_path: str, profile: Dict) -> str:
        """
        Post-process audio (adjust speed, pitch, add effects)
        
        Args:
            audio_path: Path to audio file
            profile: Voice profile settings
            
        Returns:
            Path to processed audio
        """
        try:
            from pydub import AudioSegment
            from pydub.effects import speedup
            
            # Load audio
            audio = AudioSegment.from_file(audio_path)
            
            # Adjust speed if needed
            speed = profile.get("speed", 1.0)
            if speed != 1.0:
                audio = speedup(audio, playback_speed=speed)
            
            # Add subtle reverb for natural sound
            # audio = audio.overlay(audio.apply_gain(-20), position=100)
            
            # Normalize audio
            audio = audio.normalize()
            
            # Save processed audio
            processed_path = audio_path.replace(".wav", "_processed.wav")
            audio.export(processed_path, format="wav")
            
            return processed_path
            
        except ImportError:
            logger.warning("pydub not available for post-processing")
            return audio_path
        except Exception as e:
            logger.warning(f"Audio post-processing failed: {str(e)}")
            return audio_path
    
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
    
    def create_voice_profile(self, name: str, settings: Dict) -> None:
        """
        Create a custom voice profile
        
        Args:
            name: Profile name
            settings: Profile settings
        """
        self.voice_profiles[name] = settings
        logger.info(f"Created voice profile: {name}")
    
    def list_available_voices(self) -> List[str]:
        """
        List available voice profiles
        
        Returns:
            List of voice profile names
        """
        return list(self.voice_profiles.keys())
