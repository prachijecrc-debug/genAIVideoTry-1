"""
Main pipeline for video generation
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from .prompt_generator import PromptGenerator
from .script_writer import ScriptWriter
from .voice_generator import VoiceGenerator
from .avatar_generator import AvatarGenerator
from .visual_composer import VisualComposer
from .video_exporter import VideoExporter


class VideoGenerator:
    """Main video generation pipeline"""
    
    def __init__(self, config):
        """
        Initialize video generator
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.temp_dir = Path(config.get("paths.temp", "temp"))
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.prompt_generator = PromptGenerator(config)
        self.script_writer = ScriptWriter(config)
        self.voice_generator = VoiceGenerator(config)
        self.avatar_generator = AvatarGenerator(config)
        self.visual_composer = VisualComposer(config)
        self.video_exporter = VideoExporter(config)
        
        logger.info("Video generator initialized")
        
    def generate_script(self, topic: str, style: str = "conversational", 
                       duration: int = 30) -> Dict[str, Any]:
        """
        Generate script from topic
        
        Args:
            topic: Video topic
            style: Video style
            duration: Target duration in seconds
            
        Returns:
            Script dictionary with dialogue, cues, and metadata
        """
        logger.info(f"Generating script for topic: {topic}")
        
        # Generate prompt
        prompt = self.prompt_generator.generate(
            topic=topic,
            style=style,
            duration=duration
        )
        
        # Write script
        script = self.script_writer.write(
            prompt=prompt,
            style=style,
            duration=duration
        )
        
        return script
    
    def generate_voice(self, script: Dict[str, Any], 
                      voice_profile: str = "default") -> str:
        """
        Generate voice audio from script
        
        Args:
            script: Script dictionary
            voice_profile: Voice profile to use
            
        Returns:
            Path to generated audio file
        """
        logger.info(f"Generating voice with profile: {voice_profile}")
        
        audio_path = self.voice_generator.synthesize(
            text=script["dialogue"],
            voice_profile=voice_profile,
            emotion_cues=script.get("emotion_cues", [])
        )
        
        return audio_path
    
    def generate_avatar(self, audio_path: str, avatar_name: str = "default",
                       script: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate avatar video from audio
        
        Args:
            audio_path: Path to audio file
            avatar_name: Avatar to use
            script: Optional script for gesture cues
            
        Returns:
            Path to avatar video
        """
        logger.info(f"Generating avatar: {avatar_name}")
        
        avatar_video = self.avatar_generator.generate(
            audio_path=audio_path,
            avatar_name=avatar_name,
            gesture_cues=script.get("gesture_cues", []) if script else []
        )
        
        return avatar_video
    
    def add_visuals(self, video_path: str, script: Dict[str, Any]) -> str:
        """
        Add visual elements to video
        
        Args:
            video_path: Path to video file
            script: Script for context
            
        Returns:
            Path to video with visuals
        """
        logger.info("Adding visual elements")
        
        enhanced_video = self.visual_composer.compose(
            video_path=video_path,
            script=script,
            add_background=True,
            add_effects=True
        )
        
        return enhanced_video
    
    def add_captions(self, video_path: str, audio_path: str) -> str:
        """
        Add captions to video
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file for transcription
            
        Returns:
            Path to video with captions
        """
        logger.info("Adding captions")
        
        captioned_video = self.visual_composer.add_subtitles(
            video_path=video_path,
            audio_path=audio_path
        )
        
        return captioned_video
    
    def export_video(self, video_path: str, output_path: str, 
                    format: str = "instagram") -> str:
        """
        Export final video in desired format
        
        Args:
            video_path: Path to video file
            output_path: Output path
            format: Export format preset
            
        Returns:
            Path to exported video
        """
        logger.info(f"Exporting video: {output_path}")
        
        final_video = self.video_exporter.export(
            video_path=video_path,
            output_path=output_path,
            format_preset=format
        )
        
        return final_video
    
    def create_video(self, topic: str, style: str = "conversational",
                    duration: int = 30, voice: str = "default",
                    avatar: str = "default", output_path: Optional[str] = None,
                    add_captions: bool = True, add_visuals: bool = True) -> Dict[str, Any]:
        """
        Complete video creation pipeline
        
        Args:
            topic: Video topic
            style: Video style
            duration: Duration in seconds
            voice: Voice profile
            avatar: Avatar name
            output_path: Output path
            add_captions: Whether to add captions
            add_visuals: Whether to add visual effects
            
        Returns:
            Dictionary with video path and metadata
        """
        try:
            # Generate script
            script = self.generate_script(topic, style, duration)
            
            # Generate voice
            audio_path = self.generate_voice(script, voice)
            
            # Generate avatar
            avatar_video = self.generate_avatar(audio_path, avatar, script)
            
            # Add visuals
            if add_visuals:
                video_with_visuals = self.add_visuals(avatar_video, script)
            else:
                video_with_visuals = avatar_video
            
            # Add captions
            if add_captions:
                video_with_captions = self.add_captions(video_with_visuals, audio_path)
            else:
                video_with_captions = video_with_visuals
            
            # Export final video
            if not output_path:
                topic_slug = topic.lower().replace(" ", "_")[:30]
                output_path = f"output/{topic_slug}_{style}_{duration}s.mp4"
            
            final_video = self.export_video(video_with_captions, output_path, "instagram")
            
            return {
                "video_path": final_video,
                "script": script,
                "audio_path": audio_path,
                "duration": duration,
                "style": style,
                "topic": topic
            }
            
        except Exception as e:
            logger.error(f"Error in video creation pipeline: {str(e)}")
            raise
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir(exist_ok=True)
            logger.info("Cleaned up temporary files")
