"""
Visual composer for adding backgrounds, effects, and subtitles using Whisper
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from loguru import logger


class VisualComposer:
    """Compose visuals with backgrounds, effects, and Whisper-generated subtitles"""
    
    def __init__(self, config):
        """
        Initialize visual composer
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.temp_dir = Path(config.get("paths.temp", "temp"))
        self.backgrounds_dir = Path(config.get("paths.backgrounds", "data/backgrounds"))
        self.resolution = config.get("video.resolution", [1080, 1920])
        self.fps = config.get("video.fps", 30)
        
        # Caption settings
        self.caption_settings = {
            "font": config.get("captions.font", "Arial"),
            "font_size": config.get("captions.font_size", 48),
            "color": config.get("captions.color", "white"),
            "stroke_color": config.get("captions.stroke_color", "black"),
            "stroke_width": config.get("captions.stroke_width", 2),
            "position": config.get("captions.position", "bottom")
        }
        
        # Create directories
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.backgrounds_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Visual composer initialized with Whisper for captions")
    
    def compose(self, video_path: str, script: Dict[str, Any],
               add_background: bool = True, add_effects: bool = True) -> str:
        """
        Compose video with visual enhancements
        
        Args:
            video_path: Path to input video
            script: Script dictionary for context
            add_background: Whether to add background
            add_effects: Whether to add effects
            
        Returns:
            Path to composed video
        """
        logger.info("Composing video with visual enhancements")
        
        output_path = video_path
        
        # Add background
        if add_background:
            output_path = self._add_background(output_path, script)
        
        # Add visual effects
        if add_effects:
            output_path = self._add_effects(output_path, script)
        
        # Add overlays (logos, watermarks, etc.)
        output_path = self._add_overlays(output_path)
        
        return output_path
    
    def add_subtitles(self, video_path: str, audio_path: str) -> str:
        """
        Add subtitles to video using Whisper for transcription
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file for transcription
            
        Returns:
            Path to video with subtitles
        """
        logger.info("Generating subtitles with Whisper")
        
        # Generate subtitles using Whisper
        subtitle_path = self._generate_subtitles_with_whisper(audio_path)
        
        # Burn subtitles into video using FFmpeg
        output_path = self._burn_subtitles_ffmpeg(video_path, subtitle_path)
        
        return output_path
    
    def _generate_subtitles_with_whisper(self, audio_path: str) -> str:
        """
        Generate subtitles using OpenAI Whisper
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Path to SRT subtitle file
        """
        try:
            import whisper
            
            # Load Whisper model
            model_size = self.config.get("captions.size", "base")
            logger.info(f"Loading Whisper model: {model_size}")
            model = whisper.load_model(model_size)
            
            # Transcribe audio
            result = model.transcribe(
                audio_path,
                language=self.config.get("captions.language", "en"),
                verbose=False
            )
            
            # Convert to SRT format
            srt_path = self.temp_dir / f"subtitles_{os.getpid()}.srt"
            self._write_srt(result["segments"], srt_path)
            
            logger.info(f"Generated subtitles: {srt_path}")
            return str(srt_path)
            
        except ImportError:
            logger.warning("Whisper not installed, using fallback")
            return self._generate_subtitles_fallback(audio_path)
        except Exception as e:
            logger.error(f"Whisper transcription failed: {str(e)}")
            return self._generate_subtitles_fallback(audio_path)
    
    def _write_srt(self, segments: List[Dict], output_path: Path) -> None:
        """
        Write segments to SRT file
        
        Args:
            segments: List of transcription segments from Whisper
            output_path: Path to save SRT file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                # Write subtitle number
                f.write(f"{i}\n")
                
                # Write timestamps
                start_time = self._seconds_to_srt_time(segment['start'])
                end_time = self._seconds_to_srt_time(segment['end'])
                f.write(f"{start_time} --> {end_time}\n")
                
                # Write text
                text = segment['text'].strip()
                f.write(f"{text}\n\n")
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """
        Convert seconds to SRT timestamp format
        
        Args:
            seconds: Time in seconds
            
        Returns:
            SRT timestamp string (HH:MM:SS,MMM)
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _burn_subtitles_ffmpeg(self, video_path: str, subtitle_path: str) -> str:
        """
        Burn subtitles into video using FFmpeg
        
        Args:
            video_path: Path to video
            subtitle_path: Path to SRT file
            
        Returns:
            Path to video with burned subtitles
        """
        try:
            import ffmpeg
            
            output_path = str(video_path).replace('.mp4', '_subtitled.mp4')
            
            # Build FFmpeg command with subtitle filter
            input_video = ffmpeg.input(video_path)
            
            # Create subtitle filter with custom styling
            subtitle_filter = f"subtitles={subtitle_path}:force_style='FontName={self.caption_settings['font']},FontSize={self.caption_settings['font_size']},PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,MarginV=50'"
            
            # Apply filter and output
            (
                ffmpeg
                .output(input_video, output_path, vf=subtitle_filter, codec='copy')
                .overwrite_output()
                .run(quiet=True)
            )
            
            logger.info(f"Subtitles burned into video: {output_path}")
            return output_path
            
        except Exception as e:
            logger.warning(f"FFmpeg subtitle burning failed: {str(e)}, using alternative method")
            return self._burn_subtitles_opencv(video_path, subtitle_path)
    
    def _burn_subtitles_opencv(self, video_path: str, subtitle_path: str) -> str:
        """
        Burn subtitles using OpenCV (fallback method)
        
        Args:
            video_path: Path to video
            subtitle_path: Path to SRT file
            
        Returns:
            Path to video with subtitles
        """
        # Parse SRT file
        subtitles = self._parse_srt(subtitle_path)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video
        output_path = str(video_path).replace('.mp4', '_subtitled.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate current time
            current_time = frame_count / fps
            
            # Find active subtitle
            for subtitle in subtitles:
                if subtitle['start'] <= current_time <= subtitle['end']:
                    # Add subtitle to frame
                    frame = self._add_text_to_frame(frame, subtitle['text'])
                    break
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        return output_path
    
    def _parse_srt(self, srt_path: str) -> List[Dict]:
        """
        Parse SRT file
        
        Args:
            srt_path: Path to SRT file
            
        Returns:
            List of subtitle dictionaries
        """
        import re
        
        subtitles = []
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newlines
        blocks = content.strip().split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                # Parse timestamp
                timestamp_line = lines[1]
                match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})', timestamp_line)
                if match:
                    start = int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3)) + int(match.group(4)) / 1000
                    end = int(match.group(5)) * 3600 + int(match.group(6)) * 60 + int(match.group(7)) + int(match.group(8)) / 1000
                    text = '\n'.join(lines[2:])
                    
                    subtitles.append({
                        'start': start,
                        'end': end,
                        'text': text
                    })
        
        return subtitles
    
    def _add_text_to_frame(self, frame: np.ndarray, text: str) -> np.ndarray:
        """
        Add text to frame
        
        Args:
            frame: Video frame
            text: Text to add
            
        Returns:
            Frame with text
        """
        height, width = frame.shape[:2]
        
        # Set font
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.caption_settings['font_size'] / 30
        thickness = 2
        
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate position
        if self.caption_settings['position'] == 'bottom':
            x = (width - text_width) // 2
            y = height - 100
        else:  # top
            x = (width - text_width) // 2
            y = 100
        
        # Draw text with stroke
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2)  # Black stroke
        cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)  # White text
        
        return frame
    
    def _generate_subtitles_fallback(self, audio_path: str) -> str:
        """
        Generate placeholder subtitles (fallback when Whisper unavailable)
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Path to SRT file
        """
        logger.info("Using fallback subtitle generation")
        
        # Create placeholder subtitles
        srt_path = self.temp_dir / f"subtitles_{os.getpid()}.srt"
        
        with open(srt_path, 'w', encoding='utf-8') as f:
            # Write sample subtitles
            f.write("1\n")
            f.write("00:00:00,000 --> 00:00:05,000\n")
            f.write("Welcome to our AI-generated video!\n\n")
            
            f.write("2\n")
            f.write("00:00:05,000 --> 00:00:10,000\n")
            f.write("This content was created automatically.\n\n")
            
            f.write("3\n")
            f.write("00:00:10,000 --> 00:00:15,000\n")
            f.write("Enjoy the presentation!\n\n")
        
        return str(srt_path)
    
    def _add_background(self, video_path: str, script: Dict) -> str:
        """
        Add background to video
        
        Args:
            video_path: Path to video
            script: Script context
            
        Returns:
            Path to video with background
        """
        logger.info("Adding background to video")
        
        # For now, return original video
        # In production, this would:
        # 1. Generate or fetch relevant background based on script keywords
        # 2. Composite avatar video over background
        # 3. Apply blur or other effects to background
        
        return video_path
    
    def _add_effects(self, video_path: str, script: Dict) -> str:
        """
        Add visual effects to video
        
        Args:
            video_path: Path to video
            script: Script context
            
        Returns:
            Path to video with effects
        """
        logger.info("Adding visual effects")
        
        # For now, return original video
        # In production, this would add:
        # - Zoom effects
        # - Pan effects
        # - Transitions
        # - Particle effects
        
        return video_path
    
    def _add_overlays(self, video_path: str) -> str:
        """
        Add overlays like logos, watermarks
        
        Args:
            video_path: Path to video
            
        Returns:
            Path to video with overlays
        """
        # For now, return original video
        return video_path
