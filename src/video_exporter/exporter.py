"""
Video exporter using FFmpeg for final composition and Instagram optimization
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class VideoExporter:
    """Export videos using FFmpeg for Instagram-ready format"""
    
    def __init__(self, config):
        """
        Initialize video exporter
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.resolution = config.get("video.resolution", [1080, 1920])
        self.fps = config.get("video.fps", 30)
        self.codec = config.get("video.codec", "h264")
        self.bitrate = config.get("video.bitrate", "5M")
        self.format = config.get("video.format", "mp4")
        self.output_dir = Path(config.get("paths.output", "output"))
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Format presets
        self.format_presets = {
            "instagram": {
                "resolution": [1080, 1920],  # 9:16 vertical
                "fps": 30,
                "codec": "h264",
                "bitrate": "5M",
                "max_size_mb": 100,
                "max_duration": 90,
                "audio_codec": "aac",
                "audio_bitrate": "128k"
            },
            "instagram_story": {
                "resolution": [1080, 1920],
                "fps": 30,
                "codec": "h264",
                "bitrate": "4M",
                "max_size_mb": 4,
                "max_duration": 15,
                "audio_codec": "aac",
                "audio_bitrate": "128k"
            },
            "instagram_feed": {
                "resolution": [1080, 1080],  # 1:1 square
                "fps": 30,
                "codec": "h264",
                "bitrate": "5M",
                "max_size_mb": 100,
                "max_duration": 60,
                "audio_codec": "aac",
                "audio_bitrate": "128k"
            },
            "tiktok": {
                "resolution": [1080, 1920],
                "fps": 30,
                "codec": "h264",
                "bitrate": "6M",
                "max_size_mb": 287,
                "max_duration": 60,
                "audio_codec": "aac",
                "audio_bitrate": "128k"
            },
            "youtube_shorts": {
                "resolution": [1080, 1920],
                "fps": 30,
                "codec": "h264",
                "bitrate": "8M",
                "max_size_mb": 500,
                "max_duration": 60,
                "audio_codec": "aac",
                "audio_bitrate": "192k"
            }
        }
        
        logger.info("Video exporter initialized with FFmpeg")
    
    def export(self, video_path: str, output_path: str, 
              format_preset: str = "instagram") -> str:
        """
        Export video in specified format using FFmpeg
        
        Args:
            video_path: Path to input video
            output_path: Path for output video
            format_preset: Format preset to use
            
        Returns:
            Path to exported video
        """
        logger.info(f"Exporting video with {format_preset} preset")
        
        # Get format settings
        preset = self.format_presets.get(format_preset, self.format_presets["instagram"])
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build FFmpeg command
        cmd = self._build_ffmpeg_command(video_path, str(output_path), preset)
        
        # Execute FFmpeg
        try:
            logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Video exported successfully: {output_path}")
            
            # Verify output file size
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            if file_size_mb > preset["max_size_mb"]:
                logger.warning(f"Output file ({file_size_mb:.2f} MB) exceeds max size ({preset['max_size_mb']} MB)")
                # Re-encode with lower bitrate
                output_path = self._compress_video(str(output_path), preset)
            
            return str(output_path)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e.stderr}")
            # Try with basic settings
            return self._export_fallback(video_path, str(output_path))
        except FileNotFoundError:
            logger.error("FFmpeg not found. Please install FFmpeg.")
            return video_path
    
    def _build_ffmpeg_command(self, input_path: str, output_path: str, 
                             preset: Dict) -> list:
        """
        Build FFmpeg command for video export
        
        Args:
            input_path: Input video path
            output_path: Output video path
            preset: Format preset settings
            
        Returns:
            FFmpeg command as list
        """
        width, height = preset["resolution"]
        
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-y",  # Overwrite output
            # Video settings
            "-c:v", preset["codec"],
            "-b:v", preset["bitrate"],
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black",
            "-r", str(preset["fps"]),
            # Audio settings
            "-c:a", preset["audio_codec"],
            "-b:a", preset["audio_bitrate"],
            "-ar", "44100",  # Audio sample rate
            # Format settings
            "-f", "mp4",
            "-movflags", "+faststart",  # Optimize for streaming
            # Duration limit
            "-t", str(preset["max_duration"]),
            # Output
            output_path
        ]
        
        return cmd
    
    def _compress_video(self, video_path: str, preset: Dict) -> str:
        """
        Compress video to meet size requirements
        
        Args:
            video_path: Path to video
            preset: Format preset
            
        Returns:
            Path to compressed video
        """
        logger.info("Compressing video to meet size requirements")
        
        compressed_path = video_path.replace('.mp4', '_compressed.mp4')
        
        # Calculate required bitrate
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        compression_ratio = preset["max_size_mb"] / file_size_mb
        new_bitrate = f"{int(5000 * compression_ratio)}k"
        
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-y",
            "-c:v", "h264",
            "-b:v", new_bitrate,
            "-c:a", "aac",
            "-b:a", "96k",
            compressed_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            logger.info(f"Video compressed: {compressed_path}")
            return compressed_path
        except Exception as e:
            logger.warning(f"Compression failed: {str(e)}")
            return video_path
    
    def _export_fallback(self, input_path: str, output_path: str) -> str:
        """
        Fallback export with basic settings
        
        Args:
            input_path: Input video path
            output_path: Output video path
            
        Returns:
            Path to exported video
        """
        logger.info("Using fallback export with basic settings")
        
        try:
            from moviepy.editor import VideoFileClip
            
            # Load video
            video = VideoFileClip(input_path)
            
            # Resize for Instagram
            if video.w / video.h != 9/16:
                video = video.resize(height=1920)
                # Pad to 9:16 aspect ratio
                video = video.on_color(
                    size=(1080, 1920),
                    color=(0, 0, 0),
                    pos='center'
                )
            
            # Export
            video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                fps=30,
                logger=None
            )
            
            video.close()
            logger.info(f"Fallback export complete: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Fallback export failed: {str(e)}")
            return input_path
    
    def create_thumbnail(self, video_path: str, time_position: float = 1.0) -> str:
        """
        Create thumbnail from video
        
        Args:
            video_path: Path to video
            time_position: Time position in seconds for thumbnail
            
        Returns:
            Path to thumbnail image
        """
        thumbnail_path = str(video_path).replace('.mp4', '_thumbnail.jpg')
        
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-ss", str(time_position),
            "-vframes", "1",
            "-vf", "scale=1080:1920",
            "-y",
            thumbnail_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            logger.info(f"Thumbnail created: {thumbnail_path}")
            return thumbnail_path
        except Exception as e:
            logger.warning(f"Thumbnail creation failed: {str(e)}")
            return ""
    
    def add_metadata(self, video_path: str, metadata: Dict[str, str]) -> str:
        """
        Add metadata to video file
        
        Args:
            video_path: Path to video
            metadata: Metadata dictionary
            
        Returns:
            Path to video with metadata
        """
        output_path = str(video_path).replace('.mp4', '_meta.mp4')
        
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-c", "copy",
            "-movflags", "use_metadata_tags"
        ]
        
        # Add metadata
        for key, value in metadata.items():
            cmd.extend(["-metadata", f"{key}={value}"])
        
        cmd.extend(["-y", output_path])
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            logger.info(f"Metadata added: {output_path}")
            return output_path
        except Exception as e:
            logger.warning(f"Metadata addition failed: {str(e)}")
            return video_path
    
    def optimize_for_platform(self, video_path: str, platform: str) -> Dict[str, Any]:
        """
        Optimize video for specific platform
        
        Args:
            video_path: Path to video
            platform: Target platform
            
        Returns:
            Dictionary with optimized video path and recommendations
        """
        recommendations = {
            "instagram": {
                "hashtags": ["#reels", "#instagram", "#viral", "#ai", "#contentcreation"],
                "best_time": "12:00 PM or 7:00 PM",
                "caption_length": "125-150 characters",
                "cta": "Follow for more AI content!"
            },
            "tiktok": {
                "hashtags": ["#fyp", "#foryou", "#ai", "#tech", "#viral"],
                "best_time": "6:00 AM, 10:00 AM, or 7:00 PM",
                "caption_length": "100 characters",
                "cta": "Like and follow for more!"
            },
            "youtube_shorts": {
                "hashtags": ["#shorts", "#youtube", "#ai", "#technology"],
                "best_time": "2:00 PM or 9:00 PM",
                "caption_length": "100-200 characters",
                "cta": "Subscribe for more shorts!"
            }
        }
        
        # Export with platform-specific settings
        platform_preset = platform.replace("_", "")
        if platform_preset in self.format_presets:
            output_path = str(video_path).replace('.mp4', f'_{platform}.mp4')
            optimized_path = self.export(video_path, output_path, platform_preset)
        else:
            optimized_path = video_path
        
        return {
            "video_path": optimized_path,
            "recommendations": recommendations.get(platform, {}),
            "thumbnail": self.create_thumbnail(optimized_path)
        }
