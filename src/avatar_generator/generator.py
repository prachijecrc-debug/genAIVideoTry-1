"""
Avatar generator for creating lip-synced talking head videos
Using SadTalker, Wav2Lip, and other open-source solutions
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import cv2
from loguru import logger


class AvatarGenerator:
    """Generate lip-synced avatar videos from audio using SadTalker/Wav2Lip"""
    
    def __init__(self, config):
        """
        Initialize avatar generator
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.engine = config.get("avatar.engine", "wav2lip")  # wav2lip, sadtalker
        self.avatars_dir = Path(config.get("paths.avatars", "data/avatars"))
        self.temp_dir = Path(config.get("paths.temp", "temp"))
        self.fps = config.get("avatar.fps", 25)
        self.resolution = config.get("avatar.resolution", [512, 512])
        
        # Create directories
        self.avatars_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Default avatar settings
        self.avatars = {
            "default": {
                "image": "default.jpg",
                "video": None,
                "style": "professional",
                "gender": "neutral"
            },
            "female_1": {
                "image": "female_1.jpg",
                "video": None,
                "style": "friendly",
                "gender": "female"
            },
            "male_1": {
                "image": "male_1.jpg",
                "video": None,
                "style": "professional",
                "gender": "male"
            },
            "animated": {
                "image": "animated.png",
                "video": None,
                "style": "energetic",
                "gender": "neutral"
            }
        }
        
        # Create default avatars if they don't exist
        self._create_default_avatars()
        
        logger.info(f"Avatar generator initialized with {self.engine} engine")
    
    def generate(self, audio_path: str, avatar_name: str = "default",
                gesture_cues: Optional[List[Dict]] = None) -> str:
        """
        Generate avatar video from audio
        
        Args:
            audio_path: Path to audio file
            avatar_name: Avatar to use
            gesture_cues: Optional gesture cues for animation
            
        Returns:
            Path to generated video
        """
        logger.info(f"Generating avatar video with {avatar_name}")
        
        # Get avatar configuration
        avatar = self.avatars.get(avatar_name, self.avatars["default"])
        
        # Select generation method based on engine
        if self.engine == "sadtalker":
            video_path = self._generate_with_sadtalker(audio_path, avatar, gesture_cues)
        elif self.engine == "wav2lip":
            video_path = self._generate_with_wav2lip(audio_path, avatar)
        else:
            video_path = self._generate_fallback(audio_path, avatar)
        
        # Add gesture animations if provided
        if gesture_cues:
            video_path = self._add_gestures(video_path, gesture_cues)
        
        return video_path
    
    def _generate_with_sadtalker(self, audio_path: str, avatar: Dict,
                                 gesture_cues: Optional[List[Dict]] = None) -> str:
        """
        Generate using SadTalker (more expressive, includes head movement)
        
        Args:
            audio_path: Path to audio file
            avatar: Avatar configuration
            gesture_cues: Optional gesture cues
            
        Returns:
            Path to video file
        """
        try:
            # Note: This is a placeholder for SadTalker integration
            # Actual implementation would require SadTalker installation
            # pip install sadtalker
            
            logger.info("Using SadTalker for avatar generation")
            
            # Get avatar image
            avatar_image = self.avatars_dir / avatar["image"]
            if not avatar_image.exists():
                avatar_image = self._create_placeholder_avatar(avatar["image"])
            
            # Output path
            output_path = self.temp_dir / f"avatar_{os.getpid()}.mp4"
            
            # SadTalker command (would be actual API call in production)
            # from sadtalker import SadTalker
            # generator = SadTalker()
            # generator.generate(
            #     source_image=str(avatar_image),
            #     driven_audio=audio_path,
            #     result_dir=str(self.temp_dir),
            #     enhancer="gfpgan"  # Face enhancement
            # )
            
            # For now, use fallback
            return self._generate_fallback(audio_path, avatar)
            
        except Exception as e:
            logger.warning(f"SadTalker generation failed: {str(e)}")
            return self._generate_fallback(audio_path, avatar)
    
    def _generate_with_wav2lip(self, audio_path: str, avatar: Dict) -> str:
        """
        Generate using Wav2Lip (focused on lip-sync accuracy)
        
        Args:
            audio_path: Path to audio file
            avatar: Avatar configuration
            
        Returns:
            Path to video file
        """
        try:
            logger.info("Using Wav2Lip for avatar generation")
            
            # Get avatar image
            avatar_image = self.avatars_dir / avatar["image"]
            if not avatar_image.exists():
                avatar_image = self._create_placeholder_avatar(avatar["image"])
            
            # Output path
            output_path = self.temp_dir / f"avatar_{os.getpid()}.mp4"
            
            # Wav2Lip command (would be actual API call in production)
            # import subprocess
            # cmd = [
            #     "python", "Wav2Lip/inference.py",
            #     "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
            #     "--face", str(avatar_image),
            #     "--audio", audio_path,
            #     "--outfile", str(output_path)
            # ]
            # subprocess.run(cmd, check=True)
            
            # For now, use fallback
            return self._generate_fallback(audio_path, avatar)
            
        except Exception as e:
            logger.warning(f"Wav2Lip generation failed: {str(e)}")
            return self._generate_fallback(audio_path, avatar)
    
    def _generate_fallback(self, audio_path: str, avatar: Dict) -> str:
        """
        Fallback avatar generation using basic animation
        
        Args:
            audio_path: Path to audio file
            avatar: Avatar configuration
            
        Returns:
            Path to video file
        """
        logger.info("Using fallback avatar generation")
        
        # Get avatar image
        avatar_image_path = self.avatars_dir / avatar["image"]
        if not avatar_image_path.exists():
            avatar_image_path = self._create_placeholder_avatar(avatar["image"])
        
        # Load avatar image
        avatar_img = cv2.imread(str(avatar_image_path))
        if avatar_img is None:
            # Create a placeholder if image loading fails
            avatar_img = self._create_placeholder_image()
        
        # Get audio duration
        from moviepy.editor import AudioFileClip
        try:
            audio_clip = AudioFileClip(audio_path)
            duration = audio_clip.duration
            audio_clip.close()
        except:
            duration = 10.0  # Default duration
        
        # Create video writer
        output_path = self.temp_dir / f"avatar_{os.getpid()}.mp4"
        height, width = avatar_img.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))
        
        # Generate frames with basic animation
        total_frames = int(duration * self.fps)
        for frame_idx in range(total_frames):
            # Create frame with subtle animation
            frame = avatar_img.copy()
            
            # Add simple mouth movement simulation
            if frame_idx % 5 < 3:  # Simple open/close mouth animation
                # Draw a simple mouth animation (placeholder)
                mouth_y = int(height * 0.7)
                mouth_x = int(width * 0.5)
                mouth_width = int(width * 0.1)
                mouth_height = int(height * 0.02) + (frame_idx % 5) * 2
                
                cv2.ellipse(frame, 
                          (mouth_x, mouth_y),
                          (mouth_width, mouth_height),
                          0, 0, 180, (50, 50, 50), -1)
            
            # Add subtle head movement
            if frame_idx % 30 < 15:
                M = np.float32([[1, 0, 2], [0, 1, 0]])
            else:
                M = np.float32([[1, 0, -2], [0, 1, 0]])
            frame = cv2.warpAffine(frame, M, (width, height))
            
            out.write(frame)
        
        out.release()
        
        # Add audio to video
        from moviepy.editor import VideoFileClip, CompositeAudioClip
        try:
            video = VideoFileClip(str(output_path))
            audio = AudioFileClip(audio_path)
            final_video = video.set_audio(audio)
            
            final_output = str(output_path).replace('.mp4', '_with_audio.mp4')
            final_video.write_videofile(final_output, codec='libx264', audio_codec='aac', logger=None)
            
            video.close()
            audio.close()
            final_video.close()
            
            return final_output
        except Exception as e:
            logger.warning(f"Failed to add audio: {str(e)}")
            return str(output_path)
    
    def _create_placeholder_avatar(self, filename: str) -> Path:
        """
        Create a placeholder avatar image
        
        Args:
            filename: Filename for the avatar
            
        Returns:
            Path to created avatar
        """
        avatar_path = self.avatars_dir / filename
        
        # Create a simple placeholder image
        img = self._create_placeholder_image()
        cv2.imwrite(str(avatar_path), img)
        
        return avatar_path
    
    def _create_placeholder_image(self) -> np.ndarray:
        """
        Create a placeholder avatar image
        
        Returns:
            Numpy array representing the image
        """
        # Create a simple face placeholder
        width, height = self.resolution
        img = np.ones((height, width, 3), dtype=np.uint8) * 200  # Light gray background
        
        # Draw a simple face
        center = (width // 2, height // 2)
        
        # Head
        cv2.circle(img, center, min(width, height) // 3, (150, 120, 90), -1)
        
        # Eyes
        eye_y = center[1] - 30
        cv2.circle(img, (center[0] - 40, eye_y), 15, (50, 50, 50), -1)
        cv2.circle(img, (center[0] + 40, eye_y), 15, (50, 50, 50), -1)
        
        # Mouth
        mouth_y = center[1] + 40
        cv2.ellipse(img, (center[0], mouth_y), (60, 20), 0, 0, 180, (50, 50, 50), 2)
        
        return img
    
    def _add_gestures(self, video_path: str, gesture_cues: List[Dict]) -> str:
        """
        Add gesture animations to video
        
        Args:
            video_path: Path to video
            gesture_cues: List of gesture cues
            
        Returns:
            Path to video with gestures
        """
        # This would add overlays or transformations based on gesture cues
        # For now, return the original video
        return video_path
    
    def _create_default_avatars(self):
        """Create default avatar images if they don't exist"""
        for avatar_name, avatar_config in self.avatars.items():
            avatar_path = self.avatars_dir / avatar_config["image"]
            if not avatar_path.exists():
                logger.info(f"Creating default avatar: {avatar_name}")
                self._create_placeholder_avatar(avatar_config["image"])
    
    def add_custom_avatar(self, name: str, image_path: str, 
                         style: str = "professional") -> None:
        """
        Add a custom avatar
        
        Args:
            name: Avatar name
            image_path: Path to avatar image
            style: Avatar style
        """
        import shutil
        
        # Copy image to avatars directory
        filename = f"{name}.jpg"
        dest_path = self.avatars_dir / filename
        shutil.copy(image_path, dest_path)
        
        # Add to avatars dictionary
        self.avatars[name] = {
            "image": filename,
            "video": None,
            "style": style,
            "gender": "neutral"
        }
        
        logger.info(f"Added custom avatar: {name}")
