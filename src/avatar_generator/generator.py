"""
Avatar generator for creating lip-synced talking head videos
Using Wav2Lip and other open-source solutions
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import cv2
from loguru import logger
import json


class AvatarGenerator:
    """Generate lip-synced avatar videos from audio using Wav2Lip"""
    
    def __init__(self, config):
        """
        Initialize avatar generator
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.engine = config.get("avatar.engine", "wav2lip")
        self.avatars_dir = Path(config.get("paths.avatars", "data/avatars"))
        self.temp_dir = Path(config.get("paths.temp", "temp"))
        self.fps = config.get("avatar.fps", 25)
        self.resolution = config.get("avatar.resolution", [512, 512])
        
        # Create directories
        self.avatars_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Wav2Lip paths
        self.wav2lip_dir = Path("Wav2Lip")
        self.checkpoint_path = self.wav2lip_dir / "checkpoints" / "wav2lip_gan.pth"
        
        # Avatar configurations with real human face images
        self.avatars = {
            "default": {
                "image": "real_person.jpg",
                "description": "Real human presenter",
                "style": "professional",
                "gender": "female"
            },
            "professional_female": {
                "image": "real_person.jpg",
                "description": "Professional female presenter",
                "style": "professional",
                "gender": "female"
            },
            "professional_male": {
                "image": "professional_male.jpg",
                "description": "Professional male presenter",
                "style": "professional",
                "gender": "male"
            },
            "casual_female": {
                "image": "casual_female.jpg",
                "description": "Casual female presenter",
                "style": "friendly",
                "gender": "female"
            },
            "casual_male": {
                "image": "casual_male.jpg",
                "description": "Casual male presenter",
                "style": "friendly",
                "gender": "male"
            },
            "tech_presenter": {
                "image": "tech_presenter.jpg",
                "description": "Tech-focused presenter",
                "style": "technical",
                "gender": "neutral"
            }
        }
        
        # Create realistic avatar images
        self._create_realistic_avatars()
        
        logger.info(f"Avatar generator initialized with {self.engine} engine")
    
    def generate(self, audio_path: str, avatar_name: str = "professional_female",
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
        avatar = self.avatars.get(avatar_name, self.avatars["professional_female"])
        
        # Check if Wav2Lip is available
        if self._check_wav2lip_available():
            video_path = self._generate_with_wav2lip(audio_path, avatar)
        else:
            logger.warning("Wav2Lip not fully configured, using enhanced fallback")
            video_path = self._generate_enhanced_fallback(audio_path, avatar)
        
        # Add gesture animations if provided
        if gesture_cues:
            video_path = self._add_gestures(video_path, gesture_cues)
        
        return video_path
    
    def _check_wav2lip_available(self) -> bool:
        """
        Check if Wav2Lip is properly installed and configured
        
        Returns:
            True if Wav2Lip is available
        """
        # Check if Wav2Lip directory exists
        if not self.wav2lip_dir.exists():
            logger.warning("Wav2Lip directory not found")
            return False
        
        # Check if checkpoint exists and is valid (should be ~430MB)
        if not self.checkpoint_path.exists():
            logger.warning("Wav2Lip checkpoint not found")
            return False
        
        checkpoint_size = self.checkpoint_path.stat().st_size
        if checkpoint_size < 100_000_000:  # Less than 100MB means it's not the real model
            logger.warning(f"Wav2Lip checkpoint appears invalid (size: {checkpoint_size} bytes)")
            return False
        
        # Check if inference script exists
        inference_script = self.wav2lip_dir / "inference.py"
        if not inference_script.exists():
            logger.warning("Wav2Lip inference script not found")
            return False
        
        return True
    
    def _generate_with_wav2lip(self, audio_path: str, avatar: Dict) -> str:
        """
        Generate using Wav2Lip for realistic lip-sync
        
        Args:
            audio_path: Path to audio file
            avatar: Avatar configuration
            
        Returns:
            Path to video file
        """
        try:
            logger.info("Using Wav2Lip for realistic avatar generation")
            
            # Get avatar image
            avatar_image = self.avatars_dir / avatar["image"]
            if not avatar_image.exists():
                avatar_image = self._create_realistic_avatar(avatar["image"], avatar.get("gender", "neutral"))
            
            # Output path
            output_path = self.temp_dir / f"avatar_wav2lip_{os.getpid()}.mp4"
            
            # Prepare Wav2Lip command
            cmd = [
                "python", str(self.wav2lip_dir / "inference.py"),
                "--checkpoint_path", str(self.checkpoint_path),
                "--face", str(avatar_image),
                "--audio", audio_path,
                "--outfile", str(output_path),
                "--resize_factor", "1",
                "--fps", str(self.fps)
            ]
            
            logger.info(f"Running Wav2Lip command: {' '.join(cmd)}")
            
            # Run Wav2Lip
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.wav2lip_dir)
            )
            
            if result.returncode != 0:
                logger.error(f"Wav2Lip failed: {result.stderr}")
                return self._generate_enhanced_fallback(audio_path, avatar)
            
            if not output_path.exists():
                logger.error("Wav2Lip did not produce output")
                return self._generate_enhanced_fallback(audio_path, avatar)
            
            logger.info(f"Wav2Lip generation complete: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Wav2Lip generation failed: {str(e)}")
            return self._generate_enhanced_fallback(audio_path, avatar)
    
    def _generate_enhanced_fallback(self, audio_path: str, avatar: Dict) -> str:
        """
        Enhanced fallback avatar generation with more realistic animation
        
        Args:
            audio_path: Path to audio file
            avatar: Avatar configuration
            
        Returns:
            Path to video file
        """
        logger.info("Using enhanced fallback avatar generation")
        
        # Get avatar image
        avatar_image_path = self.avatars_dir / avatar["image"]
        if not avatar_image_path.exists():
            avatar_image_path = self._create_realistic_avatar(
                avatar["image"], 
                avatar.get("gender", "neutral")
            )
        
        # Load avatar image
        avatar_img = cv2.imread(str(avatar_image_path))
        if avatar_img is None:
            avatar_img = self._create_realistic_face_image(avatar.get("gender", "neutral"))
        
        # Get audio duration and analyze for lip sync
        try:
            from moviepy import AudioFileClip
            audio_clip = AudioFileClip(audio_path)
            duration = audio_clip.duration
            
            # Get audio array for basic amplitude analysis
            audio_array = audio_clip.to_soundarray()
            audio_clip.close()
            
            # Compute amplitude envelope for lip sync
            audio_amplitude = self._compute_audio_envelope(audio_array, duration, self.fps)
        except Exception as e:
            logger.warning(f"Could not analyze audio: {e}")
            duration = 10.0
            audio_amplitude = np.random.random(int(duration * self.fps)) * 0.5
        
        # Create video writer
        output_path = self.temp_dir / f"avatar_enhanced_{os.getpid()}.mp4"
        height, width = avatar_img.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))
        
        # Face landmark detection for mouth region
        mouth_region = self._detect_mouth_region(avatar_img)
        
        # Generate frames with realistic animation
        total_frames = int(duration * self.fps)
        for frame_idx in range(total_frames):
            # Create frame with animation
            frame = avatar_img.copy()
            
            # Get amplitude for this frame
            amplitude = audio_amplitude[min(frame_idx, len(audio_amplitude) - 1)]
            
            # Animate mouth based on audio amplitude
            if mouth_region:
                frame = self._animate_mouth(frame, mouth_region, amplitude)
            
            # Add subtle head movement
            frame = self._add_head_movement(frame, frame_idx, total_frames)
            
            # Add blinking animation
            frame = self._add_blinking(frame, frame_idx)
            
            out.write(frame)
        
        out.release()
        
        # Add audio to video
        try:
            from moviepy import VideoFileClip, AudioFileClip
            video = VideoFileClip(str(output_path))
            audio = AudioFileClip(audio_path)
            final_video = video.with_audio(audio)  # Use with_audio instead of set_audio
            
            final_output = str(output_path).replace('.mp4', '_with_audio.mp4')
            final_video.write_videofile(
                final_output, 
                codec='libx264', 
                audio_codec='aac',
                logger=None,
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )
            
            video.close()
            audio.close()
            final_video.close()
            
            return final_output
        except Exception as e:
            logger.warning(f"Failed to add audio: {str(e)}")
            return str(output_path)
    
    def _create_realistic_avatars(self):
        """Create realistic avatar images if they don't exist"""
        for avatar_name, avatar_config in self.avatars.items():
            avatar_path = self.avatars_dir / avatar_config["image"]
            if not avatar_path.exists():
                logger.info(f"Creating realistic avatar: {avatar_name}")
                self._create_realistic_avatar(
                    avatar_config["image"],
                    avatar_config.get("gender", "neutral")
                )
    
    def _create_realistic_avatar(self, filename: str, gender: str = "neutral") -> Path:
        """
        Create a realistic-looking avatar image
        
        Args:
            filename: Filename for the avatar
            gender: Gender of the avatar
            
        Returns:
            Path to created avatar
        """
        avatar_path = self.avatars_dir / filename
        
        # Create a realistic-looking face placeholder
        img = self._create_realistic_face_image(gender)
        cv2.imwrite(str(avatar_path), img)
        
        return avatar_path
    
    def _create_realistic_face_image(self, gender: str = "neutral") -> np.ndarray:
        """
        Create a realistic-looking face placeholder
        
        Args:
            gender: Gender for the face
            
        Returns:
            Numpy array representing the image
        """
        width, height = 512, 512
        
        # Create a gradient background (simulating studio lighting)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            gray_value = int(200 + (55 * i / height))
            img[i, :] = [gray_value, gray_value, gray_value]
        
        # Face parameters based on gender
        if gender == "female":
            skin_color = (218, 195, 175)  # Lighter skin tone
            face_width = 160
            face_height = 200
            jaw_curve = 20
        elif gender == "male":
            skin_color = (195, 170, 150)  # Slightly darker skin tone
            face_width = 180
            face_height = 210
            jaw_curve = 10
        else:
            skin_color = (205, 180, 160)  # Neutral skin tone
            face_width = 170
            face_height = 205
            jaw_curve = 15
        
        center_x, center_y = width // 2, height // 2
        
        # Draw face shape (more realistic oval)
        face_points = []
        for angle in range(0, 360, 10):
            rad = np.radians(angle)
            if angle < 180:  # Upper half
                x = int(center_x + face_width * np.cos(rad) * 0.9)
                y = int(center_y - 20 + face_height * np.sin(rad) * 0.8)
            else:  # Lower half with jaw
                x = int(center_x + (face_width - jaw_curve) * np.cos(rad) * 0.85)
                y = int(center_y - 20 + face_height * np.sin(rad) * 0.9)
            face_points.append([x, y])
        
        face_points = np.array(face_points, np.int32)
        cv2.fillPoly(img, [face_points], skin_color)
        
        # Add subtle shading
        overlay = img.copy()
        cv2.ellipse(overlay, (center_x - 40, center_y), (30, 40), 0, 0, 180, 
                   (skin_color[0] - 20, skin_color[1] - 20, skin_color[2] - 20), -1)
        cv2.ellipse(overlay, (center_x + 40, center_y), (30, 40), 0, 0, 180,
                   (skin_color[0] - 20, skin_color[1] - 20, skin_color[2] - 20), -1)
        img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        
        # Draw eyes (more realistic)
        eye_y = center_y - 30
        # Left eye
        cv2.ellipse(img, (center_x - 45, eye_y), (25, 15), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(img, (center_x - 45, eye_y), 10, (100, 80, 60), -1)  # Iris
        cv2.circle(img, (center_x - 45, eye_y), 5, (20, 20, 20), -1)  # Pupil
        cv2.ellipse(img, (center_x - 45, eye_y - 20), (30, 10), 0, 0, 180, (50, 30, 20), 2)  # Eyebrow
        
        # Right eye
        cv2.ellipse(img, (center_x + 45, eye_y), (25, 15), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(img, (center_x + 45, eye_y), 10, (100, 80, 60), -1)  # Iris
        cv2.circle(img, (center_x + 45, eye_y), 5, (20, 20, 20), -1)  # Pupil
        cv2.ellipse(img, (center_x + 45, eye_y - 20), (30, 10), 0, 0, 180, (50, 30, 20), 2)  # Eyebrow
        
        # Draw nose (more subtle)
        nose_tip_y = center_y + 20
        cv2.line(img, (center_x, center_y - 10), (center_x, nose_tip_y), 
                (skin_color[0] - 30, skin_color[1] - 30, skin_color[2] - 30), 2)
        cv2.ellipse(img, (center_x - 8, nose_tip_y), (8, 5), 0, 0, 180,
                   (skin_color[0] - 20, skin_color[1] - 20, skin_color[2] - 20), -1)
        cv2.ellipse(img, (center_x + 8, nose_tip_y), (8, 5), 0, 0, 180,
                   (skin_color[0] - 20, skin_color[1] - 20, skin_color[2] - 20), -1)
        
        # Draw mouth (more realistic shape)
        mouth_y = center_y + 60
        mouth_width = 50 if gender == "female" else 55
        
        # Upper lip
        upper_lip_points = [
            [center_x - mouth_width, mouth_y],
            [center_x - mouth_width//2, mouth_y - 5],
            [center_x, mouth_y - 8],
            [center_x + mouth_width//2, mouth_y - 5],
            [center_x + mouth_width, mouth_y]
        ]
        
        # Lower lip
        lower_lip_points = [
            [center_x - mouth_width, mouth_y],
            [center_x - mouth_width//2, mouth_y + 8],
            [center_x, mouth_y + 10],
            [center_x + mouth_width//2, mouth_y + 8],
            [center_x + mouth_width, mouth_y]
        ]
        
        # Draw lips
        lip_color = (150, 100, 100) if gender == "female" else (140, 90, 90)
        cv2.fillPoly(img, [np.array(upper_lip_points + lower_lip_points[::-1], np.int32)], lip_color)
        
        # Add hair (simple representation)
        hair_color = (50, 30, 20) if gender != "female" else (80, 50, 30)
        if gender == "female":
            # Longer hair for female
            hair_points = [
                [center_x - 100, center_y - 80],
                [center_x - 120, center_y + 50],
                [center_x - 100, center_y + 150],
                [center_x + 100, center_y + 150],
                [center_x + 120, center_y + 50],
                [center_x + 100, center_y - 80],
                [center_x, center_y - 120]
            ]
        else:
            # Shorter hair for male/neutral
            hair_points = [
                [center_x - 90, center_y - 70],
                [center_x - 100, center_y - 30],
                [center_x - 90, center_y - 90],
                [center_x + 90, center_y - 90],
                [center_x + 100, center_y - 30],
                [center_x + 90, center_y - 70],
                [center_x, center_y - 110]
            ]
        
        hair_poly = np.array(hair_points, np.int32)
        cv2.fillPoly(img, [hair_poly], hair_color)
        
        # Apply Gaussian blur for smoother appearance
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        return img
    
    def _detect_mouth_region(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect mouth region in the image
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with mouth region coordinates or None
        """
        height, width = image.shape[:2]
        
        # Approximate mouth position (for placeholder faces)
        mouth_region = {
            'center': (width // 2, int(height * 0.65)),
            'width': int(width * 0.2),
            'height': int(height * 0.08)
        }
        
        return mouth_region
    
    def _animate_mouth(self, frame: np.ndarray, mouth_region: Dict, amplitude: float) -> np.ndarray:
        """
        Animate mouth based on audio amplitude
        
        Args:
            frame: Current frame
            mouth_region: Mouth region coordinates
            amplitude: Audio amplitude (0-1)
            
        Returns:
            Frame with animated mouth
        """
        center = mouth_region['center']
        width = mouth_region['width']
        height = int(mouth_region['height'] * (0.3 + amplitude * 1.7))  # Scale based on amplitude
        
        # Create mouth mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, center, (width // 2, height // 2), 0, 0, 180, 255, -1)
        
        # Darken the mouth area based on openness
        mouth_color = int(50 + amplitude * 30)
        frame_copy = frame.copy()
        frame_copy[mask > 0] = [mouth_color, mouth_color, mouth_color]
        
        # Blend with original
        alpha = 0.5 + amplitude * 0.3
        frame = cv2.addWeighted(frame, 1 - alpha, frame_copy, alpha, 0)
        
        return frame
    
    def _add_head_movement(self, frame: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        """
        Add subtle head movement
        
        Args:
            frame: Current frame
            frame_idx: Current frame index
            total_frames: Total number of frames
            
        Returns:
            Frame with head movement
        """
        # Subtle sinusoidal movement
        angle = (frame_idx / total_frames) * 2 * np.pi
        offset_x = int(3 * np.sin(angle * 2))
        offset_y = int(2 * np.sin(angle * 3))
        
        # Apply translation
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        
        return frame
    
    def _add_blinking(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Add blinking animation
        
        Args:
            frame: Current frame
            frame_idx: Current frame index
            
        Returns:
            Frame with blinking effect
        """
        # Blink every ~100 frames for 3 frames
        if frame_idx % 100 < 3:
            height, width = frame.shape[:2]
            eye_y = int(height * 0.4)
            
            # Draw closed eyes (simple lines)
            cv2.line(frame, (int(width * 0.35), eye_y), (int(width * 0.45), eye_y), (50, 30, 20), 3)
            cv2.line(frame, (int(width * 0.55), eye_y), (int(width * 0.65), eye_y), (50, 30, 20), 3)
        
        return frame
    
    def _compute_audio_envelope(self, audio_array: np.ndarray, duration: float, fps: int) -> np.ndarray:
        """
        Compute audio envelope for lip sync
        
        Args:
            audio_array: Audio samples
            duration: Audio duration
            fps: Video frame rate
            
        Returns:
            Array of amplitude values per frame
        """
        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        # Compute frame-wise amplitude
        samples_per_frame = len(audio_array) // int(duration * fps)
        amplitude_per_frame = []
        
        for i in range(int(duration * fps)):
            start_idx = i * samples_per_frame
            end_idx = min((i + 1) * samples_per_frame, len(audio_array))
            
            if start_idx < len(audio_array):
                frame_samples = audio_array[start_idx:end_idx]
                amplitude = np.sqrt(np.mean(frame_samples ** 2))  # RMS
                amplitude_per_frame.append(min(amplitude * 3, 1.0))  # Normalize and clip
            else:
                amplitude_per_frame.append(0)
        
        # Smooth the envelope
        from scipy.ndimage import gaussian_filter1d
        amplitude_per_frame = gaussian_filter1d(amplitude_per_frame, sigma=2)
        
        return np.array(amplitude_per_frame)
    
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
    
    def download_wav2lip_model(self) -> bool:
        """
        Download Wav2Lip model if not present
        
        Returns:
            True if model is ready
        """
        if self.checkpoint_path.exists() and self.checkpoint_path.stat().st_size > 100_000_000:
            logger.info("Wav2Lip model already downloaded")
            return True
        
        logger.info("Downloading Wav2Lip model...")
        
        # Create checkpoints directory
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download URL for Wav2Lip GAN model
        model_url = "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA"
        
        try:
            import urllib.request
            urllib.request.urlretrieve(model_url, str(self.checkpoint_path))
            logger.info(f"Downloaded Wav2Lip model to {self.checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download Wav2Lip model: {e}")
            
            # Create a note about manual download
            readme_path = self.checkpoint_path.parent / "README.md"
            with open(readme_path, 'w') as f:
                f.write("# Wav2Lip Model Download\n\n")
                f.write("Please download the Wav2Lip GAN model manually:\n\n")
                f.write("1. Visit: https://github.com/Rudrabha/Wav2Lip#getting-the-weights\n")
                f.write("2. Download 'Wav2Lip + GAN' model\n")
                f.write(f"3. Place it as: {self.checkpoint_path}\n")
            
            return False
