"""
Configuration management for the video generation system
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class Config:
    """Configuration manager for the video generation system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config/default.yaml"
        self.config = self._load_config()
        self._validate_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = self._get_default_config()
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    # Merge with defaults
                    default_config.update(file_config)
                    logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")
                logger.info("Using default configuration")
        else:
            logger.info("Using default configuration")
            
        return default_config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "llm": {
                "provider": "ollama",  # ollama, openai, huggingface
                "model": "llama3",
                "temperature": 0.7,
                "max_tokens": 2000,
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "base_url": "http://localhost:11434"  # For Ollama
            },
            "tts": {
                "engine": "edge-tts",  # edge-tts, bark, xtts
                "voice": "v2/en_speaker_0",
                "speed": 1.0,
                "pitch": 1.0,
                "sample_rate": 24000
            },
            "avatar": {
                "engine": "wav2lip",  # wav2lip, sadtalker
                "default_avatar": "default.jpg",
                "fps": 25,
                "resolution": [512, 512]
            },
            "video": {
                "resolution": [1080, 1920],  # Instagram vertical
                "fps": 30,
                "codec": "h264",
                "bitrate": "5M",
                "format": "mp4"
            },
            "captions": {
                "enabled": True,
                "model": "whisper",
                "size": "base",
                "language": "en",
                "font": "Arial",
                "font_size": 48,
                "color": "white",
                "stroke_color": "black",
                "stroke_width": 2,
                "position": "bottom"
            },
            "visuals": {
                "background_blur": 10,
                "overlay_opacity": 0.8,
                "transitions": True,
                "effects": ["zoom", "pan"],
                "b_roll_source": "pexels"  # pexels, unsplash, generated
            },
            "paths": {
                "avatars": "data/avatars",
                "backgrounds": "data/backgrounds",
                "templates": "data/templates",
                "temp": "temp",
                "output": "output",
                "models": "models"
            },
            "performance": {
                "gpu": True,
                "batch_size": 1,
                "num_workers": 4,
                "cache": True
            }
        }
    
    def _validate_config(self):
        """Validate configuration and create necessary directories"""
        # Create necessary directories
        for key, path in self.config["paths"].items():
            Path(path).mkdir(parents=True, exist_ok=True)
            
        # Validate model availability
        if self.config["llm"]["provider"] == "openai" and not self.config["llm"]["api_key"]:
            logger.warning("OpenAI API key not found. Set OPENAI_API_KEY environment variable")
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        
    def save(self, path: Optional[str] = None):
        """
        Save configuration to file
        
        Args:
            path: Path to save configuration
        """
        save_path = path or self.config_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
        logger.info(f"Configuration saved to {save_path}")
