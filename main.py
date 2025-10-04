#!/usr/bin/env python3
"""
Main entry point for the Auto-Generated Instagram Video System
"""

import argparse
import os
import sys
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pipeline import VideoGenerator
from src.config import Config

console = Console()

def setup_logging():
    """Configure logging"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logger.add("logs/video_generation.log", rotation="10 MB", retention="7 days")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate Instagram Reels with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Topic for the video (e.g., 'How photosynthesis works')"
    )
    
    parser.add_argument(
        "--style",
        type=str,
        default="conversational",
        choices=["educational", "exciting", "storytelling", "tutorial", "motivational", "conversational"],
        help="Style of the video"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        choices=[15, 30, 60, 90],
        help="Duration of the video in seconds"
    )
    
    parser.add_argument(
        "--voice",
        type=str,
        default="default",
        help="Voice profile to use"
    )
    
    parser.add_argument(
        "--avatar",
        type=str,
        default="default",
        help="Avatar to use for the video"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the video"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--no-captions",
        action="store_true",
        help="Disable automatic captions"
    )
    
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="Disable background visuals"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser.parse_args()

def main():
    """Main execution function"""
    setup_logging()
    args = parse_arguments()
    
    if args.debug:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
    
    console.print(f"[bold cyan]🎬 Auto-Generated Instagram Video System[/bold cyan]")
    console.print(f"[yellow]Topic:[/yellow] {args.topic}")
    console.print(f"[yellow]Style:[/yellow] {args.style}")
    console.print(f"[yellow]Duration:[/yellow] {args.duration} seconds\n")
    
    try:
        # Load configuration
        config = Config(args.config)
        
        # Initialize video generator
        generator = VideoGenerator(config)
        
        # Generate output filename if not provided
        if not args.output:
            topic_slug = args.topic.lower().replace(" ", "_")[:30]
            args.output = f"output/{topic_slug}_{args.style}_{args.duration}s.mp4"
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            # Stage 1: Generate Script
            task = progress.add_task("[cyan]Generating script...", total=1)
            script = generator.generate_script(
                topic=args.topic,
                style=args.style,
                duration=args.duration
            )
            progress.update(task, completed=1)
            logger.info(f"Script generated: {len(script['dialogue'])} words")
            
            # Stage 2: Generate Voice
            task = progress.add_task("[cyan]Synthesizing voice...", total=1)
            audio_path = generator.generate_voice(
                script=script,
                voice_profile=args.voice
            )
            progress.update(task, completed=1)
            logger.info(f"Voice synthesized: {audio_path}")
            
            # Stage 3: Generate Avatar
            task = progress.add_task("[cyan]Creating avatar animation...", total=1)
            avatar_video = generator.generate_avatar(
                audio_path=audio_path,
                avatar_name=args.avatar,
                script=script
            )
            progress.update(task, completed=1)
            logger.info(f"Avatar animated: {avatar_video}")
            
            # Stage 4: Add Visuals
            if not args.no_background:
                task = progress.add_task("[cyan]Adding visuals and effects...", total=1)
                video_with_visuals = generator.add_visuals(
                    video_path=avatar_video,
                    script=script
                )
                progress.update(task, completed=1)
                logger.info("Visuals added")
            else:
                video_with_visuals = avatar_video
            
            # Stage 5: Add Captions
            if not args.no_captions:
                task = progress.add_task("[cyan]Generating captions...", total=1)
                video_with_captions = generator.add_captions(
                    video_path=video_with_visuals,
                    audio_path=audio_path
                )
                progress.update(task, completed=1)
                logger.info("Captions added")
            else:
                video_with_captions = video_with_visuals
            
            # Stage 6: Export Final Video
            task = progress.add_task("[cyan]Exporting Instagram-ready video...", total=1)
            final_video = generator.export_video(
                video_path=video_with_captions,
                output_path=args.output,
                format="instagram"
            )
            progress.update(task, completed=1)
        
        console.print(f"\n[bold green]✅ Video generated successfully![/bold green]")
        console.print(f"[yellow]Output:[/yellow] {final_video}")
        console.print(f"[dim]File size: {os.path.getsize(final_video) / (1024*1024):.2f} MB[/dim]")
        
        # Show preview command
        console.print(f"\n[cyan]Preview with:[/cyan] open {final_video}")
        
    except Exception as e:
        logger.error(f"Error generating video: {str(e)}")
        console.print(f"[bold red]❌ Error:[/bold red] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
