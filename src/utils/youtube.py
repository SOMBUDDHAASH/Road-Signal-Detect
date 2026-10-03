"""
YouTube Driving Video Stream Handler using yt-dlp.
Resolves direct video stream URLs or downloads test segments for real-time sign detection.
"""

from typing import Optional, Dict, Any
import os
import yt_dlp


class YouTubeStreamHandler:
    """
    Handles resolving or downloading YouTube video streams for traffic sign processing.
    """

    def __init__(self, cache_dir: str = "outputs/youtube"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_direct_stream_url(self, youtube_url: str) -> Optional[str]:
        """
        Extracts a direct progressive MP4 stream URL (e.g. 720p/480p) without downloading.
        Can be passed directly into cv2.VideoCapture(url).
        """
        ydl_opts = {
            "format": "best[ext=mp4][height<=720]/best[ext=mp4]/best",
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                return info.get("url")
        except Exception as e:
            print(f"[Warning] Failed to resolve direct stream URL: {e}")
            return None

    def download_video_segment(
        self,
        youtube_url: str,
        output_filename: str = "youtube_test_clip.mp4",
        max_duration_sec: int = 60
    ) -> Optional[str]:
        """
        Downloads a short video segment from YouTube for reliable local playback and detection.
        """
        target_path = os.path.join(self.cache_dir, output_filename)
        
        ydl_opts = {
            "format": "best[ext=mp4][height<=720]/best[ext=mp4]/best",
            "outtmpl": target_path,
            "quiet": True,
            "no_warnings": True,
            "overwrites": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            if os.path.exists(target_path):
                return target_path
        except Exception as e:
            print(f"[Error] Failed to download YouTube video: {e}")
            return None

        return None
