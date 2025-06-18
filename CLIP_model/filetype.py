import os

class FileTypeChecker:
    """Detects if a file is an image or video based on its extension."""
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


    def is_image(file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in FileTypeChecker.IMAGE_EXTENSIONS

    def is_video(file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in FileTypeChecker.VIDEO_EXTENSIONS

    def detect_media_type(file_path: str) -> str:
        """Determines if the media is an 'image', 'video', or 'unknown'."""
        if FileTypeChecker.is_image(file_path):
            return 'image'
        elif FileTypeChecker.is_video(file_path):
            return 'video'
        else:
            return 'unknown'