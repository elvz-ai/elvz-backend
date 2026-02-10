"""
Firebase Storage utility for uploading images.
"""

import base64
import uuid
from typing import Optional
import structlog

import firebase_admin
from firebase_admin import credentials, storage

from app.core.config import settings

logger = structlog.get_logger(__name__)

# Global Firebase app instance
_firebase_app: Optional[firebase_admin.App] = None


def initialize_firebase():
    """Initialize Firebase Admin SDK."""
    global _firebase_app

    if _firebase_app is not None:
        return _firebase_app

    try:
        # Check if already initialized
        _firebase_app = firebase_admin.get_app()
        logger.info("Firebase already initialized")
        return _firebase_app
    except ValueError:
        # Not initialized yet
        pass

    try:
        if not settings.firebase_json_path:
            logger.warning("Firebase JSON path not configured")
            return None

        cred = credentials.Certificate(settings.firebase_json_path)
        _firebase_app = firebase_admin.initialize_app(cred, {
            'storageBucket': settings.firebase_storage_bucket
        })
        logger.info("Firebase initialized successfully", bucket=settings.firebase_storage_bucket)
        return _firebase_app
    except Exception as e:
        logger.error("Failed to initialize Firebase", error=str(e))
        return None


def upload_base64_image(
    base64_data: str,
    filename: Optional[str] = None,
    folder: Optional[str] = None
) -> Optional[str]:
    """
    Upload a base64-encoded image to Firebase Storage.

    Args:
        base64_data: Base64-encoded image data (with or without data URI prefix)
        filename: Optional custom filename (auto-generated if not provided)
        folder: Optional folder path (defaults to config value)

    Returns:
        Public URL of the uploaded image, or None if upload fails
    """
    try:
        # Initialize Firebase if needed
        app = initialize_firebase()
        if app is None:
            logger.error("Firebase not initialized, cannot upload image")
            return None

        # Remove data URI prefix if present
        if base64_data.startswith('data:image'):
            base64_data = base64_data.split(',', 1)[1]

        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_data)

        # Generate filename if not provided
        if filename is None:
            filename = f"{uuid.uuid4()}.png"

        # Use configured folder or provided one
        folder = folder or settings.firebase_storage_folder
        blob_path = f"{folder}/{filename}" if folder else filename

        # Get storage bucket
        bucket = storage.bucket()
        blob = bucket.blob(blob_path)

        # Upload image
        blob.upload_from_string(
            image_bytes,
            content_type='image/png'
        )

        # Make the blob publicly accessible
        blob.make_public()

        # Get public URL
        public_url = blob.public_url

        logger.info("Image uploaded to Firebase Storage",
                   blob_path=blob_path,
                   url_length=len(public_url))

        return public_url

    except Exception as e:
        logger.error("Failed to upload image to Firebase Storage", error=str(e))
        return None
