"""
Visual Agent - Generates visual content recommendations and images.
Uses Grok for description generation, then Google Gemini for actual image generation.
Simplified for speed - no RAG/Pinecone dependency.
"""

import json
from typing import Optional

import structlog

from app.core.llm_clients import LLMMessage, LLMProvider, llm_client
from app.core.config import settings

logger = structlog.get_logger(__name__)


# Platform dimensions
PLATFORM_DIMENSIONS = {
    "linkedin": {
        "feed_post": "1200 x 627",
        "carousel": "1080 x 1080",
    },
    "twitter": {
        "tweet_image": "1200 x 675",
    },
    "instagram": {
        "feed_square": "1080 x 1080",
        "feed_portrait": "1080 x 1350",
        "story": "1080 x 1920",
    },
    "facebook": {
        "feed_image": "1200 x 630",
    },
}

# Visual best practices (fallback knowledge)
VISUAL_TIPS = {
    "linkedin": """- Professional imagery performs best
- Include faces for 38% more engagement
- Blue tones align with platform aesthetics
- Infographics for data-driven content""",
    
    "twitter": """- Bold, high-contrast visuals stand out
- Keep text on images minimal
- GIFs for reactions and engagement
- Memes work well for engagement""",
    
    "instagram": """- High-quality, aesthetic imagery essential
- Consistent visual theme/filter
- Reels outperform static images
- Behind-the-scenes resonates""",
    
    "facebook": """- Native video gets priority
- Emotional imagery performs best
- Text overlays should be minimal
- Live videos have highest engagement""",
}


VISUAL_SYSTEM_PROMPT = """You are a visual content strategist for social media.
Your role is to recommend visual elements that complement written content.

Provide specific, actionable visual recommendations including:
- Type of visual (image, carousel, video, infographic)
- Description of what the visual should contain
- Style and color recommendations
- Platform-specific considerations"""


VISUAL_USER_PROMPT = """Recommend visual content for this {platform} post.

## Post Details
Topic: {topic}
Target Audience: {target_audience}
Platform: {platform}

## Platform Best Practices
{visual_tips}

## Brand Colors (if any)
{brand_colors}

## Strategic Direction (from Planner)
- Visual Style: {style}
- Visual Type: {type}
- Subject: {subject}
- Mood: {mood}

Generate a visual recommendation aligned with this strategy.

Respond with valid JSON:
{{
    "type": "image/carousel/video/infographic",
    "description": "Detailed description of the visual content",
    "style": "Style notes (colors, mood, aesthetic)",
    "dimensions": "Recommended dimensions"
}}"""


class VisualAgent:
    """
    Visual Agent for Social Media Manager.
    
    Generates visual content descriptions using Grok.
    Image generation can be added later with a separate model.
    """
    
    name = "visual_agent"
    
    async def execute(self, state: dict, context: dict) -> dict:
        """
        Generate visual content recommendations.
        
        Args:
            state: Current workflow state
            context: Execution context
            
        Returns:
            Updated state with visual advice
        """
        request = state.get("user_request") or {}
        content = state.get("content") or {}

        platform = request.get("platform", "linkedin")
        topic = request.get("topic", "")
        target_audience = content.get("target_audience", "professionals")
        
        logger.info(
            "Visual agent executing",
            platform=platform,
        )
        
        # Get platform-specific info
        visual_tips = VISUAL_TIPS.get(platform, VISUAL_TIPS["linkedin"])
        dimensions = PLATFORM_DIMENSIONS.get(platform, {})
        
        # Get brand colors if available
        context = context or {}
        brand_info = context.get("brand_info") or {}
        brand_colors = brand_info.get("brand_colors", "No specific brand colors")
        
        # Get visual strategy from planner
        plan = state.get("plan") or {}
        visual_strategy = plan.get("visual_strategy") or {}
        
        # Generate visual recommendation using Grok
        visual_advice = await self._generate_visual(
            platform=platform,
            topic=topic,
            target_audience=target_audience,
            visual_tips=visual_tips,
            brand_colors=brand_colors,
            dimensions=dimensions,
            visual_strategy=visual_strategy,
        )
        
        return {"visual_advice": visual_advice}
    
    async def _generate_visual(
        self,
        platform: str,
        topic: str,
        target_audience: str,
        visual_tips: str,
        brand_colors: str,
        dimensions: dict,
        visual_strategy: dict,
    ) -> dict:
        """Generate visual recommendation using Grok."""
        
        visual_strategy = visual_strategy or {}
        
        user_prompt = VISUAL_USER_PROMPT.format(
            platform=platform,
            topic=topic,
            target_audience=target_audience,
            visual_tips=visual_tips,
            brand_colors=brand_colors,
            style=visual_strategy.get("style", "Professional"),
            type=visual_strategy.get("type", "Image"),
            subject=visual_strategy.get("subject", topic),
            mood=visual_strategy.get("mood", "Professional"),
        )
        
        messages = [
            LLMMessage(role="system", content=VISUAL_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]
        
        from app.core.model_config import TaskType
        
        # Use OpenRouter with task-based routing
        response = await llm_client.generate_for_task(
            task=TaskType.VISUAL_DESCRIPTION,
            messages=messages,
            json_mode=True,
        )
        
        try:
            # Add null check for response
            if response is None or not response.content:
                logger.warning("Visual generation returned empty response")
                raise ValueError("Empty response from LLM")

            result = json.loads(response.content)

            # Add null check for parsed result
            if result is None or not isinstance(result, dict):
                logger.warning("Visual generation returned null or non-dict result", result=result)
                raise ValueError("Invalid result format")

            # Get default dimension for platform
            default_dim = list(dimensions.values())[0] if dimensions else "1200 x 630"

            visual_advice = {
                "type": result.get("type", "image"),
                "description": result.get("description", ""),
                "style": result.get("style", ""),
                "dimensions": result.get("dimensions", default_dim),
                "platform": platform,
                "image_url": None,
                "generation_status": "description_only",
            }

            logger.debug("Visual recommendation generated", type=visual_advice["type"])

            # Generate actual image using the description
            try:
                image_result = await self._generate_image(
                    description=visual_advice["description"],
                    style=visual_advice["style"],
                    dimensions=visual_advice["dimensions"],
                )
                if image_result:
                    visual_advice["image_url"] = image_result.get("url")
                    visual_advice["generation_status"] = "image_generated"
                    logger.info("Image generated successfully", has_url=bool(image_result.get("url")))
            except Exception as e:
                logger.warning("Image generation failed, using description only", error=str(e))
                # Continue with description only

            return visual_advice

        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Visual parsing failed", error=str(e))
            # Return fallback
            return {
                "type": "image",
                "description": f"Professional visual related to {topic}",
                "style": "Clean, modern, professional",
                "dimensions": list(dimensions.values())[0] if dimensions else "1200 x 630",
                "platform": platform,
                "image_url": None,
                "generation_status": "description_only",
            }

    async def _generate_image(
        self,
        description: str,
        style: str,
        dimensions: str,
    ) -> Optional[dict]:
        """
        Generate actual image using the description.

        Args:
            description: Text description of the image
            style: Style notes for the image
            dimensions: Target dimensions

        Returns:
            Dictionary with image URL or None if generation fails
        """
        try:
            # Build the image generation prompt
            image_prompt = f"""Generate a high-quality social media image with the following specifications:

Description: {description}
Style: {style}
Dimensions: {dimensions}

Create a visually appealing, professional image that matches this description."""

            from app.core.model_config import TaskType, get_model_config
            from app.core.config import settings

            # Get model config for image generation
            config = get_model_config(TaskType.IMAGE_GENERATION)

            # Call OpenRouter directly with proper parameters for image generation
            logger.debug("Requesting image generation", prompt_length=len(image_prompt))

            # Use OpenRouter client directly to access raw response
            openrouter_client = llm_client.openrouter.client

            request_params = {
                "model": config.model,
                "messages": [{"role": "user", "content": image_prompt}],
                "extra_headers": llm_client.openrouter.extra_headers,
                "extra_body": {
                    "models": config.fallbacks,
                }
            }

            # Make the API call
            response = await openrouter_client.chat.completions.create(**request_params)

            # OpenRouter image generation returns images in message.images field
            message = response.choices[0].message

            # Extract base64 image from response
            base64_image = None

            # Check for images field (new OpenRouter format)
            if hasattr(message, 'images') and message.images:
                try:
                    first_image = message.images[0]

                    # Try different ways to access the URL (dict vs object)
                    image_url = None

                    # Method 1: Try as object attributes
                    if hasattr(first_image, 'image_url'):
                        image_url_obj = first_image.image_url
                        if hasattr(image_url_obj, 'url'):
                            image_url = image_url_obj.url

                    # Method 2: Try as dictionary
                    if not image_url and isinstance(first_image, dict):
                        image_url = first_image.get('image_url', {}).get('url')

                    # Method 3: Try direct url field
                    if not image_url:
                        if hasattr(first_image, 'url'):
                            image_url = first_image.url
                        elif isinstance(first_image, dict) and 'url' in first_image:
                            image_url = first_image['url']

                    if image_url and image_url.startswith('data:image'):
                        base64_image = image_url

                except Exception as e:
                    logger.error("Error extracting image from images field", error=str(e))

            # Fallback: check message content for URL or base64
            external_url = None
            if not base64_image and message.content:
                content = message.content.strip()

                # Check if response is a direct URL
                if content.startswith("http://") or content.startswith("https://"):
                    external_url = content

                # Check if response is JSON with URL
                if not external_url:
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict):
                            url = parsed.get("url") or parsed.get("image_url") or parsed.get("data", {}).get("url")
                            if url:
                                if url.startswith("http"):
                                    external_url = url
                                elif url.startswith("data:image"):
                                    base64_image = url
                    except json.JSONDecodeError:
                        pass

                # If response is base64 without prefix
                if not base64_image and not external_url and len(content) > 100 and not content.startswith("http"):
                    base64_image = f"data:image/png;base64,{content}"

            from app.core.s3_storage import upload_base64_image, upload_image_from_url

            # Route external URLs through S3
            if external_url:
                s3_url = await upload_image_from_url(external_url)
                if s3_url:
                    logger.info("Image re-uploaded to S3 from external URL")
                    return {"url": s3_url}
                else:
                    logger.warning("Failed to re-upload external URL to S3, using original")
                    return {"url": external_url}

            # Upload base64 to S3
            if base64_image:
                s3_url = await upload_base64_image(base64_image)
                if s3_url:
                    logger.info("Image uploaded to S3")
                    return {"url": s3_url}
                else:
                    logger.warning("Failed to upload to S3, returning base64")
                    return {"url": base64_image}

            logger.warning("Image generation returned unexpected format")
            return None

        except Exception as e:
            logger.error("Image generation failed", error=str(e))
            return None
