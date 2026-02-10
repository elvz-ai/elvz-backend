"""
Copy Writer Elf - Generates high-quality content for various formats.
Uses sequential workflow: Strategy → Content → Tone Adaptation
"""

import asyncio
import json
import time
from typing import Any, Optional

import structlog
from langgraph.graph import END, StateGraph

from app.agents.elves.base import BaseElf
from app.core.llm_clients import LLMMessage, llm_client
from app.core.cache import cache

logger = structlog.get_logger(__name__)


class ContentStrategist:
    """Creates content framework and storytelling approach."""
    
    name = "content_strategist"
    
    async def execute(self, state: dict, context: dict) -> dict:
        """Create content strategy."""
        request = state.get("user_request", {})
        
        system_prompt = """You are an expert content strategist.
Create a content framework for the given topic.
Respond with JSON:
{
    "content_framework": {
        "angle": "main angle/hook",
        "structure": ["intro", "section1", "section2", "conclusion"],
        "key_points": ["point1", "point2", "point3"],
        "storytelling_approach": "narrative/educational/persuasive",
        "target_emotion": "emotion to evoke"
    }
}"""
        
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(
                role="user",
                content=f"Topic: {request.get('topic', '')}\n"
                       f"Audience: {request.get('target_audience', 'general')}\n"
                       f"Goals: {request.get('goals', ['engagement'])}",
            ),
        ]
        
        response = await llm_client.generate_fast(messages, json_mode=True)
        
        try:
            result = json.loads(response.content)
            return {"content_framework": result.get("content_framework", {})}
        except:
            return {"content_framework": {"angle": "informative", "structure": ["intro", "body", "conclusion"]}}


class BlogWriter:
    """Writes long-form blog content."""
    
    name = "blog_writer"
    
    async def execute(self, state: dict, context: dict) -> dict:
        """Write blog post."""
        request = state.get("user_request", {})
        framework = state.get("content_framework", {})
        
        topic = request.get("topic", "")
        keywords = request.get("target_keywords", [])
        word_count = request.get("word_count", 1000)
        tone = request.get("tone", "professional")
        
        # Get brand voice
        brand_voice = await self._get_brand_voice(context)
        
        system_prompt = f"""You are a professional blog writer.
Write engaging, well-structured blog content.
Target word count: {word_count} words.
Tone: {tone}
{brand_voice}

Include:
- Compelling headline
- Engaging introduction with hook
- Clear subheadings (H2, H3)
- Actionable takeaways
- Strong conclusion with CTA"""

        user_prompt = f"""Write a blog post about: {topic}

Content Framework:
- Angle: {framework.get('angle', 'informative')}
- Structure: {framework.get('structure', [])}
- Key Points: {framework.get('key_points', [])}

Target Keywords: {', '.join(keywords) if keywords else 'None specified'}

Respond with JSON:
{{
    "title": "blog title",
    "meta_description": "SEO meta description (150-160 chars)",
    "content": "full markdown content with headings",
    "headings": ["h2 heading 1", "h2 heading 2"]
}}"""

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        
        response = await llm_client.generate_smart(messages, json_mode=True)
        
        try:
            result = json.loads(response.content)
            return {"blog_content": result}
        except:
            return {"blog_content": {"title": topic, "content": response.content}}
    
    async def _get_brand_voice(self, context: dict) -> str:
        user_id = context.get("user_id")
        if not user_id:
            return ""
        
        profile = await cache.get_voice_profile(user_id)
        if profile:
            return f"Brand Voice: {profile.get('tone_characteristics', {})}"
        return ""


class AdCopyWriter:
    """Creates ad copy for various platforms."""
    
    name = "ad_copy_writer"
    
    async def execute(self, state: dict, context: dict) -> dict:
        """Write ad copy."""
        request = state.get("user_request", {})
        
        platform = request.get("platform", "google")
        product = request.get("product", request.get("topic", ""))
        goals = request.get("goals", ["conversions"])
        
        # Platform-specific specs
        specs = {
            "google": {"headlines": 3, "headline_limit": 30, "description_limit": 90},
            "meta": {"headlines": 1, "headline_limit": 40, "primary_text_limit": 125},
            "linkedin": {"headlines": 1, "headline_limit": 70, "intro_limit": 150},
        }
        
        spec = specs.get(platform, specs["google"])
        
        system_prompt = f"""You are an expert advertising copywriter.
Create high-converting ad copy for {platform}.
Specs: {spec}

Guidelines:
- Use power words that drive action
- Include clear value proposition
- Create urgency without being pushy
- A/B test worthy variations"""

        user_prompt = f"""Create ad copy for: {product}
Goals: {', '.join(goals)}

Respond with JSON:
{{
    "variations": [
        {{
            "headline": "headline text",
            "description": "description text",
            "cta": "call to action"
        }}
    ]
}}"""

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        
        response = await llm_client.generate(messages, json_mode=True)
        
        try:
            result = json.loads(response.content)
            return {"ad_copy": result.get("variations", [])}
        except:
            return {"ad_copy": []}


class ProductDescriptionWriter:
    """Writes product descriptions."""
    
    name = "product_description_writer"
    
    async def execute(self, state: dict, context: dict) -> dict:
        """Write product description."""
        request = state.get("user_request", {})
        
        product = request.get("product", request.get("topic", ""))
        features = request.get("features", [])
        
        system_prompt = """You are an expert e-commerce copywriter.
Write compelling product descriptions that convert.

Include:
- Benefit-focused headline
- Emotional hook
- Feature-to-benefit translations
- Social proof suggestions
- Clear CTA"""

        user_prompt = f"""Write a product description for: {product}
Features: {features}

Respond with JSON:
{{
    "headline": "product headline",
    "short_description": "1-2 sentence hook",
    "long_description": "detailed description",
    "bullet_points": ["benefit 1", "benefit 2"],
    "cta": "call to action"
}}"""

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        
        response = await llm_client.generate(messages, json_mode=True)
        
        try:
            result = json.loads(response.content)
            return {"product_description": result}
        except:
            return {"product_description": {"headline": product}}


class ToneAdapter:
    """Adapts content tone."""
    
    name = "tone_adapter"
    
    async def execute(self, state: dict, context: dict) -> dict:
        """Adapt content tone if needed."""
        request = state.get("user_request", {})
        desired_tone = request.get("tone", "professional")
        
        # Get existing content
        content = None
        if state.get("blog_content"):
            content = state["blog_content"].get("content")
        elif state.get("ad_copy"):
            return state  # Ad copy already has tone
        elif state.get("product_description"):
            content = state["product_description"].get("long_description")
        
        if not content:
            return state
        
        # Check if tone adaptation needed
        original_tone = state.get("content_framework", {}).get("target_emotion", "professional")
        if original_tone == desired_tone:
            return state
        
        system_prompt = f"""You are an expert content editor.
Rewrite the content in a {desired_tone} tone while preserving the message.

Tone characteristics:
- professional: formal, authoritative, credible
- casual: friendly, conversational, relatable
- witty: clever, humorous, engaging
- storytelling: narrative, emotional, immersive
- formal: corporate, precise, structured"""

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(
                role="user",
                content=f"Rewrite this in {desired_tone} tone:\n\n{content[:3000]}",
            ),
        ]
        
        response = await llm_client.generate(messages)
        
        # Update the appropriate content
        if state.get("blog_content"):
            state["blog_content"]["content"] = response.content
        elif state.get("product_description"):
            state["product_description"]["long_description"] = response.content
        
        state["tone_adapted"] = True
        return state


class CopywriterElf(BaseElf):
    """
    Copy Writer Elf - Generates high-quality content for various formats.
    
    Mini-Agents:
    1. Content Strategist - Creates content framework
    2. Blog Writer - Writes long-form blog posts
    3. Ad Copy Writer - Creates ad copy
    4. Product Description Writer - Writes product copy
    5. Tone Adapter - Adapts content tone
    
    Workflow: Sequential (Strategy → Content → Tone)
    """
    
    name = "copywriter"
    description = "Generates high-quality content for various formats"
    version = "1.0"
    
    def __init__(self):
        self.strategist = ContentStrategist()
        self.blog_writer = BlogWriter()
        self.ad_writer = AdCopyWriter()
        self.product_writer = ProductDescriptionWriter()
        self.tone_adapter = ToneAdapter()
        
        super().__init__()
    
    def _setup_workflow(self) -> None:
        """Set up the LangGraph workflow."""
        workflow = StateGraph(dict)
        
        workflow.add_node("strategy", self._run_strategy)
        workflow.add_node("content_generation", self._run_content_generation)
        workflow.add_node("tone_adaptation", self._run_tone_adaptation)
        workflow.add_node("seo_analysis", self._run_seo_analysis)
        workflow.add_node("synthesize", self._synthesize_results)
        
        workflow.set_entry_point("strategy")
        workflow.add_edge("strategy", "content_generation")
        workflow.add_edge("content_generation", "tone_adaptation")
        workflow.add_conditional_edges(
            "tone_adaptation",
            self._needs_seo,
            {"yes": "seo_analysis", "no": "synthesize"}
        )
        workflow.add_edge("seo_analysis", "synthesize")
        workflow.add_edge("synthesize", END)
        
        self._workflow = workflow.compile()
    
    async def execute(self, request: dict, context: dict) -> dict:
        """Execute copywriting workflow."""
        start_time = time.time()
        
        initial_state = {
            "user_request": request,
            "context": context,
            "content_framework": {},
            "blog_content": None,
            "ad_copy": None,
            "product_description": None,
            "seo_analysis": None,
            "execution_trace": [],
            "errors": [],
        }
        
        content_type = request.get("content_type", "blog")
        logger.info("Copywriter executing", content_type=content_type)
        
        try:
            final_state = await self._workflow.ainvoke(initial_state)
            execution_time_ms = int((time.time() - start_time) * 1000)
            return self._build_response(final_state, execution_time_ms)
        except Exception as e:
            logger.error("Copywriter workflow failed", error=str(e))
            return {"error": str(e)}
    
    async def _run_strategy(self, state: dict) -> dict:
        result = await self.strategist.execute(state, state.get("context", {}))
        state.update(result)
        state["execution_trace"].append({"agent": "strategist", "status": "completed"})
        return state
    
    async def _run_content_generation(self, state: dict) -> dict:
        """Run appropriate content generator based on request type."""
        request = state.get("user_request", {})
        content_type = request.get("content_type", "blog")
        
        if content_type == "blog":
            result = await self.blog_writer.execute(state, state.get("context", {}))
        elif content_type == "ad":
            result = await self.ad_writer.execute(state, state.get("context", {}))
        elif content_type == "product":
            result = await self.product_writer.execute(state, state.get("context", {}))
        else:
            result = await self.blog_writer.execute(state, state.get("context", {}))
        
        state.update(result)
        state["execution_trace"].append({"agent": "content_generator", "status": "completed"})
        return state
    
    async def _run_tone_adaptation(self, state: dict) -> dict:
        result = await self.tone_adapter.execute(state, state.get("context", {}))
        state.update(result)
        state["execution_trace"].append({"agent": "tone_adapter", "status": "completed"})
        return state
    
    def _needs_seo(self, state: dict) -> str:
        """Check if SEO analysis should run."""
        request = state.get("user_request", {})
        return "yes" if request.get("include_seo", True) and state.get("blog_content") else "no"
    
    async def _run_seo_analysis(self, state: dict) -> dict:
        """Run basic SEO analysis on blog content."""
        blog_content = state.get("blog_content", {})
        content = blog_content.get("content", "")
        keywords = state.get("user_request", {}).get("target_keywords", [])
        
        if not content:
            return state
        
        # Calculate basic metrics
        word_count = len(content.split())
        
        # Calculate keyword density
        keyword_density = {}
        content_lower = content.lower()
        for kw in keywords:
            count = content_lower.count(kw.lower())
            density = (count / max(word_count, 1)) * 100
            keyword_density[kw] = round(density, 2)
        
        # Readability (simple Flesch-like estimate)
        sentences = content.count('.') + content.count('!') + content.count('?')
        avg_sentence_length = word_count / max(sentences, 1)
        readability_score = max(0, min(100, 206.835 - 1.015 * avg_sentence_length))
        
        state["seo_analysis"] = {
            "word_count": word_count,
            "keyword_density": keyword_density,
            "readability_score": round(readability_score, 1),
            "suggestions": self._generate_seo_suggestions(keyword_density, word_count),
        }
        
        state["execution_trace"].append({"agent": "seo_analyzer", "status": "completed"})
        return state
    
    def _generate_seo_suggestions(self, keyword_density: dict, word_count: int) -> list[str]:
        suggestions = []
        
        for kw, density in keyword_density.items():
            if density < 0.5:
                suggestions.append(f"Consider using '{kw}' more frequently (current: {density}%)")
            elif density > 3:
                suggestions.append(f"Reduce usage of '{kw}' to avoid keyword stuffing")
        
        if word_count < 600:
            suggestions.append("Consider expanding content to at least 600 words for better SEO")
        
        return suggestions
    
    async def _synthesize_results(self, state: dict) -> dict:
        """Synthesize final output."""
        request = state.get("user_request", {})
        content_type = request.get("content_type", "blog")
        
        if content_type == "blog" and state.get("blog_content"):
            state["final_output"] = {
                "type": "blog",
                "title": state["blog_content"].get("title"),
                "meta_description": state["blog_content"].get("meta_description"),
                "content": state["blog_content"].get("content"),
                "structure": {"headings": state["blog_content"].get("headings", [])},
                "seo_analysis": state.get("seo_analysis"),
            }
        elif content_type == "ad" and state.get("ad_copy"):
            state["final_output"] = {
                "type": "ad",
                "variations": state["ad_copy"],
            }
        elif content_type == "product" and state.get("product_description"):
            state["final_output"] = {
                "type": "product",
                **state["product_description"],
            }
        
        return state
    
    def _build_response(self, state: dict, execution_time_ms: int) -> dict:
        final = state.get("final_output", {})
        return {
            **final,
            "execution_time_ms": execution_time_ms,
            "alternative_titles": [],  # Could generate alternatives
        }

