"""
Response aggregator for combining outputs from multiple agents.
Synthesizes results into coherent user responses.
"""

import json
from typing import Any, Optional

import structlog
from pydantic import BaseModel

from app.core.llm_clients import LLMMessage, llm_client

logger = structlog.get_logger(__name__)


class AggregatedResponse(BaseModel):
    """Aggregated response from multiple agents."""
    content: str
    elves_used: list[str]
    execution_time_ms: int
    metadata: dict = {}
    suggestions: list[str] = []


class ResponseAggregator:
    """
    Aggregates responses from single or multiple Elf agents.
    Synthesizes into coherent, user-friendly output.
    """
    
    async def aggregate_single(
        self,
        elf_type: str,
        result: dict,
        user_request: str,
        execution_time_ms: int,
    ) -> AggregatedResponse:
        """
        Process single Elf response.
        
        Args:
            elf_type: Type of Elf that produced the result
            result: The Elf's output
            user_request: Original user request
            execution_time_ms: Execution time
            
        Returns:
            Formatted response for user
        """
        # Format response based on result structure
        content = self._format_single_result(elf_type, result)
        
        # Generate suggestions for follow-up
        suggestions = self._generate_suggestions(elf_type, result)
        
        return AggregatedResponse(
            content=content,
            elves_used=[elf_type],
            execution_time_ms=execution_time_ms,
            metadata={"raw_result": result},
            suggestions=suggestions,
        )
    
    async def aggregate_multiple(
        self,
        results: list[tuple[str, dict]],  # [(elf_type, result), ...]
        user_request: str,
        execution_time_ms: int,
    ) -> AggregatedResponse:
        """
        Synthesize results from multiple Elves.
        
        Args:
            results: List of (elf_type, result) tuples
            user_request: Original user request
            execution_time_ms: Total execution time
            
        Returns:
            Synthesized response for user
        """
        if len(results) == 1:
            elf_type, result = results[0]
            return await self.aggregate_single(
                elf_type, result, user_request, execution_time_ms
            )
        
        # Use LLM to synthesize multiple results
        content = await self._synthesize_results(results, user_request)
        
        # Combine suggestions from all Elves
        all_suggestions = []
        for elf_type, result in results:
            all_suggestions.extend(self._generate_suggestions(elf_type, result))
        
        return AggregatedResponse(
            content=content,
            elves_used=[elf_type for elf_type, _ in results],
            execution_time_ms=execution_time_ms,
            metadata={"raw_results": {elf: res for elf, res in results}},
            suggestions=list(set(all_suggestions))[:5],  # Dedupe and limit
        )
    
    def _format_single_result(self, elf_type: str, result: dict) -> str:
        """Format a single Elf's result for display."""
        
        formatters = {
            "social_media": self._format_social_media_result,
            "seo": self._format_seo_result,
            "copywriter": self._format_copywriter_result,
            "assistant": self._format_assistant_result,
        }
        
        formatter = formatters.get(elf_type, self._format_generic_result)
        return formatter(result)
    
    def _format_social_media_result(self, result: dict) -> str:
        """Format social media result."""
        parts = []
        
        if "post_variations" in result:
            variations = result["post_variations"]
            parts.append("## Generated Content Variations\n")
            
            for i, var in enumerate(variations, 1):
                parts.append(f"### Option {i}: {var.get('version', 'Variation')}")
                
                content = var.get("content", {})
                if isinstance(content, dict):
                    parts.append(f"\n{content.get('post_text', '')}")
                    if content.get("reasoning"):
                        parts.append(f"\n*Reasoning: {content['reasoning']}*")
                else:
                    parts.append(f"\n{content}")
                
                # Hashtags
                hashtags = var.get("hashtags", [])
                if hashtags:
                    tags = [h.get("tag", h) if isinstance(h, dict) else h for h in hashtags]
                    parts.append(f"\n**Hashtags:** {' '.join(tags)}")
                
                # Schedule
                schedule = var.get("posting_schedule", {})
                if schedule:
                    parts.append(f"\n**Best posting time:** {schedule.get('datetime', 'TBD')}")
                
                parts.append("\n---\n")
        
        if "recommendations" in result:
            rec = result["recommendations"]
            parts.append(f"\n## Recommendation\n")
            parts.append(f"**Best option:** {rec.get('best_variation', 'Option 1')}")
            parts.append(f"\n{rec.get('reason', '')}")
        
        return "\n".join(parts) if parts else json.dumps(result, indent=2)
    
    def _format_seo_result(self, result: dict) -> str:
        """Format SEO result."""
        parts = []
        
        # Overall score
        if "overall_score" in result:
            score = result["overall_score"]
            parts.append(f"## SEO Score: {score}/100\n")
        
        # Technical issues
        if "technical_issues" in result:
            issues = result["technical_issues"]
            parts.append("### Technical Issues\n")
            
            for issue in issues[:10]:  # Limit displayed issues
                severity = issue.get("severity", "medium").upper()
                parts.append(f"- **[{severity}]** {issue.get('description', '')}")
                parts.append(f"  - Fix: {issue.get('fix_suggestion', 'N/A')}")
            
            if len(issues) > 10:
                parts.append(f"\n*...and {len(issues) - 10} more issues*")
            parts.append("")
        
        # Keyword opportunities
        if "keyword_opportunities" in result:
            keywords = result["keyword_opportunities"]
            parts.append("### Keyword Opportunities\n")
            
            for kw in keywords[:5]:
                parts.append(
                    f"- **{kw.get('keyword', '')}** - "
                    f"Volume: {kw.get('volume', 'N/A')}, "
                    f"Difficulty: {kw.get('difficulty', 'N/A')}"
                )
            parts.append("")
        
        # Recommendations
        if "recommendations" in result:
            recs = result["recommendations"]
            parts.append("### Priority Actions\n")
            
            for rec in recs[:5]:
                parts.append(f"1. **{rec.get('action', '')}** ({rec.get('priority', 'medium')} priority)")
            parts.append("")
        
        return "\n".join(parts) if parts else json.dumps(result, indent=2)
    
    def _format_copywriter_result(self, result: dict) -> str:
        """Format copywriter result."""
        parts = []
        
        # Blog post format
        if "title" in result and "content" in result:
            parts.append(f"# {result['title']}\n")
            
            if "meta_description" in result:
                parts.append(f"*{result['meta_description']}*\n")
            
            parts.append(result["content"])
            
            # SEO analysis
            if "seo_analysis" in result:
                seo = result["seo_analysis"]
                parts.append("\n---\n### SEO Analysis")
                parts.append(f"- Readability Score: {seo.get('readability_score', 'N/A')}")
                if seo.get("suggestions"):
                    parts.append("- Suggestions: " + ", ".join(seo["suggestions"]))
        
        # Ad copy format
        elif "variations" in result:
            parts.append("## Ad Copy Variations\n")
            for i, var in enumerate(result["variations"], 1):
                parts.append(f"### Variation {i}")
                parts.append(f"**Headline:** {var.get('headline', '')}")
                parts.append(f"**Body:** {var.get('body', var.get('description', ''))}")
                parts.append("")
        
        else:
            parts.append(json.dumps(result, indent=2))
        
        return "\n".join(parts)
    
    def _format_assistant_result(self, result: dict) -> str:
        """Format assistant result."""
        parts = []
        
        # Task management format
        if "tasks" in result:
            parts.append("## Tasks\n")
            for task in result["tasks"]:
                status = task.get("status", "pending")
                parts.append(f"- [{status}] {task.get('title', task.get('description', ''))}")
        
        # Email format
        elif "email" in result or "draft" in result:
            email = result.get("email") or result.get("draft", {})
            if isinstance(email, dict):
                parts.append(f"**Subject:** {email.get('subject', '')}\n")
                parts.append(email.get("body", ""))
            else:
                parts.append(email)
        
        # Research format
        elif "findings" in result or "summary" in result:
            parts.append("## Research Summary\n")
            parts.append(result.get("summary", ""))
            
            if "key_points" in result:
                parts.append("\n### Key Points")
                for point in result["key_points"]:
                    parts.append(f"- {point}")
            
            if "sources" in result:
                parts.append("\n### Sources")
                for source in result["sources"]:
                    parts.append(f"- {source}")
        
        else:
            # Generic format
            if isinstance(result, dict):
                if "response" in result:
                    parts.append(result["response"])
                elif "content" in result:
                    parts.append(result["content"])
                else:
                    parts.append(json.dumps(result, indent=2))
            else:
                parts.append(str(result))
        
        return "\n".join(parts)
    
    def _format_generic_result(self, result: dict) -> str:
        """Generic result formatting."""
        if isinstance(result, str):
            return result
        
        if "content" in result:
            return result["content"]
        if "response" in result:
            return result["response"]
        if "result" in result:
            return str(result["result"])
        
        return json.dumps(result, indent=2)
    
    def _generate_suggestions(self, elf_type: str, result: dict) -> list[str]:
        """Generate follow-up suggestions based on result."""
        suggestions_map = {
            "social_media": [
                "Schedule this post for optimal timing",
                "Create variations for other platforms",
                "Analyze competitor content for inspiration",
            ],
            "seo": [
                "Run a detailed technical audit",
                "Research more keyword opportunities",
                "Analyze competitor backlinks",
            ],
            "copywriter": [
                "Generate alternative headlines",
                "Adapt content for different audiences",
                "Create social media snippets from this content",
            ],
            "assistant": [
                "Add follow-up reminders",
                "Create related tasks",
                "Research more on this topic",
            ],
        }
        
        return suggestions_map.get(elf_type, [])[:3]
    
    async def _synthesize_results(
        self,
        results: list[tuple[str, dict]],
        user_request: str,
    ) -> str:
        """Use LLM to synthesize multiple Elf results."""
        
        # Build context from all results
        results_context = []
        for elf_type, result in results:
            formatted = self._format_single_result(elf_type, result)
            results_context.append(f"## {elf_type.title()} Output\n{formatted}")
        
        system_prompt = """You are a helpful assistant synthesizing outputs from multiple AI agents.
Your task is to combine their outputs into a coherent, well-organized response.
Maintain all important information while removing redundancy.
Present the information in a clear, user-friendly format."""
        
        user_prompt = f"""The user asked: "{user_request}"

Here are the outputs from different AI agents:

{chr(10).join(results_context)}

Please synthesize these into a single, coherent response that addresses the user's needs."""
        
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        
        response = await llm_client.generate(messages)
        return response.content


# Global aggregator instance
response_aggregator = ResponseAggregator()

