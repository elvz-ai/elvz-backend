"""
Prompt engineering utilities and templates.
Provides structured prompt building for all agents.
"""

from typing import Any, Optional

from pydantic import BaseModel


class PromptTemplate(BaseModel):
    """Template for agent prompts."""
    name: str
    version: str
    system_template: str
    user_template: str
    output_format: Optional[str] = None
    

class PromptBuilder:
    """
    Builds structured prompts for LLM agents.
    
    Prompt Structure:
    1. System Instruction (Character)
    2. Context (Dynamic)
    3. Task (Specific)
    4. Output Format
    """
    
    @staticmethod
    def build_system_prompt(
        role: str,
        expertise: list[str],
        guidelines: list[str],
        constraints: list[str],
    ) -> str:
        """Build system prompt with agent character."""
        prompt_parts = [
            f"You are {role}.",
            "",
            "## Expertise",
            *[f"- {e}" for e in expertise],
            "",
            "## Guidelines",
            *[f"- {g}" for g in guidelines],
            "",
            "## Constraints",
            *[f"- {c}" for c in constraints],
        ]
        return "\n".join(prompt_parts)
    
    @staticmethod
    def build_context_block(
        user_context: Optional[dict] = None,
        retrieved_knowledge: Optional[list[str]] = None,
        tool_results: Optional[dict] = None,
        conversation_history: Optional[list[dict]] = None,
        brand_voice: Optional[str] = None,
    ) -> str:
        """Build dynamic context block."""
        context_parts = []
        
        if user_context:
            context_parts.append("## User Context")
            for key, value in user_context.items():
                context_parts.append(f"- {key}: {value}")
            context_parts.append("")
        
        if brand_voice:
            context_parts.append("## Brand Voice")
            context_parts.append(brand_voice)
            context_parts.append("")
        
        if retrieved_knowledge:
            context_parts.append("## Relevant Knowledge")
            for i, knowledge in enumerate(retrieved_knowledge, 1):
                context_parts.append(f"{i}. {knowledge}")
            context_parts.append("")
        
        if tool_results:
            context_parts.append("## Data from Tools")
            for tool_name, result in tool_results.items():
                context_parts.append(f"### {tool_name}")
                context_parts.append(str(result))
            context_parts.append("")
        
        if conversation_history:
            context_parts.append("## Previous Conversation")
            for msg in conversation_history[-5:]:  # Last 5 messages
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")[:500]  # Truncate
                context_parts.append(f"{role}: {content}")
            context_parts.append("")
        
        return "\n".join(context_parts) if context_parts else ""
    
    @staticmethod
    def build_task_block(
        task_description: str,
        requirements: list[str],
        examples: Optional[list[dict]] = None,
    ) -> str:
        """Build task-specific block."""
        task_parts = [
            "## Task",
            task_description,
            "",
            "## Requirements",
            *[f"- {r}" for r in requirements],
        ]
        
        if examples:
            task_parts.append("")
            task_parts.append("## Examples")
            for i, example in enumerate(examples, 1):
                task_parts.append(f"### Example {i}")
                if "input" in example:
                    task_parts.append(f"Input: {example['input']}")
                if "output" in example:
                    task_parts.append(f"Output: {example['output']}")
                task_parts.append("")
        
        return "\n".join(task_parts)
    
    @staticmethod
    def build_output_format(
        format_type: str = "json",
        schema: Optional[dict] = None,
        field_descriptions: Optional[dict] = None,
    ) -> str:
        """Build output format specification."""
        format_parts = ["## Output Format"]
        
        if format_type == "json":
            format_parts.append("Respond with valid JSON only. No additional text.")
            if schema:
                import json
                format_parts.append("")
                format_parts.append("Schema:")
                format_parts.append("```json")
                format_parts.append(json.dumps(schema, indent=2))
                format_parts.append("```")
        else:
            format_parts.append(f"Respond in {format_type} format.")
        
        if field_descriptions:
            format_parts.append("")
            format_parts.append("Field Descriptions:")
            for field, desc in field_descriptions.items():
                format_parts.append(f"- {field}: {desc}")
        
        return "\n".join(format_parts)
    
    @classmethod
    def build_full_prompt(
        cls,
        role: str,
        expertise: list[str],
        guidelines: list[str],
        constraints: list[str],
        task_description: str,
        requirements: list[str],
        user_context: Optional[dict] = None,
        retrieved_knowledge: Optional[list[str]] = None,
        tool_results: Optional[dict] = None,
        brand_voice: Optional[str] = None,
        examples: Optional[list[dict]] = None,
        output_schema: Optional[dict] = None,
    ) -> tuple[str, str]:
        """
        Build complete prompt with system and user messages.
        
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = cls.build_system_prompt(
            role=role,
            expertise=expertise,
            guidelines=guidelines,
            constraints=constraints,
        )
        
        user_parts = []
        
        context = cls.build_context_block(
            user_context=user_context,
            retrieved_knowledge=retrieved_knowledge,
            tool_results=tool_results,
            brand_voice=brand_voice,
        )
        if context:
            user_parts.append(context)
        
        task = cls.build_task_block(
            task_description=task_description,
            requirements=requirements,
            examples=examples,
        )
        user_parts.append(task)
        
        if output_schema:
            output_format = cls.build_output_format(
                format_type="json",
                schema=output_schema,
            )
            user_parts.append(output_format)
        
        user_prompt = "\n".join(user_parts)
        
        return system_prompt, user_prompt


# Pre-defined prompt templates for common agent types
AGENT_TEMPLATES = {
    "content_strategy": PromptTemplate(
        name="Content Strategy Agent",
        version="1.0",
        system_template="""You are an expert content strategist specializing in {platform} marketing.
Your role is to create strategic content briefs that maximize engagement and achieve business goals.

## Expertise
- Content strategy and planning
- Audience psychology and engagement
- Platform-specific best practices
- Brand voice development

## Guidelines
- Always consider the target audience first
- Recommend data-driven strategies
- Balance creativity with brand consistency
- Provide clear, actionable recommendations

## Constraints
- Stay within platform guidelines
- Respect brand voice and values
- Be specific and measurable in recommendations""",
        user_template="""Create a content strategy brief for the following request:

## User Request
{user_request}

## Brand Context
{brand_context}

## Goals
{goals}

Provide a strategic brief including:
1. Key messaging pillars
2. Recommended content angles
3. Call-to-action strategy
4. Tone and voice recommendations""",
    ),
    
    "content_generator": PromptTemplate(
        name="Content Generator Agent",
        version="1.0",
        system_template="""You are a world-class content creator specializing in {content_type} for {platform}.
Your role is to generate engaging, on-brand content that resonates with the target audience.

## Expertise
- Compelling copywriting
- Platform-specific content formats
- Emotional storytelling
- Conversion optimization

## Guidelines
- Write in the specified brand voice
- Hook readers in the first line
- Include clear calls-to-action
- Optimize for the platform's algorithm

## Constraints
- Stay within character/word limits
- Maintain brand consistency
- Avoid controversial topics
- Use inclusive language""",
        user_template="""Generate content based on the following strategy brief:

## Strategy Brief
{strategy_brief}

## Brand Voice
{brand_voice}

## Requirements
- Platform: {platform}
- Content Type: {content_type}
- Tone: {tone}
- Length: {length}

Create {variations} variations:
1. Hook-focused version (attention-grabbing opening)
2. Story-focused version (narrative approach)
3. Value-focused version (educational/helpful)

For each variation, explain your creative reasoning.""",
    ),
    
    "hashtag_research": PromptTemplate(
        name="Hashtag Research Agent",
        version="1.0",
        system_template="""You are a social media optimization specialist focusing on hashtag strategy.
Your role is to research and recommend optimal hashtags for maximum reach and engagement.

## Expertise
- Hashtag research and analysis
- Platform algorithm understanding
- Trend identification
- Niche community discovery

## Guidelines
- Balance popular and niche hashtags
- Consider hashtag relevance to content
- Recommend mix of volumes (high/medium/low)
- Avoid banned or overused hashtags

## Constraints
- Maximum {max_hashtags} hashtags
- Must be relevant to content
- Platform-appropriate""",
        user_template="""Research hashtags for the following content:

## Content
{content}

## Platform
{platform}

## Niche/Industry
{niche}

## Target Audience
{audience}

Recommend 5-7 hashtags with:
- Volume estimate (low/medium/high)
- Relevance score (1-10)
- Strategic rationale""",
    ),
}

