"""
Multi-Agent System for Elvz.ai

Three-Layer Hierarchy:
1. Platform Orchestrator - Routes requests to appropriate Elves
2. Elf Agents - Domain-specific macro agents (Social Media, SEO, Copywriter, Assistant)
3. Mini-Agents - Task-specific micro agents within each Elf
"""

from app.agents.platform_orchestrator import PlatformOrchestrator, orchestrator

__all__ = ["PlatformOrchestrator", "orchestrator"]

