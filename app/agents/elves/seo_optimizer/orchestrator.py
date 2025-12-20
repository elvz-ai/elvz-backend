"""
SEO Optimizer Elf - Audits websites and provides SEO recommendations.
Uses sequential workflow with conditional branches.
"""

import asyncio
import json
import time
from typing import Any, Optional

import structlog
from langgraph.graph import END, StateGraph

from app.agents.elves.base import BaseElf
from app.core.llm_clients import LLMMessage, llm_client
from app.tools.registry import tool_registry

logger = structlog.get_logger(__name__)


class TechnicalSEOAuditor:
    """Audits technical SEO aspects of a website."""
    
    name = "technical_seo_auditor"
    
    async def execute(self, state: dict, context: dict) -> dict:
        """Run technical SEO audit."""
        request = state.get("user_request", {})
        website_url = request.get("website_url", "")
        
        logger.info("Technical SEO audit", url=website_url)
        
        # Use SEO audit tool
        try:
            from app.tools.seo_tools import TechnicalAuditInput
            
            tool = tool_registry.get("technical_seo_audit")
            if tool:
                input_data = TechnicalAuditInput(
                    website_url=website_url,
                    check_sitemap=True,
                    check_robots=True,
                )
                result = await tool.execute(input_data)
                
                if result.success:
                    return {"technical_issues": result.data.get("issues", [])}
        except Exception as e:
            logger.error("Technical audit failed", error=str(e))
        
        # Fallback to LLM-based analysis
        return {"technical_issues": await self._llm_audit(website_url)}
    
    async def _llm_audit(self, url: str) -> list[dict]:
        """LLM-based technical SEO analysis."""
        system_prompt = """You are a technical SEO expert. Analyze the given URL and identify potential technical issues.
Respond with JSON: {"issues": [{"type": str, "severity": "critical/high/medium/low", "description": str, "fix_suggestion": str}]}"""
        
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=f"Analyze technical SEO for: {url}"),
        ]
        
        response = await llm_client.generate_fast(messages, json_mode=True)
        
        try:
            result = json.loads(response.content)
            return result.get("issues", [])
        except:
            return []


class KeywordAnalyzer:
    """Analyzes keywords and opportunities."""
    
    name = "keyword_analyzer"
    
    async def execute(self, state: dict, context: dict) -> dict:
        """Analyze keyword opportunities."""
        request = state.get("user_request", {})
        website_url = request.get("website_url", "")
        target_keywords = request.get("target_keywords", [])
        
        logger.info("Keyword analysis")
        
        # Use keyword tool
        try:
            from app.tools.seo_tools import KeywordResearchInput
            
            tool = tool_registry.get("keyword_research")
            if tool:
                seed_keywords = target_keywords or self._extract_keywords(website_url)
                input_data = KeywordResearchInput(
                    seed_keywords=seed_keywords,
                    target_url=website_url,
                )
                result = await tool.execute(input_data)
                
                if result.success:
                    return {
                        "keyword_opportunities": result.data.get("keywords", []),
                        "related_topics": result.data.get("related_topics", []),
                    }
        except Exception as e:
            logger.error("Keyword analysis failed", error=str(e))
        
        return {"keyword_opportunities": [], "related_topics": []}
    
    def _extract_keywords(self, url: str) -> list[str]:
        """Extract keywords from URL."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "").split(".")[0]
        return [domain, "business", "services"]


class CompetitorSEOAnalyzer:
    """Analyzes competitor SEO."""
    
    name = "competitor_seo_analyzer"
    
    async def execute(self, state: dict, context: dict) -> dict:
        """Analyze competitor SEO."""
        request = state.get("user_request", {})
        competitor_urls = request.get("competitor_urls", [])
        
        if not competitor_urls:
            return {"competitor_analysis": None}
        
        logger.info("Competitor SEO analysis", count=len(competitor_urls))
        
        # Use backlink tool for comparison
        try:
            from app.tools.seo_tools import BacklinkAnalysisInput
            
            tool = tool_registry.get("backlink_analysis")
            if tool:
                input_data = BacklinkAnalysisInput(
                    website_url=request.get("website_url", ""),
                    compare_with=competitor_urls,
                )
                result = await tool.execute(input_data)
                
                if result.success:
                    return {"competitor_analysis": result.data}
        except Exception as e:
            logger.error("Competitor analysis failed", error=str(e))
        
        return {"competitor_analysis": {"competitors_analyzed": competitor_urls}}


class ContentOptimizer:
    """Provides content optimization suggestions."""
    
    name = "content_optimizer"
    
    async def execute(self, state: dict, context: dict) -> dict:
        """Generate content optimization suggestions."""
        request = state.get("user_request", {})
        website_url = request.get("website_url", "")
        keywords = state.get("keyword_opportunities", [])
        
        logger.info("Content optimization")
        
        # Use content analysis tool
        try:
            from app.tools.seo_tools import ContentAnalysisInput
            
            tool = tool_registry.get("content_analysis")
            if tool and keywords:
                target_kws = [k.get("keyword", k) if isinstance(k, dict) else k 
                             for k in keywords[:5]]
                input_data = ContentAnalysisInput(
                    page_url=website_url,
                    target_keywords=target_kws,
                )
                result = await tool.execute(input_data)
                
                if result.success:
                    return {"content_suggestions": result.data.get("suggestions", [])}
        except Exception as e:
            logger.error("Content optimization failed", error=str(e))
        
        return {"content_suggestions": self._generate_generic_suggestions()}
    
    def _generate_generic_suggestions(self) -> list[str]:
        return [
            "Add target keywords to page title",
            "Optimize meta description with call-to-action",
            "Add internal links to related pages",
            "Include heading tags (H1, H2, H3) with keywords",
            "Improve page load speed",
        ]


class MetaSchemaGenerator:
    """Generates optimized meta tags and schema markup."""
    
    name = "meta_schema_generator"
    
    async def execute(self, state: dict, context: dict) -> dict:
        """Generate meta tags and schema suggestions."""
        request = state.get("user_request", {})
        keywords = state.get("keyword_opportunities", [])
        
        logger.info("Meta/Schema generation")
        
        system_prompt = """You are an SEO expert specializing in meta tags and schema markup.
Generate optimized meta tags and schema suggestions.
Respond with JSON:
{
    "meta_tags": {
        "title": "optimized title (50-60 chars)",
        "description": "optimized description (150-160 chars)",
        "og_title": "social title",
        "og_description": "social description"
    },
    "schema_suggestions": [
        {"type": "schema type", "properties": ["key properties to include"]}
    ]
}"""
        
        kw_list = [k.get("keyword", k) if isinstance(k, dict) else k for k in keywords[:5]]
        
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(
                role="user",
                content=f"Generate meta tags for a page targeting: {', '.join(kw_list) or 'general business'}",
            ),
        ]
        
        response = await llm_client.generate_fast(messages, json_mode=True)
        
        try:
            result = json.loads(response.content)
            return {
                "meta_suggestions": result.get("meta_tags", {}),
                "schema_suggestions": result.get("schema_suggestions", []),
            }
        except:
            return {"meta_suggestions": {}, "schema_suggestions": []}


class SEOOptimizerElf(BaseElf):
    """
    SEO Optimizer Elf - Audits websites and provides SEO recommendations.
    
    Mini-Agents:
    1. Technical SEO Auditor - Checks technical issues
    2. Keyword Analyzer - Researches keywords
    3. Competitor SEO Analyzer - Analyzes competitors
    4. Content Optimizer - Suggests content improvements
    5. Meta/Schema Generator - Creates meta tags and schema
    
    Workflow: Sequential with conditional branches
    """
    
    name = "seo_optimizer"
    description = "Audits websites and provides SEO optimization recommendations"
    version = "1.0"
    
    def __init__(self):
        self.technical_auditor = TechnicalSEOAuditor()
        self.keyword_analyzer = KeywordAnalyzer()
        self.competitor_analyzer = CompetitorSEOAnalyzer()
        self.content_optimizer = ContentOptimizer()
        self.meta_generator = MetaSchemaGenerator()
        
        super().__init__()
    
    def _setup_workflow(self) -> None:
        """Set up the LangGraph workflow."""
        workflow = StateGraph(dict)
        
        workflow.add_node("technical_audit", self._run_technical_audit)
        workflow.add_node("keyword_analysis", self._run_keyword_analysis)
        workflow.add_node("competitor_analysis", self._run_competitor_analysis)
        workflow.add_node("content_optimization", self._run_content_optimization)
        workflow.add_node("meta_generation", self._run_meta_generation)
        workflow.add_node("synthesize", self._synthesize_results)
        
        workflow.set_entry_point("technical_audit")
        workflow.add_edge("technical_audit", "keyword_analysis")
        workflow.add_conditional_edges(
            "keyword_analysis",
            self._should_analyze_competitors,
            {"yes": "competitor_analysis", "no": "content_optimization"}
        )
        workflow.add_edge("competitor_analysis", "content_optimization")
        workflow.add_edge("content_optimization", "meta_generation")
        workflow.add_edge("meta_generation", "synthesize")
        workflow.add_edge("synthesize", END)
        
        self._workflow = workflow.compile()
    
    async def execute(self, request: dict, context: dict) -> dict:
        """Execute SEO audit workflow."""
        start_time = time.time()
        
        initial_state = {
            "user_request": request,
            "context": context,
            "technical_issues": [],
            "keyword_opportunities": [],
            "competitor_analysis": None,
            "content_suggestions": [],
            "meta_suggestions": {},
            "schema_suggestions": [],
            "execution_trace": [],
            "errors": [],
        }
        
        logger.info("SEO Optimizer executing", url=request.get("website_url"))
        
        try:
            final_state = await self._workflow.ainvoke(initial_state)
            execution_time_ms = int((time.time() - start_time) * 1000)
            return self._build_response(final_state, execution_time_ms)
        except Exception as e:
            logger.error("SEO workflow failed", error=str(e))
            return {"error": str(e), "overall_score": 0}
    
    async def _run_technical_audit(self, state: dict) -> dict:
        result = await self.technical_auditor.execute(state, state.get("context", {}))
        state.update(result)
        state["execution_trace"].append({"agent": "technical_audit", "status": "completed"})
        return state
    
    async def _run_keyword_analysis(self, state: dict) -> dict:
        result = await self.keyword_analyzer.execute(state, state.get("context", {}))
        state.update(result)
        state["execution_trace"].append({"agent": "keyword_analysis", "status": "completed"})
        return state
    
    def _should_analyze_competitors(self, state: dict) -> str:
        """Check if competitor analysis should run."""
        request = state.get("user_request", {})
        return "yes" if request.get("include_competitors") else "no"
    
    async def _run_competitor_analysis(self, state: dict) -> dict:
        result = await self.competitor_analyzer.execute(state, state.get("context", {}))
        state.update(result)
        state["execution_trace"].append({"agent": "competitor_analysis", "status": "completed"})
        return state
    
    async def _run_content_optimization(self, state: dict) -> dict:
        result = await self.content_optimizer.execute(state, state.get("context", {}))
        state.update(result)
        state["execution_trace"].append({"agent": "content_optimization", "status": "completed"})
        return state
    
    async def _run_meta_generation(self, state: dict) -> dict:
        result = await self.meta_generator.execute(state, state.get("context", {}))
        state.update(result)
        state["execution_trace"].append({"agent": "meta_generation", "status": "completed"})
        return state
    
    async def _synthesize_results(self, state: dict) -> dict:
        """Synthesize all results into final report."""
        
        # Calculate overall score
        issues = state.get("technical_issues", [])
        score = 100
        
        severity_deductions = {"critical": 25, "high": 15, "medium": 5, "low": 2}
        for issue in issues:
            severity = issue.get("severity", "low")
            score -= severity_deductions.get(severity, 2)
        
        score = max(0, score)
        
        # Generate prioritized recommendations
        recommendations = []
        
        # From technical issues
        for issue in issues[:5]:
            recommendations.append({
                "priority": issue.get("severity", "medium"),
                "category": "technical",
                "action": issue.get("fix_suggestion", issue.get("description", "")),
                "impact": "high" if issue.get("severity") in ["critical", "high"] else "medium",
            })
        
        # From content suggestions
        for suggestion in state.get("content_suggestions", [])[:3]:
            recommendations.append({
                "priority": "medium",
                "category": "content",
                "action": suggestion,
                "impact": "medium",
            })
        
        state["final_output"] = {
            "overall_score": score,
            "technical_issues": issues,
            "keyword_opportunities": state.get("keyword_opportunities", []),
            "content_gaps": state.get("related_topics", []),
            "recommendations": recommendations,
            "meta_suggestions": state.get("meta_suggestions", {}),
            "schema_suggestions": state.get("schema_suggestions", []),
        }
        
        if state.get("competitor_analysis"):
            state["final_output"]["competitor_analysis"] = state["competitor_analysis"]
        
        return state
    
    def _build_response(self, state: dict, execution_time_ms: int) -> dict:
        final = state.get("final_output", {})
        return {
            "overall_score": final.get("overall_score", 0),
            "technical_issues": final.get("technical_issues", []),
            "keyword_opportunities": final.get("keyword_opportunities", []),
            "content_gaps": final.get("content_gaps", []),
            "recommendations": final.get("recommendations", []),
            "meta_suggestions": final.get("meta_suggestions", {}),
            "schema_suggestions": final.get("schema_suggestions", []),
            "competitor_analysis": final.get("competitor_analysis"),
            "execution_time_ms": execution_time_ms,
        }

