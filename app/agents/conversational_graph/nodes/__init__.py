"""
Graph Nodes for the Conversational Graph.

Each node is a callable that takes the state and returns updated state.
"""

from app.agents.conversational_graph.nodes.guardrail import guardrail_node
from app.agents.conversational_graph.nodes.intent import intent_classifier_node
from app.agents.conversational_graph.nodes.decomposer import query_decomposer_node
from app.agents.conversational_graph.nodes.memory import memory_retriever_node
from app.agents.conversational_graph.nodes.context import context_builder_node
from app.agents.conversational_graph.nodes.router import router_node
from app.agents.conversational_graph.nodes.follow_up import (
    follow_up_detector_node,
    follow_up_generator_node,
)
from app.agents.conversational_graph.nodes.data_checker import data_checker_node
from app.agents.conversational_graph.nodes.orchestrator import multi_platform_orchestrator_node
from app.agents.conversational_graph.nodes.aggregator import stream_aggregator_node
from app.agents.conversational_graph.nodes.saver import memory_saver_node

__all__ = [
    "guardrail_node",
    "intent_classifier_node",
    "query_decomposer_node",
    "memory_retriever_node",
    "context_builder_node",
    "router_node",
    "follow_up_detector_node",
    "follow_up_generator_node",
    "data_checker_node",
    "multi_platform_orchestrator_node",
    "stream_aggregator_node",
    "memory_saver_node",
]
