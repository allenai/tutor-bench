"""LLM provider abstraction and shared parsing/cost utilities.

Submodules:
- ``llm_utils``: provider-agnostic JSON extraction, cost computation, retry sleep
- ``llm_client``: ``ModelClient`` protocol, batch/sync runners (added in Phase 1.4)
- ``anthropic``: concrete ``AnthropicClient`` (added in Phase 1.5)
"""
