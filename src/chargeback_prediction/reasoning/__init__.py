"""LLM reasoning layer: a risk rationale and representment strategy for each dispute.

Rules decide what is allowed (:mod:`.playbook`), the LLM writes the explanation behind a
provider-agnostic interface (:mod:`.llm`), and every reply is validated with a deterministic
fallback (:mod:`.reasoner`).
"""

from chargeback_prediction.reasoning.llm import (
    CompletionRequest,
    LLMClient,
    LLMError,
    Message,
    MockLLM,
    OpenAICompatibleClient,
    ScriptedLLM,
)
from chargeback_prediction.reasoning.playbook import (
    Leaning,
    Recommendation,
    Signal,
    Strategy,
    collect_signals,
    recommend,
)
from chargeback_prediction.reasoning.prompt import DisputeContext, build_messages, context_payload
from chargeback_prediction.reasoning.reasoner import (
    Confidence,
    RationaleError,
    RiskRationale,
    RiskReasoner,
    Source,
    parse_reply,
    template_rationale,
)
from chargeback_prediction.reasoning.run import (
    BacktestRow,
    backtest_playbook,
    build_contexts,
    dispute_reason_codes,
    parse_reason_code,
    render_backtest,
)

__all__ = [
    "BacktestRow",
    "CompletionRequest",
    "Confidence",
    "DisputeContext",
    "LLMClient",
    "LLMError",
    "Leaning",
    "Message",
    "MockLLM",
    "OpenAICompatibleClient",
    "RationaleError",
    "Recommendation",
    "RiskRationale",
    "RiskReasoner",
    "ScriptedLLM",
    "Signal",
    "Source",
    "Strategy",
    "backtest_playbook",
    "build_contexts",
    "build_messages",
    "collect_signals",
    "context_payload",
    "dispute_reason_codes",
    "parse_reason_code",
    "parse_reply",
    "recommend",
    "render_backtest",
    "template_rationale",
]
