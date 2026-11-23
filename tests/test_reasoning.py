from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

from chargeback_prediction.cli import API_KEY_ENV, main
from chargeback_prediction.explain import ExplainConfig, Factor, TransactionExplanation
from chargeback_prediction.explain.run import load_model
from chargeback_prediction.features import FEATURE_NAMES
from chargeback_prediction.modeling import FeatureMatrix, ModelKind
from chargeback_prediction.reason_codes import (
    DisputeCategory,
    Network,
    ReasonCode,
    all_reason_codes,
    lookup,
)
from chargeback_prediction.reasoning import (
    CompletionRequest,
    Confidence,
    DisputeContext,
    Leaning,
    LLMClient,
    LLMError,
    Message,
    MockLLM,
    OpenAICompatibleClient,
    RationaleError,
    Recommendation,
    RiskReasoner,
    ScriptedLLM,
    Source,
    Strategy,
    backtest_playbook,
    build_contexts,
    build_messages,
    collect_signals,
    context_payload,
    dispute_reason_codes,
    parse_reason_code,
    parse_reply,
    recommend,
    render_backtest,
    template_rationale,
)
from chargeback_prediction.reasoning.llm import extract_context
from chargeback_prediction.reasoning.prompt import (
    MAX_KEY_FACTORS,
    RATIONALE_MAX_CHARS,
    SYSTEM_PROMPT,
    citable_features,
)
from chargeback_prediction.reasoning.reasoner import feature_values
from chargeback_prediction.synthetic import read_dataset
from chargeback_prediction.synthetic.schema import Dispute

VISA_FRAUD = lookup(Network.VISA, "10.4")
MC_FRAUD = lookup(Network.MASTERCARD, "4837")
VISA_RECURRING = lookup(Network.VISA, "13.2")

# ---------------------------------------------------------------------------- helpers


def features(**overrides: float) -> dict[str, float]:
    """A neutral transaction: 30-day-old account, nothing suspicious, no loyalty history."""
    base = dict.fromkeys(FEATURE_NAMES, 0.0)
    base.update(account_age_days=30.0, amount_usd=19.99, is_renewal=1.0)
    base.update(overrides)
    return base


STOLEN = {"account_age_days": 0.5, "email_disposable": 1.0, "ip_bin_mismatch": 1.0}
LOYAL = {"account_age_days": 400.0, "prior_approved_renewals": 6.0}


def explanation(
    *factors: Factor, probability: float = 0.08, txn: str = "txn_1"
) -> TransactionExplanation:
    return TransactionExplanation(
        transaction_id=txn,
        probability=probability,
        baseline_probability=0.02,
        base_value=-3.9,
        factors=factors,
        rest=0.0,
    )


def context(
    reason: ReasonCode = VISA_FRAUD, *, feats: dict[str, float] | None = None
) -> DisputeContext:
    exp = explanation(
        Factor("account_age_days", 0.5, 2.1),
        Factor("email_disposable", 1.0, 0.9),
        Factor("plan_is_annual", 0.0, -0.3),
    )
    return DisputeContext(exp, reason, feats if feats is not None else features(**STOLEN))


def reply(**overrides: Any) -> str:
    body: dict[str, Any] = {
        "strategy": "accept",
        "confidence": "high",
        "rationale": "New account with a disposable email and a foreign IP: a stolen card.",
        "key_factors": ["account_age_days", "email_disposable"],
    }
    body.update(overrides)
    return json.dumps(body)


# --------------------------------------------------------------------------- playbook


@pytest.mark.parametrize("reason", all_reason_codes(), ids=lambda rc: rc.key)
def test_charge_after_cancel_request_is_always_accepted(reason: ReasonCode) -> None:
    rec = recommend(reason, features(charged_after_cancel_request=1.0, **LOYAL))
    assert rec.strategy is Strategy.ACCEPT
    assert rec.allowed == {Strategy.ACCEPT}, "merchant error must not be argued by the LLM"
    assert rec.count(Leaning.MERCHANT_ERROR) == 1
    assert rec.evidence == ("refund confirmation, or accept the chargeback without a response",)


@pytest.mark.parametrize("reason", [VISA_FRAUD, MC_FRAUD], ids=lambda rc: rc.key)
def test_fraud_code_with_stolen_card_signals_is_accepted(reason: ReasonCode) -> None:
    rec = recommend(reason, features(**STOLEN))
    assert rec.strategy is Strategy.ACCEPT
    assert rec.allowed == {Strategy.ACCEPT, Strategy.REVIEW}
    assert rec.count(Leaning.TRUE_FRAUD) == 3


def test_fraud_code_with_loyal_customer_is_represented_with_compelling_evidence() -> None:
    rec = recommend(VISA_FRAUD, features(**LOYAL))
    assert rec.strategy is Strategy.REPRESENT
    assert rec.allowed == {Strategy.REPRESENT, Strategy.REVIEW}
    assert "friendly fraud" in rec.reasons[0]
    assert any("Compelling Evidence 3.0" in e for e in rec.evidence)


def test_conflicting_fraud_signals_go_to_review_with_full_latitude() -> None:
    rec = recommend(MC_FRAUD, features(**LOYAL, ip_bin_mismatch=1.0, email_disposable=1.0))
    assert rec.strategy is Strategy.REVIEW
    assert rec.allowed == set(Strategy)


def test_consumer_dispute_defaults_to_represent() -> None:
    rec = recommend(VISA_RECURRING, features())
    assert rec.strategy is Strategy.REPRESENT
    assert any("cancellation policy" in e for e in rec.evidence)


def test_consumer_dispute_with_stolen_card_signals_goes_to_review() -> None:
    rec = recommend(VISA_RECURRING, features(**STOLEN))
    assert rec.strategy is Strategy.REVIEW


def test_unknown_category_goes_to_review() -> None:
    processing = ReasonCode(Network.VISA, "12.6", "Duplicate", DisputeCategory.PROCESSING, False)
    rec = recommend(processing, features())
    assert rec.strategy is Strategy.REVIEW
    assert rec.evidence == ()


def test_signals_render_values_in_natural_units() -> None:
    signals = collect_signals(features(attempts_24h=5, declines_24h=3, prior_disputes=2))
    details = {s.feature: s.detail for s in signals}
    assert details["attempts_24h"] == "5 authorization attempts in the previous 24 h"
    assert details["declines_24h"].startswith("3 declines")
    assert details["prior_disputes"].startswith("2 earlier dispute(s)")


def test_established_account_does_not_count_with_ip_mismatch() -> None:
    signals = collect_signals(features(account_age_days=500, ip_bin_mismatch=1.0))
    assert [s.leaning for s in signals] == [Leaning.TRUE_FRAUD]


def test_missing_feature_is_reported_by_name() -> None:
    feats = features()
    del feats["email_disposable"]
    with pytest.raises(KeyError, match="email_disposable"):
        collect_signals(feats)


def test_recommendation_rejects_strategy_outside_allowed() -> None:
    with pytest.raises(ValueError, match="allowed"):
        Recommendation(Strategy.ACCEPT, frozenset({Strategy.REVIEW}), (), (), ())


def test_every_catalog_code_has_evidence_guidance() -> None:
    for reason in all_reason_codes():
        rec = recommend(reason, features())
        assert rec.evidence, reason.key


# ---------------------------------------------------------------------- reply parsing

ALLOWED = frozenset({Strategy.ACCEPT, Strategy.REVIEW})
CITABLE = frozenset({"account_age_days", "email_disposable", "ip_bin_mismatch"})


def test_parse_valid_reply() -> None:
    parsed = parse_reply(reply(), allowed=ALLOWED, citable=CITABLE)
    assert parsed.strategy is Strategy.ACCEPT
    assert parsed.confidence is Confidence.HIGH
    assert parsed.key_factors == ("account_age_days", "email_disposable")


def test_parse_tolerates_code_fence_and_normalises_whitespace() -> None:
    text = "```json\n" + reply(rationale="  Line one.\n\n  Line   two. ") + "\n```"
    parsed = parse_reply(text, allowed=ALLOWED, citable=CITABLE)
    assert parsed.rationale == "Line one. Line two."


def test_parse_deduplicates_key_factors() -> None:
    text = reply(key_factors=["email_disposable", "email_disposable"])
    assert parse_reply(text, allowed=ALLOWED, citable=CITABLE).key_factors == ("email_disposable",)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("Sure! Here is my analysis.", "not valid JSON"),
        ("[1, 2]", "JSON object"),
        (json.dumps({"strategy": "accept"}), "missing keys: confidence, key_factors, rationale"),
        (reply(strategy="escalate"), "unknown strategy"),
        (reply(strategy="represent"), "not allowed here; choose from accept, review"),
        (reply(confidence="certain"), "unknown confidence"),
        (reply(rationale="   "), "non-empty string"),
        (reply(rationale=7), "non-empty string"),
        (reply(rationale="x" * (RATIONALE_MAX_CHARS + 1)), "exceeds"),
        (reply(key_factors="account_age_days"), "list of feature names"),
        (reply(key_factors=[1]), "list of feature names"),
        (reply(key_factors=[]), "1 to"),
        (reply(key_factors=["device_fingerprint_risk"]), "not in the context"),
    ],
)
def test_parse_rejects(text: str, message: str) -> None:
    with pytest.raises(RationaleError, match=message):
        parse_reply(text, allowed=ALLOWED, citable=CITABLE)


def test_parse_caps_key_factor_count() -> None:
    names = [f"f{i}" for i in range(MAX_KEY_FACTORS + 1)]
    with pytest.raises(RationaleError, match="1 to"):
        parse_reply(reply(key_factors=names), allowed=ALLOWED, citable=frozenset(names))


def test_parse_allows_no_key_factors_when_nothing_is_citable() -> None:
    parsed = parse_reply(reply(key_factors=[]), allowed=ALLOWED, citable=frozenset())
    assert parsed.key_factors == ()


# ----------------------------------------------------------------------------- prompt


def test_context_payload_carries_policy_and_formatted_values() -> None:
    ctx = context()
    rec = recommend(ctx.reason, ctx.features)
    payload = context_payload(ctx, rec)
    assert payload["dispute"]["reason_code"] == "visa:10.4"
    assert payload["model"] == {
        "dispute_probability": "8.0%",
        "typical_probability": "2.0%",
        "relative_risk": 4.0,
    }
    assert payload["top_factors"][0]["value"] == "0.5 days"
    assert payload["top_factors"][2]["effect"] == "lowers"
    assert payload["playbook"]["allowed_strategies"] == ["accept", "review"]
    assert {s["feature"] for s in payload["signals"]} == {
        "account_age_days",
        "email_disposable",
        "ip_bin_mismatch",
    }


def test_messages_embed_context_and_round_trip() -> None:
    ctx = context()
    rec = recommend(ctx.reason, ctx.features)
    messages = build_messages(ctx, rec)
    assert [m.role for m in messages] == ["system", "user"]
    assert messages[0].content == SYSTEM_PROMPT
    assert extract_context(messages) == context_payload(ctx, rec)
    assert citable_features(ctx, rec) == {
        "account_age_days",
        "email_disposable",
        "ip_bin_mismatch",
        "plan_is_annual",
    }


def test_extract_context_errors() -> None:
    with pytest.raises(ValueError, match="no <context>"):
        extract_context([Message("system", "<context>{}</context>"), Message("user", "hi")])
    with pytest.raises(ValueError, match="JSON object"):
        extract_context([Message("user", "<context>[1]</context>")])


def test_completion_request_validation() -> None:
    with pytest.raises(ValueError, match="at least one message"):
        CompletionRequest(())
    with pytest.raises(ValueError, match="max_tokens"):
        CompletionRequest((Message("user", "x"),), max_tokens=0)


# ------------------------------------------------------------------------ LLM clients


def test_clients_satisfy_protocol() -> None:
    assert isinstance(MockLLM(), LLMClient)
    assert isinstance(ScriptedLLM([]), LLMClient)
    assert isinstance(OpenAICompatibleClient("http://localhost/v1", "m"), LLMClient)


def test_mock_reply_passes_validation() -> None:
    ctx = context()
    rec = recommend(ctx.reason, ctx.features)
    text = MockLLM().complete(CompletionRequest(build_messages(ctx, rec)))
    parsed = parse_reply(text, allowed=rec.allowed, citable=citable_features(ctx, rec))
    assert parsed.strategy is rec.strategy
    assert "visa:10.4" in parsed.rationale
    assert parsed.confidence is Confidence.HIGH


def test_mock_without_context_raises_llm_error() -> None:
    with pytest.raises(LLMError, match="mock could not read"):
        MockLLM().complete(CompletionRequest((Message("user", "hello"),)))


def test_scripted_llm_raises_queued_errors_and_runs_out() -> None:
    llm = ScriptedLLM([LLMError("boom")])
    req = CompletionRequest((Message("user", "x"),))
    with pytest.raises(LLMError, match="boom"):
        llm.complete(req)
    with pytest.raises(LLMError, match="exhausted"):
        llm.complete(req)
    assert len(llm.requests) == 2


class _ChatHandler(BaseHTTPRequestHandler):
    status: ClassVar[int] = 200
    body: ClassVar[bytes] = b""
    seen: ClassVar[list[tuple[dict[str, str], dict[str, Any]]]] = []

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        _ChatHandler.seen.append((dict(self.headers), json.loads(self.rfile.read(length))))
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, format: str, *args: Any) -> None:  # silence test output
        pass


@pytest.fixture
def chat_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    _ChatHandler.status, _ChatHandler.seen = 200, []
    _ChatHandler.body = json.dumps({"choices": [{"message": {"content": reply()}}]}).encode()
    server = HTTPServer(("127.0.0.1", 0), _ChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1/"
    server.shutdown()
    server.server_close()


def test_openai_compatible_client_request_and_reply(chat_server: str) -> None:
    client = OpenAICompatibleClient(chat_server, "local-model", api_key="sk-test")
    req = CompletionRequest((Message("system", "s"), Message("user", "u")), max_tokens=50)
    assert client.complete(req) == reply()
    headers, body = _ChatHandler.seen[0]
    assert headers["Authorization"] == "Bearer sk-test"
    assert body == {
        "model": "local-model",
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        "temperature": 0.0,
        "max_tokens": 50,
        "response_format": {"type": "json_object"},
    }
    assert client.endpoint.endswith("/v1/chat/completions")
    assert client.name == "openai-compatible:local-model"


def test_openai_compatible_client_omits_auth_and_json_mode(chat_server: str) -> None:
    client = OpenAICompatibleClient(chat_server, "m")
    client.complete(CompletionRequest((Message("user", "u"),), json_mode=False))
    headers, body = _ChatHandler.seen[0]
    assert "Authorization" not in headers
    assert "response_format" not in body


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (500, b"{}", "HTTP 500"),
        (200, b"<html>", "non-JSON"),
        (200, b'{"choices": []}', "no choices"),
        (200, b'{"choices": [{"message": {"content": null}}]}', "not a string"),
    ],
)
def test_openai_compatible_client_errors(
    chat_server: str, status: int, body: bytes, message: str
) -> None:
    _ChatHandler.status, _ChatHandler.body = status, body
    with pytest.raises(LLMError, match=message):
        OpenAICompatibleClient(chat_server, "m").complete(
            CompletionRequest((Message("user", "u"),))
        )


def test_openai_compatible_client_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("http_proxy", "HTTP_PROXY"):
        monkeypatch.delenv(var, raising=False)
    client = OpenAICompatibleClient("http://127.0.0.1:9/v1", "m", timeout_s=2)
    with pytest.raises(LLMError, match="could not reach"):
        client.complete(CompletionRequest((Message("user", "u"),)))


def test_openai_compatible_client_validation_and_secret_hygiene() -> None:
    with pytest.raises(ValueError, match="http"):
        OpenAICompatibleClient("file:///etc/passwd", "m")
    with pytest.raises(ValueError, match="model"):
        OpenAICompatibleClient("http://x", "")
    with pytest.raises(ValueError, match="timeout"):
        OpenAICompatibleClient("http://x", "m", timeout_s=0)
    assert "sk-secret" not in repr(OpenAICompatibleClient("http://x", "m", api_key="sk-secret"))


# --------------------------------------------------------------------------- reasoner


def test_reasoner_with_mock_returns_llm_rationale() -> None:
    result = RiskReasoner(MockLLM()).reason(context())
    assert result.source is Source.LLM
    assert result.provider == "mock"
    assert result.strategy is Strategy.ACCEPT is result.playbook_strategy
    assert not result.overrides_playbook
    assert result.warnings == ()
    assert result.evidence == recommend(VISA_FRAUD, features(**STOLEN)).evidence


def test_reasoner_retries_with_validation_feedback() -> None:
    llm = ScriptedLLM(["not json", reply(strategy="review", confidence="medium")])
    result = RiskReasoner(llm).reason(context())
    assert result.source is Source.LLM
    assert result.strategy is Strategy.REVIEW
    assert result.overrides_playbook
    assert result.warnings == ("attempt 1: rejected reply: reply is not valid JSON",)
    retry = llm.requests[1].messages
    assert [m.role for m in retry] == ["system", "user", "assistant", "user"]
    assert retry[2].content == "not json"
    assert "rejected: reply is not valid JSON" in retry[3].content


def test_reasoner_falls_back_when_llm_keeps_choosing_disallowed_strategy() -> None:
    llm = ScriptedLLM([reply(strategy="represent")] * 2)
    result = RiskReasoner(llm).reason(context())
    assert result.source is Source.FALLBACK
    assert result.strategy is Strategy.ACCEPT
    assert result.confidence is Confidence.LOW
    assert len(result.warnings) == 2
    assert "Playbook: accept." in result.rationale


def test_reasoner_falls_back_on_provider_errors() -> None:
    llm = ScriptedLLM([LLMError("timeout"), LLMError("503")])
    result = RiskReasoner(llm, max_attempts=2).reason(context())
    assert result.source is Source.FALLBACK
    assert result.warnings == (
        "attempt 1: provider error: timeout",
        "attempt 2: provider error: 503",
    )
    assert result.provider == "scripted"
    assert set(result.key_factors) <= citable_features(
        context(), recommend(VISA_FRAUD, features(**STOLEN))
    )


def test_reasoner_validates_max_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RiskReasoner(MockLLM(), max_attempts=0)


def test_template_rationale_without_factors_or_signals() -> None:
    ctx = DisputeContext(explanation(), VISA_RECURRING, features())
    text = template_rationale(ctx, recommend(ctx.reason, ctx.features))
    assert text.startswith("visa:13.2 (Cancelled Recurring Transaction); pre-dispute risk 8.0%")
    assert "Top factors" not in text
    assert "Signals" not in text


def test_rationale_render_and_dict() -> None:
    llm = ScriptedLLM([LLMError("down"), reply()])
    result = RiskReasoner(llm).reason(context())
    text = result.render()
    assert text.splitlines()[0] == "txn_1 · visa:10.4 Other Fraud - Card Absent Environment"
    assert "strategy:    accept (playbook: accept, confidence: high, llm via scripted)" in text
    assert "warning:     attempt 1: provider error: down" in text
    data = result.to_dict()
    assert data["reason_code"] == "visa:10.4"
    assert data["source"] == "llm"
    assert data["evidence"] == list(result.evidence)
    json.dumps(data)


def test_feature_values_checks_length() -> None:
    assert feature_values(["a", "b"], [1, 2.5]) == {"a": 1.0, "b": 2.5}
    with pytest.raises(ValueError, match="length"):
        feature_values(["a"], [1.0, 2.0])


# ------------------------------------------------------------------- run + pipeline


def test_parse_reason_code() -> None:
    assert parse_reason_code("visa:13.2") is VISA_RECURRING
    assert parse_reason_code(" MasterCard:4837") is MC_FRAUD
    for bad in ("13.2", "visa:", ":13.2"):
        with pytest.raises(ValueError, match="must look like"):
            parse_reason_code(bad)
    with pytest.raises(ValueError, match="unknown reason code"):
        parse_reason_code("visa:99.9")
    with pytest.raises(ValueError, match="amex"):
        parse_reason_code("amex:F29")


def _dispute(txn: str, code: str, day: int, friendly: bool = True) -> Dispute:
    rc = parse_reason_code(code)
    return Dispute(
        dispute_id=f"dsp_{txn}_{day}",
        transaction_id=txn,
        customer_id="cus_1",
        network=rc.network,
        reason_code=code,
        category=rc.category,
        filed_at=datetime(2026, 1, day, tzinfo=UTC),
        amount_cents=999,
        is_friendly_fraud=friendly,
    )


def test_dispute_reason_codes_keeps_earliest_filing() -> None:
    disputes = [_dispute("t1", "visa:13.2", 9), _dispute("t1", "visa:10.4", 3)]
    assert dispute_reason_codes(disputes) == {"t1": VISA_FRAUD}


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("reason")
    data, feats, models = root / "data", root / "features.csv", root / "models"
    assert main(["generate", "--customers", "1500", "--seed", "11", "--out", str(data)]) == 0
    assert main(["features", "--data", str(data), "--out", str(feats)]) == 0
    assert main(["train", "--features", str(feats), "--out", str(models), "--model", "logreg"]) == 0
    return {"data": data, "features": feats, "models": models}


def test_build_contexts_and_reason_end_to_end(pipeline: dict[str, Path]) -> None:
    disputes = read_dataset(pipeline["data"]).disputes
    reasons = dict(list(dispute_reason_codes(disputes).items())[:8])
    matrix = FeatureMatrix.read_csv(pipeline["features"])
    model = load_model(pipeline["models"], ModelKind.LOGREG)
    contexts = build_contexts(model, matrix, reasons, ExplainConfig(background_size=100))
    assert [c.transaction_id for c in contexts] == list(reasons)
    assert all(set(c.features) == set(FEATURE_NAMES) for c in contexts)
    results = RiskReasoner(MockLLM()).reason_all(contexts)
    assert all(r.source is Source.LLM for r in results), [r.warnings for r in results]
    assert all(r.strategy is r.playbook_strategy for r in results)
    with pytest.raises(KeyError, match="txn_nope"):
        build_contexts(model, matrix, {"txn_nope": VISA_FRAUD})


def test_backtest_playbook_on_synthetic_ground_truth(pipeline: dict[str, Path]) -> None:
    disputes = read_dataset(pipeline["data"]).disputes
    matrix = FeatureMatrix.read_csv(pipeline["features"])
    rows = backtest_playbook([*disputes, _dispute("txn_absent", "visa:10.4", 1)], matrix)
    assert sum(r.disputes for r in rows) == len(disputes)
    by_key = {(r.category, r.strategy): r for r in rows}
    accepted_fraud = by_key[(DisputeCategory.FRAUD, Strategy.ACCEPT)]
    represented_fraud = by_key[(DisputeCategory.FRAUD, Strategy.REPRESENT)]
    # The rules are meant to be precise at the extremes, leaving ambiguity to review.
    assert accepted_fraud.friendly_share <= 0.1
    assert represented_fraud.friendly_share >= 0.9
    table = render_backtest(rows)
    assert table.startswith("| Category | Playbook strategy |")
    assert table.count("\n") == len(rows) + 2


def test_cli_reason_mock(pipeline: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    txn = next(iter(dispute_reason_codes(read_dataset(pipeline["data"]).disputes)))
    args = [
        "reason",
        "--data",
        str(pipeline["data"]),
        "--features",
        str(pipeline["features"]),
        "--models",
        str(pipeline["models"]),
        "--transaction-id",
        txn,
    ]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"{txn} · ")
    assert "llm via mock" in out

    assert main([*args, "--json", "--reason-code", "mastercard:4853"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["transaction_id"] == txn
    assert data["reason_code"] == "mastercard:4853"


def test_cli_reason_errors(
    pipeline: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    disputed = set(dispute_reason_codes(read_dataset(pipeline["data"]).disputes))
    matrix = FeatureMatrix.read_csv(pipeline["features"])
    undisputed = next(t for t in matrix.transaction_ids if t not in disputed)
    base = [
        "reason",
        "--data",
        str(pipeline["data"]),
        "--features",
        str(pipeline["features"]),
        "--models",
        str(pipeline["models"]),
    ]
    cases = [
        ([*base, "--transaction-id", undisputed], "no dispute on"),
        ([*base, "--transaction-id", "t", "--reason-code", "visa"], "must look like"),
        ([*base, "--transaction-id", "t", "--reason-code", "visa:10.4"], "unknown transaction"),
        ([*base, "--transaction-id", "t", "--provider", "openai-compatible"], "--llm-base-url"),
        (
            [
                *base,
                "--transaction-id",
                "t",
                "--provider",
                "openai-compatible",
                "--llm-base-url",
                "ftp://x",
                "--llm-model",
                "m",
            ],
            "http",
        ),
        (
            ["reason", "--transaction-id", "t", "--data", str(tmp_path)],
            "not found; run `cbp generate` or pass --reason-code",
        ),
        (
            [
                *base[:-2],
                "--models",
                str(tmp_path),
                "--transaction-id",
                "t",
                "--reason-code",
                "visa:10.4",
            ],
            "run `cbp train` first",
        ),
        (
            [
                "reason",
                "--models",
                str(pipeline["models"]),
                "--features",
                str(tmp_path / "nope.csv"),
                "--transaction-id",
                "t",
                "--reason-code",
                "visa:10.4",
            ],
            "run `cbp features` first",
        ),
    ]
    for argv, message in cases:
        with pytest.raises(SystemExit):
            main(argv)
        assert message in capsys.readouterr().err, argv


def test_cli_reason_openai_compatible(
    pipeline: dict[str, Path],
    chat_server: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(API_KEY_ENV, "sk-from-env")
    _ChatHandler.body = json.dumps({"choices": [{"message": {"content": "nope"}}]}).encode()
    txn = next(iter(dispute_reason_codes(read_dataset(pipeline["data"]).disputes)))
    argv = [
        "reason",
        "--data",
        str(pipeline["data"]),
        "--features",
        str(pipeline["features"]),
        "--models",
        str(pipeline["models"]),
        "--transaction-id",
        txn,
        "--provider",
        "openai-compatible",
        "--llm-base-url",
        chat_server,
        "--llm-model",
        "local",
        "--json",
    ]
    assert main(argv) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["source"] == "fallback"
    assert data["provider"] == "openai-compatible:local"
    assert len(_ChatHandler.seen) == 2
    assert _ChatHandler.seen[0][0]["Authorization"] == "Bearer sk-from-env"


def test_cli_playbook(
    pipeline: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["playbook", "--data", str(pipeline["data"]), "--features", str(pipeline["features"])]
    assert main(argv) == 0
    assert capsys.readouterr().out.startswith("| Category | Playbook strategy |")
    with pytest.raises(SystemExit):
        main(["playbook", "--data", str(tmp_path)])
    assert "run `cbp generate` and `cbp features` first" in capsys.readouterr().err
