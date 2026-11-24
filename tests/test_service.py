from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from chargeback_prediction.cli import main
from chargeback_prediction.explain import ExplainConfig, explain_transactions, load_model
from chargeback_prediction.explain.run import model_path
from chargeback_prediction.features import FEATURE_NAMES, FEATURES, FeatureKind
from chargeback_prediction.modeling import FeatureMatrix, ModelKind, predict_risk
from chargeback_prediction.reasoning.llm import (
    LLMError,
    MockLLM,
    OpenAICompatibleClient,
    ScriptedLLM,
)
from chargeback_prediction.service import Scorer, Settings, artifact_id, create_app
from chargeback_prediction.service.schemas import MAX_BATCH, TransactionFeatures

BACKGROUND = 120


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("service")
    data, feats, models = root / "data", root / "features.csv", root / "models"
    assert main(["generate", "--customers", "1500", "--seed", "11", "--out", str(data)]) == 0
    assert main(["features", "--data", str(data), "--out", str(feats)]) == 0
    assert main(["train", "--features", str(feats), "--out", str(models)]) == 0
    return {"features": feats, "models": models}


@pytest.fixture(scope="module")
def matrix(pipeline: dict[str, Path]) -> FeatureMatrix:
    return FeatureMatrix.read_csv(pipeline["features"])


def _scorer(pipeline: dict[str, Path], kind: ModelKind, **kwargs: Any) -> Scorer:
    return Scorer.from_artifacts(
        pipeline["models"], pipeline["features"], kind, background_size=BACKGROUND, **kwargs
    )


@pytest.fixture(scope="module")
def scorer(pipeline: dict[str, Path]) -> Scorer:
    return _scorer(pipeline, ModelKind.LOGREG)


@pytest.fixture(scope="module")
def client(scorer: Scorer) -> Iterator[TestClient]:
    with TestClient(create_app(scorer)) as c:
        yield c


def _features(matrix: FeatureMatrix, i: int) -> dict[str, float]:
    """A feature-table row as a JSON payload, ints where the schema expects ints."""
    out: dict[str, float] = {}
    for spec, value in zip(FEATURES, matrix.X[i], strict=True):
        out[spec.name] = float(value) if spec.kind is FeatureKind.NUMERIC else int(value)
    return out


def _txn(matrix: FeatureMatrix, i: int) -> dict[str, Any]:
    return {"transaction_id": matrix.transaction_ids[i], "features": _features(matrix, i)}


def _riskiest(scorer: Scorer, matrix: FeatureMatrix) -> int:
    return int(np.argmax(scorer.score(matrix.X)))


# ------------------------------------------------------------------- contract


def test_schema_matches_feature_catalog() -> None:
    """The HTTP contract and the feature catalog must not drift apart."""
    fields = TransactionFeatures.model_fields
    assert tuple(fields) == FEATURE_NAMES
    expected = {FeatureKind.FLAG: int, FeatureKind.COUNT: int, FeatureKind.NUMERIC: float}
    for spec in FEATURES:
        info = fields[spec.name]
        assert info.annotation is expected[spec.kind], spec.name
        assert info.description == spec.description
        lower = [m.ge for m in info.metadata if hasattr(m, "ge")]
        assert lower == [0], spec.name
        upper = [m.le for m in info.metadata if hasattr(m, "le")]
        assert upper == ([1] if spec.kind is FeatureKind.FLAG else []), spec.name


def test_openapi_documents_endpoints_and_features(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    assert {"/health", "/v1/model", "/v1/score", "/v1/explain"} <= set(spec["paths"])
    props = spec["components"]["schemas"]["TransactionFeatures"]["properties"]
    assert list(props) == list(FEATURE_NAMES)
    assert props["email_disposable"]["maximum"] == 1
    assert "disposable email" in props["email_disposable"]["description"]
    assert client.get("/docs").status_code == 200


def test_health_and_model_info(client: TestClient, scorer: Scorer) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/v1/model").json()
    assert info["model_id"] == scorer.model_id
    assert info["model_id"].startswith("logreg@")
    assert info["features"] == list(FEATURE_NAMES)
    assert info["background_rows"] == BACKGROUND
    assert info["llm_provider"] == "mock"
    assert 0 < info["baseline_probability"] < 0.1


# ------------------------------------------------------------------- /v1/score


def test_score_batch_matches_offline_model(
    client: TestClient, pipeline: dict[str, Path], matrix: FeatureMatrix
) -> None:
    rows = list(range(0, len(matrix), max(1, len(matrix) // 50)))
    resp = client.post("/v1/score", json={"transactions": [_txn(matrix, i) for i in rows]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    expected = predict_risk(load_model(pipeline["models"], ModelKind.LOGREG), matrix.X[rows])
    assert [r["transaction_id"] for r in body["results"]] == [
        matrix.transaction_ids[i] for i in rows
    ]
    got = np.array([r["probability"] for r in body["results"]])
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
    for r in body["results"]:
        assert r["relative_risk"] == pytest.approx(r["probability"] / body["baseline_probability"])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda f: f.pop("amount_usd"), "amount_usd"),
        (lambda f: f.update(amount_eur=1.0), "amount_eur"),
        (lambda f: f.update(email_disposable=2), "email_disposable"),
        (lambda f: f.update(attempts_24h=-1), "attempts_24h"),
        (lambda f: f.update(attempts_24h=1.5), "attempts_24h"),
        (lambda f: f.update(account_age_days=-0.5), "account_age_days"),
        (lambda f: f.update(amount_usd="lots"), "amount_usd"),
    ],
)
def test_score_rejects_bad_features(
    client: TestClient, matrix: FeatureMatrix, mutate: Any, message: str
) -> None:
    txn = _txn(matrix, 0)
    mutate(txn["features"])
    resp = client.post("/v1/score", json={"transactions": [txn]})
    assert resp.status_code == 422
    assert message in resp.text


def test_score_rejects_bad_batches(client: TestClient, matrix: FeatureMatrix) -> None:
    assert client.post("/v1/score", json={"transactions": []}).status_code == 422
    dup = [_txn(matrix, 0), _txn(matrix, 0)]
    resp = client.post("/v1/score", json={"transactions": dup})
    assert resp.status_code == 422
    assert "duplicate transaction_id" in resp.text
    too_many = [
        {"transaction_id": f"t{i}", "features": _features(matrix, 0)} for i in range(MAX_BATCH + 1)
    ]
    assert client.post("/v1/score", json={"transactions": too_many}).status_code == 422
    bad_id = {"transaction_id": "", "features": _features(matrix, 0)}
    assert client.post("/v1/score", json={"transactions": [bad_id]}).status_code == 422


# ------------------------------------------------------------------- /v1/explain


def test_explain_agrees_with_cbp_explain(
    client: TestClient, pipeline: dict[str, Path], scorer: Scorer, matrix: FeatureMatrix
) -> None:
    """Same model, same background sample: the service and the offline report must agree."""
    i = _riskiest(scorer, matrix)
    resp = client.post("/v1/explain", json={"transaction": _txn(matrix, i), "top_k": 5})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    model = load_model(pipeline["models"], ModelKind.LOGREG)
    (offline,) = explain_transactions(
        model,
        matrix,
        [matrix.transaction_ids[i]],
        ExplainConfig(background_size=BACKGROUND, top_k=5),
    )
    assert body["model_id"] == scorer.model_id
    assert body["probability"] == pytest.approx(offline.probability, abs=1e-12)
    assert body["baseline_probability"] == pytest.approx(offline.baseline_probability)
    assert [f["feature"] for f in body["factors"]] == [f.feature for f in offline.factors]
    assert body["summary"] == offline.summary()
    assert body["rationale"] is None
    # Local accuracy: base + shown factors + the rest is the model's log-odds.
    logit = body["base_log_odds"] + body["other_features_log_odds"]
    logit += sum(f["shap_log_odds"] for f in body["factors"])
    assert logit == pytest.approx(math.log(body["probability"] / (1 - body["probability"])))
    assert len(body["factors"]) <= 5
    assert all(f["direction"] in {"raises", "lowers"} for f in body["factors"])


def test_explain_top_k_bounds(client: TestClient, matrix: FeatureMatrix) -> None:
    txn = _txn(matrix, 0)
    zero = client.post("/v1/explain", json={"transaction": txn, "top_k": 0}).json()
    assert zero["factors"] == []
    assert client.post("/v1/explain", json={"transaction": txn, "top_k": 11}).status_code == 422


def test_explain_with_reason_code_adds_rationale(
    client: TestClient, scorer: Scorer, matrix: FeatureMatrix
) -> None:
    txn = _txn(matrix, _riskiest(scorer, matrix))
    body = client.post("/v1/explain", json={"transaction": txn, "reason_code": "visa:10.4"}).json()
    rationale = body["rationale"]
    assert rationale["reason_code"] == "visa:10.4"
    assert rationale["source"] == "llm"
    assert rationale["provider"] == "mock"
    assert rationale["strategy"] == rationale["playbook_strategy"]
    assert rationale["evidence"]
    assert set(rationale["key_factors"]) <= set(FEATURE_NAMES)


def test_explain_rejects_unknown_reason_code(client: TestClient, matrix: FeatureMatrix) -> None:
    for code in ("visa:99.9", "amex:1", "13.2"):
        resp = client.post(
            "/v1/explain", json={"transaction": _txn(matrix, 0), "reason_code": code}
        )
        assert resp.status_code == 422, code


def test_explain_falls_back_when_llm_is_down(
    pipeline: dict[str, Path], matrix: FeatureMatrix
) -> None:
    llm = ScriptedLLM([LLMError("connection refused"), LLMError("connection refused")])
    with TestClient(create_app(_scorer(pipeline, ModelKind.LOGREG, llm=llm))) as c:
        body = c.post(
            "/v1/explain", json={"transaction": _txn(matrix, 0), "reason_code": "visa:13.2"}
        ).json()
    assert body["rationale"]["source"] == "fallback"
    assert body["rationale"]["provider"] == "scripted"
    assert len(body["rationale"]["warnings"]) == 2
    assert body["probability"] > 0


def test_gbm_scorer_explains_exactly(pipeline: dict[str, Path], matrix: FeatureMatrix) -> None:
    gbm = _scorer(pipeline, ModelKind.GBM)
    assert gbm.model_id.startswith("gbm@")
    with TestClient(create_app(gbm)) as c:
        body = c.post("/v1/explain", json={"transaction": _txn(matrix, 3), "top_k": 10}).json()
    logit = body["base_log_odds"] + body["other_features_log_odds"]
    logit += sum(f["shap_log_odds"] for f in body["factors"])
    assert logit == pytest.approx(math.log(body["probability"] / (1 - body["probability"])))


# ------------------------------------------------------------------- scorer + settings


def test_scorer_build_validates(scorer: Scorer) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Scorer.build(scorer.model, scorer.kind, np.empty((0, 30)), model_id="x")
    with pytest.raises(ValueError, match="columns"):
        Scorer.build(scorer.model, scorer.kind, np.zeros((5, 3)), model_id="x")
    dummy = DummyClassifier(strategy="prior").fit(np.zeros((4, 30)), [0, 0, 1, 1])
    with pytest.raises(TypeError, match="no exact SHAP"):
        Scorer.build(dummy, ModelKind.LOGREG, np.zeros((5, 30)), model_id="x")


def test_artifact_id_is_a_content_hash(pipeline: dict[str, Path], tmp_path: Path) -> None:
    path = model_path(pipeline["models"], ModelKind.LOGREG)
    first = artifact_id(ModelKind.LOGREG, path)
    assert first == artifact_id(ModelKind.LOGREG, path)
    copy = tmp_path / "m.joblib"
    copy.write_bytes(path.read_bytes() + b"\0")
    assert artifact_id(ModelKind.LOGREG, copy) != first


def test_settings_from_env() -> None:
    assert Settings.from_env({}) == Settings()
    s = Settings.from_env(
        {
            "CBP_MODELS_DIR": "/m",
            "CBP_FEATURES": "/f.csv",
            "CBP_MODEL": "gbm",
            "CBP_BACKGROUND_SIZE": "50",
            "CBP_LLM_BASE_URL": "http://localhost:11434/v1",
            "CBP_LLM_MODEL": "llama3",
            "CBP_LLM_API_KEY": "",
            "UNRELATED": "x",
        }
    )
    assert (s.models_dir, s.features, s.model, s.background_size) == (
        Path("/m"),
        Path("/f.csv"),
        ModelKind.GBM,
        50,
    )
    assert s.llm_api_key is None
    llm = s.llm_client()
    assert isinstance(llm, OpenAICompatibleClient)
    assert llm.name == "openai-compatible:llama3"
    assert isinstance(Settings().llm_client(), MockLLM)
    with pytest.raises(ValueError, match="neither"):
        Settings(llm_model="llama3").llm_client()
    with pytest.raises(ValueError, match="'svm'"):
        Settings.from_env({"CBP_MODEL": "svm"})


def test_app_loads_scorer_from_settings_at_startup(
    pipeline: dict[str, Path], tmp_path: Path
) -> None:
    settings = Settings(
        models_dir=pipeline["models"], features=pipeline["features"], background_size=50
    )
    with TestClient(create_app(settings=settings)) as c:
        assert c.get("/v1/model").json()["background_rows"] == 50
    broken = create_app(settings=Settings(models_dir=tmp_path, features=pipeline["features"]))
    with pytest.raises(FileNotFoundError), TestClient(broken):
        pass  # pragma: no cover - startup fails first


# ------------------------------------------------------------------- CLI


def test_cli_serve(
    pipeline: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run(app: Any, **kwargs: Any) -> None:
        calls.append({"app": app, **kwargs})

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setenv("CBP_MODELS_DIR", str(pipeline["models"]))
    monkeypatch.setenv("CBP_FEATURES", str(pipeline["features"]))
    assert main(["serve", "--model", "gbm", "--port", "9999"]) == 0
    assert calls[0]["port"] == 9999
    assert calls[0]["host"] == "127.0.0.1"
    assert "serving gbm@" in capsys.readouterr().out

    monkeypatch.setenv("CBP_MODELS_DIR", "/nonexistent")
    with pytest.raises(SystemExit) as exc:
        main(["serve"])
    assert exc.value.code == 2
    assert "cbp train" in capsys.readouterr().err
    monkeypatch.setenv("CBP_LLM_MODEL", "llama3")
    monkeypatch.setenv("CBP_MODELS_DIR", str(pipeline["models"]))
    with pytest.raises(SystemExit):
        main(["serve"])
    assert "CBP_LLM_BASE_URL" in capsys.readouterr().err
