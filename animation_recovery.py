"""Read-only recovery of the recorded sharp LTX job; never submits a job.

The caller supplies its existing Comfy client. This module does not read, return,
or log credentials. Intended for a temporary authenticated private-Space bridge.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile


SHARP_JOB_ID = "50e5ea56-949b-49c8-aefa-ec6f247f1c52"
_SECRET_KEYS = {
    "apikey", "apikeys", "authorization", "bearertoken", "accesstoken",
    "refreshtoken", "token", "password", "secret", "secrets", "credentials",
    "clientsecret", "privatekey", "cookie", "cookies", "setcookie",
}
_SIGNED_QUERY_KEYS = {
    "token", "access_token", "api_key", "apikey", "key", "sig", "signature",
    "x-amz-signature", "x-amz-credential", "x-amz-security-token",
    "x-goog-signature", "x-goog-credential",
}


def _safe_graph(value: Any, redactions: list[str], path: str = "$graph") -> Any:
    """Exclude credential fields and credential-bearing URLs from the export."""
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            child_path = f"{path}.{key}"
            normalized = str(key).lower().replace("_", "").replace("-", "")
            if normalized in _SECRET_KEYS or normalized.endswith(("apikey", "accesstoken", "refreshtoken", "clientsecret", "password")):
                redactions.append(child_path)
                result[key] = "[REDACTED: credential field]"
            else:
                result[key] = _safe_graph(child, redactions, child_path)
        return result
    if isinstance(value, list):
        return [_safe_graph(child, redactions, f"{path}[{i}]") for i, child in enumerate(value)]
    if isinstance(value, str):
        if value.lstrip().lower().startswith("bearer "):
            redactions.append(path)
            return "[REDACTED: bearer credential]"
        if value.lower().startswith(("http://", "https://")):
            try:
                url = urlsplit(value)
                secret_query = any(key.lower() in _SIGNED_QUERY_KEYS for key, _ in parse_qsl(url.query))
                if url.username or url.password or secret_query:
                    redactions.append(path)
                    return "[REDACTED: credential-bearing URL; rebind asset]"
            except ValueError:
                redactions.append(path)
                return "[REDACTED: malformed URL]"
    return value


def recover_sharp_workflow(client: Any, output_dir: str | Path) -> tuple[str, str]:
    """Return (ZIP path, status) for the fixed recorded job using SDK GETs only.

    ``client.jobs.get(...).get_workflow()`` in comfy-sdk 0.5.0 returns format
    ``api`` or ``save``; a save-format graph is not silently converted. The SDK
    version has no documented cost-estimation method, so none is guessed here.
    """
    try:
        workflow = client.jobs.get(SHARP_JOB_ID).get_workflow()
    except Exception as error:
        # Do not leak SDK responses, request headers, or exception bodies through
        # a Gradio error. The account may not own the job or it may have expired.
        kind = type(error).__name__
        raise RuntimeError(f"Read-only workflow recovery failed ({kind}); no job was submitted.") from None

    graph_format = workflow.format
    if graph_format not in {"api", "save"} or not isinstance(workflow.graph, dict):
        raise RuntimeError("The SDK returned an unsupported workflow shape; no job was submitted.")
    redactions: list[str] = []
    graph = _safe_graph(workflow.graph, redactions)
    graph_bytes = json.dumps(graph, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    api_nodes = {
        key: node for key, node in graph.items()
        if isinstance(node, dict) and isinstance(node.get("class_type"), str)
    } if graph_format == "api" else {}
    api_shape = bool(api_nodes) and len(api_nodes) == len(graph) and all(
        isinstance(node.get("inputs"), dict) for node in api_nodes.values()
    )
    now = datetime.now(timezone.utc)
    provenance = {
        "recovered_at_utc": now.isoformat(),
        "source_job_id": SHARP_JOB_ID,
        "source": "Existing account job via comfy-sdk client.jobs.get(id).get_workflow()",
        "operations": ["GET job", "GET job workflow"],
        "generation_submitted": False,
        "workflow_format": graph_format,
        "exported_graph_sha256": hashlib.sha256(graph_bytes).hexdigest(),
        "api_node_count": len(api_nodes),
        "api_shape_checked": api_shape,
        "class_types": sorted({node["class_type"] for node in api_nodes.values()}),
        "credential_redaction_paths": redactions,
        "requires_asset_rebinding": True,
        "execution_verified_by_recovery": False,
        "cost_estimate": None,
        "cost_estimate_note": "No estimate attempted: comfy-sdk 0.5.0 exposes no documented estimator. Retrieval is not generation, and zero partner/API nodes would not imply zero compute cost.",
        "sdk_source": "https://github.com/Comfy-Org/comfy-python-sdk/blob/v0.5.0/src/comfy_sdk/jobs.py",
        "format_contract": "https://github.com/Comfy-Org/comfy-python-sdk/blob/v0.5.0/spec/openapi.yaml#L493",
        "scope": "Private recovery artifact. Graph provenance and format only; no current model availability, license eligibility, visual quality, or reproducibility certification.",
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / f"sharp-ltx-recovery-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}.zip"
    with ZipFile(archive_path, "x", compression=ZIP_DEFLATED) as archive:
        archive.writestr("sharp-ltx-workflow.json", graph_bytes)
        archive.writestr("provenance.json", json.dumps(provenance, indent=2))
    if graph_format == "save":
        status = "Recovered the recorded LTX workflow in frontend save format; API export is still required."
    elif not api_shape:
        status = "Recovered an API-labeled workflow; its node structure needs inspection before execution."
    else:
        status = f"Recovered {len(api_nodes)} API nodes from the recorded LTX job; rebind assets and validate before execution."
    if redactions:
        status += f" Removed {len(redactions)} credential fields or URLs; inspect the redaction list."
    return str(archive_path), status + " No job was submitted."
