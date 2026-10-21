"""Replay one saved develop graph on local ComfyUI and retain execution evidence.

    python replay_desktop.py recipe.zip --server http://127.0.0.1:8188 --output NEW_FOLDER

No server launch, package install, Cloud request, credentials, retries or job cancellation.
Only supported saved develop graphs are allowed, never frame-generation/scan graphs.
Timeout retains evidence and leaves the queued job alone. The output directory must
not exist. A successful history AND numerical image agreement are required for replay_pass.
"""

import argparse
import copy
import hashlib
import ipaddress
import json
from pathlib import Path, PurePosixPath
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from compare_replay import compare_recipe
from validate_recipe import validate
import guide_eye


ALLOWED_CLASSES = frozenset({
    "LoadImage", "HDRExpandDynamicRange", "RadianceDepthOfField", "HDRToneMap",
    "ColorSpaceConvert", "RadianceFilmGrain", "SaveImage", "ImageToMask", "ImageCompositeMasked",
    "ImageBlur", "ImageScale", "RadianceGrade",
})

ROUND_GRAPH_VERSION = "core-round-gaussian-legacy-footprint-v1"


def inspect_round_graph(graph):
    """Recognize the versioned still builder's topology, including its zero-blur path.

    The title alone is not sufficient: every node must belong to the fixed finishing
    chain, optional subject/plate composite, or a complete saved-mask blur branch.
    This does not certify mask semantics or renderer parity; replay records those
    inputs and compares the actual output to the recipe reference separately.
    """
    errors, expected = [], set()
    title = graph.get("32", {}).get("_meta", {}).get("title")
    guided = title == "Save hero frame / " + ROUND_GRAPH_VERSION + " / " + guide_eye.VERSION
    guide_values = []

    def node(node_id, kind, connections=None, literals=None):
        expected.add(node_id)
        value = graph.get(node_id, {})
        if value.get("class_type") != kind:
            errors.append(f"Round graph node {node_id}: expected {kind}.")
            return {}
        inputs = value.get("inputs", {})
        for key, wanted in {**(connections or {}), **(literals or {})}.items():
            if inputs.get(key) != wanted:
                errors.append(f"Round graph node {node_id}.{key}: unexpected connection/value.")
        return inputs

    def branch(prefix, image_ref):
        stem = f"round_{prefix}"
        members = {key for key in graph if key.startswith(stem + "_")}
        if not members:
            return image_ref  # explicit versioned zero-blur identity
        scale = stem + "_scale"
        scaled = node(scale, "ImageScale", {"image": image_ref}, {"upscale_method": "area", "crop": "disabled"})
        previous, full_size = image_ref, None
        for level in range(1, 6):
            tag = f"{stem}_{level}"
            if not any(key.startswith(tag + "_") for key in members):
                continue
            load, mask, blur, up, mix = (tag + suffix for suffix in ("_load", "_mask", "_blur", "_up", "_mix"))
            node(load, "LoadImage")
            node(mask, "ImageToMask", {"image": [load, 0]}, {"channel": "red"})
            node(blur, "ImageBlur", {"image": [scale, 0]})
            enlarged = node(up, "ImageScale", {"image": [blur, 0]}, {"upscale_method": "bilinear", "crop": "disabled"})
            size = (enlarged.get("width"), enlarged.get("height"))
            if any(type(v) is not int or v < 2 for v in size):
                errors.append(f"Round graph node {up}: invalid output dimensions.")
            if full_size is not None and full_size != size:
                errors.append("Round graph branch output dimensions differ between levels.")
            full_size = size
            node(mix, "ImageCompositeMasked", {"destination": previous, "source": [up, 0], "mask": [mask, 0]},
                 {"x": 0, "y": 0, "resize_source": False})
            previous = [mix, 0]
        if any(type(scaled.get(key)) is not int or scaled[key] < 2 for key in ("width", "height")):
            errors.append(f"Round graph node {scale}: invalid scale dimensions.")
        return previous

    def guidance(image_ref, role):
        if not guided:
            return image_ref
        stem = f"guide_{role}"
        load, mask, grade, mix = (stem + suffix for suffix in ("_load", "_mask", "_grade", "_mix"))
        node(load, "LoadImage")
        node(mask, "ImageToMask", {"image": [load, 0]}, {"channel": "red"})
        try:
            value = guide_eye.amount(graph.get(grade, {}).get("_meta", {}).get("guide"))
            if value == 0:
                raise ValueError("Zero guide must use the original graph.")
            grade_values = guide_eye.grade_inputs(value)
            guide_values.append(value)
        except ValueError:
            errors.append(f"Guide node {grade}: invalid versioned guide amount.")
            grade_values = guide_eye.grade_inputs(0)
        actual = node(grade, "RadianceGrade", {"image": image_ref}, grade_values)
        # No preset files, reference images or unrecognized future options on replay.
        if set(actual) != {"image", *grade_values}:
            errors.append(f"Guide node {grade}: unexpected grade inputs.")
        if graph.get(grade, {}).get("_meta", {}).get("title") != guide_eye.VERSION:
            errors.append(f"Guide node {grade}: missing rendering version.")
        node(mix, "ImageCompositeMasked", {"destination": image_ref, "source": [grade, 0], "mask": [mask, 0]},
             {"x": 0, "y": 0, "resize_source": False})
        return [mix, 0]

    node("20", "LoadImage")
    node("21", "LoadImage")
    node("30", "HDRExpandDynamicRange", {"image": ["20", 0]})
    subject = branch("31", ["30", 0])
    subject = guidance(subject, "subject" if "39" in graph else "main")
    lens_output = subject
    if "39" in graph:
        node("22", "LoadImage")
        node("23", "LoadImage")
        node("24", "LoadImage")
        matte = branch("26", ["24", 0])
        node("25", "ImageToMask", {"image": matte}, {"channel": "red"})
        node("37", "HDRExpandDynamicRange", {"image": ["22", 0]})
        plate = branch("38", ["37", 0])
        plate = guidance(plate, "plate")
        node("39", "ImageCompositeMasked", {"destination": plate, "source": subject, "mask": ["25", 0]},
             {"x": 0, "y": 0, "resize_source": False})
        lens_output = ["39", 0]
    node("33", "HDRToneMap", {"image": lens_output})
    node("34", "ColorSpaceConvert", {"image": ["33", 0]})
    node("35", "RadianceFilmGrain", {"image": ["34", 0]})
    node("36", "HDRToneMap", {"image": ["35", 0]})
    node("32", "SaveImage", {"images": ["36", 0]})
    if not guided and title != "Save hero frame / " + ROUND_GRAPH_VERSION:
        errors.append("Round graph is missing the exact supported rendering-version marker.")
    if len(set(guide_values)) > 1:
        errors.append("Guide amount differs between colour layers.")
    for node_id in sorted(set(graph) - expected):
        errors.append(f"Round graph contains an unexpected node: {node_id}.")
    return errors


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def loopback_server(url):
    """No DNS lookups or environment proxies; localhost is normalized to 127.0.0.1."""
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme not in {"http", "https"} or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"}):
        raise ValueError("Server must be a plain loopback HTTP(S) origin without credentials or a path.")
    host = parsed.hostname
    if host == "localhost":
        host = "127.0.0.1"
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("Server must use localhost or a literal loopback IP address.") from exc
    if not address.is_loopback:
        raise ValueError("Only loopback ComfyUI servers are allowed.")
    port = parsed.port
    authority = f"[{address}]" if address.version == 6 else str(address)
    if port is not None:
        authority += f":{port}"
    return f"{parsed.scheme}://{authority}"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LocalHTTPError(RuntimeError):
    def __init__(self, code, body):
        super().__init__(f"Local ComfyUI returned HTTP {code}.")
        self.code, self.body = code, body.decode("utf-8", errors="replace")


class LocalClient:
    def __init__(self, server, deadline):
        self.server, self.deadline = loopback_server(server), deadline
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, path, data=None, headers=None):
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("Only local absolute endpoint paths are supported.")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Replay deadline reached; no cancellation was sent.")
        request = urllib.request.Request(self.server + path, data=data, headers=headers or {})
        try:
            with self.opener.open(request, timeout=min(30, remaining)) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            raise LocalHTTPError(exc.code, exc.read()) from exc

    def get_json(self, path):
        return json.loads(self.request(path))

    def post_json(self, path, payload):
        return json.loads(self.request(path, json.dumps(payload).encode(), {"Content-Type": "application/json"}))

    def get_bytes(self, path):
        return self.request(path)

    def upload(self, data, filename, subfolder):
        boundary = "hero_replay_" + uuid.uuid4().hex
        parts = []
        for name, value in (("type", "input"), ("subfolder", subfolder), ("overwrite", "false")):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{filename}"\r\nContent-Type: image/png\r\n\r\n'.encode())
        parts.extend((data, f"\r\n--{boundary}--\r\n".encode()))
        return json.loads(self.request("/upload/image", b"".join(parts),
                                       {"Content-Type": f"multipart/form-data; boundary={boundary}"}))


def inspect_graph(graph, schemas=None):
    """Restrict execution and check required inputs, literal choices/types and links."""
    errors, loads, saves = [], {}, []
    for node_id, node in graph.items():
        kind, inputs = node["class_type"], node.get("inputs", {})
        if kind not in ALLOWED_CLASSES:
            errors.append(f"Node {node_id}: class {kind} is not allowed in develop replay.")
        if kind == "LoadImage":
            image = inputs.get("image")
            if not isinstance(image, str) or not image.lower().endswith(".png"):
                errors.append(f"Node {node_id}: only bundled input PNGs can be loaded.")
            else:
                loads[node_id] = image
        if kind == "SaveImage":
            saves.append(node_id)
        if schemas is None:
            continue
        schema = schemas.get(kind)
        if not schema:
            errors.append(f"Node {node_id}: {kind} is missing from local /object_info.")
            continue
        required = schema.get("input", {}).get("required", {})
        permitted = {**required, **schema.get("input", {}).get("optional", {})}
        for key in required.keys() - inputs.keys():
            errors.append(f"Node {node_id} ({kind}): missing required input {key}.")
        for key in inputs.keys() - permitted.keys():
            errors.append(f"Node {node_id} ({kind}): input {key} is absent from the local schema.")
        for key, value in inputs.items():
            spec = permitted.get(key)
            if not isinstance(spec, (list, tuple)) or not spec:
                continue
            expected = spec[0]
            linked = isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and type(value[1]) is int
            if linked:
                upstream = graph.get(value[0])
                outputs = schemas.get(upstream["class_type"], {}).get("output", []) if upstream else []
                if value[1] < 0 or value[1] >= len(outputs):
                    errors.append(f"Node {node_id}.{key}: invalid local output connection {value}.")
                elif isinstance(expected, str) and expected != "*" and outputs[value[1]] not in {expected, "*"}:
                    errors.append(f"Node {node_id}.{key}: local connection type differs from {expected}.")
            elif isinstance(expected, list) or expected == "COMBO":
                # V1 nodes use [choices, metadata]; V3 uses ["COMBO", {"options": choices}].
                metadata = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                choices = expected if isinstance(expected, list) else metadata.get("options")
                if not isinstance(choices, list) or metadata.get("multiselect", False):
                    errors.append(f"Node {node_id}.{key}: unsupported local COMBO options schema.")
                    continue
                # New uploads are not yet in LoadImage's dynamic filename choices.
                if not (kind == "LoadImage" and key == "image") and value not in choices:
                    errors.append(f"Node {node_id}.{key}: {value!r} is not a local schema choice.")
            elif expected in {"INT", "FLOAT", "BOOLEAN", "STRING"}:
                valid = {"INT": type(value) is int, "FLOAT": type(value) in {int, float},
                         "BOOLEAN": type(value) is bool, "STRING": isinstance(value, str)}[expected]
                if not valid:
                    errors.append(f"Node {node_id}.{key}: expected literal {expected}.")
                elif expected in {"INT", "FLOAT"} and len(spec) > 1 and isinstance(spec[1], dict):
                    options = spec[1]
                    if value < options.get("min", value) or value > options.get("max", value):
                        errors.append(f"Node {node_id}.{key}: value is outside local schema bounds.")
            else:
                errors.append(f"Node {node_id}.{key}: {expected} requires a connection.")
    if len(saves) != 1:
        errors.append("Develop replay requires exactly one SaveImage output.")
    has_round_nodes = any(n["class_type"] in {"ImageBlur", "ImageScale", "RadianceGrade"} for n in graph.values())
    round_marked = graph.get("32", {}).get("_meta", {}).get("title") in {
        "Save hero frame / " + ROUND_GRAPH_VERSION,
        "Save hero frame / " + ROUND_GRAPH_VERSION + " / " + guide_eye.VERSION}
    if has_round_nodes or round_marked:
        errors.extend(inspect_round_graph(graph))
    elif not loads or not any(n["class_type"] == "RadianceDepthOfField" for n in graph.values()):
        errors.append("Selected graph is not a supported saved depth-of-field develop graph.")
    return {"ok": not errors, "errors": errors, "load_images": loads, "save_nodes": saves}


def view_path(filename, subfolder, kind):
    return "/view?" + urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": kind})


def execution_graph(graph):
    """Compare execution inputs; Comfy can append cache/UI metadata to history."""
    if not isinstance(graph, dict) or not all(isinstance(node, dict) for node in graph.values()):
        return None
    return {node_id: {"class_type": node.get("class_type"), "inputs": node.get("inputs", {})}
            for node_id, node in graph.items()}


def run(recipe_path, server, output_dir, take_number=None, timeout=600, client=None):
    """Execute once. client injection is for offline tests, never a CLI option."""
    server = loopback_server(server)
    if not 0 < timeout <= 86400:
        raise ValueError("Timeout must be positive and at most 86400 seconds.")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + timeout
    state = {"server": server, "status": "started", "replay_pass": False, "history_success": False,
             "namespace": "hero_replay_" + uuid.uuid4().hex, "timeout_seconds": timeout,
             "stage": "validate", "no_cancellation_sent": True}
    def save(name, value):
        (output_dir / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    save("run.json", state)
    try:
        recipe_data = Path(recipe_path).read_bytes()
        local_recipe = output_dir / "recipe.zip"
        local_recipe.write_bytes(recipe_data)
        state["recipe_sha256"] = sha256(recipe_data)
        validation, contents = validate(local_recipe)
        save("archive_validation.json", validation)
        if not validation["ok"]:
            raise ValueError("Recipe validation failed; see archive_validation.json.")
        recipe = json.loads(contents["recipe.json"])
        number = take_number if take_number is not None else recipe.get("selected_take")
        if type(number) is not int:
            raise ValueError("Specify --take for a recipe without selected_take.")
        matches = [t for t in recipe["takes"] if t["take"] == number]
        if len(matches) != 1:
            raise ValueError("The requested take is not uniquely present in the recipe.")
        take = matches[0]
        graph = json.loads(contents[take["graph"]])
        (output_dir / "original_graph.json").write_bytes(contents[take["graph"]])
        (output_dir / "reference.png").write_bytes(contents[take["hero"]])
        state.update(take=number, graph_member=take["graph"], original_graph_sha256=sha256(contents[take["graph"]]))
        preflight = inspect_graph(graph)
        save("preflight.json", preflight)
        if not preflight["ok"]:
            raise ValueError("Graph is not permitted; see preflight.json.")
        client = client or LocalClient(server, deadline)
        state["stage"] = "preflight"
        save("system_stats.json", client.get_json("/system_stats"))
        info = client.get_json("/object_info")
        subset = {kind: info[kind] for kind in {n["class_type"] for n in graph.values()} if kind in info}
        save("node_schemas.json", subset)
        preflight = inspect_graph(graph, subset)
        save("preflight.json", preflight)
        if not preflight["ok"]:
            raise ValueError("Local node schema is incompatible; see preflight.json.")
        submitted, changes, bindings, uploads = copy.deepcopy(graph), [], {}, []
        inputs_dir = output_dir / "inputs"
        inputs_dir.mkdir()
        state["stage"] = "upload"
        for index, name in enumerate(sorted(set(preflight["load_images"].values()))):
            data = contents[name]
            filename = f"input_{index:02d}.png"
            (inputs_dir / filename).write_bytes(data)
            response = client.upload(data, filename, state["namespace"])
            item = {"member": name, "sha256": sha256(data), "upload_response": response, "verified": False}
            uploads.append(item)
            save("uploads.json", uploads)
            if response.get("name") != filename or response.get("subfolder") != state["namespace"] or response.get("type") != "input":
                raise ValueError("Upload returned an unexpected input location.")
            downloaded = client.get_bytes(view_path(filename, state["namespace"], "input"))
            item["view_sha256"] = sha256(downloaded)
            item["verified"] = downloaded == data
            save("uploads.json", uploads)
            if not item["verified"]:
                raise ValueError("Uploaded input bytes differ from the archived PNG.")
            bindings[name] = f"{state['namespace']}/{filename}"
        for node_id, name in preflight["load_images"].items():
            submitted[node_id]["inputs"]["image"] = bindings[name]
            changes.append({"node": node_id, "input": "image", "before": name, "after": bindings[name]})
        save_node = preflight["save_nodes"][0]
        prefix = submitted[save_node]["inputs"].get("filename_prefix")
        submitted[save_node]["inputs"]["filename_prefix"] = state["namespace"] + "/hero"
        changes.append({"node": save_node, "input": "filename_prefix", "before": prefix,
                        "after": submitted[save_node]["inputs"]["filename_prefix"]})
        reversed_changes = copy.deepcopy(submitted)
        for change in changes:
            if change["before"] is None and change["input"] not in graph[change["node"]]["inputs"]:
                del reversed_changes[change["node"]]["inputs"][change["input"]]
            else:
                reversed_changes[change["node"]]["inputs"][change["input"]] = change["before"]
        if reversed_changes != graph:
            raise ValueError("Unexpected graph mutation outside input paths and output prefix.")
        save("submitted_graph.json", submitted)
        save("graph_changes.json", changes)
        state["submitted_graph_sha256"] = sha256((output_dir / "submitted_graph.json").read_bytes())
        state["stage"] = "submit"
        payload = {"prompt": submitted, "client_id": state["namespace"]}
        save("prompt_request.json", payload)
        try:
            response = client.post_json("/prompt", payload)
        except LocalHTTPError as exc:
            save("prompt_response.json", {"http_status": exc.code, "body": exc.body})
            raise
        save("prompt_response.json", response)
        prompt_id = response.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id or response.get("node_errors"):
            raise ValueError("Prompt was not accepted without node errors; see prompt_response.json.")
        state["prompt_id"], state["stage"] = prompt_id, "wait"
        save("run.json", state)
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Replay deadline reached; queued job was not cancelled.")
            history = client.get_json("/history/" + urllib.parse.quote(prompt_id, safe=""))
            save("history_last.json", history)
            entry = history.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("completed") is True or status.get("status_str") == "error":
                    save("history_terminal.json", history)
                    break
            time.sleep(min(1.0, max(0, deadline - time.monotonic())))
        if status.get("completed") is not True or status.get("status_str") != "success":
            raise ValueError("ComfyUI history reports an unsuccessful execution.")
        if any(message and message[0] in {"execution_error", "execution_interrupted"} for message in status.get("messages", [])):
            raise ValueError("ComfyUI history contains an execution error/interruption.")
        recorded_prompt = entry.get("prompt", [])
        if (not isinstance(recorded_prompt, list) or len(recorded_prompt) < 3
                or recorded_prompt[1] != prompt_id
                or execution_graph(recorded_prompt[2]) != execution_graph(submitted)):
            raise ValueError("Successful history does not identify the submitted prompt's classes/inputs exactly.")
        state["history_success"] = True
        images = entry.get("outputs", {}).get(save_node, {}).get("images", [])
        if len(images) != 1:
            raise ValueError("Successful prompt must have one image from its selected SaveImage node.")
        output = images[0]
        filename, subfolder = output.get("filename"), output.get("subfolder")
        if (not isinstance(filename, str) or PurePosixPath(filename).name != filename
                or "\\" in filename or ":" in filename or output.get("type") != "output"
                or subfolder != state["namespace"]):
            raise ValueError("SaveImage returned an output outside this run's unique namespace.")
        state["stage"] = "download"
        result = client.get_bytes(view_path(filename, subfolder, "output"))
        result_path = output_dir / "output.png"
        result_path.write_bytes(result)
        state["output"] = {**output, "file_sha256": sha256(result), "save_node": save_node}
        state["stage"] = "compare"
        comparison = compare_recipe(local_recipe, result_path, number)
        save("comparison.json", comparison)
        state["comparison_status"] = comparison["comparison_status"]
        state["replay_pass"] = state["history_success"] and comparison.get("within_tolerance", False)
        state["status"] = "matched" if state["replay_pass"] else "different"
        state["stage"] = "complete"
    except Exception as exc:
        state["status"] = "timeout" if isinstance(exc, TimeoutError) else "failed"
        state["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, LocalHTTPError):
            state["error"].update(http_status=exc.code, response_body=exc.body)
    state["elapsed_seconds"] = round(time.monotonic() - started, 3)
    save("run.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("zip", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--take", type=int)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    try:
        state = run(args.zip, args.server, args.output, args.take, args.timeout)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(state, indent=2))
    raise SystemExit(0 if state["replay_pass"] else 1)


if __name__ == "__main__":
    main()
