#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shlex
import subprocess
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = REPO_ROOT / "visualizer"
DEFAULT_ASSETS = {
    "/": STATIC_DIR / "console_gui.html",
    "/console_gui.html": STATIC_DIR / "console_gui.html",
    "/console_gui.css": STATIC_DIR / "console_gui.css",
    "/console_gui.js": STATIC_DIR / "console_gui.js",
}


class MiniZeroRecord:
    def __init__(self, raw_record: str, tags: dict[str, str], actions: list[dict[str, Any]]) -> None:
        self.raw_record = raw_record
        self.tags = tags
        self.actions = actions
        self.board_size = int(tags.get("SZ", "0"))
        self.game = tags.get("GM", "")
        self.result = tags.get("RE", "")


def _extract_record(text: str) -> str:
    """Return the first complete MiniZero SGF-like game tree in pasted text."""
    start = text.find("(")
    if start == -1:
        raise ValueError("Record must contain a MiniZero SGF-like game string.")

    in_value = False
    escape_next = False
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if in_value:
            if char == "\\" and not escape_next:
                escape_next = True
            elif char == "]" and not escape_next:
                in_value = False
            else:
                escape_next = False
            continue

        if char == "[":
            in_value = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start: index + 1]

    raise ValueError("Record is missing the closing ')'.")


def _split_sgf_nodes(record: str) -> list[str]:
    """Split a single game tree into root and move nodes without breaking bracket values."""
    body = record.strip()[1:-1]
    nodes: list[str] = []
    current: list[str] = []
    in_value = False
    escape_next = False
    seen_node = False

    for char in body:
        if in_value:
            current.append(char)
            if char == "\\" and not escape_next:
                escape_next = True
            elif char == "]" and not escape_next:
                in_value = False
            else:
                escape_next = False
            continue

        if char == ";":
            if seen_node:
                nodes.append("".join(current))
                current = []
            else:
                seen_node = True
            continue
        if char == "[":
            in_value = True
        current.append(char)

    if seen_node:
        nodes.append("".join(current))
    return [node for node in nodes if node]


def _parse_sgf_node(node: str) -> list[tuple[str, str]]:
    """Parse SGF-style key[value] pairs while preserving escaped value text."""
    props: list[tuple[str, str]] = []
    index = 0
    while index < len(node):
        while index < len(node) and node[index].isspace():
            index += 1
        key_start = index
        while index < len(node) and node[index] != "[":
            index += 1
        if index >= len(node):
            break
        key = node[key_start:index]
        index += 1

        value: list[str] = []
        escape_next = False
        while index < len(node):
            char = node[index]
            index += 1
            if char == "\\" and not escape_next:
                escape_next = True
                continue
            if char == "]" and not escape_next:
                break
            value.append(char)
            escape_next = False
        if key:
            props.append((key, "".join(value)))
    return props


def _action_id_to_coord(action_id: int, board_size: int) -> str:
    if action_id == board_size * board_size:
        return "PASS"
    if action_id < 0 or action_id > board_size * board_size:
        raise ValueError(f"Action id {action_id} is outside board size {board_size}.")
    column = action_id % board_size
    row = action_id // board_size
    return f"{chr(ord('A') + column + (1 if column >= 8 else 0))}{row + 1}"


def _record_move_to_action_id(value: str, board_size: int) -> int:
    """Accept MiniZero action ids and normal SGF coordinates for replay import."""
    move = value.split("|", 1)[0].strip()
    if not move or move.upper() == "PASS" or move.lower() == "tt":
        return board_size * board_size
    if move[0].isdigit():
        return int(move)
    if len(move) != 2:
        raise ValueError(f"Unsupported move format: {value}")
    column = ord(move[0].upper()) - ord("A")
    row = (board_size - 1) - (ord(move[1].upper()) - ord("A"))
    action_id = row * board_size + column
    if column < 0 or column >= board_size or row < 0 or row >= board_size:
        raise ValueError(f"Move {value} is outside board size {board_size}.")
    return action_id


def parse_minizero_record(text: str) -> MiniZeroRecord:
    """Parse the subset of MiniZero internal records needed by GUI replay."""
    record = _extract_record(text)
    nodes = _split_sgf_nodes(record)
    if not nodes:
        raise ValueError("Record has no SGF nodes.")

    tags = dict(_parse_sgf_node(nodes[0]))
    board_size = int(tags.get("SZ", "0"))
    if board_size <= 0:
        raise ValueError("MiniZero record must include a positive SZ tag.")

    actions: list[dict[str, Any]] = []
    for node in nodes[1:]:
        props = _parse_sgf_node(node)
        if not props:
            continue
        player, move_value = props[0]
        if player not in {"B", "W"}:
            continue
        action_id = _record_move_to_action_id(move_value, board_size)
        actions.append(
            {
                "player": player,
                "raw": move_value,
                "action_id": action_id,
                "move": _action_id_to_coord(action_id, board_size),
                "info": dict(props[1:]),
            }
        )

    return MiniZeroRecord(record, tags, actions)


def _same_actions(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        a.get("player") == b.get("player") and a.get("action_id") == b.get("action_id")
        for a, b in zip(left, right)
    )


class ConsoleSession:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_lines: deque[str] = deque(maxlen=400)
        self._command = ""
        self._cwd = str(REPO_ROOT)
        self._default_command = ""
        self._default_cwd = str(REPO_ROOT)

    def _append_log(self, line: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self._stderr_lines.append(f"[{timestamp}] {line}")

    def _shell(self) -> str:
        shell = os.environ.get("SHELL")
        if shell and Path(shell).exists():
            return shell
        for candidate in ("/bin/zsh", "/bin/bash", "/bin/sh"):
            if Path(candidate).exists():
                return candidate
        return "/bin/sh"

    def _consume_stderr(self, process: subprocess.Popen[str]) -> None:
        assert process.stderr is not None
        for line in process.stderr:
            cleaned = line.rstrip("\r\n")
            if cleaned:
                self._append_log(cleaned)
        return_code = process.poll()
        if return_code is not None:
            self._append_log(f"[process exited] code={return_code}")

    def _ensure_running(self) -> subprocess.Popen[str]:
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("Console session is not running.")
        return self._process

    def _read_response(self, process: subprocess.Popen[str]) -> dict[str, Any]:
        """Read one GTP response block from console stdout."""
        assert process.stdout is not None
        lines: list[str] = []
        while True:
            line = process.stdout.readline()
            if line == "":
                raise RuntimeError("Console process terminated while waiting for a response.")
            cleaned = line.rstrip("\r\n")
            if not lines and cleaned == "":
                continue
            if cleaned == "":
                break
            lines.append(cleaned)

        if not lines:
            raise RuntimeError("Received an empty response from console mode.")

        header = lines[0]
        status = header[0]
        payload_head = header[1:]
        index = 0
        while index < len(payload_head) and payload_head[index].isdigit():
            index += 1
        command_id = payload_head[:index]
        payload_head = payload_head[index:]
        if payload_head.startswith(" "):
            payload_head = payload_head[1:]

        payload_lines = []
        if payload_head:
            payload_lines.append(payload_head)
        payload_lines.extend(lines[1:])

        return {
            "success": status == "=",
            "status": status,
            "command_id": command_id,
            "message": "\n".join(payload_lines),
            "raw_lines": lines,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            return {
                "running": running,
                "pid": self._process.pid if running and self._process else None,
                "command": self._command,
                "cwd": self._cwd,
                "default_command": self._default_command,
                "default_cwd": self._default_cwd,
            }

    def configure_default(self, command: str, cwd: str | None = None) -> None:
        with self._lock:
            self._default_command = command
            self._default_cwd = str(Path(cwd).expanduser().resolve()) if cwd else str(REPO_ROOT)

    def logs(self) -> list[str]:
        with self._lock:
            return list(self._stderr_lines)

    def _startup_error(self, message: str) -> RuntimeError:
        """Attach recent stderr so failed model/config launches are actionable."""
        process = self._process
        if self._stderr_thread:
            self._stderr_thread.join(timeout=0.5)
        stderr = "\n".join(self.logs()[-80:])
        return_code = process.poll() if process else None

        details = [message]
        if return_code is not None:
            details.append(f"exit_code={return_code}")
        if stderr:
            details.append("stderr:\n" + stderr)
        return RuntimeError("\n".join(details))

    def start(self, command: str | None = None, cwd: str | None = None) -> dict[str, Any]:
        """Start console mode and verify it is responsive before serving requests."""
        launch_command = (command or "").strip() or self._default_command
        launch_cwd = cwd or self._default_cwd

        if not launch_command.strip():
            raise ValueError("A console launch command is required.")

        working_directory = Path(launch_cwd).expanduser().resolve() if launch_cwd else REPO_ROOT
        if not working_directory.exists():
            raise ValueError(f"Working directory does not exist: {working_directory}")

        self.stop()
        self._stderr_lines.clear()

        process = subprocess.Popen(
            [self._shell(), "-lc", launch_command],
            cwd=str(working_directory),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self._process = process
        self._command = launch_command
        self._cwd = str(working_directory)
        self._stderr_thread = threading.Thread(target=self._consume_stderr, args=(process,), daemon=True)
        self._stderr_thread.start()

        time.sleep(0.1)
        if process.poll() is not None:
            raise self._startup_error("Console process exited immediately.")

        try:
            handshake = self.send_command("name")
        except RuntimeError as exc:
            raise self._startup_error(str(exc)) from exc
        if not handshake["success"]:
            raise RuntimeError(f"Console handshake failed: {handshake['message']}")
        return self.status()

    def stop(self) -> None:
        with self._lock:
            process = self._process
            self._process = None

        if process is None:
            return

        if process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write("quit\n")
                    process.stdin.flush()
            except OSError:
                pass

            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)

    def send_command(self, command: str) -> dict[str, Any]:
        if not command.strip():
            raise ValueError("Command must not be empty.")

        with self._lock:
            process = self._ensure_running()
            assert process.stdin is not None
            process.stdin.write(command.rstrip("\n") + "\n")
            process.stdin.flush()
            return self._read_response(process)

    def state(self) -> dict[str, Any]:
        response = self.send_command("board_state")
        if not response["success"]:
            raise RuntimeError(response["message"] or "board_state command failed.")
        try:
            return json.loads(response["message"])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to decode board_state JSON: {exc}") from exc


class ReplaySession:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._record: MiniZeroRecord | None = None
        self._current_step = 0

    def clear(self) -> None:
        with self._lock:
            self._record = None
            self._current_step = 0

    def load(self, record_text: str) -> dict[str, Any]:
        record = parse_minizero_record(record_text)
        with self._lock:
            self._record = record
            self._current_step = 0
        return self.status()

    def sync_from_current(self, session: ConsoleSession, only_if_empty: bool = False) -> dict[str, Any]:
        """Use the engine's current game_string as replay source without losing future steps."""
        with self._lock:
            if only_if_empty and self._record is not None:
                return self.status()

        response = session.send_command("game_string")
        if not response["success"]:
            return self.status()

        record = parse_minizero_record(response["message"])
        with self._lock:
            if not record.actions and self._record is None:
                return {"loaded": False}

            if self._record is not None:
                current_prefix = self._record.actions[: self._current_step]
                if _same_actions(record.actions, current_prefix):
                    return self.status()

            self._record = record
            self._current_step = len(record.actions)
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._record is None:
                return {"loaded": False}
            return {
                "loaded": True,
                "current_step": self._current_step,
                "total_steps": len(self._record.actions),
                "board_size": self._record.board_size,
                "game": self._record.game,
                "result": self._record.result,
                "tags": self._record.tags,
            }

    def replay_to_step(self, session: ConsoleSession, step: int) -> dict[str, Any]:
        """Rebuild engine state from the record prefix up to the requested step."""
        with self._lock:
            if self._record is None:
                raise ValueError("No record is loaded.")
            record = self._record
            target_step = max(0, min(step, len(record.actions)))

        state = session.state()
        board_size = state.get("board_size")
        if board_size and int(board_size) != record.board_size:
            raise ValueError(f"Record board size {record.board_size} does not match engine board size {board_size}.")
        engine_game = str(state.get("game", "")).split("_", 1)[0]
        record_game = str(record.game).split("_", 1)[0]
        if record_game and not record_game.isdigit() and engine_game and record_game != engine_game:
            raise ValueError(f"Record game {record.game} does not match running engine {state.get('game')}.")

        session.send_command("clear_board")
        for action in record.actions[:target_step]:
            # IIG failed tries may return '?', but they can still update hidden-information state.
            session.send_command(f"play {action['player']} {action['move']}")

        with self._lock:
            self._current_step = target_step
        return session.state()


SESSION = ConsoleSession()
REPLAY = ReplaySession()


def _redact_failed_play_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("success"):
        return result
    redacted = dict(result)
    redacted["message"] = ""
    redacted["raw_lines"] = []
    return redacted


def _conf_keys(conf_string: str) -> set[str]:
    keys: set[str] = set()
    for item in conf_string.split(":"):
        if "=" not in item:
            continue
        key = item.split("=", 1)[0].strip()
        if key:
            keys.add(key)
    return keys


def _build_console_command(args: argparse.Namespace) -> str:
    """Build a quoted console-mode command from the web bridge CLI arguments."""
    if args.discriminator_model and (not args.game or not args.model):
        raise ValueError("--discriminator_model requires --game and --model.")
    if not args.game and not args.model:
        return ""
    if args.game and not args.model:
        raise ValueError("--model is required when --game is set.")
    if args.model and not args.game:
        raise ValueError("--game is required when --model is set.")

    executable = args.executable or f"./build/{args.game}/minizero_{args.game}"
    conf_file = args.conf_file or f"cfg/{args.game}_train.cfg"
    conf_parts = [args.conf_str.strip()] if args.conf_str.strip() else []
    keys = _conf_keys(args.conf_str)

    conf_parts.append(f"nn_file_name={args.model}")
    if args.discriminator_model:
        conf_parts.append("iig_use_discriminator=true")
        conf_parts.append(f"iig_discriminator_file_name={args.discriminator_model}")
    elif "iig_use_discriminator" not in keys and "iig_discriminator_file_name" not in keys:
        conf_parts.append("iig_use_discriminator=false")

    if "program_quiet" not in keys:
        conf_parts.append("program_quiet=true")

    conf_string = ":".join(conf_parts)
    return (
        f"{shlex.quote(executable)} -mode console "
        f"-conf_file {shlex.quote(conf_file)} "
        f"-conf_str {shlex.quote(conf_string)}"
    )


def _resolve_launch_path(cwd: str, path: str) -> Path:
    launch_path = Path(path).expanduser()
    if not launch_path.is_absolute():
        launch_path = Path(cwd).expanduser() / launch_path
    return launch_path.resolve()


def _validate_launch_paths(args: argparse.Namespace) -> None:
    """Fail before binding HTTP if the configured engine assets are missing."""
    if not args.game and not args.model:
        return

    cwd = str(Path(args.cwd).expanduser().resolve())
    checks = [
        ("executable", args.executable or f"./build/{args.game}/minizero_{args.game}"),
        ("config", args.conf_file or f"cfg/{args.game}_train.cfg"),
        ("model", args.model),
    ]
    if args.discriminator_model:
        checks.append(("discriminator_model", args.discriminator_model))

    missing = []
    for label, path in checks:
        if path and not _resolve_launch_path(cwd, path).exists():
            missing.append(f"{label}: {path}")

    if missing:
        raise ValueError("Missing launch file(s): " + ", ".join(missing))


class ConsoleGUIHandler(BaseHTTPRequestHandler):
    server_version = "MiniZeroConsoleGUI/0.1"

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        payload = self.rfile.read(content_length).decode("utf-8")
        return json.loads(payload) if payload else {}

    def _write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _write_error(self, status: HTTPStatus, message: str) -> None:
        self._write_json({"error": message}, status)

    def _serve_asset(self, path: str) -> None:
        asset = DEFAULT_ASSETS.get(path)
        if asset is None or not asset.exists():
            self._write_error(HTTPStatus.NOT_FOUND, f"Unknown path: {path}")
            return

        mime_type, _ = mimetypes.guess_type(str(asset))
        content = asset.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/api/status":
                self._write_json({"status": SESSION.status()})
            elif path == "/api/state":
                self._write_json({"state": SESSION.state(), "status": SESSION.status()})
            elif path == "/api/replay":
                status = SESSION.status()
                replay = REPLAY.sync_from_current(SESSION, only_if_empty=True) if status["running"] else REPLAY.status()
                self._write_json({"replay": replay, "status": status})
            elif path == "/api/logs":
                self._write_json({"logs": SESSION.logs(), "status": SESSION.status()})
            elif path == "/api/final_score":
                self._write_json({"result": SESSION.send_command("final_score"), "status": SESSION.status()})
            elif path == "/api/export_record":
                self._write_json({"result": SESSION.send_command("game_string"), "status": SESSION.status()})
            else:
                self._serve_asset(path)
        except ValueError as exc:
            self._write_error(HTTPStatus.BAD_REQUEST, str(exc))
        except RuntimeError as exc:
            self._write_error(HTTPStatus.CONFLICT, str(exc))
        except Exception as exc:  # pragma: no cover - last-resort server guard
            self._write_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            payload = self._read_json()

            if path == "/api/stop":
                REPLAY.clear()
                SESSION.stop()
                self._write_json({"status": SESSION.status(), "replay": REPLAY.status()})
                return

            if path == "/api/play":
                player = payload.get("player", "").strip()
                move = payload.get("move", "").strip()
                if not player or not move:
                    raise ValueError("Both player and move are required.")
                result = SESSION.send_command(f"play {player} {move}")
                state = SESSION.state()
                replay = REPLAY.sync_from_current(SESSION)
                self._write_json({
                    "result": _redact_failed_play_result(result),
                    "state": state,
                    "status": SESSION.status(),
                    "replay": replay,
                })
                return

            if path == "/api/genmove":
                player = payload.get("player", "").strip()
                if not player:
                    raise ValueError("Player is required.")
                result = SESSION.send_command(f"genmove {player}")
                state = SESSION.state()
                replay = REPLAY.sync_from_current(SESSION)
                self._write_json({
                    "result": result,
                    "state": state,
                    "status": SESSION.status(),
                    "replay": replay,
                })
                return

            if path == "/api/clear":
                REPLAY.clear()
                result = SESSION.send_command("clear_board")
                self._write_json({"result": result, "state": SESSION.state(), "status": SESSION.status(), "replay": REPLAY.status()})
                return

            if path == "/api/load_model":
                model_path = payload.get("path", "").strip()
                if not model_path:
                    raise ValueError("Model path is required.")
                result = SESSION.send_command(f"load_model {model_path}")
                self._write_json({"result": result, "state": SESSION.state(), "status": SESSION.status(), "replay": REPLAY.status()})
                return

            if path == "/api/import_record":
                record = payload.get("record", "")
                if not record.strip():
                    raise ValueError("Record text is required.")
                REPLAY.load(record)
                state = REPLAY.replay_to_step(SESSION, 0)
                self._write_json({"state": state, "status": SESSION.status(), "replay": REPLAY.status()})
                return

            if path == "/api/replay_step":
                try:
                    step = int(payload.get("step", 0))
                except (TypeError, ValueError) as exc:
                    raise ValueError("Replay step must be an integer.") from exc
                REPLAY.sync_from_current(SESSION, only_if_empty=True)
                state = REPLAY.replay_to_step(SESSION, step)
                self._write_json({"state": state, "status": SESSION.status(), "replay": REPLAY.status()})
                return

            if path == "/api/command":
                REPLAY.clear()
                command = payload.get("command", "")
                result = SESSION.send_command(command)
                state = None
                try:
                    state = SESSION.state()
                except RuntimeError:
                    state = None
                self._write_json({"result": result, "state": state, "status": SESSION.status(), "replay": REPLAY.status()})
                return

            self._write_error(HTTPStatus.NOT_FOUND, f"Unknown API path: {path}")
        except json.JSONDecodeError as exc:
            self._write_error(HTTPStatus.BAD_REQUEST, f"Invalid JSON payload: {exc}")
        except ValueError as exc:
            self._write_error(HTTPStatus.BAD_REQUEST, str(exc))
        except RuntimeError as exc:
            self._write_error(HTTPStatus.CONFLICT, str(exc))
        except Exception as exc:  # pragma: no cover - last-resort server guard
            self._write_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a lightweight web GUI for MiniZero console mode.")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    parser.add_argument("--port", default=8765, type=int, help="HTTP bind port")
    parser.add_argument("--game", choices=("phantomgo", "darkhex"), help="Game executable/config family")
    parser.add_argument("--model", help="Player model path for nn_file_name")
    parser.add_argument("--discriminator_model", help="Discriminator model path for iig_discriminator_file_name")
    parser.add_argument("--executable", help="MiniZero console executable path")
    parser.add_argument("--cwd", default=str(REPO_ROOT), help="Working directory for the console process")
    parser.add_argument("-conf_file", "--conf_file", dest="conf_file", help="MiniZero configuration file")
    parser.add_argument("-conf_str", "--conf_str", dest="conf_str", default="", help="Extra MiniZero config string")
    args = parser.parse_args()

    try:
        default_command = _build_console_command(args)
        _validate_launch_paths(args)
    except ValueError as exc:
        parser.error(str(exc))

    SESSION.configure_default(default_command, args.cwd)
    if default_command:
        try:
            SESSION.start()
        except Exception as exc:
            raise SystemExit(f"Failed to start MiniZero console: {exc}") from exc

    try:
        server = ThreadingHTTPServer((args.host, args.port), ConsoleGUIHandler)
    except OSError as exc:
        raise SystemExit(f"Failed to bind http://{args.host}:{args.port}: {exc}") from exc
    print(f"Console GUI listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        SESSION.stop()
        server.server_close()


if __name__ == "__main__":
    main()
