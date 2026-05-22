const stateStore = {
    status: { running: false, pid: null, command: "", cwd: "", default_command: "", default_cwd: "" },
    state: null,
    logs: [],
    lastResult: null,
    finalScore: null,
    finalScoreKey: null,
    replay: { loaded: false },
    ui: {
        mode: "human-ai",
        humanColor: "B",
        autoRunning: false,
        busy: false,
    },
    autoTimer: null,
    aiSequence: 0,
};

const boardBindings = {
    perfect: document.getElementById("board-perfect"),
    black_view: document.getElementById("board-black"),
    white_view: document.getElementById("board-white"),
};

const panelBindings = {
    perfect: document.querySelector('[data-board-panel="perfect"]'),
    black_view: document.querySelector('[data-board-panel="black"]'),
    white_view: document.querySelector('[data-board-panel="white"]'),
};

const visibilityBindings = {
    perfect: document.getElementById("toggle-perfect"),
    black_view: document.getElementById("toggle-black"),
    white_view: document.getElementById("toggle-white"),
};

const elements = {
    sessionPill: document.getElementById("session-pill"),
    statusDot: document.getElementById("status-dot"),
    stderrLog: document.getElementById("stderr-log"),
    modeHint: document.getElementById("mode-hint"),
    modeSelect: document.getElementById("mode-select"),
    humanColorSelect: document.getElementById("human-color-select"),
    humanColorField: document.getElementById("human-color-field"),
    delaySelect: document.getElementById("delay-select"),
    delayField: document.getElementById("delay-field"),
    autoToggleButton: document.getElementById("auto-toggle-button"),
    recordText: document.getElementById("record-text"),
    importRecordButton: document.getElementById("import-record-button"),
    exportRecordButton: document.getElementById("export-record-button"),
    replayPrev5Button: document.getElementById("replay-prev-5-button"),
    replayPrevButton: document.getElementById("replay-prev-button"),
    replayNextButton: document.getElementById("replay-next-button"),
    replayNext5Button: document.getElementById("replay-next-5-button"),
    replayStepRange: document.getElementById("replay-step-range"),
    replayStepLabel: document.getElementById("replay-step-label"),
};

const SVG_NS = "http://www.w3.org/2000/svg";

function sleep(ms) {
    return new Promise((resolve) => {
        window.setTimeout(resolve, ms);
    });
}

function fileLabel(index) {
    return String.fromCharCode("A".charCodeAt(0) + index + (index >= 8 ? 1 : 0));
}

function coordFromTopRow(column, rowFromTop, boardSize) {
    return `${fileLabel(column)}${boardSize - rowFromTop}`;
}

function pointFromCoord(coord, boardSize) {
    const match = /^([A-Z])(\d+)$/i.exec(coord || "");
    if (!match) {
        return null;
    }

    const letterCode = match[1].toUpperCase().charCodeAt(0);
    let column = letterCode - "A".charCodeAt(0);
    if (match[1].toUpperCase() >= "I") {
        column -= 1;
    }

    const rowFromTop = boardSize - Number(match[2]);
    if (column < 0 || column >= boardSize || rowFromTop < 0 || rowFromTop >= boardSize) {
        return null;
    }
    return { column, rowFromTop };
}

function usesRedBluePlayers() {
    const state = stateStore.state;
    return state?.board_type === "hex" || state?.game?.startsWith("darkhex");
}

function playerCopy() {
    if (usesRedBluePlayers()) {
        return {
            B: "Red",
            W: "Blue",
            blackViewEyebrow: "Red",
            blackViewTitle: "Red View",
            whiteViewEyebrow: "Blue",
            whiteViewTitle: "Blue View",
        };
    }
    return {
        B: "Black",
        W: "White",
        blackViewEyebrow: "Black",
        blackViewTitle: "Black View",
        whiteViewEyebrow: "White",
        whiteViewTitle: "White View",
    };
}

function colorName(player) {
    const copy = playerCopy();
    if (copy[player]) { return copy[player]; }
    return player || "Player";
}

function setInputWhenIdle(id, value) {
    if (!value) {
        return;
    }
    const input = document.getElementById(id);
    if (document.activeElement !== input) {
        input.value = value;
    }
}

function currentTurn() {
    return stateStore.state?.turn || "B";
}

function aiColor() {
    return stateStore.ui.humanColor === "B" ? "W" : "B";
}

function playColor() {
    if (stateStore.ui.mode === "manual") {
        return currentTurn();
    }
    return stateStore.ui.humanColor;
}

function replayLoaded() {
    return stateStore.replay?.loaded === true;
}

function viewingReplayPrefix() {
    if (!replayLoaded()) {
        return false;
    }
    return (stateStore.replay.current_step || 0) < (stateStore.replay.total_steps || 0);
}

function ownerForBoard(boardKey) {
    if (boardKey === "black_view") {
        return "B";
    }
    if (boardKey === "white_view") {
        return "W";
    }
    return null;
}

function boardForPlayer(player) {
    return player === "B" ? "black_view" : "white_view";
}

function canPlayOnBoard(boardKey) {
    if (boardKey === "perfect") {
        return true;
    }
    return boardKey === boardForPlayer(playColor());
}

function setMessage(message, success = true) {
    stateStore.lastResult = { success, message };
}

function cancelPendingAi() {
    stateStore.aiSequence += 1;
}

function canManualPlay() {
    if (!stateStore.status.running || stateStore.ui.busy || !stateStore.state || stateStore.state.is_terminal) {
        return false;
    }
    if (stateStore.ui.mode === "ai-auto") {
        return false;
    }
    return stateStore.ui.mode !== "human-ai" || currentTurn() === stateStore.ui.humanColor;
}

function canUseGenmove() {
    if (!stateStore.status.running || stateStore.ui.busy || !stateStore.state || stateStore.state.is_terminal) {
        return false;
    }
    return stateStore.ui.mode !== "human-ai" || currentTurn() !== stateStore.ui.humanColor;
}

function finalScoreKey() {
    if (!stateStore.state?.is_terminal) {
        return null;
    }
    return String(stateStore.state.action_history?.length || 0);
}

function formatFinalScore(score) {
    if (score > 0) {
        return `${colorName("B")} wins.`;
    }
    if (score < 0) {
        return `${colorName("W")} wins.`;
    }
    return "Draw.";
}

function parseFinalScore(message) {
    const score = Number.parseFloat(message);
    if (!Number.isFinite(score)) {
        return null;
    }
    return {
        score,
        label: formatFinalScore(score),
    };
}

function boardStateProblem() {
    const state = stateStore.state;
    if (!state) {
        return "Connect to an engine.";
    }
    if (state.unsupported) {
        return `${state.game || "This game"} does not expose GUI board_state. Rebuild this game target.`;
    }
    if (!state.board_size) {
        return "board_state is missing board_size. Rebuild this game target.";
    }
    if (!state.boards) {
        return "board_state is missing boards. Rebuild this game target.";
    }

    const missingBoards = ["perfect", "black_view", "white_view"].filter((key) => !state.boards[key]?.rows);
    if (missingBoards.length) {
        return `board_state is missing ${missingBoards.join(", ")} rows. Rebuild this game target.`;
    }
    return "";
}

async function api(path, options = {}) {
    const response = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options,
    });
    const contentType = response.headers.get("Content-Type") || "";
    const payload = contentType.includes("application/json")
        ? await response.json()
        : { error: await response.text() || response.statusText };
    if (!response.ok) {
        throw new Error(payload.error || "Request failed.");
    }
    return payload;
}

function setPanelCopy(key, eyebrow, title) {
    const panel = panelBindings[key];
    panel.querySelector(".board-card-header span").textContent = eyebrow;
    panel.querySelector("h2").textContent = title;
}

function renderDynamicCopy() {
    const copy = playerCopy();
    document.body.dataset.gameSkin = usesRedBluePlayers() ? "hex" : "go";
    setPanelCopy("perfect", "Perfect", "Full Board");
    setPanelCopy("black_view", copy.blackViewEyebrow, copy.blackViewTitle);
    setPanelCopy("white_view", copy.whiteViewEyebrow, copy.whiteViewTitle);
    elements.humanColorSelect.options[0].textContent = copy.B;
    elements.humanColorSelect.options[1].textContent = copy.W;
}

function renderStatus() {
    const running = stateStore.status.running;
    const mode = stateStore.ui.mode;
    const game = stateStore.state?.game || "MiniZero IIG";
    const passEnabled = stateStore.state?.pass_enabled !== false;

    renderDynamicCopy();
    elements.sessionPill.textContent = running ? `Connected PID ${stateStore.status.pid}` : "Disconnected";
    elements.statusDot.classList.toggle("is-on", running);

    document.getElementById("game-label").textContent = game;
    setInputWhenIdle("cwd-input", stateStore.status.cwd || stateStore.status.default_cwd);
    setInputWhenIdle("command-input", stateStore.status.command || stateStore.status.default_command);

    elements.humanColorField.hidden = mode !== "human-ai";
    elements.delayField.hidden = mode !== "ai-auto";
    elements.autoToggleButton.hidden = mode !== "ai-auto";
    elements.autoToggleButton.textContent = stateStore.ui.autoRunning ? "Pause AI" : "Run AI";

    document.getElementById("stop-button").disabled = !running || stateStore.ui.busy;
    document.getElementById("genmove-auto-button").disabled = !canUseGenmove();
    document.getElementById("pass-button").hidden = !passEnabled;
    document.getElementById("pass-button").disabled = !passEnabled || !canManualPlay();
    document.getElementById("clear-button").disabled = !running || stateStore.ui.busy;
}

function renderModeHint() {
    // Keep the main status line short; detailed traces stay in History/Advanced.
    elements.modeHint.classList.toggle("is-error", stateStore.lastResult?.success === false);
    if (!stateStore.status.running) {
        elements.modeHint.textContent = "Connect to an engine.";
        return;
    }
    if (!stateStore.state) {
        elements.modeHint.textContent = "Waiting for board state.";
        return;
    }

    const problem = boardStateProblem();
    if (problem) {
        elements.modeHint.textContent = problem;
        return;
    }

    const turn = colorName(currentTurn());
    if (viewingReplayPrefix()) {
        const current = stateStore.replay.current_step || 0;
        const total = stateStore.replay.total_steps || 0;
        if (stateStore.state.is_terminal && stateStore.finalScore) {
            elements.modeHint.textContent = `Replay ${current}/${total}. Game over. ${stateStore.finalScore.label}`;
        } else {
            elements.modeHint.textContent = `Replay ${current}/${total}. ${turn} to play.`;
        }
    } else if (stateStore.state.is_terminal) {
        if (stateStore.finalScore) {
            elements.modeHint.textContent = `Game over. ${stateStore.finalScore.label}`;
        } else if (stateStore.lastResult?.success === false) {
            elements.modeHint.textContent = stateStore.lastResult.message;
        } else {
            elements.modeHint.textContent = "Game over.";
        }
    } else if (stateStore.lastResult?.message) {
        elements.modeHint.textContent = stateStore.lastResult.message;
    } else if (stateStore.ui.mode === "human-ai" && currentTurn() !== stateStore.ui.humanColor) {
        elements.modeHint.textContent = `${turn} thinking.`;
    } else if (stateStore.ui.mode === "ai-auto") {
        elements.modeHint.textContent = stateStore.ui.autoRunning ? `AI running. ${turn} to play.` : `AI paused. ${turn} to play.`;
    } else {
        elements.modeHint.textContent = `${turn} to play.`;
    }
}

function renderLogs() {
    elements.stderrLog.textContent = stateStore.logs.length ? stateStore.logs.join("\n") : "No stderr output yet.";
    elements.stderrLog.scrollTop = elements.stderrLog.scrollHeight;
}

function renderHistory(listId, items, limit = 36) {
    const list = document.getElementById(listId);
    list.innerHTML = "";

    if (!items || !items.length) {
        const empty = document.createElement("li");
        empty.textContent = "Empty";
        list.appendChild(empty);
        return;
    }

    items.slice(-limit).forEach((item, index) => {
        const entry = document.createElement("li");
        entry.textContent = `${items.length - Math.min(items.length, limit) + index + 1}. ${colorName(item.player)} ${item.move}`;
        list.appendChild(entry);
    });
}

function renderReplayControls() {
    const replay = stateStore.replay || { loaded: false };
    const loaded = replay.loaded === true;
    const current = loaded ? replay.current_step : 0;
    const total = loaded ? replay.total_steps : 0;
    const busyOrStopped = stateStore.ui.busy || !stateStore.status.running;

    elements.replayStepRange.max = String(total || 0);
    elements.replayStepRange.value = String(current || 0);
    elements.replayStepRange.disabled = !loaded || busyOrStopped;
    elements.replayPrev5Button.disabled = !loaded || busyOrStopped || current <= 0;
    elements.replayPrevButton.disabled = !loaded || busyOrStopped || current <= 0;
    elements.replayNextButton.disabled = !loaded || busyOrStopped || current >= total;
    elements.replayNext5Button.disabled = !loaded || busyOrStopped || current >= total;
    elements.importRecordButton.disabled = busyOrStopped;
    elements.exportRecordButton.disabled = busyOrStopped;

    if (!loaded) {
        elements.replayStepLabel.textContent = "No record";
        return;
    }

    elements.replayStepLabel.textContent = `${current}/${total}`;
}

function createBoardCell(coord, token, tried, lastMove, boardKey, position = {}) {
    const cell = document.createElement("button");
    cell.className = "cell";
    cell.type = "button";
    cell.title = coord;
    cell.dataset.coord = coord;
    cell.tabIndex = canPlayOnBoard(boardKey) ? 0 : -1;
    cell.classList.toggle("is-playable", canPlayOnBoard(boardKey));

    if (token === "B" || token === "W") {
        cell.classList.add(token === "B" ? "black" : "white");
        const stone = document.createElement("span");
        stone.className = "stone";
        cell.appendChild(stone);
    }

    if (tried && token === ".") {
        cell.classList.add("tried");
        const marker = document.createElement("span");
        marker.className = "marker";
        cell.appendChild(marker);
    }

    if (lastMove && lastMove.move === coord) {
        cell.classList.add("last-move");
    }

    Object.entries(position).forEach(([className, enabled]) => {
        if (enabled) {
            cell.classList.add(className);
        }
    });

    cell.addEventListener("click", () => handleBoardClick(coord, boardKey));
    return cell;
}

function renderSquareBoard(host, boardState, boardSize, lastMove, boardKey) {
    // Build one CSS grid with labels on all four sides and cells in top-to-bottom order.
    const grid = document.createElement("div");
    grid.className = "board-grid square-board";
    grid.style.gridTemplateColumns = `1.5rem repeat(${boardSize}, minmax(0, 1fr)) 1.5rem`;

    const topLeft = document.createElement("div");
    topLeft.className = "axis-spacer";
    grid.appendChild(topLeft);

    for (let column = 0; column < boardSize; column += 1) {
        const label = document.createElement("div");
        label.className = "axis-label";
        label.textContent = fileLabel(column);
        grid.appendChild(label);
    }

    const topRight = document.createElement("div");
    topRight.className = "axis-spacer";
    grid.appendChild(topRight);

    for (let rowFromTop = 0; rowFromTop < boardSize; rowFromTop += 1) {
        const rowLabel = document.createElement("div");
        rowLabel.className = "axis-label";
        rowLabel.textContent = String(boardSize - rowFromTop);
        grid.appendChild(rowLabel);

        for (let column = 0; column < boardSize; column += 1) {
            const coord = coordFromTopRow(column, rowFromTop, boardSize);
            const token = boardState.rows[rowFromTop][column];
            const tried = boardState.tried_rows?.[rowFromTop]?.[column] || false;
            grid.appendChild(createBoardCell(coord, token, tried, lastMove, boardKey, {
                "first-row": rowFromTop === 0,
                "last-row": rowFromTop === boardSize - 1,
                "first-col": column === 0,
                "last-col": column === boardSize - 1,
            }));
        }

        const rightRowLabel = document.createElement("div");
        rightRowLabel.className = "axis-label";
        rightRowLabel.textContent = String(boardSize - rowFromTop);
        grid.appendChild(rightRowLabel);
    }

    const bottomLeft = document.createElement("div");
    bottomLeft.className = "axis-spacer";
    grid.appendChild(bottomLeft);

    for (let column = 0; column < boardSize; column += 1) {
        const label = document.createElement("div");
        label.className = "axis-label";
        label.textContent = fileLabel(column);
        grid.appendChild(label);
    }

    const bottomRight = document.createElement("div");
    bottomRight.className = "axis-spacer";
    grid.appendChild(bottomRight);

    host.appendChild(grid);
}

function createSvgElement(tag, attributes = {}) {
    const element = document.createElementNS(SVG_NS, tag);
    Object.entries(attributes).forEach(([key, value]) => {
        element.setAttribute(key, String(value));
    });
    return element;
}

function hexPoints(centerX, centerY, radius) {
    const halfWidth = Math.sqrt(3) * radius / 2;
    return [
        [centerX, centerY - radius],
        [centerX + halfWidth, centerY - radius / 2],
        [centerX + halfWidth, centerY + radius / 2],
        [centerX, centerY + radius],
        [centerX - halfWidth, centerY + radius / 2],
        [centerX - halfWidth, centerY - radius / 2],
    ];
}

function pointsAttribute(points) {
    return points.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" ");
}

function renderHexBoard(host, boardState, boardSize, lastMove, boardKey) {
    // DarkHex is rendered as SVG so the rhombus shape can scale without layout hacks.
    const board = document.createElement("div");
    board.className = "hex-board";
    const radius = 12;
    const halfWidth = Math.sqrt(3) * radius / 2;
    const horizontalStep = Math.sqrt(3) * radius;
    const verticalStep = 1.5 * radius;
    const labelPad = 18;
    const boardWidth = labelPad * 2 + horizontalStep * (boardSize + (boardSize - 1) / 2);
    const boardHeight = labelPad * 2 + radius * 2 + verticalStep * (boardSize - 1);
    const svg = createSvgElement("svg", {
        class: "hex-svg",
        viewBox: `0 0 ${boardWidth.toFixed(2)} ${boardHeight.toFixed(2)}`,
        role: "img",
        "aria-label": `${boardSize} by ${boardSize} DarkHex board`,
    });

    const centerX = (column, rowFromTop) => labelPad + halfWidth + horizontalStep * column + halfWidth * rowFromTop;
    const centerY = (rowFromTop) => labelPad + radius + verticalStep * rowFromTop;

    for (let column = 0; column < boardSize; column += 1) {
        const topLabel = createSvgElement("text", {
            class: "hex-axis-label is-blue",
            x: centerX(column, 0),
            y: labelPad * 0.55,
        });
        topLabel.textContent = fileLabel(column);
        svg.appendChild(topLabel);

        const bottomLabel = createSvgElement("text", {
            class: "hex-axis-label is-blue",
            x: centerX(column, boardSize - 1),
            y: boardHeight - labelPad * 0.25,
        });
        bottomLabel.textContent = fileLabel(column);
        svg.appendChild(bottomLabel);
    }

    for (let rowFromTop = 0; rowFromTop < boardSize; rowFromTop += 1) {
        const rowText = String(boardSize - rowFromTop);
        const leftLabel = createSvgElement("text", {
            class: "hex-axis-label is-red",
            x: centerX(0, rowFromTop) - halfWidth - 7,
            y: centerY(rowFromTop) + 2,
        });
        leftLabel.textContent = rowText;
        svg.appendChild(leftLabel);

        const rightLabel = createSvgElement("text", {
            class: "hex-axis-label is-red",
            x: centerX(boardSize - 1, rowFromTop) + halfWidth + 7,
            y: centerY(rowFromTop) + 2,
        });
        rightLabel.textContent = rowText;
        svg.appendChild(rightLabel);

        for (let column = 0; column < boardSize; column += 1) {
            const coord = coordFromTopRow(column, rowFromTop, boardSize);
            const token = boardState.rows[rowFromTop][column];
            const points = hexPoints(centerX(column, rowFromTop), centerY(rowFromTop), radius);
            const playable = canPlayOnBoard(boardKey);
            const cell = createSvgElement("polygon", {
                class: `hex-cell${playable ? " is-playable" : ""}${token === "B" ? " is-red" : ""}${token === "W" ? " is-blue" : ""}`,
                points: pointsAttribute(points),
                tabindex: playable ? 0 : -1,
                "data-coord": coord,
            });
            cell.addEventListener("click", () => handleBoardClick(coord, boardKey));
            cell.addEventListener("keydown", (event) => {
                if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    handleBoardClick(coord, boardKey);
                }
            });
            svg.appendChild(cell);

            if (lastMove && lastMove.move === coord) {
                svg.appendChild(createSvgElement("polygon", {
                    class: "hex-last-move",
                    points: pointsAttribute(points),
                }));
            }

            if (rowFromTop === 0) {
                svg.appendChild(createSvgElement("polyline", {
                    class: "hex-edge is-blue",
                    points: pointsAttribute([points[5], points[0], points[1]]),
                }));
            }
            if (rowFromTop === boardSize - 1) {
                svg.appendChild(createSvgElement("polyline", {
                    class: "hex-edge is-blue",
                    points: pointsAttribute([points[4], points[3], points[2]]),
                }));
            }
            if (column === 0) {
                svg.appendChild(createSvgElement("line", {
                    class: "hex-edge is-red",
                    x1: points[5][0],
                    y1: points[5][1],
                    x2: points[4][0],
                    y2: points[4][1],
                }));
            }
            if (column === boardSize - 1) {
                svg.appendChild(createSvgElement("line", {
                    class: "hex-edge is-red",
                    x1: points[1][0],
                    y1: points[1][1],
                    x2: points[2][0],
                    y2: points[2][1],
                }));
            }
        }
    }

    board.appendChild(svg);
    host.appendChild(board);
}

function visibleLastMove(boardKey, boardState, boardSize, lastMove) {
    // Imperfect views must not reveal a hidden opponent move marker.
    if (!lastMove || boardKey === "perfect") {
        return lastMove;
    }

    const point = pointFromCoord(lastMove.move, boardSize);
    if (!point) {
        return null;
    }

    const token = boardState.rows?.[point.rowFromTop]?.[point.column];
    return token === lastMove.player ? lastMove : null;
}

function renderBoard(host, boardState, boardSize, lastMove, boardType, boardKey) {
    host.innerHTML = "";
    host.classList.remove("is-message");
    if (!boardState) {
        host.textContent = "Connect to an engine.";
        host.classList.add("is-message");
        return;
    }

    const boardLastMove = visibleLastMove(boardKey, boardState, boardSize, lastMove);
    if (boardType === "hex") {
        renderHexBoard(host, boardState, boardSize, boardLastMove, boardKey);
    } else {
        renderSquareBoard(host, boardState, boardSize, boardLastMove, boardKey);
    }
}

function renderBoards() {
    const state = stateStore.state;

    Object.entries(visibilityBindings).forEach(([key, checkbox]) => {
        panelBindings[key].classList.toggle("is-collapsed", !checkbox.checked);
    });

    const problem = boardStateProblem();
    if (problem) {
        Object.values(boardBindings).forEach((host) => {
            host.textContent = problem;
            host.classList.add("is-message");
        });
        return;
    }

    const boardSize = state.board_size;
    renderBoard(boardBindings.perfect, state.boards.perfect, boardSize, state.last_move, state.board_type, "perfect");
    renderBoard(boardBindings.black_view, state.boards.black_view, boardSize, state.last_move, state.board_type, "black_view");
    renderBoard(boardBindings.white_view, state.boards.white_view, boardSize, state.last_move, state.board_type, "white_view");
}

function renderAll() {
    renderStatus();
    renderModeHint();
    renderLogs();
    renderBoards();
    renderReplayControls();
    renderHistory("history-observed", stateStore.state?.action_history, 24);
}

async function refreshStatus() {
    const payload = await api("/api/status");
    stateStore.status = payload.status;
}

async function refreshLogs() {
    const payload = await api("/api/logs");
    stateStore.logs = payload.logs;
}

async function refreshReplay() {
    const payload = await api("/api/replay");
    stateStore.replay = payload.replay;
}

async function refreshState() {
    if (!stateStore.status.running) {
        stateStore.state = null;
        stateStore.finalScore = null;
        stateStore.finalScoreKey = null;
        stateStore.replay = { loaded: false };
        return;
    }
    const payload = await api("/api/state");
    stateStore.state = payload.state;
    stateStore.status = payload.status;
}

async function refreshFinalScoreIfNeeded() {
    // final_score is only meaningful after terminal board_state is observed.
    if (!stateStore.status.running) {
        stateStore.finalScore = null;
        stateStore.finalScoreKey = null;
        return;
    }
    const key = finalScoreKey();
    if (!key) {
        stateStore.finalScore = null;
        stateStore.finalScoreKey = null;
        return;
    }
    if (stateStore.finalScore && stateStore.finalScoreKey === key) {
        return;
    }

    const payload = await api("/api/final_score");
    if (!payload.result?.success) {
        throw new Error(payload.result?.message || "final_score command failed.");
    }
    const parsed = parseFinalScore(payload.result.message);
    if (!parsed) {
        throw new Error(`Unexpected final_score response: ${payload.result.message}`);
    }
    stateStore.finalScore = parsed;
    stateStore.finalScoreKey = key;
    stateStore.status = payload.status || stateStore.status;
}

async function refreshAll() {
    if (stateStore.ui.busy) {
        return;
    }

    try {
        await refreshStatus();
        await Promise.all([refreshLogs(), refreshState(), refreshReplay()]);
        await refreshFinalScoreIfNeeded();
    } catch (error) {
        setMessage(error.message, false);
    } finally {
        renderAll();
    }
}

async function performMutation(path, body) {
    // Serialize engine mutations; console mode is single-stdin/single-stdout.
    if (stateStore.ui.busy) {
        return null;
    }
    cancelPendingAi();
    if (path === "/api/stop" || path === "/api/clear" || path === "/api/import_record") {
        stateStore.finalScore = null;
        stateStore.finalScoreKey = null;
    }
    stateStore.ui.busy = true;
    renderAll();

    try {
        const payload = await api(path, {
            method: "POST",
            body: JSON.stringify(body || {}),
        });
        stateStore.status = payload.status || stateStore.status;
        stateStore.state = payload.state ?? stateStore.state;
        stateStore.replay = payload.replay ?? stateStore.replay;
        if (path === "/api/stop") {
            stateStore.state = null;
            stateStore.replay = { loaded: false };
        }
        if (payload.result?.success === false) {
            if (path !== "/api/play") {
                stateStore.lastResult = { success: false, message: payload.result.message || "Command failed." };
            }
        } else {
            stateStore.lastResult = { success: true, message: mutationMessage(path, body || {}, payload.result) };
        }
        await refreshLogs();
        try {
            await refreshFinalScoreIfNeeded();
        } catch (error) {
            setMessage(error.message, false);
        }
        return payload;
    } catch (error) {
        setMessage(error.message, false);
        try {
            await refreshStatus();
            await refreshLogs();
            if (stateStore.status.running) {
                try {
                    await refreshState();
                    await refreshReplay();
                } catch (_) {
                    stateStore.state = null;
                }
            } else {
                stateStore.state = null;
                stateStore.replay = { loaded: false };
            }
        } catch (_) {
            stateStore.status = { ...stateStore.status, running: false, pid: null };
            stateStore.state = null;
            stateStore.replay = { loaded: false };
        }
        return null;
    } finally {
        stateStore.ui.busy = false;
        renderAll();
    }
}

function mutationMessage(path, body, result = null) {
    const turn = colorName(currentTurn());
    if (path === "/api/stop") {
        return "Disconnected.";
    }
    if (path === "/api/clear") {
        return `Board cleared. ${turn} to play.`;
    }
    if (path === "/api/play") {
        const verb = body.move === "PASS" ? "passed" : "played";
        return `${colorName(body.player)} ${verb}. ${turn} to play.`;
    }
    if (path === "/api/genmove") {
        if (result?.message === "PASS") {
            return `${colorName(body.player)} passed. ${turn} to play.`;
        }
        if (result?.message === "Resign") {
            return `${colorName(body.player)} resigned.`;
        }
        return `${colorName(body.player)} moved. ${turn} to play.`;
    }
    if (path === "/api/load_model") {
        return "Model loaded.";
    }
    if (path === "/api/import_record") {
        return "Record loaded.";
    }
    if (path === "/api/replay_step") {
        return "Replay updated.";
    }
    if (path === "/api/command") {
        return "Command sent.";
    }
    return "OK.";
}

async function maybeAiRespond() {
    // Guard with aiSequence so stale delayed replies cannot play after user changes state.
    if (stateStore.ui.mode !== "human-ai" || !stateStore.status.running || stateStore.state?.is_terminal) {
        return;
    }
    if (currentTurn() !== aiColor()) {
        return;
    }
    const sequence = stateStore.aiSequence;
    await sleep(220);
    if (
        sequence !== stateStore.aiSequence ||
        stateStore.ui.busy ||
        stateStore.ui.mode !== "human-ai" ||
        !stateStore.status.running ||
        stateStore.state?.is_terminal ||
        currentTurn() !== aiColor()
    ) {
        return;
    }
    await performMutation("/api/genmove", { player: aiColor() });
}

function stopAutoLoop() {
    stateStore.ui.autoRunning = false;
    if (stateStore.autoTimer) {
        window.clearTimeout(stateStore.autoTimer);
        stateStore.autoTimer = null;
    }
    renderAll();
}

function scheduleAutoLoop(delay = Number(elements.delaySelect.value)) {
    if (!stateStore.ui.autoRunning) {
        return;
    }
    stateStore.autoTimer = window.setTimeout(runAutoStep, delay);
}

async function runAutoStep() {
    // AI Auto advances exactly one engine move per timer tick.
    if (!stateStore.ui.autoRunning || stateStore.ui.mode !== "ai-auto") {
        stopAutoLoop();
        return;
    }
    if (!stateStore.status.running || stateStore.state?.is_terminal) {
        stopAutoLoop();
        return;
    }

    const payload = await performMutation("/api/genmove", { player: currentTurn() });
    if (!payload || stateStore.state?.is_terminal) {
        stopAutoLoop();
        return;
    }
    scheduleAutoLoop();
}

async function goToReplayStep(step) {
    const total = stateStore.replay?.total_steps || 0;
    const requested = Number(step);
    const fallback = stateStore.replay?.current_step || 0;
    const target = Number.isFinite(requested) ? Math.max(0, Math.min(requested, total)) : fallback;
    return performMutation("/api/replay_step", { step: target });
}

async function importRecord() {
    const record = elements.recordText.value.trim();
    if (!record) {
        setMessage("Paste a MiniZero record first.", false);
        renderAll();
        return;
    }
    stopAutoLoop();
    const payload = await performMutation("/api/import_record", { record });
    if (payload?.replay?.loaded) {
        setMessage(`Record loaded. ${payload.replay.total_steps} moves.`);
        renderAll();
    }
}

async function exportRecord() {
    if (!stateStore.status.running || stateStore.ui.busy) {
        setMessage("Engine is not connected.", false);
        renderAll();
        return;
    }
    try {
        const payload = await api("/api/export_record");
        if (!payload.result?.success) {
            throw new Error(payload.result?.message || "Export failed.");
        }
        elements.recordText.value = payload.result.message;
        stateStore.status = payload.status || stateStore.status;
        setMessage("Record exported.");
    } catch (error) {
        setMessage(error.message, false);
    } finally {
        renderAll();
    }
}

async function handleBoardClick(coord, boardKey) {
    // Board clicks are accepted only when the selected test mode allows human input.
    if (!stateStore.status.running) {
        setMessage("Engine is not connected.", false);
        renderAll();
        return;
    }
    if (stateStore.ui.busy) {
        return;
    }
    if (stateStore.state?.is_terminal) {
        setMessage("Game over.", false);
        renderAll();
        return;
    }
    if (stateStore.ui.mode === "ai-auto") {
        setMessage("Pause AI Auto before manual play.", false);
        renderAll();
        return;
    }
    if (!canManualPlay()) {
        setMessage("Waiting for AI.", false);
        renderAll();
        return;
    }
    if (!canPlayOnBoard(boardKey)) {
        renderAll();
        return;
    }

    const payload = await performMutation("/api/play", { player: playColor(), move: coord });
    if (payload?.result?.success) {
        await maybeAiRespond();
    }
}

function bindEvents() {
    // Event handlers delegate all engine-changing work through performMutation().
    document.getElementById("stop-button").addEventListener("click", async () => {
        stopAutoLoop();
        await performMutation("/api/stop", {});
    });

    document.getElementById("clear-button").addEventListener("click", async () => {
        stopAutoLoop();
        const payload = await performMutation("/api/clear", {});
        if (payload) {
            await maybeAiRespond();
        }
    });

    document.getElementById("pass-button").addEventListener("click", async () => {
        if (!canManualPlay()) {
            setMessage(stateStore.state?.is_terminal ? "Game over." : "Waiting for AI.", false);
            renderAll();
            return;
        }
        const payload = await performMutation("/api/play", { player: playColor(), move: "PASS" });
        if (payload?.result?.success) {
            await maybeAiRespond();
        }
    });

    document.getElementById("genmove-auto-button").addEventListener("click", async () => {
        if (!canUseGenmove()) {
            setMessage(stateStore.state?.is_terminal ? "Game over." : "Not AI's turn.", false);
            renderAll();
            return;
        }
        await performMutation("/api/genmove", { player: currentTurn() });
    });

    elements.autoToggleButton.addEventListener("click", () => {
        if (stateStore.ui.autoRunning) {
            stopAutoLoop();
            return;
        }
        stateStore.ui.autoRunning = true;
        renderAll();
        scheduleAutoLoop(0);
    });

    document.getElementById("load-model-button").addEventListener("click", async () => {
        const path = document.getElementById("model-input").value.trim();
        await performMutation("/api/load_model", { path });
    });

    document.getElementById("send-command-button").addEventListener("click", async () => {
        const command = document.getElementById("command-raw-input").value.trim();
        await performMutation("/api/command", { command });
    });

    elements.importRecordButton.addEventListener("click", importRecord);
    elements.exportRecordButton.addEventListener("click", exportRecord);

    elements.replayPrev5Button.addEventListener("click", async () => {
        await goToReplayStep((stateStore.replay?.current_step || 0) - 5);
    });

    elements.replayPrevButton.addEventListener("click", async () => {
        await goToReplayStep((stateStore.replay?.current_step || 0) - 1);
    });

    elements.replayNextButton.addEventListener("click", async () => {
        await goToReplayStep((stateStore.replay?.current_step || 0) + 1);
    });

    elements.replayNext5Button.addEventListener("click", async () => {
        await goToReplayStep((stateStore.replay?.current_step || 0) + 5);
    });

    elements.replayStepRange.addEventListener("input", () => {
        elements.replayStepLabel.textContent = `${elements.replayStepRange.value}/${stateStore.replay?.total_steps || 0}`;
    });

    elements.replayStepRange.addEventListener("change", async () => {
        await goToReplayStep(Number(elements.replayStepRange.value));
    });

    elements.modeSelect.addEventListener("change", () => {
        stopAutoLoop();
        cancelPendingAi();
        stateStore.lastResult = null;
        stateStore.ui.mode = elements.modeSelect.value;
        renderAll();
    });

    elements.humanColorSelect.addEventListener("change", async () => {
        cancelPendingAi();
        stateStore.lastResult = null;
        stateStore.ui.humanColor = elements.humanColorSelect.value;
        renderAll();
        await maybeAiRespond();
    });

    Object.values(visibilityBindings).forEach((checkbox) => {
        checkbox.addEventListener("change", renderBoards);
    });
}

bindEvents();
refreshAll();
window.setInterval(refreshAll, 2500);
