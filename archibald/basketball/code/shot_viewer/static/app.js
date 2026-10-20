const TEAMS = {
  ATL: "#E03A3E", BOS: "#007A33", BKN: "#000000", CHA: "#1D1160", CHI: "#CE1141",
  CLE: "#860038", DAL: "#00538C", DEN: "#0E2240", DET: "#C8102E", GSW: "#1D428A",
  HOU: "#CE1141", IND: "#002D62", LAC: "#C8102E", LAL: "#552583", MEM: "#5D76A9",
  MIA: "#98002E", MIL: "#00471B", MIN: "#0C2340", NOP: "#0C2340", NYK: "#006BB6",
  OKC: "#007AC1", ORL: "#0077C0", PHI: "#006BB6", PHX: "#1D1160", POR: "#E03A3E",
  SAC: "#5A2D81", SAS: "#C4CED4", TOR: "#CE1141", UTA: "#002B5C", WAS: "#002B5C",
};

const court = document.querySelector("#court");
const tag = document.querySelector("#tag");
const loading = document.querySelector("#loading");
const gameSelect = document.querySelector("#game");

const state = {
  games: [],
  game: null,
  selected: null,
  playing: false,
  speed: 1,
  t: 0,
  lastTick: 0,
  request: null,
};

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

function teamColor(abbr) {
  return TEAMS[abbr] || "#8d8376";
}

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${url}`);
  return response.json();
}

function option(value, text) {
  const item = document.createElement("option");
  item.value = value;
  item.textContent = text;
  return item;
}

function periodLabel(period) {
  if (period == null) return "";
  if (period <= 4) return `Q${period}`;
  return `OT${period - 4}`;
}

function currentAction() {
  const game = state.game;
  if (!game || !game.actions.length) return null;
  const index = Math.max(0, Math.min(game.actions.length - 1, Math.round(state.t)));
  return game.actions[index];
}

function toCourt(x, y) {
  return { x: 25 + x, y: 41.75 - y };
}

function drawCourt() {
  const line = `stroke="#f4efe4" stroke-width="0.12" fill="none"`;
  const parts = [];
  parts.push(`<rect x="0" y="0" width="50" height="47" fill="#c9864a"/>`);
  for (let i = 0; i < 50; i += 2) {
    parts.push(`<rect x="${i}" y="0" width="2" height="47" fill="${i % 4 === 0 ? "#c9864a" : "#c17d40"}"/>`);
  }
  parts.push(`<rect x="0" y="0" width="50" height="47" ${line}/>`);
  parts.push(`<line x1="0" y1="0" x2="50" y2="0" ${line}/>`);
  parts.push(`<rect x="17" y="28" width="16" height="19" fill="rgba(215,161,90,0.45)" stroke="#f4efe4" stroke-width="0.12"/>`);
  parts.push(`<line x1="19" y1="47" x2="19" y2="28" stroke="#f4efe4" stroke-width="0.08"/>`);
  parts.push(`<line x1="31" y1="47" x2="31" y2="28" stroke="#f4efe4" stroke-width="0.08"/>`);
  parts.push(`<path d="M 19 28 A 6 6 0 0 1 31 28" ${line}/>`);
  parts.push(`<path d="M 19 28 A 6 6 0 0 0 31 28" stroke="#f4efe4" stroke-width="0.1" fill="none" stroke-dasharray="0.35 0.28"/>`);
  parts.push(`<path d="M 21.2 41.2 A 4 4 0 0 1 28.8 41.2" ${line}/>`);
  parts.push(`<line x1="22" y1="43" x2="28" y2="43" stroke="#f4efe4" stroke-width="0.18"/>`);
  parts.push(`<circle cx="25" cy="41.75" r="0.75" ${line}/>`);
  parts.push(`<line x1="3" y1="47" x2="3" y2="32.8" ${line}/>`);
  parts.push(`<line x1="47" y1="47" x2="47" y2="32.8" ${line}/>`);
  parts.push(`<path d="M 3 32.8 A 23.75 23.75 0 0 1 47 32.8" ${line}/>`);
  parts.push(`<circle cx="25" cy="0" r="6" ${line}/>`);
  parts.push(`<g id="shots"></g>`);
  court.innerHTML = parts.join("");
}

function renderShots() {
  const game = state.game;
  const layer = court.querySelector("#shots");
  if (!game || !layer) return;
  const index = Math.round(state.t);
  const html = [];
  game.actions.forEach((action, i) => {
    if (!action.shot || action.x == null) return;
    if (state.selected) {
      if (action.playerId !== state.selected) return;
    } else if (i > index) {
      return;
    }
    const point = toCourt(action.x, action.y);
    const made = action.result === "Made";
    const classes = [
      "shot",
      made ? "made" : "missed",
      i === index ? "current" : "",
    ].filter(Boolean).join(" ");
    const fill = made ? teamColor(action.team) : "none";
    const stroke = made ? "rgba(0,0,0,0.35)" : teamColor(action.team);
    html.push(`<g class="${classes}" data-id="${esc(action.playerId)}" data-index="${i}">`);
    html.push(`<circle cx="${point.x}" cy="${point.y}" r="1.35" fill="transparent"/>`);
    html.push(`<circle cx="${point.x}" cy="${point.y}" r="${i === index ? 0.85 : 0.48}" fill="${fill}" stroke="${stroke}" stroke-width="${made ? 0.08 : 0.16}" pointer-events="none"/>`);
    html.push(`</g>`);
  });
  layer.innerHTML = html.join("");
  layer.querySelectorAll(".shot").forEach((dot) => {
    dot.addEventListener("click", (event) => {
      event.stopPropagation();
      state.selected = dot.dataset.id || null;
      state.t = Number(dot.dataset.index);
      state.playing = false;
      document.querySelector("#toggle").textContent = "▶";
      render();
    });
  });
  placeTag();
}

function placeTag() {
  const action = currentAction();
  if (!action || !action.shot || action.x == null || (state.selected && action.playerId !== state.selected)) {
    tag.hidden = true;
    return;
  }
  const point = toCourt(action.x, action.y);
  const svgPoint = court.createSVGPoint();
  svgPoint.x = point.x;
  svgPoint.y = point.y;
  const screen = svgPoint.matrixTransform(court.getScreenCTM());
  const box = document.querySelector("#field-wrap").getBoundingClientRect();
  tag.hidden = false;
  tag.textContent = action.player || action.team;
  tag.style.left = `${screen.x - box.left}px`;
  tag.style.top = `${screen.y - box.top - 8}px`;
}

function playerById(id) {
  return state.game && state.game.players.find((player) => player.id === id);
}

function renderRail() {
  const game = state.game;
  const action = currentAction();
  if (!game || !action) return;
  const homeScore = action.scoreHome || "0";
  const awayScore = action.scoreAway || "0";
  document.querySelector("#score").textContent = `${game.away} ${awayScore}  ·  ${game.home} ${homeScore}`;
  document.querySelector("#description").textContent = action.description;
  const chips = [
    periodLabel(action.period),
    action.clock,
    action.team,
    action.action,
    action.shot && action.distance != null ? `${action.distance} ft` : null,
    action.result,
  ].filter(Boolean);
  document.querySelector("#chips").innerHTML = chips.map((chip) => `<span class="chip">${esc(chip)}</span>`).join("");
  const phase = document.querySelector("#phase");
  phase.innerHTML = `<strong>${esc(periodLabel(action.period))} ${esc(action.clock)}</strong>`;
  const scrub = document.querySelector("#scrub");
  if (document.activeElement !== scrub) scrub.value = String(Math.round(state.t));

  const focus = playerById(state.selected) || playerById(action.playerId);
  const card = document.querySelector("#card");
  if (!focus) {
    card.innerHTML = `<p class="empty">Click a shot</p>`;
  } else {
    const line = focus.fga == null ? "—" : `${focus.fgm}-${focus.fga}`;
    const threes = focus.tpa == null ? "—" : `${focus.tpm}-${focus.tpa}`;
    card.innerHTML = `
      <h2>${esc(focus.name)}</h2>
      <div class="role">${esc(focus.team)} · ${esc(focus.position || "—")} · game line</div>
      <div class="stats">
        <div class="stat"><b>${focus.points ?? "—"}</b><span>Points</span></div>
        <div class="stat"><b>${esc(line)}</b><span>Field goals</span></div>
        <div class="stat"><b>${esc(threes)}</b><span>Threes</span></div>
        <div class="stat"><b>${focus.reb ?? "—"} / ${focus.ast ?? "—"}</b><span>Reb / Ast</span></div>
      </div>`;
  }

  const roster = document.querySelector("#roster");
  roster.innerHTML = game.players.map((player) => `
    <li><button type="button" data-id="${esc(player.id)}" class="${player.id === state.selected ? "active" : ""}">
      <span class="who">${esc(player.name)} <small>${esc(player.team)} ${esc(player.position)}</small></span>
      <b>${player.points ?? ""}</b>
    </button></li>`).join("");
  roster.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.selected = state.selected === button.dataset.id ? null : button.dataset.id;
      state.playing = false;
      document.querySelector("#toggle").textContent = "▶";
      render();
    });
  });
}

function render() {
  renderShots();
  renderRail();
}

function carryScore(actions) {
  let home = "0";
  let away = "0";
  actions.forEach((action) => {
    if (action.scoreHome) home = action.scoreHome;
    else action.scoreHome = home;
    if (action.scoreAway) away = action.scoreAway;
    else action.scoreAway = away;
  });
}

function showGame(game) {
  carryScore(game.actions);
  state.game = game;
  state.t = 0;
  state.selected = null;
  state.playing = false;
  document.querySelector("#toggle").textContent = "▶";
  const scrub = document.querySelector("#scrub");
  const last = Math.max(0, game.actions.length - 1);
  scrub.max = String(last);
  scrub.value = "0";
  tag.hidden = true;
  drawCourt();
  render();
}

async function loadGame(id) {
  if (state.request) state.request.abort();
  state.request = new AbortController();
  loading.classList.add("show");
  try {
    const response = await fetch(`/api/game?id=${id}`, { signal: state.request.signal });
    if (!response.ok) throw new Error("missing game");
    showGame(await response.json());
  } catch (error) {
    if (error.name !== "AbortError") {
      document.querySelector("#description").textContent = "Could not load that game.";
    }
  } finally {
    loading.classList.remove("show");
  }
}

function seek(frame) {
  if (!state.game) return;
  const last = state.game.actions.length - 1;
  state.t = Math.max(0, Math.min(last, frame));
  render();
}

function setPlaying(playing) {
  state.playing = playing && state.game && state.game.actions.length > 1;
  const toggle = document.querySelector("#toggle");
  toggle.textContent = state.playing ? "❚❚" : "▶";
  toggle.setAttribute("aria-label", state.playing ? "Pause" : "Play");
  if (state.playing) {
    if (state.t >= state.game.actions.length - 1) state.t = 0;
    state.lastTick = performance.now();
    requestAnimationFrame(tick);
  }
}

function tick(now) {
  if (!state.playing || !state.game) return;
  const dt = now - state.lastTick;
  state.lastTick = now;
  state.t += (dt / 450) * state.speed;
  if (state.t >= state.game.actions.length - 1) {
    state.t = state.game.actions.length - 1;
    setPlaying(false);
  }
  render();
  if (state.playing) requestAnimationFrame(tick);
}

court.addEventListener("click", () => {
  state.selected = null;
  render();
});
document.querySelector("#toggle").addEventListener("click", () => setPlaying(!state.playing));
document.querySelector("#back").addEventListener("click", () => { setPlaying(false); seek(Math.round(state.t) - 1); });
document.querySelector("#forward").addEventListener("click", () => { setPlaying(false); seek(Math.round(state.t) + 1); });
document.querySelector("#scrub").addEventListener("input", (event) => {
  setPlaying(false);
  seek(Number(event.target.value));
});
document.querySelectorAll(".speed-btn").forEach((button) => {
  button.addEventListener("click", () => {
    state.speed = Number(button.dataset.speed);
    document.querySelectorAll(".speed-btn").forEach((item) => item.classList.toggle("active", item === button));
  });
});
gameSelect.addEventListener("change", () => loadGame(gameSelect.value));
document.querySelector("#surprise").addEventListener("click", () => {
  if (!state.games.length) return;
  const pick = state.games[Math.floor(Math.random() * state.games.length)];
  gameSelect.value = pick.id;
  loadGame(pick.id);
});
window.addEventListener("keydown", (event) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return;
  if (event.code === "Space") {
    event.preventDefault();
    setPlaying(!state.playing);
  } else if (event.code === "ArrowRight") {
    setPlaying(false);
    seek(Math.round(state.t) + 1);
  } else if (event.code === "ArrowLeft") {
    setPlaying(false);
    seek(Math.round(state.t) - 1);
  }
});
window.addEventListener("resize", placeTag);

getJSON("/api/games").then((games) => {
  state.games = games;
  games.forEach((game) => {
    gameSelect.appendChild(option(game.id, `${game.date} · ${game.away} at ${game.home} · ${game.shots} shots`));
  });
  if (games[0]) return loadGame(games[0].id);
  document.querySelector("#description").textContent = "No games have finished downloading yet.";
  return null;
});
