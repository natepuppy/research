const TEAMS = {
  ARI: "#97233F", ATL: "#A71930", BAL: "#241773", BUF: "#00338D", CAR: "#0085CA",
  CHI: "#0B162A", CIN: "#FB4F14", CLE: "#311D00", DAL: "#003594", DEN: "#FB4F14",
  DET: "#0076B6", GB: "#203731", HOU: "#03202F", IND: "#002C5F", JAX: "#006778",
  KC: "#E31837", LA: "#003594", LAC: "#0080C6", LV: "#000000", MIA: "#008E97",
  MIN: "#4F2683", NE: "#002244", NO: "#D3BC8D", NYG: "#0B2265", NYJ: "#125740",
  PHI: "#004C54", PIT: "#FFB612", SEA: "#002244", SF: "#AA0000", TB: "#D50A0A",
  TEN: "#0C2340", WAS: "#5A1414",
};

const SHORT = { ILB: "LB", MLB: "LB", OLB: "LB", FB: "FB", NT: "NT" };

const field = document.querySelector("#field");
const tag = document.querySelector("#tag");
const loading = document.querySelector("#loading");
const weekSelect = document.querySelector("#week");
const gameSelect = document.querySelector("#game");
const playSelect = document.querySelector("#play");

const state = {
  mode: "plays",
  index: null,
  combine: null,
  plays: [],
  play: null,
  selected: null,
  align: true,
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
  return TEAMS[abbr] || "#8d9386";
}

function ink(hex) {
  const n = parseInt(hex.slice(1), 16);
  const r = (n >> 16) & 255;
  const g = (n >> 8) & 255;
  const b = n & 255;
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.64 ? "#1a1a1a" : "#fff";
}

function ordinal(n) {
  if (n === 1) return "1st";
  if (n === 2) return "2nd";
  if (n === 3) return "3rd";
  return `${n}th`;
}

function resultLabel(code) {
  return { C: "Complete", I: "Incomplete", IN: "Intercepted" }[code] || code || "";
}

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${url}`);
  return response.json();
}

function mapPoint(x, y) {
  if (state.align && state.play && state.play.direction === "left") {
    return { x: 120 - x, y: 53.3 - y };
  }
  return { x, y };
}

function mapAngle(deg) {
  if (deg == null) return 0;
  if (state.align && state.play && state.play.direction === "left") return (deg + 180) % 360;
  return deg;
}

function toSvg(x, y) {
  const p = mapPoint(x, y);
  return { x: p.x, y: 53.3 - p.y };
}

function heading(deg) {
  const r = (mapAngle(deg) * Math.PI) / 180;
  return { x: Math.sin(r), y: -Math.cos(r) };
}

function totalFrames() {
  if (!state.play) return 1;
  return Math.max(1, state.play.inputFrames + state.play.outputFrames);
}

function buildTrack(player) {
  const play = state.play;
  const track = player.input.map((frame) => ({
    t: frame.f - 1,
    x: frame.x,
    y: frame.y,
    s: frame.s,
    a: frame.a,
    o: frame.o,
    tracked: true,
  }));
  const last = track[track.length - 1];
  player.output.forEach((frame) => {
    const prev = track[track.length - 1];
    const dx = frame.x - prev.x;
    const dy = frame.y - prev.y;
    track.push({
      t: play.inputFrames + frame.f - 1,
      x: frame.x,
      y: frame.y,
      s: Math.hypot(dx, dy) * 10,
      a: null,
      o: last ? last.o : null,
      tracked: true,
    });
  });
  if (!player.output.length && last) {
    track.push({ ...last, t: totalFrames() - 1, s: 0, a: 0, tracked: false });
  }
  player.track = track;
}

function sample(player, t) {
  const track = player.track;
  if (!track.length) return null;
  if (t <= track[0].t) return { ...track[0] };
  for (let i = 1; i < track.length; i += 1) {
    const b = track[i];
    const a = track[i - 1];
    if (t <= b.t) {
      const span = b.t - a.t || 1;
      const u = (t - a.t) / span;
      return {
        x: a.x + (b.x - a.x) * u,
        y: a.y + (b.y - a.y) * u,
        s: a.s,
        a: a.a,
        o: a.o,
        tracked: t < state.play.inputFrames ? true : b.tracked && a.tracked,
      };
    }
  }
  return { ...track[track.length - 1] };
}

function fitField(play) {
  const wrap = document.querySelector("#field-wrap");
  if (play && play.kind === "combine" && play.bounds) {
    const pad = 4;
    const [minX, minY, maxX, maxY] = play.bounds;
    const x0 = minX - pad;
    const yTop = maxY + pad;
    const width = Math.max(12, maxX - minX + pad * 2);
    const height = Math.max(10, maxY - minY + pad * 2);
    field.setAttribute("viewBox", `${x0} ${53.3 - yTop} ${width} ${height}`);
    const ratio = width / height;
    wrap.style.aspectRatio = String(ratio);
    wrap.style.width = `min(100%, calc((100vh - 210px) * ${ratio}))`;
    return;
  }
  field.setAttribute("viewBox", "-1 -3 122 59.3");
  wrap.style.aspectRatio = "2.06";
  wrap.style.width = "min(100%, calc((100vh - 210px) * 2.06))";
}

function drawField() {
  const play = state.play;
  const svg = [];
  svg.push(`<rect x="0" y="0" width="120" height="53.3" fill="#1e7a3a"/>`);
  for (let x = 0; x < 120; x += 5) {
    const shade = (x / 5) % 2 === 0 ? "#1e7a3a" : "#197233";
    svg.push(`<rect x="${x}" y="0" width="5" height="53.3" fill="${shade}"/>`);
  }
  svg.push(`<rect x="0" y="0" width="10" height="53.3" fill="#14562a"/>`);
  svg.push(`<rect x="110" y="0" width="10" height="53.3" fill="#14562a"/>`);
  for (let x = 10; x <= 110; x += 5) {
    const major = x % 10 === 0;
    svg.push(`<line x1="${x}" y1="0" x2="${x}" y2="53.3" stroke="rgba(255,255,255,${major ? 0.85 : 0.35})" stroke-width="${major ? 0.12 : 0.06}"/>`);
  }
  svg.push(`<line x1="0" y1="0" x2="120" y2="0" stroke="white" stroke-width="0.18"/>`);
  svg.push(`<line x1="0" y1="53.3" x2="120" y2="53.3" stroke="white" stroke-width="0.18"/>`);
  for (let x = 11; x < 110; x += 1) {
    if (x % 5 === 0) continue;
    svg.push(`<line x1="${x}" y1="23.2" x2="${x}" y2="23.9" stroke="rgba(255,255,255,0.7)" stroke-width="0.08"/>`);
    svg.push(`<line x1="${x}" y1="29.4" x2="${x}" y2="30.1" stroke="rgba(255,255,255,0.7)" stroke-width="0.08"/>`);
  }
  for (let x = 20; x <= 100; x += 10) {
    const fromLeft = x - 10;
    const yard = fromLeft <= 50 ? fromLeft : 100 - fromLeft;
    const label = `<text x="${x}" y="10" fill="rgba(255,255,255,0.8)" font-size="3.2" text-anchor="middle" transform="rotate(180 ${x} 10)">${yard}</text>
      <text x="${x}" y="46" fill="rgba(255,255,255,0.8)" font-size="3.2" text-anchor="middle">${yard}</text>`;
    svg.push(label);
  }

  if (play && play.kind !== "combine" && play.yardline != null) {
    const los = toSvg(play.yardline, 0).x;
    svg.push(`<line x1="${los}" y1="0" x2="${los}" y2="53.3" stroke="#ffe14a" stroke-width="0.22"/>`);
    if (play.distance != null) {
      const raw = play.direction === "right" ? play.yardline + play.distance : play.yardline - play.distance;
      const fd = toSvg(Math.max(10, Math.min(110, raw)), 0).x;
      svg.push(`<line x1="${fd}" y1="0" x2="${fd}" y2="53.3" stroke="#7ec8ff" stroke-width="0.16" stroke-dasharray="0.8 0.55"/>`);
    }
    const land = toSvg(play.ballLand[0], play.ballLand[1]);
    svg.push(`<g id="landing" transform="translate(${land.x} ${land.y})">
      <line x1="-0.7" y1="-0.7" x2="0.7" y2="0.7" stroke="#fff" stroke-width="0.16"/>
      <line x1="-0.7" y1="0.7" x2="0.7" y2="-0.7" stroke="#fff" stroke-width="0.16"/>
    </g>`);
    const passer = play.players.find((player) => player.role === "Passer");
    if (passer) {
      const release = toSvg(passer.input[passer.input.length - 1].x, passer.input[passer.input.length - 1].y);
      svg.push(`<line id="flight" x1="${release.x}" y1="${release.y}" x2="${land.x}" y2="${land.y}" stroke="rgba(255,255,255,0.45)" stroke-width="0.1" stroke-dasharray="0.45 0.35"/>`);
    }
  }
  svg.push(`<g id="actors"></g>`);
  svg.push(`<g id="ball"></g>`);
  field.innerHTML = svg.join("");
}

function renderActors() {
  const play = state.play;
  if (!play) return;
  const actors = field.querySelector("#actors");
  const ball = field.querySelector("#ball");
  actors.innerHTML = "";
  const order = [...play.players].sort((a, b) => Number(a.role === "Targeted Receiver") - Number(b.role === "Targeted Receiver"));
  order.forEach((player) => {
    const spot = sample(player, state.t);
    if (!spot) return;
    const p = toSvg(spot.x, spot.y);
    const color = play.kind === "combine" ? "#f0c84b" : teamColor(player.side === "Offense" ? play.offense : play.defense);
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", [
      "player",
      player.role === "Targeted Receiver" ? "targeted" : "",
      player.id === state.selected ? "selected" : "",
      spot.tracked ? "" : "faded",
    ].filter(Boolean).join(" "));
    const viewWidth = field.viewBox.baseVal.width || 122;
    const px = field.getBoundingClientRect().width / viewWidth;
    const scale = Math.min(1.9, Math.max(1, 13 / (1.35 * px || 13)));
    g.setAttribute("transform", `translate(${p.x} ${p.y}) scale(${scale})`);
    g.dataset.id = player.id;
    const nose = heading(spot.o);
    g.innerHTML = `
      <line x1="0" y1="0" x2="${nose.x * 2.1}" y2="${nose.y * 2.1}" stroke="${ink(color)}" stroke-width="0.18" stroke-linecap="round"/>
      <circle class="ring" r="1.7"/>
      <circle class="body" r="1.35" fill="${color}"/>
      <text fill="${ink(color)}" font-size="1.15">${esc(SHORT[player.position] || player.position)}</text>`;
    g.addEventListener("click", (event) => {
      event.stopPropagation();
      state.selected = player.id;
      renderActors();
      renderCard();
    });
    actors.appendChild(g);
  });

  const passer = play.players.find((player) => player.role === "Passer");
  let ballPoint = null;
  if (play.kind === "combine" && play.ballFrames && play.ballFrames.length) {
    const ball = sample({ track: play.ballFrames.map((frame) => ({ t: frame.f - 1, x: frame.x, y: frame.y, tracked: true })) }, state.t);
    ballPoint = ball ? toSvg(ball.x, ball.y) : null;
  } else if (state.t < play.inputFrames - 1 && passer) {
    const spot = sample(passer, state.t);
    ballPoint = toSvg(spot.x, spot.y);
  } else if (passer) {
    const start = passer.input[passer.input.length - 1];
    const span = Math.max(play.outputFrames, 1);
    const u = Math.min(1, Math.max(0, (state.t - (play.inputFrames - 1)) / span));
    const x = start.x + (play.ballLand[0] - start.x) * u;
    const y = start.y + (play.ballLand[1] - start.y) * u;
    ballPoint = toSvg(x, y);
  }
  ball.innerHTML = ballPoint
    ? `<circle cx="${ballPoint.x}" cy="${ballPoint.y}" r="0.55" fill="#f6f1e4" stroke="#6b4a2b" stroke-width="0.12"/>`
    : "";
  placeTag();
  renderPhase();
  if (state.selected) renderCard();
}

function placeTag() {
  const play = state.play;
  const player = play && play.players.find((item) => item.id === state.selected);
  if (!player) {
    tag.hidden = true;
    return;
  }
  const spot = sample(player, state.t);
  const p = toSvg(spot.x, spot.y);
  const point = field.createSVGPoint();
  point.x = p.x;
  point.y = p.y;
  const screen = point.matrixTransform(field.getScreenCTM());
  const box = document.querySelector("#field-wrap").getBoundingClientRect();
  tag.hidden = false;
  tag.textContent = player.name;
  tag.style.left = `${screen.x - box.left}px`;
  tag.style.top = `${screen.y - box.top - 8}px`;
}

function renderPhase() {
  const play = state.play;
  const phase = document.querySelector("#phase");
  const scrub = document.querySelector("#scrub");
  if (!play) return;
  const air = state.t >= play.inputFrames - 1 && play.outputFrames > 0;
  const seconds = (state.t / 10).toFixed(1);
  const label = play.kind === "combine" ? "Drill" : (air ? "Ball in the air" : "Before the throw");
  phase.innerHTML = `<strong>${label}</strong> ${seconds}s`;
  if (document.activeElement !== scrub) scrub.value = String(Math.round(state.t));
}

function renderCard() {
  const card = document.querySelector("#card");
  const play = state.play;
  const player = play && play.players.find((item) => item.id === state.selected);
  if (!player) {
    card.innerHTML = `<p class="empty">Click a player</p>`;
    return;
  }
  const spot = sample(player, state.t);
  const knownSpeed = spot.tracked && spot.s != null;
  const speed = knownSpeed ? `${spot.s.toFixed(1)} yd/s` : "—";
  const mph = knownSpeed ? `${(spot.s * 2.045).toFixed(1)} mph` : "Speed";
  const accel = spot.tracked && spot.a != null ? `${spot.a.toFixed(1)} yd/s²` : "—";
  card.innerHTML = `
    <h2>${esc(player.name)}</h2>
    <div class="role">${play.kind === "combine"
      ? `${esc(player.position)} · ${esc(play.college || "Combine")}`
      : `${esc(player.position)} · ${esc(player.side)} · ${esc(player.role)}`}</div>
    <div class="stats">
      <div class="stat"><b>${esc(player.height || "—")}</b><span>Height</span></div>
      <div class="stat"><b>${player.weight ? `${esc(player.weight)} lb` : "—"}</b><span>Weight</span></div>
      <div class="stat"><b>${esc(speed)}</b><span>${esc(mph)}</span></div>
      <div class="stat"><b>${esc(accel)}</b><span>Acceleration</span></div>
    </div>
    <p class="note">${play.kind === "combine" ? esc(play.drill) : (player.predict ? "Tracked after the throw." : "Not tracked after the throw.")} ${play.kind === "combine" ? "" : `nfl id ${esc(player.id)}`}</p>`;
}

function renderSummary() {
  const play = state.play;
  if (play.kind === "combine") {
    document.querySelector("#description").textContent = play.description;
    document.querySelector("#score").textContent = `${play.year} Combine`;
    const chips = [
      play.drill,
      play.attempt ? `Rep ${play.attempt}` : null,
      ...(play.marks || []).slice(0, 2).map(([label, value]) => `${label} ${value}`),
    ].filter(Boolean);
    document.querySelector("#chips").innerHTML = chips.map((chip) => `<span class="chip">${esc(chip)}</span>`).join("");
    const facts = [["College", play.college], ["Draft", play.draft], ...(play.marks || [])];
    document.querySelector("#facts").innerHTML = facts
      .filter(([, value]) => value != null && value !== "")
      .map(([label, value]) => `<dt>${esc(label)}</dt><dd>${esc(value)}</dd>`)
      .join("");
    document.querySelector("#off-swatch").style.background = "#f0c84b";
    document.querySelector("#def-swatch").style.background = "#f6f1e4";
    document.querySelector("#off-name").textContent = play.players[0].position || "Athlete";
    const hasBall = play.ballFrames && play.ballFrames.length;
    document.querySelector("#def-name").textContent = hasBall ? "Ball" : "";
    document.querySelector("#def-name").parentElement.hidden = !hasBall;
    return;
  }
  document.querySelector("#description").textContent = play.description;
  const home = play.homeScore == null ? "" : `${play.home} ${play.homeScore}, ${play.away} ${play.awayScore}`;
  document.querySelector("#score").textContent = `${play.away} at ${play.home} · Q${play.quarter} ${play.clock}${home ? ` · ${home}` : ""}`;
  const chips = [
    play.down ? `${ordinal(play.down)} & ${play.distance}` : null,
    resultLabel(play.result),
    play.yards == null ? null : `${play.yards > 0 ? "+" : ""}${play.yards} yards`,
    play.epa == null ? null : `${play.epa > 0 ? "+" : ""}${play.epa.toFixed(2)} EPA`,
    play.playAction ? "Play action" : null,
  ].filter(Boolean);
  document.querySelector("#chips").innerHTML = chips.map((chip) => `<span class="chip">${esc(chip)}</span>`).join("");
  const facts = [
    ["Formation", play.formation],
    ["Coverage", play.coverage],
    ["Route", play.route],
    ["Offense", play.offense],
    ["Defense", play.defense],
  ];
  document.querySelector("#facts").innerHTML = facts
    .filter(([, value]) => value)
    .map(([label, value]) => `<dt>${esc(label)}</dt><dd>${esc(value)}</dd>`)
    .join("");
  document.querySelector("#off-swatch").style.background = teamColor(play.offense);
  document.querySelector("#def-swatch").style.background = teamColor(play.defense);
  document.querySelector("#off-name").textContent = play.offense;
  document.querySelector("#def-name").textContent = play.defense;
  document.querySelector("#def-name").parentElement.hidden = false;
}

function showPlay(play) {
  state.play = play;
  state.t = 0;
  state.selected = null;
  state.playing = false;
  document.querySelector("#toggle").textContent = "▶";
  play.players.forEach(buildTrack);
  fitField(play);
  const scrub = document.querySelector("#scrub");
  scrub.max = String(totalFrames() - 1);
  scrub.value = "0";
  tag.hidden = true;
  renderSummary();
  drawField();
  renderActors();
  renderCard();
}

async function loadPlay(week, gameId, playId) {
  if (state.request) state.request.abort();
  state.request = new AbortController();
  loading.classList.add("show");
  try {
    const response = await fetch(`/api/play?week=${week}&game=${gameId}&play=${playId}`, { signal: state.request.signal });
    if (!response.ok) throw new Error("missing play");
    const play = await response.json();
    showPlay(play);
  } catch (error) {
    if (error.name !== "AbortError") {
      document.querySelector("#description").textContent = "Could not load that play.";
    }
  } finally {
    loading.classList.remove("show");
  }
}

function option(value, text) {
  const item = document.createElement("option");
  item.value = value;
  item.textContent = text;
  return item;
}

async function loadRep(eventId) {
  if (state.request) state.request.abort();
  state.request = new AbortController();
  loading.classList.add("show");
  try {
    const response = await fetch(`/api/combine/rep?event=${encodeURIComponent(eventId)}`, { signal: state.request.signal });
    if (!response.ok) throw new Error("missing rep");
    showPlay(await response.json());
  } catch (error) {
    if (error.name !== "AbortError") {
      document.querySelector("#description").textContent = "Could not load that rep.";
    }
  } finally {
    loading.classList.remove("show");
  }
}

function fillReps() {
  const bucket = (state.combine.reps[gameSelect.value] || []);
  playSelect.innerHTML = "";
  bucket.forEach((rep) => {
    const attempt = rep.attempt ? ` · rep ${rep.attempt}` : "";
    playSelect.appendChild(option(rep.event, `${rep.name} · ${rep.position || ""}${attempt}`));
  });
  if (bucket[0]) return loadRep(bucket[0].event);
  return null;
}

function fillDrills() {
  const drills = state.combine.drills[weekSelect.value] || [];
  gameSelect.innerHTML = "";
  drills.forEach((drill) => gameSelect.appendChild(option(drill.key, drill.label)));
  return fillReps();
}

function fillYears() {
  weekSelect.innerHTML = "";
  state.combine.years.forEach((year) => weekSelect.appendChild(option(String(year), String(year))));
  return fillDrills();
}

function setMode(mode) {
  state.mode = mode;
  state.playing = false;
  document.querySelector("#mode-plays").classList.toggle("active", mode === "plays");
  document.querySelector("#mode-combine").classList.toggle("active", mode === "combine");
  document.querySelector("#align").hidden = mode === "combine";
  document.querySelector("#week-label").textContent = mode === "combine" ? "Year" : "Week";
  document.querySelector("#game-label").textContent = mode === "combine" ? "Drill" : "Game";
  document.querySelector("#play-label").textContent = mode === "combine" ? "Rep" : "Play";
  document.querySelector("#subtitle").textContent = mode === "combine"
    ? "Combine drills, 10 Hz"
    : "2026 tracking, ball in the air";
  document.querySelector("#legend-note").textContent = mode === "combine"
    ? "The view closes in on the drill. The ball is drawn when the rep has one."
    : "Gold ring is the target. Faded players stop being tracked after the throw.";
  document.querySelector("#footnote").textContent = mode === "combine"
    ? "Space plays and pauses. Arrow keys step one frame. Each rep is one athlete."
    : "Space plays and pauses. Arrow keys step one frame. This file is passing plays only, and it leaves out offensive linemen.";
  if (mode === "plays") {
    weekSelect.innerHTML = "";
    state.index.weeks.forEach((week) => weekSelect.appendChild(option(String(week), `Week ${week}`)));
    return fillGames();
  }
  if (state.combine) return fillYears();
  return getJSON("/api/combine").then((combine) => {
    state.combine = combine;
    return fillYears();
  });
}

function fillPlays() {
  const week = weekSelect.value;
  const gameId = gameSelect.value;
  state.plays = [];
  playSelect.innerHTML = "";
  return getJSON(`/api/plays?week=${week}&game=${gameId}`).then((plays) => {
    state.plays = plays;
    plays.forEach((play) => {
      const situation = play.down ? `${ordinal(play.down)} & ${play.distance}` : "Play";
      playSelect.appendChild(option(play.playId, `Q${play.quarter} ${play.clock} · ${situation} · ${resultLabel(play.result)}`));
    });
    if (plays[0]) return loadPlay(week, gameId, plays[0].playId);
    return null;
  });
}

function fillGames() {
  const week = weekSelect.value;
  const games = state.index.games[week] || [];
  gameSelect.innerHTML = "";
  games
    .slice()
    .sort((a, b) => a.id.localeCompare(b.id))
    .forEach((game) => {
      gameSelect.appendChild(option(game.id, `${game.away} at ${game.home}`));
    });
  return fillPlays();
}

function seek(frame) {
  state.t = Math.max(0, Math.min(totalFrames() - 1, frame));
  renderActors();
}

function setPlaying(playing) {
  state.playing = playing;
  document.querySelector("#toggle").textContent = playing ? "❚❚" : "▶";
  if (playing) {
    if (state.t >= totalFrames() - 1) state.t = 0;
    state.lastTick = performance.now();
    requestAnimationFrame(tick);
  }
}

function tick(now) {
  if (!state.playing) return;
  const dt = now - state.lastTick;
  state.lastTick = now;
  state.t += (dt / 100) * state.speed;
  if (state.t >= totalFrames() - 1) {
    state.t = totalFrames() - 1;
    setPlaying(false);
  }
  renderActors();
  if (state.playing) requestAnimationFrame(tick);
}

field.addEventListener("click", () => {
  state.selected = null;
  tag.hidden = true;
  renderActors();
  renderCard();
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
document.querySelector("#align").addEventListener("click", () => {
  state.align = !state.align;
  document.querySelector("#align").textContent = state.align ? "Offense to the right" : "Actual direction";
  if (state.play) {
    drawField();
    renderActors();
  }
});
weekSelect.addEventListener("change", () => (state.mode === "combine" ? fillDrills() : fillGames()));
gameSelect.addEventListener("change", () => (state.mode === "combine" ? fillReps() : fillPlays()));
playSelect.addEventListener("change", () => {
  if (state.mode === "combine") loadRep(playSelect.value);
  else loadPlay(weekSelect.value, gameSelect.value, playSelect.value);
});
document.querySelector("#mode-plays").addEventListener("click", () => { if (state.mode !== "plays") setMode("plays"); });
document.querySelector("#mode-combine").addEventListener("click", () => { if (state.mode !== "combine") setMode("combine"); });
document.querySelector("#surprise").addEventListener("click", async () => {
  if (state.mode === "combine") {
    if (!state.combine) await setMode("combine");
    const years = state.combine.years;
    weekSelect.value = String(years[Math.floor(Math.random() * years.length)]);
    await fillDrills();
    const drills = [...gameSelect.options];
    gameSelect.value = drills[Math.floor(Math.random() * drills.length)].value;
    await fillReps();
    const reps = [...playSelect.options];
    const pick = reps[Math.floor(Math.random() * reps.length)];
    playSelect.value = pick.value;
    loadRep(pick.value);
    return;
  }
  const weeks = state.index.weeks;
  const week = weeks[Math.floor(Math.random() * weeks.length)];
  weekSelect.value = String(week);
  await fillGames();
  const games = [...gameSelect.options];
  gameSelect.value = games[Math.floor(Math.random() * games.length)].value;
  await fillPlays();
  const options = [...playSelect.options];
  const pick = options[Math.floor(Math.random() * options.length)];
  playSelect.value = pick.value;
  loadPlay(week, gameSelect.value, pick.value);
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

getJSON("/api/index").then((index) => {
  state.index = index;
  index.weeks.forEach((week) => weekSelect.appendChild(option(String(week), `Week ${week}`)));
  return fillGames();
});
