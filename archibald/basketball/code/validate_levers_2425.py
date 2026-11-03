"""The new levers on the 2024-25 sample (schema s2425)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import levers as L  # noqa: E402
from scoreboard import build_possessions  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "levers"


def main() -> None:
    pbp = L.load_pbp("s2425")
    raw = L.read("SELECT * FROM s2425.pbp")
    p = build_possessions(raw)
    p["half"] = L.half(p["game_id"])
    p["pushed"] = p["seconds_to_first_action"] <= L.EARLY
    p["opp"] = p["side"].map({"h": "v", "v": "h"})
    p = p.sort_values(["game_id", "period", "start_time"]).reset_index(drop=True)
    res = {"games": int(p["game_id"].nunique())}
    res["two_for_one"] = L.two_for_one(p, teams=False)
    res["bonus"] = L.bonus(p, pbp, teams=False)
    res["after_oreb"] = L.after_oreb(pbp)
    res["push_by_period"] = L.push_by_period(p)
    res["run_timeouts"] = L.run_timeouts(p)
    (OUT / "validate_2425.json").write_text(json.dumps(res, indent=2, default=str))
    t = res["two_for_one"]
    print("games", res["games"])
    print("2-for-1 rest-of-period net:", {k: (round(v["quick"]["mean"], 3), round(v["held"]["mean"], 3), v["quick"]["n"], v["held"]["n"]) for k, v in t["rest_of_period_net"].items()})
    print("2-for-1 this possession:", {k: (round(v["quick"]["mean"], 3), round(v["held"]["mean"], 3)) for k, v in t["this_possession_ppp"].items()})
    b = res["bonus"]
    print("bonus lift", round(b["lift_weighted"], 3), "even", round(b["lift_even"], 3), "odd", round(b["lift_odd"], 3), "ft trip", b["ft_trip_rate"])
    print("after oreb", {k: (v["mean"], v["count"]) for k, v in res["after_oreb"]["points_after_all"].items()})
    print("reset by seconds", res["after_oreb"]["reset_by_seconds"])
    print("push by period", res["push_by_period"])
    r = res["run_timeouts"]["net_next_6_possessions"]["all"]
    print("timeouts after a run: timeout", r["timeout"], "none", r["no_timeout"])


if __name__ == "__main__":
    main()
