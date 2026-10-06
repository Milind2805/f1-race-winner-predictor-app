"""Model + data layer for the F1 Race Winner Predictor (no Streamlit)."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import time
import requests

STANDINGS_URL = "https://api.jolpi.ca/ergast/f1/current/driverStandings.json"
_cache = {"time": 0, "data": None}

def fetch_live_standings(ttl=6 * 3600):
    """Returns {driver_code: (position, points)} or None if the API fails."""
    if _cache["data"] and time.time() - _cache["time"] < ttl:
        return _cache["data"]
    try:
        r = requests.get(STANDINGS_URL, timeout=10)
        r.raise_for_status()
        lists = r.json()["MRData"]["StandingsTable"]["StandingsLists"]
        rows = lists[0]["DriverStandings"]
        data = {row["Driver"]["code"]: (int(row["position"]), float(row["points"]))
                for row in rows}
    except Exception:
        return None
    _cache.update(time=time.time(), data=data)
    return data

DATA_DIR = Path(__file__).parent / "data"

# Display name -> what the user sees
TEAM_DISPLAY_ALIAS = {
    "Racing Point": "Aston Martin",
    "AlphaTauri": "Racing Bulls",
    "RB": "Racing Bulls",
    "Alfa Romeo": "Audi",
    "Kick Sauber": "Audi",
    "Alfa Romeo Racing": "Cadillac",
}
# What the user sees -> the label the encoder was fitted on
TEAM_ENCODE_ALIAS = {
    "Racing Bulls": "RB",
    "Audi": "Kick Sauber",
    "Cadillac": "Alfa Romeo Racing",
}


FEATURE_NAMES = [
    "Driver", "Team", "Circuit", "GridPosition", "QualiPosition",
    "Rainfall", "AirTemp", "TrackTemp", "StandingPoints", "HomeRace",
]


def _records(df: pd.DataFrame) -> list:
    """DataFrame -> JSON-safe list of dicts (avoids numpy types)."""
    return json.loads(df.to_json(orient="records"))


class F1Engine:
    def __init__(self, data_dir: Path = DATA_DIR):
        with open(data_dir / "f1_model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open(data_dir / "f1_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        with open(data_dir / "f1_encoders.pkl", "rb") as f:
            self.encoders = pickle.load(f)

        self.df = pd.read_csv(data_dir / "f1_multiyear.csv")
        with open(data_dir / "f1_feature_importance.json") as f:
            self.importance = json.load(f)
        with open(data_dir / "f1_shap_values.json") as f:
            self.shap = json.load(f)
        with open(data_dir / "model_metrics.json") as f:
            self.metrics = json.load(f)
        with open(data_dir / "default_grids.json") as f:
            self.default_grids = json.load(f)

        self.drivers = sorted(self.encoders["driver"].classes_.tolist())
        self.circuits = sorted(self.encoders["circuit"].classes_.tolist())
        raw_teams = self.encoders["team"].classes_.tolist()
        self.teams = sorted({TEAM_DISPLAY_ALIAS.get(t, t) for t in raw_teams})
        self.years = sorted(int(y) for y in self.df["Year"].unique())

    # ── Prediction ────────────────────────────────────────────
    def _encode(self, key: str, value: str) -> int:
        try:
            return int(self.encoders[key].transform([value])[0])
        except ValueError:
            raise ValueError(f"Unknown {key}: {value!r}")

    def _scaled(self, driver, team, circuit, grid_position, quali_position,
                rainfall, air_temp, track_temp, standing_points, home_race):
        x = np.array([[
            self._encode("driver", driver),
            self._encode("team", TEAM_ENCODE_ALIAS.get(team, team)),
            self._encode("circuit", circuit),
            grid_position, quali_position,
            int(rainfall), air_temp, track_temp,
            standing_points, int(home_race),
        ]])
        return self.scaler.transform(x)

    def predict(self, driver, team, circuit, grid_position, quali_position,
                rainfall, air_temp, track_temp, standing_points, home_race):
        """Returns (predicted_win: bool, win_probability: float in 0..1)."""
        xs = self._scaled(driver, team, circuit, grid_position, quali_position,
                          rainfall, air_temp, track_temp, standing_points, home_race)
        pred = int(self.model.predict(xs)[0]) == 1
        prob = float(self.model.predict_proba(xs)[0][1])
        return pred, prob

    # ── Season analytics ──────────────────────────────────────
    def _season(self, year: int) -> pd.DataFrame:
        s = self.df[self.df["Year"] == year]
        if s.empty:
            raise ValueError(f"No data for season {year}")
        return s

    def driver_progression(self, year: int, top_n: int = 5) -> dict:
        s = self._season(year)
        top = s.groupby("Abbreviation")["Points"].sum().nlargest(top_n).index
        # Assumes the CSV rows are in chronological order
        race_order = s["Race"].drop_duplicates().tolist()
        rows = s[s["Abbreviation"].isin(top)][["Race", "Abbreviation", "Points"]]
        return {"race_order": race_order, "rows": _records(rows)}

    def driver_wins(self, year: int) -> list:
        s = self._season(year)
        w = s[s["Position"] == 1]["Abbreviation"].value_counts().reset_index()
        w.columns = ["driver", "wins"]
        return _records(w)

    def constructor_points(self, year: int) -> list:
        s = self._season(year)
        c = s.groupby("TeamName")["Points"].sum().reset_index()
        c.columns = ["team", "points"]
        return _records(c.sort_values("points", ascending=False))

    def constructor_wins(self, year: int) -> list:
        s = self._season(year)
        w = s[s["Position"] == 1]["TeamName"].value_counts().reset_index()
        w.columns = ["team", "wins"]
        return _records(w)

    # ── Explainability ────────────────────────────────────────
    def explain(self, driver, team, circuit, grid_position, quali_position,
                rainfall, air_temp, track_temp, standing_points, home_race):
        """Per-prediction SHAP contributions (log-odds) toward a win."""
        xs = self._scaled(driver, team, circuit, grid_position, quali_position,
                          rainfall, air_temp, track_temp, standing_points, home_race)
        try:
            # XGBoost's built-in SHAP: same values as TreeExplainer, no `shap` dependency
            import xgboost as xgb
            booster = self.model.get_booster()
            dm = xgb.DMatrix(xs, feature_names=booster.feature_names)
            contribs = booster.predict(dm, pred_contribs=True)[0][:-1]
        except Exception:
            import shap
            contribs = shap.TreeExplainer(self.model).shap_values(xs)[0]
        pred = int(self.model.predict(xs)[0]) == 1
        prob = float(self.model.predict_proba(xs)[0][1])
        return pred, prob, {n: float(v) for n, v in zip(FEATURE_NAMES, contribs)}

    # ── Grid simulator ────────────────────────────────────────
    def current_grid(self, season: str = "2026") -> list:
        """The JSON decides WHO is on the grid; the live API (2026 only)
        decides their points and order. Falls back to the JSON if offline."""
        if season not in self.default_grids:
            raise ValueError(f"No default grid for {season}")
        rows = [dict(r) for r in self.default_grids[season]]
        live = fetch_live_standings() if season == "2026" else None
        if not live:
            return rows
        for r in rows:
            if r["driver"] in live:
                r["_live_pos"], r["points"] = live[r["driver"]]
        rows.sort(key=lambda r: r.get("_live_pos", 999))
        for i, r in enumerate(rows, start=1):
            r["position"] = i
            r.pop("_live_pos", None)
        return rows
    def default_grid(self, season: int) -> list:
        grid = self.current_grid(str(season))
        return [
            {
                **row,
                "team": TEAM_DISPLAY_ALIAS.get(row["team"], row["team"]),
                "supported": row["driver"] in self.drivers,
            }
            for row in grid
        ]

    def simulate_grid(self, entries, circuit, rainfall, air_temp, track_temp):
        """Win probability for every car on the grid. Drivers the model has
        never seen are returned with supported=False and no probability."""
        out = []
        for e in entries:
            row = {"driver": e["driver"], "team": e["team"],
                   "quali_position": e["quali_position"]}
            try:
                _, prob = self.predict(
                    e["driver"], e["team"], circuit,
                    e["grid_position"], e["quali_position"],
                    rainfall, air_temp, track_temp,
                    e["standing_points"], e["home_race"],
                )
                out.append({**row, "supported": True, "win_probability": round(prob, 4)})
            except ValueError:
                out.append({**row, "supported": False, "win_probability": None})
        out.sort(key=lambda r: (-1 if r["win_probability"] is None else r["win_probability"]),
                 reverse=True)
        return out

    # ── Season projection (Monte Carlo) ───────────────────────
    POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=float)

    def project_season(self, remaining_races, n_sims=500, seed=None):
        """Monte Carlo projection of the 2026 standings.

        Same logic as the Streamlit tab, vectorised: for each race, ONE batched
        model call covers every simulation and driver. Finishing orders are drawn
        from Plackett-Luce via the Gumbel-max trick (identical distribution to
        sequential weighted sampling without replacement).
        """
        grid = self.current_grid("2026")
        names = [r["driver"] for r in grid]
        D, S = len(grid), int(n_sims)
        rng = np.random.default_rng(seed)

        current = np.array([r["points"] for r in grid], dtype=float)

        # Recent form (2025 averages); neutral midfield guess for rookies
        form_grid = np.full(D, 12.0)
        form_quali = np.full(D, 12.0)
        for i, drv in enumerate(names):
            rows = self.df[(self.df["Abbreviation"] == drv) & (self.df["Year"] == 2025)]
            if len(rows):
                form_grid[i] = rows["GridPosition"].mean()
                form_quali[i] = rows["QualiPosition"].mean()

        # Which drivers can the model score? Others get a small floor probability.
        sup_idx, drv_enc, team_enc = [], [], []
        for i, r in enumerate(grid):
            try:
                d = self._encode("driver", r["driver"])
                t = self._encode("team", TEAM_ENCODE_ALIAS.get(r["team"], r["team"]))
            except ValueError:
                continue
            sup_idx.append(i); drv_enc.append(d); team_enc.append(t)
        sup_idx = np.array(sup_idx, dtype=int)
        drv_enc = np.array(drv_enc, dtype=float)
        team_enc = np.array(team_enc, dtype=float)
        Ds = len(sup_idx)

        season_pts = np.tile(current, (S, 1))
        skipped = []
        rows_idx = np.arange(S)

        for race in remaining_races:
            try:
                circuit_enc = self._encode("circuit", race)
            except ValueError:
                skipped.append(race)
                continue

            win = np.full((S, D), 0.01)
            if Ds:
                gp = np.maximum(1, rng.normal(form_grid[sup_idx], 2, (S, Ds)).astype(int))
                qp = np.maximum(1, rng.normal(form_quali[sup_idx], 2, (S, Ds)).astype(int))
                X = np.empty((S, Ds, 10))
                X[:, :, 0] = drv_enc
                X[:, :, 1] = team_enc
                X[:, :, 2] = circuit_enc
                X[:, :, 3] = gp
                X[:, :, 4] = qp
                X[:, :, 5] = 0      # dry
                X[:, :, 6] = 25     # air temp
                X[:, :, 7] = 35     # track temp
                X[:, :, 8] = season_pts[:, sup_idx]
                X[:, :, 9] = 0      # home race
                p = self.model.predict_proba(self.scaler.transform(X.reshape(-1, 10)))[:, 1]
                win[:, sup_idx] = np.maximum(p.reshape(S, Ds), 0.001)

            keys = np.log(win) + rng.gumbel(size=(S, D))
            order = np.argsort(-keys, axis=1)[:, : min(10, D)]
            for rank in range(order.shape[1]):
                season_pts[rows_idx, order[:, rank]] += self.POINTS[rank]

        champs = np.argmax(season_pts, axis=1)
        title_pct = np.bincount(champs, minlength=D) * 100.0 / S

        summary = []
        for i, r in enumerate(grid):
            v = season_pts[:, i]
            summary.append({
                "driver": r["driver"],
                "team": TEAM_DISPLAY_ALIAS.get(r["team"], r["team"]),
                "current_points": float(current[i]),
                "projected_mean": round(float(v.mean()), 1),
                "p10": round(float(np.percentile(v, 10)), 1),
                "p90": round(float(np.percentile(v, 90)), 1),
                "title_win_pct": round(float(title_pct[i]), 1),
                "modelled": i in set(sup_idx.tolist()),
            })
        summary.sort(key=lambda x: x["projected_mean"], reverse=True)
        return {"n_sims": S, "skipped_races": skipped, "standings": summary}
