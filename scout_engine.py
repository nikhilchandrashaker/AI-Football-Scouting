"""
scout_engine.py - Team-level football scouting engine (Premier League 2010-11 .. 2024-25).

Modules:  data cleaning -> team profiles -> form/home-away flags -> trajectory
          -> PCA + clustering + similarity -> scout search -> match model -> scouting report

Design notes
* Every match row carries PRE-match season-to-date stats for both teams, so they are leak-free features.
* Rounds 1-10 of each season are not in the file (rows start when a team has played >= 10 games).
  Season-to-date totals still cover the whole season because they come from the cumulative columns.
* Points are recomputed as 3*W + D ("on-pitch points"): the raw PTS column includes deductions
  (Everton and Nottingham Forest 2023-24) that would distort performance analysis.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline

NAME_FIX = {"Leeds United": "Leeds", "Sheffield": "Sheffield Utd"}
STATE = ["P", "W", "D", "L", "GF", "GA", "hP", "hW", "hD", "hL", "hGF", "hGA", "aP", "aW", "aD", "aL", "aGF", "aGA"]
LEVEL = ["ppg", "gf_pg", "ga_pg", "gd_pg", "h_ppg", "a_ppg", "h_gf_pg", "h_ga_pg", "a_gf_pg", "a_ga_pg"]

# similarity / clustering feature vector: (column, readable name, text if high, text if low)
FEATS = [
    ("ppg", "points per match", "high points accumulation", "low points accumulation"),
    ("gf_pg", "goals scored", "above-average scoring", "below-average scoring"),
    ("ga_pg", "goals conceded", "leaky defence", "tight defence"),
    ("gd_pg", "goal difference", "positive goal difference", "negative goal difference"),
    ("h_ppg", "home points", "strong home performance", "weak home performance"),
    ("a_ppg", "away points", "strong away performance", "weak away performance"),
    ("h_gf_pg", "home scoring", "prolific at home", "blunt at home"),
    ("h_ga_pg", "home goals conceded", "leaky at home", "solid at home"),
    ("a_gf_pg", "away scoring", "prolific away", "blunt away"),
    ("a_ga_pg", "away goals conceded", "leaky away", "solid away"),
    ("form6", "recent form (last 6)", "strong recent form", "poor recent form"),
    ("pts_std", "results volatility", "volatile results", "consistent results"),
]
SCORE_COLS = ["Overall", "Attack", "Defense", "Home", "Away", "Consistency", "Form", "Trajectory"]


# --------------------------------------------------------------------------- data
def load_matches(path) -> pd.DataFrame:
    d = pd.read_csv(path, sep=";", encoding="utf-8-sig").dropna(how="all").drop_duplicates().reset_index(drop=True)
    d["Date"] = pd.to_datetime(d["Date"], dayfirst=True)
    for c in ["Home Team", "Away Team"]:
        d[c] = d[c].replace(NAME_FIX)
    start = np.where(d.Date.dt.month >= 8, d.Date.dt.year, d.Date.dt.year - 1).astype(int)  # handles COVID-delayed 2019-20
    d["Season"] = [f"{s}-{str(s + 1)[-2:]}" for s in start]
    inv = 1 / d[["B365H", "B365D", "B365A"]]
    inv = inv.div(inv.sum(axis=1), axis=0)  # remove bookmaker margin
    d["pH"], d["pD"], d["pA"] = inv["B365H"], inv["B365D"], inv["B365A"]
    d = d.sort_values(["Date", "Home Team"]).reset_index(drop=True)
    d["match_id"] = np.arange(len(d))
    return d


def _side(d, s):
    home = s == "H"
    o = pd.DataFrame({
        "match_id": d.match_id, "Date": d.Date, "Season": d.Season,
        "team": d["Home Team" if home else "Away Team"], "opp": d["Away Team" if home else "Home Team"],
        "venue": "H" if home else "A"})
    for k in ["P", "W", "D", "L", "GF", "GA"]:
        o["b_" + k] = d[f"Total_{s}_{k}"].values
        o["b_h" + k] = d[f"Home_{s}_{k}"].values
        o["b_a" + k] = d[f"Away_{s}_{k}"].values
    gf, ga = (d.FTHG, d.FTAG) if home else (d.FTAG, d.FTHG)
    o["gf"], o["ga"] = gf.values, ga.values
    o["pts"] = np.where(gf > ga, 3, np.where(gf == ga, 1, 0))
    o["xpts"] = (3 * d["pH" if home else "pA"] + d.pD).values  # market-implied expected points
    inc = {"P": 1, "W": (o.pts == 3) * 1, "D": (o.pts == 1) * 1, "L": (o.pts == 0) * 1, "GF": o.gf, "GA": o.ga}
    mine, other = ("h", "a") if home else ("a", "h")
    for k, v in inc.items():
        o["f_" + k] = o["b_" + k] + v
        o["f_" + mine + k] = o["b_" + mine + k] + v
        o["f_" + other + k] = o["b_" + other + k]
    o["game_no"] = o["b_P"] + 1
    return o


def _rates(S):
    P, hP, aP = max(S["P"], 1), max(S["hP"], 1), max(S["aP"], 1)
    return dict(ppg=(3 * S["W"] + S["D"]) / P, gf_pg=S["GF"] / P, ga_pg=S["GA"] / P, gd_pg=(S["GF"] - S["GA"]) / P,
                d_rate=S["D"] / P, h_ppg=(3 * S["hW"] + S["hD"]) / hP, a_ppg=(3 * S["aW"] + S["aD"]) / aP,
                h_gf_pg=S["hGF"] / hP, h_ga_pg=S["hGA"] / hP, a_gf_pg=S["aGF"] / aP, a_ga_pg=S["aGA"] / aP)


def _bar(v, n=10):
    k = int(round(v / 100 * n))
    return "█" * k + "░" * (n - k)


def _level(v):
    return "strong" if v >= 75 else "solid" if v >= 55 else "average" if v >= 35 else "weak"


# --------------------------------------------------------------------------- match model helpers
def _state_from_match(d, s):
    out = {}
    for k in ["P", "W", "D", "L", "GF", "GA"]:
        out[k] = d[f"Total_{s}_{k}"]
        out["h" + k] = d[f"Home_{s}_{k}"]
        out["a" + k] = d[f"Away_{s}_{k}"]
    return out


def _features(hs, as_):
    def rt(S, p):
        P = np.maximum(S[p + "P"], 1)
        return (3 * S[p + "W"] + S[p + "D"]) / P, S[p + "GF"] / P, S[p + "GA"] / P
    htp, htf, hta = rt(hs, "")
    hvp, hvf, hva = rt(hs, "h")
    atp, atf, ata = rt(as_, "")
    avp, avf, ava = rt(as_, "a")
    return pd.DataFrame({
        "h_tot_ppg": htp, "h_tot_gf": htf, "h_tot_ga": hta, "h_home_ppg": hvp, "h_home_gf": hvf, "h_home_ga": hva,
        "a_tot_ppg": atp, "a_tot_gf": atf, "a_tot_ga": ata, "a_away_ppg": avp, "a_away_gf": avf, "a_away_ga": ava,
        "d_tot_ppg": htp - atp, "d_venue_ppg": hvp - avp,
        "d_tot_gd": (htf - hta) - (atf - ata), "d_venue_gd": (hvf - hva) - (avf - ava)})


def _mm(y, P):
    y = np.asarray(y)
    oh = np.eye(3)[y]
    P = np.clip(P, 1e-6, 1)
    P = P / P.sum(1, keepdims=True)
    cp, cy = np.cumsum(P, 1)[:, :2], np.cumsum(oh, 1)[:, :2]
    return dict(accuracy=float((P.argmax(1) == y).mean()), log_loss=float(-np.log(P[np.arange(len(y)), y]).mean()),
                brier=float(((P - oh) ** 2).sum(1).mean()), rps=float(((cp - cy) ** 2).sum(1).mean() / 2))


def _lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))


def _hgb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=120, l2_regularization=1.0, random_state=0)


# --------------------------------------------------------------------------- engine
class ScoutEngine:
    def __init__(self, csv_path):
        self.matches = load_matches(csv_path)
        d = self.matches
        self.L = pd.concat([_side(d, "H"), _side(d, "A")]).sort_values(["Season", "team", "game_no"]).reset_index(drop=True)
        self.groups = {k: g.reset_index(drop=True) for k, g in self.L.groupby(["Season", "team"])}
        self.seasons = sorted(d.Season.unique())
        self._snaps = {}
        self.ts = self.snapshot(99).reset_index(drop=True)  # final state of every team-season
        self._build_scores()
        self._build_embedding()
        self._build_clusters()

    # ---- basic lookups
    def teams(self, season):
        return sorted(self.ts[self.ts.Season == season].team)

    def latest_season(self):
        return self.seasons[-1]

    # ---- snapshots (state of every team-season after `as_of` games) ------------------------------
    def _snap_row(self, g, n=6):
        last = g.iloc[-1]
        w = g[g.game_no >= last.game_no - n + 1]
        first = w.iloc[0]
        aft = {k: float(last["f_" + k]) for k in STATE}
        bef = {k: float(first["b_" + k]) for k in STATE}
        win = {k: aft[k] - bef[k] for k in STATE}
        a, r, b = _rates(aft), _rates(win), _rates(bef)
        k = len(g) // 2
        return dict(games=aft["P"], n_rows=len(g), w_games=win["P"], **a,
                    r_ppg=r["ppg"], r_gf=r["gf_pg"], r_ga=r["ga_pg"], b_ppg=b["ppg"], b_gf=b["gf_pg"], b_ga=b["ga_pg"],
                    form6=r["ppg"], d_pts=r["ppg"] - b["ppg"], d_att=r["gf_pg"] - b["gf_pg"], d_def=r["ga_pg"] - b["ga_pg"],
                    over=float((g.pts - g.xpts).mean()), xppg=float(g.xpts.mean()), pts_std=float(g.pts.std(ddof=0)),
                    half_delta=float(g.pts.iloc[k:].mean() - g.pts.iloc[:k].mean()) if k else 0.0,
                    gap=a["h_ppg"] - a["a_ppg"], on_pitch_pts=3 * aft["W"] + aft["D"])

    def snapshot(self, as_of=99, min_rows=8):
        as_of = int(as_of)
        if as_of in self._snaps:
            return self._snaps[as_of]
        rows = []
        for (s, t), g in self.groups.items():
            g = g[g.game_no <= as_of]
            if len(g) >= min_rows:
                rows.append(dict(Season=s, team=t, **self._snap_row(g)))
        out = pd.DataFrame(rows).set_index(["Season", "team"], drop=False)
        out.index.names = ["s", "t"]
        self._snaps[as_of] = out
        return out

    @staticmethod
    def _center(snap):
        c = snap.copy()
        c[LEVEL] = c[LEVEL] - c.groupby(c["Season"].values)[LEVEL].transform("mean")
        return c

    # ---- team profile -------------------------------------------------------------------------
    def _build_scores(self):
        c = self._center(self.ts.set_index(["Season", "team"], drop=False))
        pct = lambda s: s.rank(pct=True) * 100
        sc = pd.DataFrame({
            "Overall": pct(c.ppg), "Attack": pct(c.gf_pg), "Defense": pct(-c.ga_pg), "Home": pct(c.h_ppg), "Away": pct(c.a_ppg),
            "Consistency": pct(-c.pts_std), "Form": pct(c.form6), "Trajectory": pct(c.half_delta),
            "Goal difference": pct(c.gd_pg)})
        self.scores, self.centered = sc, c

    def profile(self, team, season):
        i = (season, team)
        t = self.ts[(self.ts.Season == season) & (self.ts.team == team)].iloc[0]
        lg = self.ts[self.ts.Season == season]
        rank = int(lg.ppg.rank(ascending=False, method="min")[t.name])
        return dict(scores=self.scores.loc[i].to_dict(), raw=t, league=lg.mean(numeric_only=True), rank=rank, n_teams=len(lg))

    def profile_bars(self, team, season):
        sc = self.scores.loc[(season, team)]
        return "\n".join(f"{k:<12}{_bar(sc[k])} {sc[k]:>3.0f}" for k in SCORE_COLS if k != "Trajectory")

    def team_history(self, team):
        return self.ts[self.ts.team == team].sort_values("Season")

    # ---- form scout / home-away scout ----------------------------------------------------------
    def scout_flags(self, team, season, as_of=None, z=1.25):
        snap = self.snapshot(as_of or 99)
        if (season, team) not in snap.index:
            return []
        cs = self._center(snap)
        cols = ["d_att", "d_def", "d_pts", "over", "h_ppg", "a_ppg", "h_ga_pg", "a_ga_pg", "h_gf_pg", "a_gf_pg", "gap"]
        zs = (cs.loc[(season, team), cols] - cs[cols].mean()) / cs[cols].std()
        t = snap.loc[(season, team)]
        lg = snap[snap.Season == season].mean(numeric_only=True)
        n = int(t.w_games)
        R = {
            "d_att": (("🟢", "Form", "Attacking improvement", f"Scoring {t.r_gf:.2f}/game over the last {n} vs {t.b_gf:.2f} earlier in the season."),
                      ("🔴", "Form", "Attacking decline", f"Scoring {t.r_gf:.2f}/game over the last {n} vs {t.b_gf:.2f} earlier in the season.")),
            "d_def": (("🔴", "Form", "Defensive decline", f"Conceding {t.r_ga:.2f}/game over the last {n} vs {t.b_ga:.2f} earlier in the season."),
                      ("🟢", "Form", "Defensive improvement", f"Conceding {t.r_ga:.2f}/game over the last {n} vs {t.b_ga:.2f} earlier in the season.")),
            "d_pts": (("🟢", "Form", "Points surge", f"{t.r_ppg:.2f} pts/game over the last {n} vs {t.b_ppg:.2f} earlier."),
                      ("🔴", "Form", "Points slump", f"{t.r_ppg:.2f} pts/game over the last {n} vs {t.b_ppg:.2f} earlier.")),
            "over": (("🟡", "Results vs market", "Results outperform market expectation", f"Averaging {t.over:+.2f} pts/game vs what pre-match odds implied - possible regression risk."),
                     ("🟡", "Results vs market", "Results below market expectation", f"Averaging {t.over:+.2f} pts/game vs what pre-match odds implied - possible positive-regression candidate.")),
            "h_ppg": (("🟢", "Home", "Strong home form", f"{t.h_ppg:.2f} home pts/game (league avg {lg.h_ppg:.2f})."),
                      ("🔴", "Home", "Weak home form", f"{t.h_ppg:.2f} home pts/game (league avg {lg.h_ppg:.2f}).")),
            "a_ppg": (("🟢", "Away", "Strong away form", f"{t.a_ppg:.2f} away pts/game (league avg {lg.a_ppg:.2f})."),
                      ("🔴", "Away", "Poor away form", f"{t.a_ppg:.2f} away pts/game (league avg {lg.a_ppg:.2f}).")),
            "a_ga_pg": (("🔴", "Away", "Away defensive vulnerability", f"Conceding {t.a_ga_pg:.2f} per away game (league avg {lg.a_ga_pg:.2f})."),
                        ("🟢", "Away", "Away defensive resilience", f"Conceding {t.a_ga_pg:.2f} per away game (league avg {lg.a_ga_pg:.2f}).")),
            "h_ga_pg": (("🔴", "Home", "Home defensive vulnerability", f"Conceding {t.h_ga_pg:.2f} per home game (league avg {lg.h_ga_pg:.2f})."),
                        ("🟢", "Home", "Home defensive fortress", f"Conceding {t.h_ga_pg:.2f} per home game (league avg {lg.h_ga_pg:.2f}).")),
            "a_gf_pg": (("🟢", "Away", "Away attacking threat", f"Scoring {t.a_gf_pg:.2f} per away game (league avg {lg.a_gf_pg:.2f})."),
                        ("🔴", "Away", "Toothless away", f"Scoring {t.a_gf_pg:.2f} per away game (league avg {lg.a_gf_pg:.2f}).")),
            "h_gf_pg": (("🟢", "Home", "Home attacking strength", f"Scoring {t.h_gf_pg:.2f} per home game (league avg {lg.h_gf_pg:.2f})."),
                        ("🔴", "Home", "Home attacking struggles", f"Scoring {t.h_gf_pg:.2f} per home game (league avg {lg.h_gf_pg:.2f}).")),
            "gap": (("🟡", "Home/Away", "Unusually home-dependent", f"Home {t.h_ppg:.2f} vs away {t.a_ppg:.2f} pts/game (gap {t.gap:+.2f}; league avg {lg.gap:+.2f})."),
                    ("🟡", "Home/Away", "Better away than home", f"Home {t.h_ppg:.2f} vs away {t.a_ppg:.2f} pts/game (gap {t.gap:+.2f}; league avg {lg.gap:+.2f}).")),
        }
        flags = []
        for m in cols:
            if abs(zs[m]) >= z:
                icon, cat, title, detail = R[m][0 if zs[m] > 0 else 1]
                flags.append(dict(icon=icon, category=cat, flag=title, detail=detail, z=float(zs[m])))
        return sorted(flags, key=lambda f: -abs(f["z"]))

    def league_flags(self, season, as_of=None, z=1.25):
        rows = []
        for t in self.teams(season):
            fl = self.scout_flags(t, season, as_of, z)
            rows.append(dict(team=t, red=sum(f["icon"] == "🔴" for f in fl), green=sum(f["icon"] == "🟢" for f in fl),
                             yellow=sum(f["icon"] == "🟡" for f in fl), flags="; ".join(f["icon"] + " " + f["flag"] for f in fl)))
        return pd.DataFrame(rows)

    def summary_lines(self, team, season):
        sc, t = self.scores.loc[(season, team)], self.ts[(self.ts.Season == season) & (self.ts.team == team)].iloc[0]
        traj = "improving" if t.half_delta > 0.25 else "declining" if t.half_delta < -0.25 else "flat"
        return {"Overall record": _level(sc.Overall), "Home performance": _level(sc.Home), "Away performance": _level(sc.Away),
                "Points trajectory": f"{traj} ({t.half_delta:+.2f} pts/game, 2nd vs 1st half)"}

    # ---- trajectory -------------------------------------------------------------------------------
    def trajectory(self, team, season, window=6):
        g = self.groups[(season, team)].copy()
        g["ppg_roll"] = g.pts.rolling(window, min_periods=3).mean()
        g["gf_roll"] = g.gf.rolling(window, min_periods=3).mean()
        g["ga_roll"] = g.ga.rolling(window, min_periods=3).mean()
        g["cum_ppg"] = (3 * g.f_W + g.f_D) / g.f_P
        g["cum_gd"] = g.f_GF - g.f_GA
        g["cum_vs_market"] = (g.pts - g.xpts).cumsum()
        return g[["game_no", "Date", "opp", "venue", "gf", "ga", "pts", "xpts", "ppg_roll", "gf_roll", "ga_roll", "cum_ppg", "cum_gd", "cum_vs_market"]]

    # ---- embedding, clustering, similarity ------------------------------------------------------------
    def _build_embedding(self):
        X = self.centered[[f[0] for f in FEATS]].values
        self.scaler = StandardScaler().fit(X)
        self.X = self.scaler.transform(X)
        self.pca = PCA(n_components=0.90, svd_solver="full").fit(self.X)
        self.Z = self.pca.transform(self.X)
        self.dmax = float(np.percentile(pdist(self.Z), 95))
        self.idx = {(s, t): i for i, (s, t) in enumerate(zip(self.ts.Season, self.ts.team))}

    def _build_clusters(self):
        best = max(((k, silhouette_score(self.Z, KMeans(k, n_init=20, random_state=0).fit_predict(self.Z))) for k in range(4, 7)), key=lambda x: x[1])
        self.k, self.silhouette = best
        km = KMeans(self.k, n_init=50, random_state=0).fit(self.Z)
        lab = km.labels_
        names = {4: ["Elite", "Contenders", "Mid-table", "Strugglers"], 5: ["Elite", "Contenders", "Upper mid-table", "Lower mid-table", "Strugglers"],
                 6: ["Elite", "Contenders", "Upper mid-table", "Mid-table", "Lower mid-table", "Strugglers"]}[self.k]
        f = {c[0]: j for j, c in enumerate(FEATS)}
        cent = pd.DataFrame(self.X).groupby(lab).mean()
        order = cent[f["ppg"]].sort_values(ascending=False).index.tolist()
        self.cluster_names = {}
        for rank, c in enumerate(order):
            z = cent.loc[c]
            style = []
            d = z[f["gf_pg"]] - (-z[f["ga_pg"]])
            if z[f["ppg"]] > 0:
                if d > 0.4: style.append("attack-led")
                elif d < -0.4: style.append("defence-led")
            else:
                if d > 0.4: style.append("defence is the weak side")
                elif d < -0.4: style.append("attack is the weak side")
            if z[f["h_ppg"]] - z[f["a_ppg"]] > 0.4: style.append("home-reliant")
            self.cluster_names[c] = names[rank] + (f" ({', '.join(style)})" if style else "")
        self.ts["cluster"] = [self.cluster_names[c] for c in lab]
        self.cluster_id = lab

    def similar(self, team, season, k=5, scope="all", same_club=False):
        i = self.idx[(season, team)]
        d = cdist(self.Z[i:i + 1], self.Z)[0]
        df = self.ts[["Season", "team", "ppg", "cluster"]].copy()
        df["similarity"] = np.clip(1 - d / self.dmax, 0, 1) * 100
        m = np.arange(len(df)) != i
        if scope == "season": m &= (df.Season == season).values
        if not same_club: m &= (df.team != team).values
        out = df[m].sort_values("similarity", ascending=False).head(k).copy()
        sh, diff = [], []
        for n in out.index:
            zt, zn = self.X[i], self.X[n]
            s_, d_ = [], []
            for j in np.argsort(-np.abs(zt)):
                if zt[j] * zn[j] > 0 and abs(zt[j]) >= 0.5 and abs(zt[j] - zn[j]) <= 0.6:
                    s_.append(FEATS[j][2] if zt[j] > 0 else FEATS[j][3])
                if abs(zt[j] - zn[j]) >= 1.0:
                    d_.append(f"{FEATS[j][1]} ({'higher' if zn[j] > zt[j] else 'lower'})")
            sh.append(s_[:4]); diff.append(d_[:3])
        out["shared"], out["differs_on"] = sh, diff
        return out.reset_index(drop=True)

    def pca_frame(self):
        df = self.ts[["Season", "team", "cluster"]].copy()
        df["PC1"], df["PC2"] = self.Z[:, 0], self.Z[:, 1]
        return df

    # ---- scout search ---------------------------------------------------------------------------------
    LEVELS = {"strong": 90, "above average": 70, "average": 50, "below average": 30, "weak": 10}

    def scout_search(self, criteria: dict, season=None, top=10):
        """criteria: {score_name: (target_percentile 0-100, weight)}.  Returns similarity-to-request, NOT a quality ranking."""
        sc = self.scores.copy()
        if season: sc = sc[sc.index.get_level_values(0) == season]
        w = np.array([criteria[c][1] for c in criteria], float)
        diffs = np.column_stack([(sc[c] - criteria[c][0]).abs() / 100 for c in criteria])
        match = 100 * (1 - (diffs * w).sum(1) / w.sum())
        out = sc[list(criteria)].round(0).copy()
        out.insert(0, "match_%", match.round(1))
        out = out.reset_index()
        return out.sort_values("match_%", ascending=False).head(top).reset_index(drop=True)

    # ---- match outcome model ------------------------------------------------------------------------------
    def _match_xy(self):
        d = self.matches
        X = _features(_state_from_match(d, "H"), _state_from_match(d, "A"))
        y = d.FTR.map({"A": 0, "D": 1, "H": 2}).values
        odds = d[["pA", "pD", "pH"]].values
        return X, y, odds

    def fit_match_model(self, first_test_idx=3):
        """Walk-forward evaluation: for each season s, train on all earlier seasons, test on s."""
        X, y, odds = self._match_xy()
        Xo = X.assign(pA=odds[:, 0], pD=odds[:, 1], pH=odds[:, 2])
        seas = self.matches.Season.values
        store = {k: [] for k in ["Class-frequency baseline", "Bookmaker (Bet365 odds)", "Logistic (team stats)", "Gradient boosting (team stats)", "Logistic (stats + odds)"]}
        ys = []
        for s in self.seasons[first_test_idx:]:
            tr, te = seas < s, seas == s
            ys.append(y[te])
            store["Class-frequency baseline"].append(np.tile(np.bincount(y[tr], minlength=3) / tr.sum(), (te.sum(), 1)))
            store["Bookmaker (Bet365 odds)"].append(odds[te])
            store["Logistic (team stats)"].append(_lr().fit(X[tr], y[tr]).predict_proba(X[te]))
            store["Gradient boosting (team stats)"].append(_hgb().fit(X[tr], y[tr]).predict_proba(X[te]))
            store["Logistic (stats + odds)"].append(_lr().fit(Xo[tr], y[tr]).predict_proba(Xo[te]))
        y_all = np.concatenate(ys)
        self.eval_y = y_all
        self.eval_probs = {k: np.vstack(v) for k, v in store.items()}
        self.eval = pd.DataFrame({k: _mm(y_all, p) for k, p in self.eval_probs.items()}).T
        self.eval.attrs["n_test_matches"] = int(len(y_all))
        self.model = _lr().fit(X, y)
        self.feature_names = list(X.columns)
        coef = self.model[-1].coef_
        self.importance = pd.DataFrame({"feature": self.feature_names, "coef_home_win": coef[2], "coef_away_win": coef[0]}) \
            .sort_values("coef_home_win", key=abs, ascending=False).reset_index(drop=True)
        return self.eval

    def calibration(self, model="Logistic (team stats)", bins=8):
        p = self.eval_probs[model][:, 2]
        hit = (self.eval_y == 2).astype(float)
        b = pd.qcut(p, bins, duplicates="drop")
        return pd.DataFrame({"pred": p, "actual": hit}).groupby(b, observed=True).mean().reset_index(drop=True)

    def _state_at(self, team, season, as_of):
        g = self.groups[(season, team)]
        g = g[g.game_no <= (as_of or 99)]
        last = g.iloc[-1]
        return {k: pd.Series([float(last["f_" + k])]) for k in STATE}

    def predict_matchup(self, home, away, season, as_of=None):
        """Hypothetical fixture using each side's season-to-date stats (home team's home split, away team's away split)."""
        if not hasattr(self, "model"): self.fit_match_model()
        X = _features(self._state_at(home, season, as_of), self._state_at(away, season, as_of))
        p = self.model.predict_proba(X[self.feature_names])[0]
        return {"Home win": p[2], "Draw": p[1], "Away win": p[0]}

    # ---- scouting report ------------------------------------------------------------------------------------
    def scouting_report(self, team, season, as_of=None, k_similar=5) -> str:
        p = self.profile(team, season)
        sc, t, lg = p["scores"], p["raw"], p["league"]
        fl = self.scout_flags(team, season, as_of)
        sim = self.similar(team, season, k_similar)
        summ = self.summary_lines(team, season)
        traj = self.trajectory(team, season)
        best = traj.ppg_roll.idxmax(); worst = traj.ppg_roll.idxmin()
        L = [f"# Scouting Report: {team} ({season})", "",
             f"*Team-level analysis of {int(t.games)} matches. Cluster: **{t.cluster}**. League position by on-pitch points: {p['rank']}/{p['n_teams']}.*", "",
             "## Profile", "```", self.profile_bars(team, season), "```", "",
             "| Metric | " + team + " | League avg |", "|---|---|---|"]
        for lab, c in [("Points / match", "ppg"), ("Goals scored / match", "gf_pg"), ("Goals conceded / match", "ga_pg"), ("Home points / match", "h_ppg"),
                       ("Away points / match", "a_ppg"), ("Draw rate", "d_rate"), ("Recent form (last 6, pts/match)", "form6")]:
            L.append(f"| {lab} | {t[c]:.2f} | {lg[c]:.2f} |")
        L += ["", "## Summary", *[f"- **{k}:** {v}" for k, v in summ.items()], "", "## Scouting flags"]
        L += [f"- {f['icon']} **{f['flag']}** ({f['category']}): {f['detail']}" for f in fl] or ["- No statistically unusual flags at the current threshold."]
        L += ["", "## Trajectory",
              f"- Second half vs first half of the season: {t.half_delta:+.2f} points/match.",
              f"- Best 6-game run: {traj.ppg_roll[best]:.2f} pts/match ending game {int(traj.game_no[best])}; worst: {traj.ppg_roll[worst]:.2f} ending game {int(traj.game_no[worst])}.",
              f"- Cumulative results vs market expectation: {traj.cum_vs_market.iloc[-1]:+.1f} points across {len(traj)} matches.", "",
              "## Most similar team profiles (all seasons, other clubs)", "", "| # | Team | Season | Similarity | Shared traits |", "|---|---|---|---|---|"]
        for i, r in sim.iterrows():
            L.append(f"| {i + 1} | {r.team} | {r.Season} | {r.similarity:.0f}% | {', '.join(r.shared) or '-'} |")
        bench = self.similar(team, season, 15)
        bench = bench[bench.ppg > t.ppg + 0.15].head(3)
        if len(bench):
            L += ["", "**Benchmarks** (similar profile, but more points - study what they do differently):"]
            L += [f"- {r.team} {r.Season}: {r.ppg:.2f} pts/match vs {t.ppg:.2f}. Differs on: {', '.join(r.differs_on) or 'no single dominant factor'}." for r in bench.itertuples()]
        L += ["", "## Areas to investigate (hypotheses from team-level data - validate with event/player data)"]
        pri = []
        if sc["Attack"] <= 35: pri.append(f"**Goal output** - scores {t.gf_pg:.2f}/match vs league {lg.gf_pg:.2f}. Check chance creation vs finishing (xG, shot quality). Profiles: creative midfielders, progressive carriers, efficient forwards.")
        if sc["Defense"] <= 35: pri.append(f"**Goals conceded** - {t.ga_pg:.2f}/match vs league {lg.ga_pg:.2f}. Check line height, set-piece and transition concessions, goalkeeper shot-stopping. Profiles: ball-winning midfielders, aerially dominant centre-backs, recovery pace.")
        if t.gap > lg.gap + 0.5: pri.append(f"**Away weakness** - home {t.h_ppg:.2f} vs away {t.a_ppg:.2f} pts/match. Check press-resistance and game management on the road. Profiles: press-resistant midfielders, experienced game-managers.")
        if t.gap < lg.gap - 0.5: pri.append(f"**Home underperformance** - home {t.h_ppg:.2f} vs away {t.a_ppg:.2f}. Check breaking down low blocks. Profiles: 1v1 wide players, set-piece specialists, creators.")
        if t.d_rate > lg.d_rate + 0.06: pri.append(f"**Draw-heavy** ({t.d_rate:.0%} of matches vs {lg.d_rate:.0%}) - look for game-changers who convert 1 point into 3 (finishing, late-game impact).")
        if sc["Consistency"] <= 25: pri.append("**Volatile results** - investigate squad depth and rotation, plus injury or suspension patterns behind the swings.")
        if any(f["flag"] in ("Defensive decline", "Attacking decline", "Points slump") for f in fl): pri.append("**Recent decline** - check injuries, tactical change or fixture difficulty behind the recent slide.")
        if any(f["flag"] == "Results outperform market expectation" for f in fl): pri.append("**Overperformance vs market** - results may flatter the underlying quality; avoid over-investing in the current positional mix.")
        if any(f["flag"] == "Results below market expectation" for f in fl): pri.append("**Underperformance vs market** - results trail what odds implied; the process may be better than the table suggests.")
        L += [f"- {x}" for x in pri] or ["- No clear weak spots: profile is at or above league norm on the main axes. Focus on depth and succession planning."]
        strengths = [k for k in ["Attack", "Defense", "Home", "Away", "Consistency"] if sc[k] >= 75]
        if strengths: L += ["", f"**Strengths to protect:** {', '.join(strengths)}."]
        L += ["", "*Limitations: match-level team data only (no xG, positions or player events). Percentile scores are relative to all team-seasons 2010-11 to 2024-25, adjusted for each season's league average.*"]
        return "\n".join(L)
