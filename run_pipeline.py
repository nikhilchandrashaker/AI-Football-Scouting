"""Batch pipeline: python run_pipeline.py  -> writes figures, tables, model metrics and sample reports to outputs/."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scout_engine import ScoutEngine
import plots

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"; (OUT / "reports").mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
e = ScoutEngine(ROOT / "data" / "England_1011_2425.csv")
m = e.matches
print(f"matches: {len(m)} | seasons: {len(e.seasons)} | team-seasons: {len(e.ts)} | result split: {m.FTR.value_counts(normalize=True).round(3).to_dict()}")
print(f"PCA components: {e.pca.n_components_} | k-means k={e.k} silhouette={e.silhouette:.2f}")

e.ts.drop(columns=["n_rows"]).round(3).to_csv(OUT / "team_season_profiles.csv", index=False)
e.scores.round(1).to_csv(OUT / "profile_scores.csv")
latest = e.latest_season()
e.league_flags(latest).to_csv(OUT / f"league_flags_{latest}.csv", index=False)

ev = e.fit_match_model()
ev.round(4).to_csv(OUT / "model_metrics.csv")
json.dump({"n_test_matches": ev.attrs["n_test_matches"], "metrics": ev.round(4).to_dict("index")}, open(OUT / "model_metrics.json", "w"), indent=2)
e.importance.round(3).to_csv(OUT / "model_feature_importance.csv", index=False)
print(ev.round(4).to_string())

figs = {"eda": plots.eda_fig(e), "model": plots.model_fig(e), "clusters": plots.cluster_fig(e), "home_away": plots.home_away_scatter(e, latest),
        "dendrogram": plots.dendrogram_fig(e, latest)}
for t in ["Arsenal", "Aston Villa"]:
    s = t.replace(" ", "_")
    figs[f"trajectory_{s}"] = plots.trajectory_fig(e, t, latest)
    figs[f"radar_{s}"] = plots.radar(e.profile(t, latest)["scores"], f"{t} {latest}")
    (OUT / "reports" / f"scouting_{s}_{latest}.md").write_text(e.scouting_report(t, latest), encoding="utf-8")
for n, f in figs.items():
    f.savefig(OUT / "figures" / f"{n}.png", dpi=110, bbox_inches="tight"); plt.close(f)
print("done ->", OUT)
