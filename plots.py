"""plots.py - matplotlib figures shared by the Streamlit app and the batch pipeline."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scout_engine import SCORE_COLS, FEATS

PAL = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def radar(scores: dict, title="", overlay=None, overlay_label=""):
    keys = [k for k in SCORE_COLS if k in scores and k != "Trajectory"]
    ang = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    fig, ax = plt.subplots(figsize=(4.6, 4.6), subplot_kw=dict(polar=True))
    for sc, lab, col in [(scores, title, PAL[0])] + ([(overlay, overlay_label, PAL[1])] if overlay else []):
        v = [sc[k] for k in keys]
        ax.plot(ang + ang[:1], v + v[:1], color=col, lw=2, label=lab)
        ax.fill(ang + ang[:1], v + v[:1], color=col, alpha=0.18)
    ax.set_xticks(ang); ax.set_xticklabels(keys, fontsize=9); ax.set_ylim(0, 100); ax.set_yticks([25, 50, 75]); ax.set_yticklabels([])
    if overlay: ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.set_title(title, fontsize=11, pad=14)
    return fig


def home_away_scatter(e, season, highlight=None):
    d = e.ts[e.ts.Season == season]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(d.a_ppg, d.h_ppg, c=[PAL[0]] * len(d), s=45)
    lim = [0, max(d.h_ppg.max(), d.a_ppg.max()) + 0.2]
    ax.plot(lim, lim, "--", c="grey", lw=1); ax.axvline(d.a_ppg.mean(), c="lightgrey", lw=1); ax.axhline(d.h_ppg.mean(), c="lightgrey", lw=1)
    for r in d.itertuples():
        ax.annotate(r.team, (r.a_ppg, r.h_ppg), fontsize=7, xytext=(3, 3), textcoords="offset points",
                    weight="bold" if r.team == highlight else None, color=PAL[1] if r.team == highlight else "black")
    if highlight:
        h = d[d.team == highlight].iloc[0]; ax.scatter([h.a_ppg], [h.h_ppg], s=140, facecolors="none", edgecolors=PAL[1], lw=2)
    ax.set_xlabel("Away points / match"); ax.set_ylabel("Home points / match"); ax.set_xlim(lim[0], lim[1] - 0.2 + 0.2 * 0); ax.set_ylim(lim)
    ax.set_title(f"Home vs away performance, {season}\n(above dashed line = better at home than away)", fontsize=10)
    return fig


def trajectory_fig(e, team, season):
    t = e.trajectory(team, season)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(t.game_no, t.ppg_roll, c=PAL[0], lw=2, label="rolling 6-game"); ax[0].plot(t.game_no, t.cum_ppg, c=PAL[4], lw=1.5, label="season to date")
    ax[0].set_title("Points per match"); ax[0].legend(fontsize=8)
    ax[1].plot(t.game_no, t.gf_roll, c=PAL[2], lw=2, label="scored"); ax[1].plot(t.game_no, t.ga_roll, c=PAL[1], lw=2, label="conceded")
    ax[1].set_title("Goals per match (rolling 6)"); ax[1].legend(fontsize=8)
    ax[2].bar(t.game_no, t.pts - t.xpts, color=np.where(t.pts - t.xpts >= 0, PAL[2], PAL[1]), alpha=0.6)
    ax[2].plot(t.game_no, t.cum_vs_market, c="black", lw=2); ax[2].axhline(0, c="grey", lw=0.8)
    ax[2].set_title("Points vs market expectation (bars: per match, line: cumulative)")
    for a in ax: a.set_xlabel("Games played")
    fig.suptitle(f"{team} {season}", y=1.02); fig.tight_layout()
    return fig


def cluster_fig(e, highlight=None, neighbours=None):
    f = e.pca_frame()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for i, (c, g) in enumerate(f.groupby("cluster")):
        ax.scatter(g.PC1, g.PC2, s=22, alpha=0.55, label=c, c=PAL[i % len(PAL)])
    if neighbours is not None:
        for r in neighbours.itertuples():
            p = f[(f.Season == r.Season) & (f.team == r.team)].iloc[0]
            ax.scatter(p.PC1, p.PC2, s=90, facecolors="none", edgecolors="black"); ax.annotate(f"{r.team} {r.Season[2:]}", (p.PC1, p.PC2), fontsize=7, xytext=(4, 4), textcoords="offset points")
    if highlight:
        p = f[(f.Season == highlight[0]) & (f.team == highlight[1])].iloc[0]
        ax.scatter(p.PC1, p.PC2, s=170, marker="*", c="gold", edgecolors="black", zorder=5)
        ax.annotate(f"{highlight[1]} {highlight[0][2:]}", (p.PC1, p.PC2), fontsize=8, weight="bold", xytext=(6, 6), textcoords="offset points")
    ev = e.pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.0%} var)"); ax.set_ylabel(f"PC2 ({ev[1]:.0%} var)"); ax.legend(fontsize=7, loc="best")
    ax.set_title("Team-season style map (PCA of 12 profile features, K-Means clusters)", fontsize=10)
    return fig


def dendrogram_fig(e, season):
    idx = np.where(e.ts.Season.values == season)[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    dendrogram(linkage(e.Z[idx], "ward"), labels=e.ts.team.values[idx], leaf_rotation=75, leaf_font_size=8, ax=ax, color_threshold=0)
    ax.set_title(f"Hierarchical (Ward) clustering of {season} teams"); fig.tight_layout()
    return fig


def model_fig(e):
    ev = e.eval
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ev["log_loss"].sort_values(ascending=False).plot.barh(ax=ax[0], color=PAL[0]); ax[0].set_title("Log loss (lower = better)")
    ev["accuracy"].sort_values().plot.barh(ax=ax[1], color=PAL[2]); ax[1].set_title("Accuracy")
    ax[1].set_xlim(0.4, 0.6); ax[0].set_xlim(0.9, 1.08)
    for m, c in [("Logistic (team stats)", PAL[0]), ("Bookmaker (Bet365 odds)", PAL[1])]:
        cal = e.calibration(m); ax[2].plot(cal.pred, cal.actual, "o-", c=c, label=m)
    ax[2].plot([0, 1], [0, 1], "--", c="grey"); ax[2].set_title("Home-win calibration"); ax[2].set_xlabel("predicted"); ax[2].set_ylabel("actual"); ax[2].legend(fontsize=7)
    for a in ax[:2]: a.set_ylabel("")
    fig.tight_layout()
    return fig


def eda_fig(e):
    m, ts = e.matches, e.ts
    g = m.groupby("Season")
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    res = g.FTR.value_counts(normalize=True).unstack()[["H", "D", "A"]]
    res.plot(ax=ax[0, 0], marker="o", color=[PAL[0], "grey", PAL[1]]); ax[0, 0].set_title("Result share by season"); ax[0, 0].tick_params(axis="x", rotation=60)
    (g.FTHG.mean() + g.FTAG.mean()).plot(ax=ax[0, 1], marker="o", c=PAL[2]); ax[0, 1].set_title("Goals per match"); ax[0, 1].tick_params(axis="x", rotation=60)
    ax[1, 0].scatter(ts.gd_pg * 38, ts.on_pitch_pts / ts.games * 38, s=14, alpha=0.6); ax[1, 0].set_xlabel("Goal difference (per 38)"); ax[1, 0].set_ylabel("Points (per 38)"); ax[1, 0].set_title("Goal difference vs points")
    ax[1, 1].hist(ts.ppg, bins=20, color=PAL[0], alpha=0.7); ax[1, 1].set_title("Points/match across 300 team-seasons")
    fig.tight_layout()
    return fig
