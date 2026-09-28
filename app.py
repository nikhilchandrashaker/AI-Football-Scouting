"""Streamlit dashboard:  streamlit run app.py"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from scout_engine import ScoutEngine, SCORE_COLS
import plots

st.set_page_config(page_title="Football Scouting Dashboard", page_icon="🕵️", layout="wide")
CSV = Path(__file__).parent / "data" / "England_1011_2425.csv"


@st.cache_resource(show_spinner="Building scouting engine...")
def get_engine():
    e = ScoutEngine(CSV)
    e.fit_match_model()
    return e


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


e = get_engine()
st.title("🕵️ Football Scouting & Recruitment Dashboard")
st.caption("Premier League 2010-11 to 2024-25 · team-level data · similarity scores measure closeness to a profile, not quality.")

with st.sidebar:
    season = st.selectbox("Season", e.seasons[::-1])
    team = st.selectbox("Team", e.teams(season))
    as_of = st.slider("Analyse as of game #", 16, 38, 38, help="Rolls the flags back to an earlier point in the season.")
    zt = st.slider("Flag sensitivity (z-threshold)", 0.75, 2.5, 1.25, 0.25, help="Lower = more flags.")

tabs = st.tabs(["📊 Profile", "🔎 Form Scout", "🏠 Home/Away", "📈 Trajectory", "🧩 Styles", "👯 Similarity", "🎯 Scout Search", "🤖 Match Model", "🕵️ Report"])

with tabs[0]:
    p = e.profile(team, season)
    c1, c2 = st.columns([1, 1])
    with c1:
        show(plots.radar(p["scores"], f"{team} {season}"))
    with c2:
        st.code(e.profile_bars(team, season))
        st.write(f"**League position (on-pitch points):** {p['rank']}/{p['n_teams']} · **Style cluster:** {p['raw'].cluster}")
        for k, v in e.summary_lines(team, season).items():
            st.write(f"**{k}:** {v}")
    r, lg = p["raw"], p["league"]
    st.dataframe(pd.DataFrame({"Metric": ["Points/match", "Goals for/match", "Goals against/match", "Home pts/match", "Away pts/match", "Draw rate", "Form (last 6)"],
                               team: [r[c] for c in ["ppg", "gf_pg", "ga_pg", "h_ppg", "a_ppg", "d_rate", "form6"]],
                               "League avg": [lg[c] for c in ["ppg", "gf_pg", "ga_pg", "h_ppg", "a_ppg", "d_rate", "form6"]]}).round(2), hide_index=True)
    st.subheader(f"{team} across seasons")
    st.line_chart(e.team_history(team).set_index("Season")[["ppg", "gf_pg", "ga_pg"]])

with tabs[1]:
    st.subheader(f"Scouting flags: {team} {season} (after {as_of} games)")
    fl = e.scout_flags(team, season, as_of, zt)
    if not fl: st.info("Nothing statistically unusual at this sensitivity. Lower the threshold in the sidebar.")
    for f in fl:
        st.markdown(f"{f['icon']} **{f['flag']}** · *{f['category']}* · z={f['z']:+.1f}  \n{f['detail']}")
    st.subheader("League-wide flag table")
    st.dataframe(e.league_flags(season, as_of, zt).sort_values("red", ascending=False), hide_index=True, use_container_width=True)
    st.caption("🔴 concern · 🟢 positive · 🟡 worth investigating. z = how unusual vs the same metric across all 15 seasons at the same point in the season.")

with tabs[2]:
    show(plots.home_away_scatter(e, season, team))
    d = e.ts[e.ts.Season == season][["team", "h_ppg", "a_ppg", "gap", "h_ga_pg", "a_ga_pg"]].sort_values("gap", ascending=False).round(2)
    st.dataframe(d.rename(columns={"h_ppg": "home pts", "a_ppg": "away pts", "gap": "home-away gap", "h_ga_pg": "home GA", "a_ga_pg": "away GA"}), hide_index=True)

with tabs[3]:
    show(plots.trajectory_fig(e, team, season))
    tj = e.trajectory(team, season); tj["Date"] = tj.Date.dt.date
    st.dataframe(tj.round(2), hide_index=True)

with tabs[4]:
    show(plots.cluster_fig(e, (season, team)))
    st.write(f"K-Means with k={e.k} (silhouette {e.silhouette:.2f}) on {e.pca.n_components_} principal components explaining 90% of variance.")
    st.dataframe(e.ts[e.ts.Season == season][["team", "cluster", "ppg", "gf_pg", "ga_pg"]].sort_values("ppg", ascending=False).round(2), hide_index=True)
    with st.expander("Hierarchical clustering cross-check"):
        show(plots.dendrogram_fig(e, season))

with tabs[5]:
    st.subheader(f"Teams that play like {team} {season}")
    c1, c2, c3 = st.columns(3)
    scope = c1.radio("Compare against", ["All seasons", "Same season"], horizontal=True)
    k = c2.slider("How many", 3, 10, 5)
    same = c3.checkbox("Include the club's own other seasons", False)
    sim = e.similar(team, season, k, "season" if scope == "Same season" else "all", same)
    left, right = st.columns([1, 1])
    with left:
        show(plots.radar(e.profile(team, season)["scores"], f"{team} {season}",
                         overlay=e.scores.loc[(sim.Season[0], sim.team[0])].to_dict(), overlay_label=f"{sim.team[0]} {sim.Season[0]}"))
    with right:
        show(plots.cluster_fig(e, (season, team), sim))
    for i, r in sim.iterrows():
        st.markdown(f"**{i + 1}. {r.team} {r.Season}** · {r.similarity:.0f}% similar · {r.ppg:.2f} pts/match  \n"
                    f"✓ {' · '.join(r.shared) or 'general overall shape'}  \n" + (f"↔ differs on: {', '.join(r.differs_on)}" if r.differs_on else ""))

with tabs[6]:
    st.subheader("Describe the profile you want")
    st.caption("Match % = closeness to your request. It is not a claim that one team is better.")
    cols = st.columns(4)
    crit = {}
    names = ["Attack", "Defense", "Home", "Away", "Goal difference", "Consistency", "Form", "Trajectory"]
    for i, n in enumerate(names):
        with cols[i % 4]:
            lv = st.selectbox(n, ["(ignore)"] + list(e.LEVELS), key="lv" + n, index=0)
            if lv != "(ignore)":
                crit[n] = (e.LEVELS[lv], st.slider("importance", 1, 3, 1, key="w" + n))
    scope = st.radio("Search in", ["All seasons", f"{season} only"], horizontal=True)
    if crit:
        res = e.scout_search(crit, None if scope == "All seasons" else season, 12)
        res["match_%"] = res["match_%"].map("{:.0f}%".format)
        st.dataframe(res, hide_index=True, use_container_width=True)
    else:
        st.info("Pick at least one criterion.")

with tabs[7]:
    st.subheader("Match outcome model (analytical component)")
    st.caption(f"Walk-forward validation: each season is predicted using only earlier seasons ({e.eval.attrs['n_test_matches']} test matches).")
    st.dataframe(e.eval.round(4))
    show(plots.model_fig(e))
    st.write("**Reading this:** team stats alone beat the class-frequency baseline but not the bookmaker, and adding odds gives essentially the bookmaker's accuracy. The model is a sanity check on the profile features, not a betting edge.")
    st.subheader("Hypothetical fixture")
    c1, c2 = st.columns(2)
    h = c1.selectbox("Home", e.teams(season), index=e.teams(season).index(team))
    a = c2.selectbox("Away", [t for t in e.teams(season) if t != h])
    pr = e.predict_matchup(h, a, season, as_of)
    st.bar_chart(pd.Series(pr, name="probability"))
    with st.expander("Model drivers (standardised logistic coefficients)"):
        st.dataframe(e.importance.round(3), hide_index=True)

with tabs[8]:
    md = e.scouting_report(team, season, as_of)
    st.markdown(md)
    st.download_button("Download report (.md)", md, file_name=f"scouting_{team}_{season}.md".replace(" ", "_"))
