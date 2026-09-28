# Football Scouting & Recruitment Dashboard (team-level)

Premier League 2010-11 to 2024-25 (4,036 matches, 300 team-seasons). Answers: *given a team's profile, which teams/patterns should we scout and why?*

## Run
```bash
pip install -r requirements.txt
python run_pipeline.py     # tables, figures, model metrics, sample reports -> outputs/
streamlit run app.py       # interactive dashboard
```

## Modules (all in `scout_engine.py`)
| Module | Method |
|---|---|
| Team Profile | Season-adjusted percentile scores (Overall, Attack, Defense, Home, Away, Consistency, Form, Trajectory) vs all 300 team-seasons |
| Form Scout | Last-6-game vs earlier-season attack/defence/points, plus results vs Bet365-implied expected points; flagged when z >= 1.25 against the pooled distribution at the same point of the season |
| Home/Away Scout | Home and away points, scoring and conceding, and the home-away gap, each z-scored |
| Trajectory | Rolling 6-game points/goals, cumulative points vs market expectation |
| Style Clustering | 12 features, centred on each season's league average, standardised, PCA (90% variance), K-Means (k by silhouette, 4-6), Ward hierarchical cross-check |
| Similarity Engine | Euclidean nearest neighbours in PCA space; similarity % = 1 - distance / 95th-percentile pairwise distance; shared/differing traits explained |
| Scout Search | Weighted closeness of percentile scores to a requested profile. The % measures match to the request, not quality |
| ML Model | Multinomial logistic regression and gradient boosting on pre-match features, walk-forward by season, compared with the bookmaker and a class-frequency baseline |
| Scouting Report | Markdown report: profile, flags, similar teams, benchmarks, and "areas to investigate" hypotheses |

## Data notes
* `;`-delimited; 2 blank rows and 1 duplicate removed; `Leeds United`→`Leeds`, `Sheffield`→`Sheffield Utd`.
* Season derived from date (Aug-Jul) so the COVID-delayed 2019-20 finish stays in 2019-20.
* Rows begin at each team's 11th game, so per-match rolling views start there; season totals still cover the full season.
* Points recomputed as 3W+D because raw `PTS` includes the Everton and Nottingham Forest 2023-24 deductions.
* Odds are converted to probabilities with the margin removed.

## Model results (3,226 test matches, walk-forward 2013-14..2024-25)
| Model | Accuracy | Log loss |
|---|---|---|
| Class-frequency baseline | 45.7% | 1.060 |
| Logistic (team stats) | 53.5% | 0.986 |
| Gradient boosting (team stats) | 50.4% | 1.015 |
| Bookmaker (Bet365) | 55.5% | 0.958 |
| Logistic (stats + odds) | 55.4% | 0.965 |

Team stats carry real signal but don't beat the market; boosting overfits at this sample size. Treat the model as an analytical component, not a betting edge.

## Limits
Team-level data only: no xG, positions or player events. "Areas to investigate" are hypotheses to test with player/event data (Phases 4-5 of the plan: add a player dataset such as FBref and reuse the same profile → similarity → search pattern).
