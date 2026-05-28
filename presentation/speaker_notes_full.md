# Speaker Notes — Full Presentation
## Wind-Assisted Propulsion on North Sea Routes

**Speakers by section:**

| Section | Speaker | Slides |
|---|---|---|
| Part I — Background & Data | Aleksandra Kłos | 1–6 |
| Part II — Methodology & Models | Olga Grigorieva | 7–11 |
| Part III — Rotor Performance | Elen Mouradian | 12–15 |
| Part IV — Route Intelligence | Bartosz Maj | 16–19 |
| Part V — Uncertainty & Conclusions | Hubert Jaczyński | 20–22 |

---

# PART I — Background & Data
## Speaker: Aleksandra Kłos

---

## Slide 1 — Title

Good morning everyone. Our team has been studying wind-assisted propulsion on North Sea cargo routes. Specifically, we looked at Flettner rotors — rotating cylinders mounted on a ship's deck that use wind energy to reduce engine load. The question we set out to answer was: given real AIS ship tracks combined with weather and sea-state data, when do rotors actually help, and by how much?

The project covers three years of data, five analytical extensions, and a complete pipeline from raw data to route optimisation. I will start with the background and data, and then hand over to each team member in turn.

---

## Slide 2 — What this presentation covers

The presentation is structured into five parts. I will cover the background: what Flettner rotors are, why they matter now, the vessel we studied, and the dataset we built. Olga will then take you through the modelling methodology and speed prediction results. Elen covers the rotor performance analysis — how much uplift the rotor delivers, where, and what it means for fuel and emissions. Bartosz explains the route intelligence extensions — how heading direction and routing choices affect rotor opportunity. And Hubert closes with Monte Carlo routing and the overall conclusions.

Please hold questions until the end and we will be happy to go into detail on any part of the analysis.

---

## Slide 3 — Part I divider

---

## Slide 4 — Project context: why Flettner rotors?

The physics behind this technology is the Magnus effect. When a cylinder spins in an airflow, the airspeed on one side increases and on the other decreases, generating a transverse lift force — the same principle that makes a spinning ball curve in flight. On a ship, if part of that force aligns with the vessel's heading, it directly reduces the engine power needed to maintain speed. A single rotor of the type fitted to modern vessels can contribute anywhere from 40 to 300 kilowatts depending on wind speed and the angle the wind arrives from.

The commercial case is strong right now for three reasons. The IMO has committed to a 50 percent reduction in fleet greenhouse gas emissions by 2050 relative to 2008 levels. Heavy fuel oil currently costs between 600 and 900 dollars per tonne, so even a one percent reduction in consumption across a fleet is worth millions annually. And Flettner rotors are retrofittable — you can bolt them onto an existing vessel without redesigning the propulsion system or drydocking for an extended period.

The analytical challenge, however, is that the benefit is not constant. It depends on wind speed, the relative angle of the wind to the vessel heading, wave height as an operational safety limit, and the vessel's own course. That conditional nature is exactly what makes this a data science problem rather than a simple engineering calculation, and it is what our project addresses.

---

## Slide 5 — Vessel and rotor specification

The vessel we are working with is a North Sea cargo ship with a design speed of 11.5 knots and a design brake power of 3,500 kilowatts. Throughout the project we use a cubic resistance model, meaning that required engine power scales with the cube of speed — this is the standard assumption in naval architecture for displacement hulls at normal operating speeds.

The specific fuel oil consumption is 195 grams per kilowatt-hour, and the CO2 conversion factor for heavy fuel oil is 3.114 tonnes of CO2 per tonne of fuel. These two numbers together allow us to convert any power change from the rotor directly into a fuel and emissions estimate.

One key parameter that runs through the entire project is the conversion of 200 kilowatts to approximately one knot at the operating point. This is the linearisation we use to translate rotor power into a speed contribution in the scenario analysis. All vessel parameters and all rotor deactivation conditions — wave height above 6 metres or wind speed above 42 metres per second — are applied consistently across every extension. There are no hidden assumption changes between analyses.

---

## Slide 6 — Data sources and trajectories

The dataset covers exactly three years, from February 2023 to February 2026, and contains 125,160 AIS position observations from 528 voyages. The map on the left shows all vessel tracks plotted on an OpenStreetMap basemap. You can see the routes concentrate along the main North Sea corridors — Rotterdam to Norway, cross-Channel routes, and North Sea basin transits.

For each AIS position fix we attached environmental data from three sources. ERA5 reanalysis gives us 10-metre wind components and gust data. Copernicus Marine Service provides significant wave height, mean wave direction, and wave period. And the Copernicus Ocean Physics product gives us eastward and northward surface current components. All three are attached via spatio-temporal interpolation — we find the nearest grid point and time step in each reanalysis product for each AIS observation.

The split for model training and evaluation is 422 voyages for training, covering 88,346 rows, and 106 voyages for testing, covering 35,758 rows. The split is strictly chronological by voyage — the test set contains only voyages that occurred later than any voyage in the training set. Olga will explain why this matters for the validity of the results.

---

# PART II — Methodology & Models
## Speaker: Olga Grigorieva

---

## Slide 7 — Part II divider

---

## Slide 8 — Feature engineering: leakage-aware design

Thank you Aleksandra. I will walk you through how we turned those raw data layers into model-ready features, and why we were careful about leakage throughout.

Starting with the feature groups. The wind and wave block gives us true wind speed, significant wave height, wave period, a squared wind term to capture the nonlinear drag relationship, and a wave-wind alignment angle that measures how much the swell direction differs from the wind direction. For wind direction we encode the relative wind angle as its sine and cosine rather than the raw angle. This is important: a model that sees angle as a number treats 359 degrees and 1 degree as very different, but they are almost identical physically. Sine and cosine encoding removes that discontinuity entirely.

Apparent wind is different from true wind because the vessel is moving. A ship travelling at 11 knots into a 10-knot headwind experiences 21 knots of apparent wind from dead ahead. A ship travelling at the same speed across a 10-knot beam wind experiences a different speed and angle of apparent wind. We include both the apparent speed and the apparent angle as separate features because this is what the rotor and the vessel's resistance actually respond to, not the meteorological true wind.

For currents we decompose the vector into along-course and cross-course components relative to the vessel's heading. The along-course component directly adds to or subtracts from the effective speed over ground. The cross-course component creates a set that the helmsman must compensate for, which affects both course over ground and indirectly fuel consumption. Keeping these as separate features rather than using the vector magnitude alone preserves that directional information.

The nowcast lag features are the most predictively powerful group: speed over ground at the previous time step, two steps back, and a rolling mean over three steps. These are all computed strictly within each voyage — the rolling window resets at the start of every voyage. The reason these features work so well is that recent SOG encodes everything we cannot observe directly: the engine setting, cargo weight, hull fouling state, whether the vessel is accelerating or decelerating, and operational decisions made by the crew.

Finally, we include interaction terms — wind speed times angle, wave height times wind speed, and current magnitude times course alignment. These allow the model to capture effects that are not additive in the original variables. For example, a strong following current at a beam angle has a very different effect from the same current aligned with the course.

Now the leakage controls, which I consider equally important as the features themselves. First, lag features are computed only from rows that come before the current target observation — the current row's speed never feeds into its own prediction, which would be a textbook case of target leakage. Second, rolling windows reset at voyage boundaries, so there is no information leaking across voyages. Third — and this is a deliberate design choice that carries through the whole project — rotor variables are never used as training features. The rotor is applied as a post-prediction scenario layer. This means our speed predictions are clean baselines, and the rotor effect is always separately attributable and auditable. Fourth, the data split is chronological by voyage, which avoids the situation where the model has seen observations from the same voyage in both training and test.

The result is two completely separate feature sets used in two separate models. The weather-only model answers the explanatory question: how much does the environment drive speed variation? The nowcast model answers the operational question: given what the vessel was doing one step ago, what will its speed be next? These are different scientific questions and they need different evaluation criteria.

---

## Slide 9 — Model hierarchy and rotor scenario methodology

We built four model levels to make the comparison honest and transparent. B1 is a constant — the mean training speed applied to every test observation. It represents zero predictive power beyond knowing the average. B2 is a linear regression on wind speed, wave height, and the sine and cosine of relative wind angle. M1 is XGBoost on the full weather-only feature set. M2 is XGBoost with weather plus all the lagged SOG features.

For XGBoost we used 300 trees, maximum depth 6, learning rate 0.05, and 80 percent subsampling of both rows and columns at each tree. These settings were kept identical for M1 and M2 so that the comparison between them is a fair test of the feature sets, not a test of different hyperparameter choices.

We also trained a quantile regression version of M2 using the same XGBoost framework, fitting separate models at the 2.5th and 97.5th percentiles to produce 95 percent prediction intervals. The mean interval width on the test set is 3.161 knots with an empirical coverage of 96.5 percent — the intervals are slightly conservative, meaning they are a little wider than strictly necessary, but that is the safe direction for any tool used operationally. You would rather have an interval that is slightly too wide than one that is overconfident.

The rotor scenario sits entirely outside the models and is applied as a post-prediction layer. Rotor power for each observation is read from the manufacturer's polar response table, which maps wind speed on one axis and absolute relative wind angle on the other to a kilowatt output. We then apply the formula: delta SOG equals P rotor divided by 200 kilowatts per knot. The rotor is switched off when wave height exceeds 6 metres or wind speed exceeds 42 metres per second, following the operational specification. This design means the baseline model and the rotor-assisted scenario are always directly comparable — there is no confounding between what the model learned and what the rotor contributes.

The weather window criterion — speed above 10 knots and wave height below 3 metres simultaneously — defines periods of good operating conditions. We count contiguous stretches of observations satisfying both conditions as a single window, and compare the number and coverage of windows before and after adding the rotor, to see whether rotor assistance extends or merges favourable operating periods.

---

## Slide 10 — Model results: held-out test performance

The results table tells a very clear story. The constant mean baseline gives an MAE of 1.388 knots and an R-squared close to zero. The linear weather model gives almost identical numbers — 1.377 MAE — confirming that wind and wave variables have very limited linear predictive power for speed over ground on their own. Adding nonlinear weather features with XGBoost in M1 actually performs slightly worse than the linear baseline, with MAE rising to 1.468. This is not a failure of the model — it is a genuine finding: weather variables without operational context do not generalise well to held-out voyages, because the same weather conditions can produce different vessel speeds depending on the engine setting, cargo, and route.

The nowcast model M2 is a completely different story. MAE drops from around 1.4 to 0.406 knots, RMSE drops from around 1.9 to 0.758 knots, R-squared reaches 0.839, and 91.7 percent of predictions land within one knot of the true observed speed. The explanation is that recent SOG is an extremely strong proxy for the vessel's entire operational state. One lagged observation effectively summarises engine setting, cargo weight, hull condition, and current operational intent in a single number.

The practical implication is that this model is strong enough to use as a real-time nowcast on the bridge. In operational use, the previous speed reading is always available, so the model's most powerful feature is always present. The quantile regression intervals give a calibrated uncertainty band around each prediction.

---

## Slide 11 — Feature importance and prediction intervals

The feature importance plots confirm what the metrics already suggested, but add useful detail about the structure of the signal. In the weather-only model on the left, the top-ranked features are course-current interaction terms — the angle between the vessel's heading and the current direction, and the along-course current component. Pure wind and wave features are present in the ranking but are not dominant. The reason is that weather effects are confounded with operational decisions: a vessel that is fighting a headwind may slow down, or the operator may increase engine power to maintain schedule. Without observing the engine setting, the model cannot cleanly separate the weather signal from the operational response.

In the nowcast model on the right, lagged SOG completely dominates. The first and second lag together explain the large majority of the model's predictive power. Apparent wind angle enters the top features as the highest-ranked weather variable, which makes physical sense — it directly affects both the aerodynamic force on the hull and the rotor gain. True wind speed and wave height appear further down the ranking. They are still informative as secondary signals, particularly for identifying periods where the model should widen its uncertainty.

The prediction interval figure in the appendix shows that the 96.5 percent empirical coverage is well calibrated across the range of true speeds. The intervals widen at the extremes — very low speeds and very high speeds — where the model is less certain, as expected. The mean width of 3.161 knots covers the practical operating range while giving the crew useful information about forecast confidence.

---

# PART III — Rotor Performance
## Speaker: Elen Mouradian

---

## Slide 12 — Part III divider

---

## Slide 13 — Rotor scenario: speed uplift and weather-window impact

Thank you Olga. Now let us look at what the rotor actually delivers when we apply it as a scenario overlay on the held-out test voyages.

The first thing to note is how often the rotor was active: 98.1 percent of observations. The deactivation conditions — wave height above 6 metres or wind speed above 42 metres per second — are rarely met during normal North Sea operations. The mean power while active was 41.2 kilowatts, translating to a mean speed contribution of 0.206 knots when active, or 0.202 knots averaged over all observations including the small fraction when the rotor was off.

Point by point, 0.202 knots sounds modest. But the picture becomes more interesting when we look at weather windows. A weather window here is a contiguous stretch of time where speed is above 10 knots and wave height is below 3 metres simultaneously — essentially good operating conditions. Without the rotor there were 866 such windows covering 76.0 percent of test observations. With the rotor there were only 808 windows — 58 fewer — but they covered 77.5 percent of observations. The number of windows went down, but the coverage went up. The rotor is merging adjacent windows that were previously separated by brief marginal periods. The total count of observations inside weather windows increased by 529. That is operationally meaningful because sustained good-condition periods allow more efficient route planning than many short fragmented windows.

---

## Slide 14 — Rotor gain surface: where does it help?

This heatmap is one of the most interpretable outputs of the project. The horizontal axis is the relative wind angle — zero is a headwind, 90 degrees is a beam wind, 180 degrees is a following wind. The vertical axis is wind speed in metres per second. The colour shows the mean rotor power at each combination.

The gain is strongly concentrated at beam and quartering angles, roughly 50 to 130 degrees, above about 8 metres per second. At those conditions rotor power regularly exceeds 100 kilowatts and can reach 200 to 300 kilowatts in the strongest wind conditions. The headwind sector — angles below about 30 degrees — and the following wind sector — above about 150 degrees — deliver very little, often less than 10 kilowatts.

The important lesson here is that good wind speed is necessary but not sufficient. A 15 metre per second wind directly on the bow contributes less rotor power than a 10 metre per second wind on the beam. Sea state appears in this analysis as an operational gate — high wave height switches the rotor off entirely — but within the operating envelope, it is the combination of angle and speed that determines the gain, not wave height.

For operational planning this means the question to ask when looking at a weather forecast is not just "how strong is the wind?" but "what angle will the wind arrive at for each leg of the route?" That directional specificity is what makes route optimisation, which Bartosz will cover, a worthwhile extension of this analysis.

---

## Slide 15 — Extension 1: Energy, fuel, and CO2 impact

Scaling the rotor scenario across the full three-year dataset gives us the cumulative impact numbers. The rotor was active in 97.6 percent of observations — slightly lower than the test-set figure because the full dataset includes some older voyages with more extreme conditions.

The baseline fuel consumption across all 528 voyages was 11,067 tonnes of heavy fuel oil. With the rotor applied, this drops to 10,954 tonnes — a saving of 112.2 tonnes, representing 1.01 percent of total consumption. The corresponding CO2 reduction is 349.4 tonnes, also exactly 1.01 percent, as expected from a fixed CO2 conversion factor.

Annualised across the three-year period, this works out to 37.4 tonnes of fuel saved per year and 116.4 tonnes of CO2 avoided per year for a single vessel. The CII — Carbon Intensity Indicator — improves from 53.75 to 53.20 grams of CO2 per tonne-nautical mile. That is a real regulatory improvement, though the magnitude of the shift depends on total fleet activity.

I want to be clear about what these numbers are. They are scenario estimates derived from a cubic resistance model and the 200 kilowatt per knot conversion assumption. They are not certified sea trial results. The value of this analysis is establishing the order of magnitude and identifying the conditions that drive most of the saving — which then informs where to invest in more detailed measurement. The 1 percent figure is robust to reasonable changes in the model assumptions.

---

# PART IV — Route Intelligence
## Speaker: Bartosz Maj

---

## Slide 16 — Part IV divider

---

## Slide 17 — Extension 2: Route segmentation by heading regime

Thank you Elen. My part covers three route intelligence extensions that ask: can we identify where and when the rotor opportunity concentrates, and can we plan routes to take advantage of it?

The first extension splits all voyages into four heading regimes — northbound, eastbound, southbound, and westbound — using 45-degree bins. The table shows the mean speed over ground and mean rotor speed contribution for each regime. Northbound and eastbound legs are faster on average, around 12 knots, while southbound and westbound legs are slower, around 10.5 to 11 knots.

The rotor contribution is similar across regimes in absolute terms — between 0.185 and 0.209 knots — but the weather window improvement tells a more interesting story. Westbound legs show the largest gain: weather-window coverage increases from 85.1 to 90.1 percent, a 5 percentage point improvement. Southbound legs gain 2.2 points. Northbound and eastbound legs already have high baseline coverage, so the rotor's relative impact there is smaller.

The physical reason is straightforward. The prevailing wind direction in the North Sea is westerly or south-westerly. For a vessel heading west, that wind arrives from ahead or the quarter — angles between roughly 120 and 180 degrees. For a vessel heading east, the same wind arrives from astern, which the gain surface showed is a poor sector. This means the same vessel on the same route at different times of day, or on different legs of a round trip, encounters completely different rotor opportunity just because of heading geometry.

---

## Slide 18 — Extension 3: Route optimisation — Dijkstra proof of concept

The second extension asks whether small deviations from the direct route can improve rotor gain. We built a grid-based routing model over the North Sea where each grid cell's edge cost is rotor-adjusted fuel consumption rather than distance. We then applied Dijkstra's shortest-path algorithm to find the route minimising cumulative fuel cost.

On Voyage 456, Rotterdam to western Norway, the results are striking despite the simple setup. The direct route is 571 nautical miles and achieves a mean rotor power of 1.9 kilowatts — a very low gain because the direct heading puts the wind at a poor angle for most of the voyage. The optimal route according to the algorithm is 568 nautical miles — actually slightly shorter — and achieves a mean rotor power of 7.5 kilowatts. That is a 288 percent relative increase in rotor contribution for a route that is slightly shorter.

The seasonal analysis shows December is the best month for rotor operation on this route, with a 59.8 kilowatt average, compared to 26.6 kilowatts in July. Winter winds are stronger but wave height is also higher, so the gain surface and the deactivation threshold interact seasonally.

This is a proof of concept on a coarse grid built from historical average weather fields, not a forecast. Production use would require operational weather forecast input, a proper navigational chart with land mask and traffic separation zones, safety rules, and compliance with routing regulations. The point of this extension is to show that the mathematical framework works and that the potential gain from heading optimisation is real and quantifiable.

---

## Slide 19 — Extension 4: High-gain moment detection

The third extension identifies the specific conditions under which the rotor delivers its largest contributions, and builds a tool to predict those moments from a weather forecast.

We defined a high-gain moment as any observation where the rotor contributes at least 0.5 knots of speed — roughly 100 kilowatts of power. Across the full dataset, 14,856 observations meet this criterion, representing 11.9 percent of all data points. These are relatively rare, but they carry a disproportionate share of the total rotor value. The median wind speed in high-gain moments is 11.5 metres per second and the median relative angle is 82 degrees, with 80 percent of observations falling between 50 and 113 degrees. This matches very precisely what the gain surface showed.

We trained an XGBoost binary classifier to predict whether a given weather observation will produce a high-gain moment, and achieved 99.8 percent accuracy, 99.4 percent precision, 99.0 percent recall, F1 of 0.992, and a ROC-AUC of 1.000. These numbers are unusually strong, and the reason is that high-gain moments are very cleanly defined by the wind speed and angle conditions — the classifier is essentially learning the gain surface boundary.

But the most operationally useful output is not the machine learning model — it is the transparent rule. Wind speed at or above 8 metres per second with a relative angle between 50 and 130 degrees achieves 95.8 percent recall and 94.2 percent accuracy. A navigator can apply this rule using a 24-hour weather forecast without any software. It translates directly into a bridge-level decision: when the forecast shows these conditions on the next leg, engage and plan around the rotor. When it does not, the rotor will contribute little and route planning should focus on other factors.

---

# PART V — Uncertainty & Conclusions
## Speaker: Hubert Jaczyński

---

## Slide 20 — Part V divider

---

## Slide 21 — Extension 5: Monte Carlo routing under uncertainty

Thank you Bartosz. The Dijkstra extension assumed we know future weather exactly. My extension relaxes that assumption by introducing uncertainty explicitly through Monte Carlo simulation.

The routing graph is land-masked and built from historical AIS density data to ensure navigable edges — 537 nodes and 3,784 edges covering the North Sea. We also apply corridor overrides near ports and narrow passages where the grid geometry would otherwise produce unrealistic paths. Importantly, we detect the actual voyage structure: rather than treating every voyage as a single origin-to-destination trip, we identify intermediate port stops using gaps in the AIS signal. For Voyage 364 we detect four sailing legs and three port stop periods. The routing optimisation is applied to each sailing leg independently, preserving the real itinerary.

Instead of optimising against a single weather scenario, we sample 1,000 weather realisations and compute the expected fuel cost under each. The route that minimises expected cost across scenarios is not necessarily the shortest route — it is the route that consistently positions the vessel in good rotor conditions across a range of possible futures.

The results for Voyage 364 show four options. The observed route used 50.72 tonnes of fuel and saved 0.88 tonnes with the rotor. The shortest feasible route through the navigable graph used 48.47 tonnes. The Monte Carlo expected-cost route used 47.81 tonnes, saving 1.22 tonnes with the rotor — 38 percent more saving than the observed route despite being shorter. The scenario upper bound, the best single scenario found, reaches 46.75 tonnes with 1.70 tonnes saved. The Monte Carlo route beats the shortest feasible path in 85.6 percent of the 1,000 simulated scenarios, and the CO2 saving reaches up to 5.3 tonnes per voyage in the upper bound case.

The key message is that uncertainty-aware routing is genuinely better than shortest-path routing under a rotor-gain objective, and we can now quantify how often and by how much.

---

## Slide 22 — Conclusion

Pulling everything together across all five parts.

On modelling: the lagged nowcast XGBoost achieves MAE 0.406 knots, R-squared 0.839, and 91.7 percent of predictions within one knot on held-out test voyages. Weather-only models are limited by missing operational variables — without knowing engine setting, draught, or cargo, the weather signal alone cannot fully explain speed variation. The chronological voyage split ensures we are testing genuine future generalisation, not interpolation within a voyage.

On rotor performance: the benefit is real but conditional. The rotor was active in 98 percent of observations but only 11.9 percent of moments qualify as high-gain. The practical operating rule is wind at or above 8 metres per second with a relative angle between 50 and 130 degrees. Outside that envelope the rotor contributes little. The gain surface makes this spatial and directional pattern explicit.

On operational impact: the three-year scenario estimate is 112 tonnes of fuel and 349 tonnes of CO2 avoided across 528 voyages — approximately 37 tonnes of fuel and 116 tonnes of CO2 per year for one vessel. The CII rating improves from 53.75 to 53.20. Westbound legs benefit most because the heading geometry aligns with prevailing North Sea winds.

On route intelligence: small deviations from the direct route can produce a 288 percent relative increase in rotor gain. Monte Carlo routing outperforms shortest-path routing in 86 percent of simulated scenarios. Segmenting by heading regime shows the same vessel on a westbound leg gains five percentage points of weather-window coverage that it would not gain on an eastbound leg.

The overall conclusion is that wind-assisted propulsion is an opportunity map, not a constant bonus. The project has built the tools to read that map — from prediction to scenario analysis to routing — using a transparent, auditable, and reproducible pipeline.

---

## Slide 23 — Thank you

Thank you for your attention. The appendix contains the prediction interval calibration plot, the full rotor polar diagram and seasonal variation charts, voyage-level opportunity maps, the classifier confusion matrix and ROC curve, the wave-height interaction surface, and the complete rotor scenario formula with all assumptions stated. We are happy to take questions on any part of the analysis.
