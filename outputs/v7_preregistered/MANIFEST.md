| file | notebook | cell | description |
|---|---|---|---|
| bootstrap_check.csv | 07_results_CPU | bootstrap | full bootstrap vs approximations: ratio of bootstrap SD of the 2024 level to its s.e., and median 5-95% widths (bootstrap vs approximation) for E1, E2 and E3 for every metric |
| bootstrap_subsample.csv | 06_sensitivity_CPU | subsample | pre-set 20% subsample for the full refitting bootstrap (fixed seed) |
| data_flow.csv | 02_build_CPU | data flow | series counts after each inclusion rule, by pollutant |
| download_log.csv | 01_download_CPU | download | EEA E1a files per country, failed files, daily rows, download time |
| environment.jsonl | all notebooks | first cell | package versions, CPU count, seed per notebook run |
| figures/fig_e1_maps.pdf | 07_results_CPU | fig_e1_maps | Station maps of E1 (weather-standardised attainment probability at the 2024 level) |
| figures/fig_e1_maps.png | 07_results_CPU | fig_e1_maps | Station maps of E1 (weather-standardised attainment probability at the 2024 level) (PNG copy) |
| figures/fig_e3_headroom.pdf | 07_results_CPU | fig_e3_headroom | Change in the weather-adjusted level needed for a 90% probability of attaining both metrics (negative = reduction needed) |
| figures/fig_e3_headroom.png | 07_results_CPU | fig_e3_headroom | Change in the weather-adjusted level needed for a 90% probability of attaining both metrics (negative = reduction needed) (PNG copy) |
| figures/fig_e5_variance.pdf | 07_results_CPU | fig_e5_variance | Median shares of the 2030 predictive variance from trend, weather and residual noise |
| figures/fig_e5_variance.png | 07_results_CPU | fig_e5_variance | Median shares of the 2030 predictive variance from trend, weather and residual noise (PNG copy) |
| figures/fig_hindcast_skill.pdf | 07_results_CPU | fig_hindcast_skill | Brier skill score for joint attainment by horizon, scenario and baseline, with 90% two-way cluster-bootstrap intervals |
| figures/fig_hindcast_skill.png | 07_results_CPU | fig_hindcast_skill | Brier skill score for joint attainment by horizon, scenario and baseline, with 90% two-way cluster-bootstrap intervals (PNG copy) |
| figures/fig_marginal_by_country.pdf | 07_results_CPU | fig_marginal_by_country | Share of stations with E1 (both metrics) between 0.1 and 0.9, by member state and pollutant |
| figures/fig_marginal_by_country.png | 07_results_CPU | fig_marginal_by_country | Share of stations with E1 (both metrics) between 0.1 and 0.9, by member state and pollutant (PNG copy) |
| figures/fig_reliability.pdf | 07_results_CPU | fig_reliability | Reliability diagrams for joint attainment (1 year ahead, level frozen; 6 years ahead, trend scenario) |
| figures/fig_reliability.png | 07_results_CPU | fig_reliability | Reliability diagrams for joint attainment (1 year ahead, level frozen; 6 years ahead, trend scenario) (PNG copy) |
| figures/fig_worked_examples.pdf | 07_results_CPU | fig_worked_examples | Weather-adjusted yearly levels for the two pre-specified example stations |
| figures/fig_worked_examples.png | 07_results_CPU | fig_worked_examples | Weather-adjusted yearly levels for the two pre-specified example stations (PNG copy) |
| gam_subsample.csv | 06_sensitivity_CPU | subsample | pre-set 10% subsample for the GAM sensitivity (fixed seed) |
| hindcast_NO2_rows.csv.gz | 05_hindcast_no2_CPU | hindcast | one row per series x origin x horizon x scenario: forecast probabilities, observed outcomes, baselines, CRPS, PIT, RPS |
| hindcast_PM10_rows.csv.gz | 05_hindcast_pm10_CPU | hindcast | one row per series x origin x horizon x scenario: forecast probabilities, observed outcomes, baselines, CRPS, PIT, RPS |
| hindcast_PM2.5_rows.csv.gz | 05_hindcast_pm25_CPU | hindcast | one row per series x origin x horizon x scenario: forecast probabilities, observed outcomes, baselines, CRPS, PIT, RPS |
| hindcast_coverage.csv | 07_results_CPU | coverage | 80%/90% interval coverage, mean CRPS (annual mean), mean RPS and PIT KS distance (exceedance days scored on observed days) by set x pollutant x scenario x horizon |
| hindcast_scores.csv | 07_results_CPU | scores | Brier score, binned decomposition (+ remainder) and Brier skill scores vs climatology/persistence/naive trend with 90% two-way cluster-bootstrap CIs, by set (hindcast / 2025 check) x pollutant x scenario x horizon x metric |
| hyperparams.json | 03_pilot_CPU | tuning | global LightGBM hyperparameters chosen on the 1% pilot (test on the latest year <= 2016 with >=100 usable days, fit on earlier years; day-to-day RMSE of log concentration) |
| main_NO2_station_results.csv | 04_main_no2_CPU | main | station-level E1 (with 90% interval), E2 (WLS, Theil-Sen, with 2020), E3 headroom, E5 variance parts, yearly weather-adjusted levels |
| main_PM10_station_results.csv | 04_main_pm10_CPU | main | station-level E1 (with 90% interval), E2 (WLS, Theil-Sen, with 2020), E3 headroom, E5 variance parts, yearly weather-adjusted levels |
| main_PM2.5_station_results.csv | 04_main_pm25_CPU | main | station-level E1 (with 90% interval), E2 (WLS, Theil-Sen, with 2020), E3 headroom, E5 variance parts, yearly weather-adjusted levels |
| mc_error.csv | 06_sensitivity_CPU | mc error | SD across 3 independent integrations at a fixed fit, for every reported estimator (2% subsample) |
| mc_error_summary.csv | 06_sensitivity_CPU | mc error | median Monte Carlo SD per estimator and pollutant |
| missingness_audit.csv | 02_build_CPU | missingness | share of missing days by pollutant and season (2013-2024), Spearman correlation of station-year level with missing share |
| no2_hourly_observed.csv | 02_build_CPU | NO2 1-hour | observed share of included NO2 stations meeting the 1-hour limit (<=3 hours >200) per year, valid station-years only |
| pilot_hp_grid.csv | 03_pilot_CPU | tuning | mean held-out RMSE for each hyperparameter setting on the pilot |
| pilot_sample.csv | 03_pilot_CPU | sample | 1% pilot sample stratified by country x pollutant x station type |
| pilot_timing.csv | 03_pilot_CPU | timing | pilot run time and projected CPU hours for the full main and hindcast runs |
| posthoc_diag_crps.csv | 08_posthoc_diagnostics_CPU | 3 | POST-HOC D4: CRPS blow-up rows and robust median CRPS |
| posthoc_diag_errors.csv | 08_posthoc_diagnostics_CPU | 1 | POST-HOC D1/D2: signed log error, spread ratio and bias-removed 90% coverage of the annual-mean prediction, by pollutant x scenario x horizon |
| posthoc_diag_perfect_weather.csv | 08_posthoc_diagnostics_CPU | 2 | POST-HOC D3: Brier score of joint attainment with resampled vs perfect (target-year) weather |
| reliability_bins.csv | 07_results_CPU | reliability | reliability diagram bins for joint attainment |
| sens_bootstrap.csv | 06_sensitivity_CPU | bootstrap | full refitting bootstrap (50 reps) of E1, E2, E3 and the 2024 level, 20% subsample (§3.1) |
| sens_breaks.csv | 06_sensitivity_CPU | breaks | stations excluded only for a relocation or method change, refit with a trend break (§3.6 item 7) |
| sens_gam_hindcast_h1.csv | 06_sensitivity_CPU | gam_hindcast_h1 | GAM (M2), 1-year-ahead hindcast rows, 10% subsample (§3.6 item 1) |
| sens_gam_main.csv | 06_sensitivity_CPU | gam_main | GAM (M2) instead of LightGBM, main estimates, 10% subsample (§3.6 item 1) |
| sensitivity_summary.csv | 07_results_CPU | sensitivity | every pre-specified sensitivity analysis next to the main E1/E2 (joint) on the same series (§3.6) |
| station_manifest.csv | 02_build_CPU | manifest | every station x pollutant series with sampling point, metadata, met match and exclusion reason |
| success_criteria.csv | 07_results_CPU | criteria | pre-specified success criteria (BSS lower 90% bound > 0 vs all baselines; 90% coverage in 85-95%), horizons 5-6 pooled, by pollutant x scenario |
| summary_by_group.csv | 07_results_CPU | summary | weather-marginal shares and mean E1 (joint) by country, station type and area |
| summary_by_pollutant.csv | 07_results_CPU | summary | E1/E2/E3 summaries by pollutant and metric (headline numbers) |
| summary_dust_sanding.csv | 07_results_CPU | summary | E1 joint by dust-affected / sanding / other member states (§3.6 item 6a) |
| summary_e5.csv | 07_results_CPU | summary | median shares of 2030 variance due to trend, weather and residual (both orders) |
| worked_examples.csv | 07_results_CPU | examples | The two example stations chosen by the pre-specified median rule |
