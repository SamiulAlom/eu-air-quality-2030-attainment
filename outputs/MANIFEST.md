| file | notebook | cell | description |
|---|---|---|---|
| bootstrap_check.csv | 07_results_CPU | bootstrap | full bootstrap vs approximations: ratio of bootstrap SD of the 2024 level to its s.e., and median 5-95% widths (bootstrap vs approximation) for E1, E2 and E3 for every metric |
| bootstrap_subsample.csv | 06_sensitivity_CPU | subsample | pre-set 20% subsample for the full refitting bootstrap (fixed seed) |
| data_flow.csv | 02_build_CPU | data flow | series counts after each inclusion rule, by pollutant |
| download_log.csv | 01_download_CPU | download | EEA E1a files per country, failed files, daily rows, download time |
| environment.jsonl | all notebooks | first cell | package versions, CPU count, seed per notebook run |
| exploratory_recal_coverage.csv | 09_exploratory_recalibration_CPU | 2 | EXPLORATORY recalibration, not validated (post-hoc, deviation 16): 80%/90% coverage and median 90% width, stored vs widened intervals, on origins >= 2020 and the 2025 check |
| exploratory_recal_factors.csv | 09_exploratory_recalibration_CPU | 2 | EXPLORATORY recalibration, not validated (post-hoc, deviation 16): split-conformal widening factors k80/k90 per rows version x pollutant x scenario x horizon, from targets <= 2020; h >= 4 extrapolated in log h |
| exploratory_recal_summary.csv | 09_exploratory_recalibration_CPU | 3 | EXPLORATORY recalibration, not validated (post-hoc, deviation 16): the same coverage pooled over horizons, with a 90% station-bootstrap interval for the widened 90% coverage |
| figures/fig_e1_maps.pdf | 07_results_CPU | fig_e1_maps | Map of E1 for joint attainment (both metrics) at each station's 2024 level |
| figures/fig_e1_maps.png | 07_results_CPU | fig_e1_maps | Map of E1 for joint attainment (both metrics) at each station's 2024 level (PNG copy) |
| figures/fig_e1_maps.png | 07_results_CPU | fig_e1_maps | Map of E1 for joint attainment (both metrics) at each station's 2024 level (PNG copy) |
| figures/fig_e1_maps_all.pdf | 07_results_CPU | fig_e1_maps_all | Station maps of E1 by pollutant and metric (supplementary) |
| figures/fig_e1_maps_all.png | 07_results_CPU | fig_e1_maps_all | Station maps of E1 by pollutant and metric (supplementary) (PNG copy) |
| figures/fig_e1_maps_all.png | 07_results_CPU | fig_e1_maps_all | Station maps of E1 by pollutant and metric (supplementary) (PNG copy) |
| figures/fig_e3_headroom.pdf | 07_results_CPU | fig_e3_headroom | Change in the weather-adjusted level needed for a 90% probability of attaining both metrics (negative = reduction needed) |
| figures/fig_e3_headroom.png | 07_results_CPU | fig_e3_headroom | Change in the weather-adjusted level needed for a 90% probability of attaining both metrics (negative = reduction needed) (PNG copy) |
| figures/fig_e3_headroom.png | 07_results_CPU | fig_e3_headroom | Change in the weather-adjusted level needed for a 90% probability of attaining both metrics (negative = reduction needed) (PNG copy) |
| figures/fig_e5_variance.pdf | 07_results_CPU | fig_e5_variance | Median shares of the 2030 predictive variance from trend, weather and residual noise |
| figures/fig_e5_variance.png | 07_results_CPU | fig_e5_variance | Median shares of the 2030 predictive variance from trend, weather and residual noise (PNG copy) |
| figures/fig_e5_variance.png | 07_results_CPU | fig_e5_variance | Median shares of the 2030 predictive variance from trend, weather and residual noise (PNG copy) |
| figures/fig_hindcast_skill.pdf | 07_results_CPU | fig_hindcast_skill | Brier skill score for joint attainment by horizon, scenario and baseline, with 90% two-way cluster-bootstrap intervals |
| figures/fig_hindcast_skill.png | 07_results_CPU | fig_hindcast_skill | Brier skill score for joint attainment by horizon, scenario and baseline, with 90% two-way cluster-bootstrap intervals (PNG copy) |
| figures/fig_hindcast_skill.png | 07_results_CPU | fig_hindcast_skill | Brier skill score for joint attainment by horizon, scenario and baseline, with 90% two-way cluster-bootstrap intervals (PNG copy) |
| figures/fig_marginal_by_country.pdf | 07_results_CPU | fig_marginal_by_country | Share of stations with E1 (both metrics) between 0.1 and 0.9, by member state and pollutant |
| figures/fig_marginal_by_country.png | 07_results_CPU | fig_marginal_by_country | Share of stations with E1 (both metrics) between 0.1 and 0.9, by member state and pollutant (PNG copy) |
| figures/fig_marginal_by_country.png | 07_results_CPU | fig_marginal_by_country | Share of stations with E1 (both metrics) between 0.1 and 0.9, by member state and pollutant (PNG copy) |
| figures/fig_reliability.pdf | 07_results_CPU | fig_reliability | Reliability diagrams for joint attainment (level frozen 1 year ahead; trend 1 and 6 years ahead); marker area proportional to the square root of the number of forecasts in the bin |
| figures/fig_reliability.png | 07_results_CPU | fig_reliability | Reliability diagrams for joint attainment (level frozen 1 year ahead; trend 1 and 6 years ahead); marker area proportional to the square root of the number of forecasts in the bin (PNG copy) |
| figures/fig_reliability.png | 07_results_CPU | fig_reliability | Reliability diagrams for joint attainment (level frozen 1 year ahead; trend 1 and 6 years ahead); marker area proportional to the square root of the number of forecasts in the bin (PNG copy) |
| figures/fig_worked_examples.pdf | 07_results_CPU | fig_worked_examples | Weather-adjusted yearly levels for the two pre-specified example stations |
| figures/fig_worked_examples.png | 07_results_CPU | fig_worked_examples | Weather-adjusted yearly levels for the two pre-specified example stations (PNG copy) |
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
| posthoc_by_origin.csv | 08_posthoc_diagnostics_CPU | 5 | POST-HOC D8: one-year Brier score and skill by forecast origin |
| posthoc_by_target_year.csv | 08_posthoc_diagnostics_CPU | 4 | POST-HOC D5: one-year coverage, log error and Brier score (model and naive trend) by target year |
| posthoc_diag_crps.csv | 08_posthoc_diagnostics_CPU | 3 | POST-HOC D4: CRPS blow-up rows and robust median CRPS |
| posthoc_diag_errors.csv | 08_posthoc_diagnostics_CPU | 1 | POST-HOC D1/D2: signed log error, spread ratio and bias-removed 90% coverage of the annual-mean prediction, by run x pollutant x scenario x horizon |
| posthoc_diag_perfect_weather.csv | 08_posthoc_diagnostics_CPU | 2 | POST-HOC D3: Brier score of joint attainment with resampled vs the target year's own weather |
| posthoc_exclusions_methods.csv | 10_exclusion_audit_CPU | 2 | POST-HOC, descriptive: sequences of EEA measurement-method codes among series excluded for a method change |
| posthoc_exclusions_summary.csv | 10_exclusion_audit_CPU | 1 | POST-HOC, descriptive: series excluded by the relocation / method-change rule, by reason, with break year and meteorological match |
| posthoc_forecast_bands.csv | 08_posthoc_diagnostics_CPU | 5 | POST-HOC D7: one-year joint-attainment forecasts by probability band, with the naive trend on the same rows, and AUC |
| posthoc_mc_convergence.csv | 11_mc_convergence_CPU | 2 | POST-HOC: coverage and median width of those intervals by pollutant, scenario, horizon and draw size |
| posthoc_mc_convergence_rows.csv | 11_mc_convergence_CPU | 1 | POST-HOC: 90% intervals of the annual mean at a fixed fit with default and 5x outer draws (2% subsample, h = 1 and 3) |
| posthoc_without_2020_2021.csv | 08_posthoc_diagnostics_CPU | 4 | POST-HOC D6: coverage and Brier skill (point estimates) with and without target years 2020-2021 |
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
| posthoc_negative_values.csv | 12_negative_values_audit_CPU | 2 | POST-HOC audit (deviation 18a): negative daily means in the analysis set by pollutant, and the effect of setting them to zero on observed annual means and annual attainment (run locally after the Colab upload failed; deviation 20) |
