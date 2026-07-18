# Copy from config/2025/ and edit when rolling to the 2026 prediction season.
# Required files in this folder:
#   firings.yaml    — coaches fired after 2025 (label HISTORY_END)
#   hires.yaml      — new head coaches + ages
#   sb_futures.csv  — Super Bowl futures odds by team
#   wins_exp.csv    — expected win totals (rank-aligned with futures)
#
# Then bump config/settings.yaml:
#   season: 2026
#   history_end: 2025
