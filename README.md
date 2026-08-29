# Crowd Risk Prediction

Crowd-Risk-Prediction is a final-year project pipeline for estimating crowd density, motion, regional convergence, and crowd-risk levels from video frames.

## Project Structure

- `data/frames/` - extracted video frames used by the processing pipeline
- `models/csrnet/` - CSRNet model definition and weights
- `outputs/` - generated results, summaries, visualized frames, and charts
- `src/` - application and analysis scripts
- `tests/` - project test and validation scripts

## Main Scripts

- `python src/main.py` processes frame pairs and writes `outputs/results.csv`, `outputs/summary.txt`, and visualized frames.
- `python src/visualize_results.py` generates charts from `outputs/results.csv`.

