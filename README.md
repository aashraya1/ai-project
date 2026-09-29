# Rainfall Amount Prediction Using Machine Learning

## Overview
This project predicts the **amount of rainfall (in mm)** expected on the next day
at a given Australian weather station, using historical weather observations.

Unlike typical "Will it rain tomorrow? Yes/No" projects, this is a **regression**
project — the model outputs a specific rainfall amount rather than a binary label.

## Dataset
- **Source:** [Kaggle - Rain in Australia (weatherAUS.csv)](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
- Contains ~10 years of daily weather observations from 49 Australian locations.
- Original size: 145,460 rows × 23 columns.

## Target Variable
- **RainfallTomorrow**: created manually by shifting each location's `Rainfall`
  column by one day (grouped by `Location`, sorted by `Date`), so that each row's
  target is the rainfall recorded on the following day at that same location.

## Features Used
- MinTemp, MaxTemp, Rainfall
- Humidity9am, Humidity3pm
- Pressure9am, Pressure3pm
- WindSpeed9am, WindSpeed3pm
- Sunshine, Cloud9am, Cloud3pm
- RainToday (converted from Yes/No to 1/0)

## Project Structure