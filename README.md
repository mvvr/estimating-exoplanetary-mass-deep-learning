
---

# Estimating Exoplanetary Mass with Deep Learning

## Overview
This project aims to develop a deep learning model to estimate the mass of exoplanets based on observational data. The model leverages neural networks to predict planetary mass from various features, such as transit depth and orbital period, obtained from space missions and observatories.

##### This Project is part of my Final Semester Project Training Program under Planetary Science Division at Physical Research Laboratory. 
##### Scope of work: Study of exoplanet with deep learning
Exoplanets - planets outside the solar system – show a wide range of diversity, and their study is at the
forefront of modern-day astronomy. Although about 6000 exoplanets have been confirmed, in many cases,
known properties about individual exoplanets are sparse. Deep learning and artificial intelligence (AI) tools
have recently been used to estimate missing values. In this project, we first plan to use available tool-kits to
understand various facets of the topic and then use available data for parameters such as planet mass, planet
radius, orbital period, stellar mass, equilibrium temperature, etc., to derive/estimate the unknown.



## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Future Work](#future-work)
- [Refernce](#refernce)

## Dataset
The dataset used in this project is sourced from:
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [Open Exoplanet Catalogue](https://www.openexoplanetcatalogue.com/)
- [MAST Archive](https://archive.stsci.edu/)
- [NASA Exoplanet Exploration Program](https://science.nasa.gov/exoplanets/)

### Preprocessing
Data preprocessing steps include:
- Cleaning and normalizing the dataset
- Handling missing values and outliers
- Feature engineering to improve model performance

## Model Architecture
The project utilizes various deep learning models:
- **Feedforward Neural Network:** For baseline performance
- **Convolutional Neural Network (CNN):** For handling time-series or spatial data
- **Recurrent Neural Network (RNN) / LSTM:** For sequential data analysis
- **Transformer Models:** For leveraging attention mechanisms


## Future Work
- Explore advanced ensemble methods and transfer learning
- Implement techniques for model explainability
- Develop a web or standalone application for model deployment

## Refernce
- [Estimating Planetary Mass with Deep Learning](https://arxiv.org/abs/1911.11035)

