# Metrics Ablation Analysis

## Purpose

This analysis evaluates the sensitivity and susceptibility to confounding of common image quality assessment metrics across different biological contexts. 

## Analysis

1. Apply known image ablations to images across various biological contexts
2. Evaluate metric values for each ablation type
3. Conduct nested regression analysis per ablation type
4. Identify the best and worst performing metrics for measuring each type of image degradation
5. Determine which metrics are least susceptible to biological confounding

## Goal

Determine optimal image quality metrics that can accurately measure specific types of image degradation without being significantly confounded by biological context variations.

## Dependency installation

To install the required dependencies:

```bash
cd 2.metrics_ablation_analysis # where this README lives
pip install .
```
