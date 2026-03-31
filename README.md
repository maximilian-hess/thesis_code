# Code and data for PhD thesis of Maximilian Hess

This repository contains the code and data necessary to reproduce the numeric results shown in my PhD thesis.

## Constraint encodings in QAOA

In Section 4.1, we present different encodings for linear inequality constraints in QAOA-like algorithms.
We compare four methods in a small numerical experiment involving the Multiple Knapsack Problem (MKP). The experiment is set up in `MKP/constraint_encoding.ipynb`.
The implementation of the constraint encoding methods can be found in `MKP/sources/MKPsolver.py`.
The results which have been generated using the code in the notebook can be found in serialized format in `MKP/results`.

## Biased state preparation for Grover algorithms

Section 5.4 introduces a range of branching criteria in state preparation circuits which are combined with Grover search / Amplitude Amplification.
We test different branching criteria on a selection of MKP instances.
The experiment is set up in `MKP/grover_branching_criteria.ipynb`.
The implementation of the methods used can be found in `MKP/sources/MKPgrover.py`.
The results which have been generated using the code in the notebook can be found in serialized format in `MKP/results`.

## Grover-boosted neighborhood search

Section 5.5 introduces state preparation circuits for the Travelling Salesperson Problem (TSP). We introduce two different protcols (edge replacement and vertex replacement) and compare them in a numerical study.
The experiment is set up in `TSP/tsp_neighborhood_search.ipynb`.
The implementation of the methods used can be found in `TSP/sources/TSPsampler.py`.
The results which have been generated using the code in the notebook can be found in serialized format in `TSP/results`.

## Grover probabilities notebook

We provide a small notebook to plot success probabilities of Grover's algorithm in different settings.
The notebook can be found at `Grover_probabilities/grover_probs.ipynb`.
We also provide a notebook which carries out the fitting of a degree $2$ polynomial to translate between number of Amplitude Amplification iterations and classical samples taken as described in Section 5.2.1.
The notebook can be found at `Grover_probabilities/approximate_benchmarking.ipynb`.