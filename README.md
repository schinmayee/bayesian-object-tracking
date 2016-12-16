# Tracking Multiple Objects From Partial Observations


## Introduction

The aim of this project is to track objects, given noisy observed positions.  This project uses simulated data, where objects move with a random acceleration at every time step. Once objects move out of the domain, they are deleted. Observed positions are true positions plus a random noise.

Occluded objects are not visible, and so their observed positions are not available. Also, the association information between objects and observations is not available. There are three main challenges:
  - estimating true position of objects
  - determing which observations correspond to which tracks
  - determining when to create and delete tracks

This project tries a Bayesian inference approach to tracking objects. There are two parts to the project:
  - maximizing probability of an association given position estimates
  - estimating track positions given association betweeb tracks and observations

For more information, and results, check out the codalab worksheet [here](https://worksheets.codalab.org/worksheets/
0xee9369450dfa40f8a006a2f54da38876/).

## Dependencies

  - numpy
  - pillow
  - matplotlib

You can install all these dependencies with pip, as

  ```
    pip install numpy pillow matplotlib
  ```
