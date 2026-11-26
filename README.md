# CrackView

CrackView selects camera views for 3D reconstruction of pavement cracks. Given 50 images of a pavement scene and their camera poses, a reinforcement learning policy generates ten candidate subsets of 25 views each. A frozen scoring network evaluates the ten subsets together. It predicts reconstruction quality for both the overall pavement surface and the crack region. The subset with the highest combined score is selected for reconstruction.

## Release status

This repository contains a **preliminary research implementation** covering feature extraction, subset quality scoring, and reinforcement learning–based view selection.

We have submitted a formal request to our institution’s research management office for approval of the final public release. The review covers intellectual property arising from the funded research program and permission to redistribute third-party components. Once the review is complete, we will publish the approved components and update the documentation in stages **at this same repository address**.

## Scope of the current version

The scripts provide the core view-selection pipeline. Reproducing the results reported in the paper also requires the original scene splits, trained weights, and reconstruction-derived quality labels. The current scripts assume that the heuristic baseline subsets and their quality labels have already been prepared. The full point-cloud reconstruction, registration, and independent test evaluation workflows are outside the scope of this initial release.
