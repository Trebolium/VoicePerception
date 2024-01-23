# Perceptual Spaces of the Singing Voice

This repository is a revised version of the work we did for the paper [An Exploratory Study on Perceptual Spaces of the Singing Voice](https://arxiv.org/abs/2111.08196).
It consists of a small dataset documenting participants' pairwise dissimilarity ratings between a total of 15 stimuli (and therefore 120 ratings) each, and non-identifying details about each participant.
There is also an accompanying notebook to reproduce our results. We hope this documentation and results will provide some helpful methodologies and insights to future researchers that are interested in the use of singing voice timbral maps. Questions can be addressed to the first author of the paper.

## Reproducing the research
0. Clone this repo.
1. Download the repository and use the `requirements.txt` when making a virtual environment.
2. The `anonimisedData.hdf5` file contains all non-identifiable participant data.
3. Open the `Exploratory Study on Perceptual Spaces of the Singing Voice.ipynb` in jupyter notebook, which provides a walkthrough for all data analysis mentioned in the paper. This is a walk through of how the data is interpretted. It describes how the data is collected, screened, grouped by participant and experimental conditions, checked for normal distributions, and underoges cluster, correlation and statistical analysis processes.

*Disclaimer: This research was conducted in 2019/2020. It will be clear from the codebase that the researcher was less experienced in Python programming at the time of writing (analysis techniques have been revised, but many functions have not been refactored).*