# Perceptual Spaces of the Singing Voice

This repository is a revised version of the work we did for the paper [An Exploratory Study on Perceptual Spaces of the Singing Voice](https://arxiv.org/abs/2111.08196).
It consists of a small dataset documenting participants' pairwise dissimilarity ratings between a total of 15 stimuli (and therefore 120 ratings) each (and non-identifying details about each participant).
There is also an accompnaying notebook to reproduce our results. Hopefully this documentation will provide a few ideas to future beginner researchers that are interested in the use of timbral maps

## Reproducing the research
0. Clone this repo.
1. Download the repository and use the `requirements.txt` if necessary (the environment used was embedded within an Anaconda environment).
2. The `anonimisedData.hdf5` file contains all non-identifiable participant data.
3. Open the `Exploratory Study on Perceptual Spaces of the Singing Voice.ipynb` in jupyter notebook, which provides a walkthrough for all data analysis mentioned in the paper. This is a walk through of how the data is interpretted. It includes clustering, statistical analysis, and multidimensional scaling.

*Disclaimer: This research was conducted in 2019/2020. It has been revised, but not drastically changed. There are many embarassing tells that it was my first time writing Python scripts. My documentation, functionality, and readability has significantly improved since then*