# Datasets
## Xeno-Canto
The Xeno-Canto data has to be downloaded on your side. You can find the metadata of my dataset here : https://www.kaggle.com/datasets/antoinemasq/xeno-canto-pretraining-data.

The dataset used contains 295,976 recordings (~38.4 GB) spanning the 5 groups available on Xeno-Canto (i.e. birds, grasshoppers, bats, frogs, and land mammals).

The recordings were resampled to 32kHz.


## BirdSet (soundscape datasets)
Those datasets can be found at https://huggingface.co/datasets/DBD-research-group/BirdSet. To use them, you need to download each individual dataset (e.g. PER at https://zenodo.org/records/7079124) with at least the annotations.csv, the soundscape_data.zip (decompressed in a folder named "soundscape_data/"), and the species.csv files.
Each of those datasets should be in different sub-folder in a general soundscape folder (e.g. "Soundscape datasets/PER/").

You can then build the dataset with the following code snippet:
```
from dataset import SoundscapeDataset

SoundscapeDataset.process_dataset(path="Datasets/Soundscapes/", dataset="PER")
```

# Citations

* Clements, J. F., P. C. Rasmussen, T. S. Schulenberg, M. J. Iliff, J. A. Gerbracht, D. Lepage, A. Spencer, S. M. Billerman, B. L. Sullivan, M. Smith, and C. L. Wood. 2025. The eBird/Clements checklist of Birds of the World: v2025. Downloaded from https://www.birds.cornell.edu/clementschecklist/download/

* Lukas Rauch and Raphael Schwinger and Moritz Wirth and René Heinrich and Denis Huseljic and Marek Herde and Jonas Lange and Stefan Kahl and Bernhard Sick and Sven Tomforde and Christoph Scholz. 2025. BirdSet: A Large-Scale Dataset for Audio Classification in Avian Bioacoustics.
