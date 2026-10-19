# Beijing Opera Percussion Instruments Classification

**Classifying traditional Chinese percussion sounds with classic machine learning**

---

## About

This project applies two conventional machine learning algorithms, **Support Vector Machines (SVM)** and **K-Nearest Neighbours (KNN)**, to classify Beijing Opera percussion instruments from their audio recordings.

The goal is to compare how well these two classic approaches distinguish between the four percussion instrument classes used in Beijing Opera.

## Dataset

We use the public **[Beijing Opera Percussion Instrument Dataset](https://zenodo.org/records/1285212)** from Zenodo.

It contains **236 recordings** covering **4 instrument classes**:

| Instrument               | Description                                                                                                       |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| **Bangu** (Clapper-drum) | Made of the _ban_ (a wooden board-shaped clapper) and the _danpigu_ (a wooden drum struck with two wooden sticks) |
| **Naobo** (Cymbals)      | A pair of cymbal instruments: _Qibo_ and _Danao_                                                                  |
| **Daluo**                | Large gong                                                                                                        |
| **Xiaoluo**              | Small gong                                                                                                        |

For a thorough explanation of the dataset, read the authors' paper (free to access):
[Beijing Opera percussion instrument dataset, IEEE Xplore](https://ieeexplore.ieee.org/document/6853981/)

## Models

| Model   | Type                   |
| ------- | ---------------------- |
| **SVM** | Support Vector Machine |
| **KNN** | K-Nearest Neighbours   |

## Getting Started

### 1. Get the project

Clone the repository:

```bash
git clone https://github.com/KrustyCode/beijing-percussion-classification.git
cd beijing-percussion-classification
```

Or download the ZIP file from GitHub and extract it.

### 2. Download the dataset

> **Note:** The dataset is **not included** in this repository.

Download it from the [Zenodo page](https://zenodo.org/records/1285212).

### 3. Add the dataset to the project

Move the downloaded dataset into a folder named `dataset` at the root of the project:

```
project-folder/
├── dataset/              <- put the downloaded dataset here
├── experiment.ipynb
├── requirements.txt
└── README.md
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the experiment

1. Open `experiment.ipynb`
2. Select your Python kernel
3. Run all cells

## References

- Tian, M., Srinivasamurthy, A., Sandler, M., Serra, X. _A study of instrument-wise onset detection in Beijing opera percussion ensembles._ ICASSP 2014. [IEEE Xplore](https://ieeexplore.ieee.org/document/6853981/)
- [Beijing Opera Percussion Instrument Dataset, Zenodo](https://zenodo.org/records/1285212)
