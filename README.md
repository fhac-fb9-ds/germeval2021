## GermEval 2021

Repository containing the experiments described in our GermEval21 paper ([available online](http://)).

The train and test data can be found in the folder [dataset](./dataset).
Experiments and their results are filed in [experiments](./experiments). 

---
### Experiments

|number|description|single- or multi-label|
|---|---|---|
| |*model exploration*| |
|1|50 gelectra|multi-label|
|2|25 gelectra + 25 gbert|multi-label|
|3|25 gelectra + 25 gbert|single-label|
| |*submissions*| |
|4|submission 1, 200 gelectra|multi-label|
|5|submission 2, 200 gelectra + 200 gbert|multi-label|
|6|submission 3, 30 gelectra + 30 gbert|single-label|

---
## Installation

The experiments were run in a conda environment with python 3.9.
You can find most of the required packages and the command for creating a new environemnt in the requirements.txt.
In addition to the packages described in the requirements.txt, two packages were installed using pip:
* `pip install emoji`
* `pip install transformers`