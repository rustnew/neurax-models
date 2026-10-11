# UntitledDesign

Trained by NEURAX: 43,393 parameters, task regression, input features of shape [273].

## Files

- `best.pt` — the weights at the best validation loss, step 2538.
- `best.safetensors` — the weights at the best validation loss, step 2538.
- `last.pt` — the weights at the end of the run, step 5640.
- `last.safetensors` — the weights at the end of the run, step 5640.
- `model.py` — the model the weights load into.
- `config.json` — this run's record.

## Data

Trained on `/home/fossouomartial/archive/sample_10k_rows.csv`.
1000 samples were held out of training to validate on (at random); 9000 were trained on. Which ones: `split.json`.

## Validation

- Loss: 0.09445

## How it trained

- The training loss fell from 0.3505 to 0.02202 (94%).
- On validation the prediction is off by 6 on average (root mean squared error, in the target's own units).

## Use it

`predict.py` prepares a raw sample as training did (`preprocessing.json`, `tokenizer.json`) and decodes the output (`labels.json`):

```python
from predict import Model

model = Model()
model.predict({...})  # a row of the table, by column name → {'value': … in the target's units}
```

## Load it

```python
import torch
from model import UntitledDesign

model = UntitledDesign()
from safetensors.torch import load_model
load_model(model, "best.safetensors", strict=True)
model.eval()
```
