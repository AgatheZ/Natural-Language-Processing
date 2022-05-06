# Natural-Language-Processing

Participation to the SemEval 2022 task on Patronizing and Condescending Language (PCL) Detection. This task is based on the paper Don't Patronize Me! An annotated Dataset with Patronizing and Condescending Language Towards Vulnerable Communities (Perez-Almendros et al., 2020).

The aim of this task is to build a model able to differenciate patronizing from non-patronizing sentences. 

* **evaluation.py**: contains the metrics and the loss curve plot functions to evaluate the models.

* **main.py**: main class. 

* **PCLDataset.py**: Map-style dataset class tailored to our problem.

* **preprocessing.py**: contains the preprocessing functions (eg stop word removal, tokenization...).

* **Roberta_PCL.py**: Instantiation of the model and definition of the forward pass.

* **Trainer_PCL.py**: Trainer class tailored to our problem. 