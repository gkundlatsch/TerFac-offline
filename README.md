# TerFac Offline Version

This repository contains the **offline version** of the **Terminator Factory (TerFac)**, originally available online at the following URL:
[TerFac Online](https://www2.fcfar.unesp.br/#!/instituicao/departamentos/bioprocessos-e-biotecnologia-novo/laboratorios/synbio/terfac/)

This version includes two main scripts:

* **`sequence_generator.py`**: Generates a terminator sequence that matches a specified strength.
* **`sequence_generator_maximum.py`**: Generates the strongest terminator sequence possible for a chosen length.

The following files **must** be downloaded:

* **CSV Mapping files** (`Hairpin_feature_mapping.csv`, `Atract_feature_mapping_normalized.csv`, `Loop_feature_mapping_normalized.csv`, and `Utract_feature_mapping_normalized.csv`).

  * Available directly in the repository.
* **Pre-trained model (`terminator_strength_predictor.joblib`)**:

  * Download from the [Releases section](https://github.com/gkundlatsch/TerFac-offline/releases).


Developed by Guilherme E. Kundlatsch under supervision of Prof. Danielle B. Pedrolli, Prof. Elibio Leopoldo Rech Filho and Prof. Leonardo Tomazeli Duarte.

The data used to train this model was originally published by Chen et al. in Nature Methods: Chen, Y.J., Liu, P., Nielsen, A., et al. (2013). Characterization of 582 natural and synthetic terminators and quantification of their design constraints. Nature Methods, 10, 659–664.

This work was funded by the São Paulo State Foundation (FAPESP) grants 2023/02133-0 and 2020/09838-0, the National Council for Scientific and Technological Development (CNPq) grants 305324/2023-3 and 405953/2024-0, and the National Institute of Science and Technology – Synthetic Biology (CNPq/FAP-DF) grant 465603/2014-9.
