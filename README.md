# pygformulaICU: a python implementation of the parametric g-formula for dynamic treatment lengths typical in ICU care

**Authors: Sagar Nagaraj Simha, Ameen Abu-Hanna, Giovanni Cinà**

Modified original package pygformula at https://github.com/CausalInference/pygformula/
**Original Authors: Jing Li, Sophia Rein, Sean McGrath, Roger Logan, Ryan O’Dea, Miguel Hernán**


## Overview
The pygformula package implements the non-iterative conditional expectation (NICE) estimator of the g-formula algorithm
(Robins, 1986). The g-formula can estimate an outcome’s counterfactual mean or risk under hypothetical treatment strategies
(interventions) when there is sufficient information on time-varying treatments and confounders. The package works with 
treatment strategies of variable length.


### Features

* Treatments: discrete or continuous time-varying treatments.
* Outcomes: failure time outcomes or continuous/binary end of follow-up outcomes.
* Interventions: interventions on a single treatment or joint interventions on multiple treatments.
* Random measurement/visit process.
* Incorporation of a priori knowledge of the data structure.
* Censoring events.
* Competing events.
* Variable treatment length


## Requirements

The package requires python 3.8+ and these necessary dependencies:

- cmprsk
- joblib
- lifelines
- matplotlib
- numpy
- pandas
- prettytable
- pytruncreg
- scipy
- seaborn
- statsmodels
- tqdm


## Documentation

The online documentation is available at [pygformula documentation](https://pygformula.readthedocs.io).

## Contact

For any questions or comments, please email s.n.simha@amsterdamumc.nl.
