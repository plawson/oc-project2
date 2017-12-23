# oc-project2
Réaliser un apprentissage distribué

## Files
- **animals_emr.py :** Script python de Classification des images
- **create_cluster.sh :** Script de création  d'un cluster EMR
- **emr_config.json :** Configuration customisée du cluster EMR
- **bootstrap-emr.sh :** Script d'installation des packages python3 additionnels

## Synopsys
```
usage: animals_emr.py [-h] (--1vs1 1VS1 | --1vsAll 1VSALL) --size SIZE
                      [--iter ITER] [--graph]

optional arguments:
  -h, --help       show this help message and exit
  --1vs1 1VS1      Classification type, provide classX,classY
  --1vsAll 1VSALL  Classification type, provide classX
  --size SIZE      Sizes of the training set separated by commas
  --iter ITER      Number of iterations
  --graph          Create a classification performance graph
```

