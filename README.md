# Dynamic Global Memory for Document-level Argument Extraction

Code and data for paper the ACL 22 paper.


## Dependencies 
- pytorch=1.6 
- transformers=3.1.0
- pytorch-lightning=1.0.6
- spacy=3.0 # conflicts with transformers
- pytorch-struct=0.4 
- sentence-transformers=2.1.0

## Datasets
<!-- - RAMS (Download at [https://nlp.jhu.edu/rams/]) -->
<!-- - ACE05 (Access from LDC[https://catalog.ldc.upenn.edu/LDC2006T06] and preprocessing following OneIE[http://blender.cs.illinois.edu/software/oneie/]) -->
- WikiEvents (included in this repo)


## Running


- Normal Data Setting

	Train:``./scripts/train_kairos.sh`` Test: ``./scripts/test_kairos.sh``

- Adversarial Examples Setting


	Train:``./scripts/train_kairos_adv.sh`` Test: ``./scripts/test_kairos_adv.sh``
	
## Cite

If you use our code or data/outputs, please cite:

	@InProceedings{memory_ie,
	  author = {Xinya Du, Sha Li and Heng Ji},
	  title = {Dynamic Global Memory for Document-level Argument Extraction},
	  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
	  year = {2022},
	}
