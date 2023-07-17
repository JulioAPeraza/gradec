.. include:: links.rst

About Gradec
============

Gradec is a Python package for performing meta-analytic functional decoding of surface-based 
neuroimaging data, particularly cortical gradient of functional connectivity.
While others meta-analytic packages, shuch as NiMARE, perform such decoding task, they  do not
currently support surface-based data. Gradec is designed to fill this gap. In the future, 
our plan is to incorporate some of the Gradec's functionality into NiMARE, once surface 
masker are implemented. 
Gradec provides a standard syntax for performing functional decoding with standard databases
(e.g., Neurosynth, and NeuroQuery) via fetchers to reduce computation time. However, it is also
possible to train the decoder from scratch if you have an HPC available for such task.