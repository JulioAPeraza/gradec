.. include:: <isonum.txt>
.. include:: links.rst

Gradec: Meta-analytic Gradient Decoding
==========================================================

Gradec is a Python package for meta-analytic functional decoding of 
cortical gradient of functional connectivity.

To install Gradec check out our `installation guide`_.

.. _installation guide: installation.html

Citing Gradec
-------------

If you use gradec in your research, we recommend citing the Zenodo DOI associated with the gradec 
version you used, as well as the Imaging Neuroscience journal 
article: https://doi.org/10.1162/imag_a_00081.
You can find the Zenodo DOI associated with each gradec release 
at https://zenodo.org/record/8161766.

.. code-block:: bibtex
    :caption: BibTeX entries for Gradec version 0.0.1rc3.

    # This is the Imaging Neuroscience paper.
    @article{10.1162/imag_a_00081,
        author = {Peraza, Julio A. and Salo, Taylor and Riedel, Michael C. and Bottenhorn, Katherine L. and Poline, Jean-Baptiste and Dockès, Jérôme and Kent, James D. and Bartley, Jessica E. and Flannery, Jessica S. and Hill-Bowen, Lauren D. and Lobo, Rosario Pintos and Poudel, Ranjita and Ray, Kimberly L. and Robinson, Jennifer L. and Laird, Robert W. and Sutherland, Matthew T. and de la Vega, Alejandro and Laird, Angela R.},
        title = "{Methods for decoding cortical gradients of functional connectivity}",
        journal = {Imaging Neuroscience},
        volume = {2},
        pages = {1-32},
        year = {2024},
        month = {02},
        abstract = "{Macroscale gradients have emerged as a central principle for understanding functional brain organization. Previous studies have demonstrated that a principal gradient of connectivity in the human brain exists, with unimodal primary sensorimotor regions situated at one end and transmodal regions associated with the default mode network and representative of abstract functioning at the other. The functional significance and interpretation of macroscale gradients remains a central topic of discussion in the neuroimaging community, with some studies demonstrating that gradients may be described using meta-analytic functional decoding techniques. However, additional methodological development is necessary to fully leverage available meta-analytic methods and resources and quantitatively evaluate their relative performance. Here, we conducted a comprehensive series of analyses to investigate and improve the framework of data-driven, meta-analytic methods, thereby establishing a principled approach for gradient segmentation and functional decoding. We found that a two-segment solution determined by a k-means segmentation approach and an LDA-based meta-analysis combined with the NeuroQuery database was the optimal combination of methods for decoding functional connectivity gradients. Finally, we proposed a method for decoding additional components of the gradient decomposition. The current work aims to provide recommendations on best practices and flexible methods for gradient-based functional decoding of fMRI data.}",
        issn = {2837-6056},
        doi = {10.1162/imag_a_00081},
        url = {https://doi.org/10.1162/imag\_a\_00081},
        eprint = {https://direct.mit.edu/imag/article-pdf/doi/10.1162/imag\_a\_00081/2326234/imag\_a\_00081.pdf},
    }

    # This is the Zenodo citation for version 0.0.1rc3.
    @software{peraza_2023_8161766,
       author = {Peraza, Julio A. and Kent, James D. and Salo, Taylor and De La Vega, Alejandro and Laird, Angela R.},
       title = {JulioAPeraza/gradec: 0.0.1rc3},
       month = jul,
       year = 2023,
       publisher = {Zenodo},
       version = {0.0.1rc3},
       doi = {10.5281/zenodo.8161766},
       url = {https://doi.org/10.5281/zenodo.8161766}
    }

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   about
   installation
   api
   auto_examples/index
   contributing
   dev_guide
   outputs
   methods
   changelog
   glossary

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. note::
    This document is based on the documentations from the `NiMARE`_ project.