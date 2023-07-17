.. include:: links.rst

Gradec Developer Guide
======================

This guide provides a more detailed description of the organization and preferred coding style for Gradec, for prospective code contributors.

Coding Style
------------

Gradec code should follow PEP8 recommendations.

To enforce Gradec's preferred coding style,
we use `flake8`_ with plugins for `isort <https://pypi.org/project/flake8-isort/>`_,
`black <https://pypi.org/project/flake8-black/>`_, and `docstrings <https://pypi.org/project/flake8-docstrings/>`_.
These plugins automatically evaluate imports, code formatting, and docstring formatting as part of our continuous integraton.

Additionally, we have modeled Gradec's code on `scikit-learn`_.
By this we mean that most of Gradec user-facing tools are implemented as classes.
These classes generally accept a number of parameters at initialization,
and then use ``fit`` or ``transform`` methods to apply the algorithm to data.

Installation for Development
----------------------------

Installation with Conda
```````````````````````

Perhaps the easiest way to install Gradec for development is with Conda.

In this setup, you simply create a conda environment,
then install your local version of Gradec in editable mode (``pip install -e``).

.. code-block:: bash

  cd /path/to/gradec_repo

  conda create -p /path/to/gradec_env pip python=3.9
  conda activate /path/to/gradec_env
  pip install -e .[all]

In this setup, any changes you make to your local clone of Gradec will automatically be reflected in your environment.

Maintaining Gradec
------------------

Labeling PRs
````````````

All PRs should be appropriately labeled.
PR labels determine how PRs will be reported in the next release's release notes.
For example, PRs with the "enhancement" label will be placed in the "🎉 Exciting New Features" section.

If you forget to add the appropriate labels to any PRs that you merge,
you can add them after they've been merged (and even change the titles),
as long as you do so before the next release has been published.

Making a Release
````````````````

To make a Gradec release, use GitHub's online release tool.
Choose a new version tag, according to the semantic versioning standard.
The release title should be the same as the new tag (e.g., ``0.0.2``).
For pre-releases, we use release candidate terminology (e.g., ``0.0.2rc1``) and we select the "This is a pre-release" option.

At the top of the release notes, add some information summarizing the release.
After you have written the summary, use the "Generate release notes" button;
this will add the full list of changes in the new release based on our template.

Once the release notes have been completed, you can publish the release.
This will make the release on GitHub and will also trigger GitHub Actions to
(1) publish the new release to PyPi and (2) update the changelog file.