# AbLM
This repo contains the antibody language model used in the paper [Physics-driven structural docking and protein language models accelerate antibody screening and design for broad-spectrum antiviral therapy](https://www.biorxiv.org/content/10.1101/2024.03.01.582176v1)

A series of language models were finetuned for antibody sequences based on a meta in-house protein language model pre-trained over pfam domain sequences. The input is composed of paired variable regions from heavy and light chains of the antibody. Since the pre-trained PLM only admits single chain inputs, two strategies were applied to facilitate double chain inputs:

1. "seqConcate": concatenate a set of paired VH and VL sequences into a single sequence with a special token `<SEP>`.
2. "seqIndiv": utilize two encoders admitting VH and VL sequences respectively. The weights of two encoders are always tied together.

![facilitate double chains](illustrations/double_chains.png)

The in-house meta pLM was trained with the Masked Language Modeling objective. For antibody finetuning, we applied three additional antibody specific masking pipelines over CDR regions apart from the original masking:

1. "vanilla": apply the original masking strategy over the whole VH and VL sequences
2. "CDR-vanilla": apply the original masking strategy only within the six CDR regions
3. "CDR-margin": randomly pick one CDR out of six and mask all residues within
4. "CDR-pair": mask all residues within one heavy CDR region one light CDR region.

We seleced the best configuration in the sense of language modeling based on the perplexity over SabDab test set and two other independent test sets

![test performance](illustrations/perf_config.png)

### Data/Model availability
The weights of pre-trained meta pLM, as well as finetuned antibody LMs can be downloaded from this Zenodo repo. The paired VH-VL sequence set for finetuning is also available.

### Create environment
Use `conda` command and the provided `.yaml` file to create an environment to run models.
```shell
conda env create -f environment.yaml
```