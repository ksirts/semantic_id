# Semantic Idea Density (SID)

The scripts in this repository implement the K-means clustering features and the SID feature
following the description in [Vector-space topic models for detecting Alzheimer's disease](https://www.aclweb.org/anthology/P16-1221.pdf) by Yancheva and Ruczicz (2016).

This implementation was used for experiments presented in the paper [Idea density for predicting Alzheimer’s disease from transcribed speech](https://www.aclweb.org/anthology/K17-1033.pdf) by Sirts et al. (2017). If you use this implementation, please cite the following reference:


    Sirts, K., Piguet, O., & Johnson, M. (2017). Idea density for predicting Alzheimer’s disease from transcribed speech. 
    In Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017) (pp. 322-332).

    @inproceedings{sirts2017idea,
      title={Idea density for predicting Alzheimer’s disease from transcribed speech},
      author={Sirts, Kairit and Piguet, Olivier and Johnson, Mark},
      booktitle={Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017)},
      pages={322--332},
      year={2017}
    }

# Usage

The script can be used on both training and evaluation mode.

## Training

    python sid_kmeans.py --input <input_file.jl> --output <output_file> --cluster --sid --mode train --model <model_name>
    
input_file.jl - json lines file where each item is expected to have the following attributes:
* id - subject id
* label - group label for classification
* num - unique id of the items of the same subject
* text - the text itself

output_file - a tab-separated file containing both the id, label and num columns as well as the computed features

--cluster - a flag specfifying whether to compute cluster features, see Yancheva and Ruczicz (2016)

--sid - a flag specifying whether to compute the SID feature, see Yancheva and Ruczicz (2016)

In training mode, the relevant model parameters are saved into a json file. The default model file location is /tmp/model and that can be changed with the --model flag.

## Evaluation

    python sid_kmeans.py --input <input_file.jl> --output <output_file> --cluster --sid --mode predict --model <model_name>
    
The parameters of a trained K-means model are read from the model file and used to compute features for evaluation data.
