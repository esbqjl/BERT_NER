# BERT NER
<img src="https://github.com/esbqjl/text_classification/blob/bert_lora/image/Bert.jpg" alt="drawing" width="600" height="400"/>

## What Bert is?

"As we learned what a Transformer is and how we might train the Transformer model, we notice that it is a great tool to make a computer understand human language. However, the Transformer was originally designed as a model to translate one language to another. If we repurpose it for a different task, we would likely need to retrain the whole model from scratch. Given the time it takes to train a Transformer model is enormous, we would like to have a solution that enables us to readily reuse the trained Transformer for many different tasks. BERT is such a model. It is an extension of the encoder part of a Transformer."
from https://machinelearningmastery.com/a-brief-introduction-to-bert/

<img src="https://github.com/esbqjl/text_classification/blob/bert_lora/image/BERT_Overall.jpg" alt="drawing" width="1000" height="450"/>

### Pre-training and Fine-tuning
BERT is designed to be pre-trained on a large corpus of text in an unsupervised manner using two strategies:

Masked Language Model (MLM): Randomly masking some of the words in the input and predicting them based only on their context.
Next Sentence Prediction (NSP): Given pairs of sentences as input, the model predicts if the second sentence logically follows the first one.
After pre-training, BERT can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering, sentiment analysis, and language inference.

### Transfer Learning
BERT exemplifies the concept of transfer learning in NLP. The idea is to take a model pre-trained on a large dataset and fine-tune it on a smaller, task-specific dataset. This approach allows BERT to perform well on tasks even when there is relatively little labeled training data available.


## Goal of This Project

"This project explores whether incorporating a simple GRU layer can enhance a BERT-based model (110M parameters, 12 layers, 768 hidden units, 12 heads) during subsequent fine-tuning. I utilized a complex dataset from a concluded algorithm competition, the Ruijin Hospital MMC Artificial Intelligence Assisted Knowledge Graph Construction Competition. Additionally, we tested the efficacy of a CRF layer in BERT fine-tuning (with and without the GRU layer). I also noted that the learning rate of the CRF could be a potential issue; we adjusted it to 1-1000 times the original rate (5e-5) to accelerate the CRF's convergence speed. For more details, please refer to the chart below."

## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

see requirements.txt and setup.py

2080 Ti GPU or above might work, Windows OS

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Clone the repo
   ```sh
   git clone [https://github.com/esbqjl/text_classification.git](https://github.com/esbqjl/BERT_NER.git)
   ```
2. Install various packages
   ```sh
   pip install -e .
   ```
3. model and datasets
   you need to download model from https://huggingface.co/google-bert/bert-base-uncased/
   
   you need to extract datasets using process.py, more information please see data_process.py and config.py
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Train emotion classifier using pretrained model 
 ```sh
   python3 train.py
 ```
you can run the predictor using predict.py, please decide which model you want to use
 ```sh
   python3 predict.py
 ```
## Model

Due to the model is too large, it is not on Github repo, you can contact me to get the model of the experiment.

## Result

Please checkout the result using `tensorboard --logdir ./runs`.

## Conclusion

"From the results, I understand that simply adding a new training layer after BERT contributes minimally to fine-tuning; the final F1 score did not improve significantly. This may be due to the complexity of the dataset. If we decide to add a CRF layer, it is crucial to increase the learning rate to 100 or 1000 times that of the original BERT training rate, as it will enhance the convergence speed of our CRF layer.

This is the first time I am participating in this competition, and the data is very difficult for training a decent model. On the leaderboard, the top score is 72, which is quite impressive."

<!-- Contact -->
## Contact

Wenjun - 1378555845gg@gmail.com

Project Link: [https://github.com/esbqjl/BERT_NER](https://github.com/esbqjl/BERT_NER)

## Useful Link 

Transformers: [https://github.com/huggingface/transformers]

Bert: [https://github.com/codertimo/BERT-pytorch]

CRF: [[https://github.com/microsoft/LoRA]](https://people.cs.umass.edu/~mccallum/papers/crf-tutorial.pdf)

GRU: https://arxiv.org/abs/1412.3555

Personal Website + project demonstration: [https://www.happychamber.xyz/deep_learning]
<p align="right">(<a href="#readme-top">back to top</a>)</p>





