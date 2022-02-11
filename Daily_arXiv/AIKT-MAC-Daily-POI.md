# MA C.'s Daily Paper Of Interest - February, 2022

# Index


- [2022-02-11](#2022-02-11)

  - [1. SHAS: Approaching optimal Segmentation for End-to-End Speech Translation](#2022-02-11-1)
  - [2. AdaPrompt: Adaptive Model Training for Prompt-based NLP](#2022-02-11-2)
  - [3. Slovene SuperGLUE Benchmark: Translation and Evaluation](#2022-02-11-3)
  - [4. Improving Automatic Speech Recognition for Non-Native English with Transfer Learning and Language Model Decoding](#2022-02-11-4)
  
- [2022-02-10](#2022-02-10)

  - [1. Machine Explanations and Human Understanding](#2022-02-10-1)
  - [2. Image Difference Captioning with Pre-training and Contrastive Learning](#2022-02-10-2)
  - [3. Exploring the Limits of Domain-Adaptive Training for Detoxifying Large-Scale Language Models](#2022-02-10-3)
  - [4. pNLP-Mixer: an Efficient all-MLP Architecture for Language](#2022-02-10-4)
  - [5. Generating Training Data with Language Models: Towards Zero-Shot Language Understanding](#2022-02-10-5)

- [2022-02-09](#2022-02-09)

  - [1. DALL-Eval: Probing the Reasoning Skills and Social Biases of Text-to-Image Generative Transformers](#2022-02-09-1)

- [2022-02-08](#2022-02-08)

  - [1. Machine Translation from Signed to Spoken Languages: State of the Art and Challenges](#2022-02-08-1)
  - [2. Efficient Adapter Transfer of Self-Supervised Speech Models for Automatic Speech Recognition](#2022-02-08-2)
  - [3. Red Teaming Language Models with Language Models](#2022-02-08-3)

- [2022-02-07](#2022-02-07)

  - [1. Data Scaling Laws in NMT: The Effect of Noise and Architecture](#2022-02-07-1)
  - [2. Temporal Attention for Language Models](#2022-02-07-2)
  - [3. The Ecological Footprint of Neural Machine Translation Systems](#2022-02-07-3)

- [2022-01-28](#2022-01-28)
  - [1. Tackling data scarcity in speech translation using zero-shot multilingual machine translation techniques](#2022-01-28-1)
- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



# 2022-02-11

[Return to Index](#Index)



<h2 id="2022-02-11-1">1. SHAS: Approaching optimal Segmentation for End-to-End Speech Translation
</h2>

Title: [SHAS: Approaching optimal Segmentation for End-to-End Speech Translation](https://arxiv.org/abs/2202.04774)

Authors: [Ioannis Tsiamas](https://arxiv.org/search/cs?searchtype=author&query=Tsiamas%2C+I), [Gerard I. Gállego](https://arxiv.org/search/cs?searchtype=author&query=Gállego%2C+G+I), [José A. R. Fonollosa](https://arxiv.org/search/cs?searchtype=author&query=Fonollosa%2C+J+A+R), [Marta R. Costa-jussà](https://arxiv.org/search/cs?searchtype=author&query=Costa-jussà%2C+M+R)

> Speech translation models are unable to directly process long audios, like TED talks, which have to be split into shorter segments. Speech translation datasets provide manual segmentations of the audios, which are not available in real-world scenarios, and existing segmentation methods usually significantly reduce translation quality at inference time. To bridge the gap between the manual segmentation of training and the automatic one at inference, we propose Supervised Hybrid Audio Segmentation (SHAS), a method that can effectively learn the optimal segmentation from any manually segmented speech corpus. First, we train a classifier to identify the included frames in a segmentation, using speech representations from a pre-trained wav2vec 2.0. The optimal splitting points are then found by a probabilistic Divide-and-Conquer algorithm that progressively splits at the frame of lowest probability until all segments are below a pre-specified length. Experiments on MuST-C and mTEDx show that the translation of the segments produced by our method approaches the quality of the manual segmentation on 5 languages pairs. Namely, SHAS retains 95-98% of the manual segmentation's BLEU score, compared to the 87-93% of the best existing methods. Our method is additionally generalizable to different domains and achieves high zero-shot performance in unseen languages.

| Comments: | 7 pages including appendix                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Sound (cs.SD)**; Computation and Language (cs.CL); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2202.04774](https://arxiv.org/abs/2202.04774) [cs.SD]** |
|           | (or **[arXiv:2202.04774v1](https://arxiv.org/abs/2202.04774v1) [cs.SD]** for this version) |





<h2 id="2022-02-11-2">2. AdaPrompt: Adaptive Model Training for Prompt-based NLP
</h2>

Title: [AdaPrompt: Adaptive Model Training for Prompt-based NLP](https://arxiv.org/abs/2202.04824)

Authors: [Yulong Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Y), [Yang Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y), [Li Dong](https://arxiv.org/search/cs?searchtype=author&query=Dong%2C+L), [Shuohang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+S), [Chenguang Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+C), [Michael Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+M), [Yue Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y)

> Prompt-based learning, with its capability to tackle zero-shot and few-shot NLP tasks, has gained much attention in community. The main idea is to bridge the gap between NLP downstream tasks and language modeling (LM), by mapping these tasks into natural language prompts, which are then filled by pre-trained language models (PLMs). However, for prompt learning, there are still two salient gaps between NLP tasks and pretraining. First, prompt information is not necessarily sufficiently present during LM pretraining. Second, task-specific data are not necessarily well represented during pretraining. We address these two issues by proposing AdaPrompt, adaptively retrieving external data for continual pretraining of PLMs by making use of both task and prompt characteristics. In addition, we make use of knowledge in Natural Language Inference models for deriving adaptive verbalizers. Experimental results on five NLP benchmarks show that AdaPrompt can improve over standard PLMs in few-shot settings. In addition, in zero-shot settings, our method outperforms standard prompt-based methods by up to 26.35\% relative error reduction.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2202.04824](https://arxiv.org/abs/2202.04824) [cs.CL]** |
|           | (or **[arXiv:2202.04824v1](https://arxiv.org/abs/2202.04824v1) [cs.CL]** for this version) |





<h2 id="2022-02-11-3">3. Slovene SuperGLUE Benchmark: Translation and Evaluation
</h2>

Title: [Slovene SuperGLUE Benchmark: Translation and Evaluation](https://arxiv.org/abs/2202.04994)

Authors: [Aleš Žagar](https://arxiv.org/search/cs?searchtype=author&query=Žagar%2C+A), [Marko Robnik-Šikonja](https://arxiv.org/search/cs?searchtype=author&query=Robnik-Šikonja%2C+M)

> We present a Slovene combined machine-human translated SuperGLUE benchmark. We describe the translation process and problems arising due to differences in morphology and grammar. We evaluate the translated datasets in several modes: monolingual, cross-lingual, and multilingual, taking into account differences between machine and human translated training sets. The results show that the monolingual Slovene SloBERTa model is superior to massively multilingual and trilingual BERT models, but these also show a good cross-lingual performance on certain tasks. The performance of Slovene models still lags behind the best English models.

| Comments: | arXiv admin note: text overlap with [arXiv:2107.10614](https://arxiv.org/abs/2107.10614) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2202.04994](https://arxiv.org/abs/2202.04994) [cs.CL]** |
|           | (or **[arXiv:2202.04994v1](https://arxiv.org/abs/2202.04994v1) [cs.CL]** for this version) |





<h2 id="2022-02-11-4">4. Improving Automatic Speech Recognition for Non-Native English with Transfer Learning and Language Model Decoding
</h2>

Title: [Improving Automatic Speech Recognition for Non-Native English with Transfer Learning and Language Model Decoding](https://arxiv.org/abs/2202.05209)

Authors: [Peter Sullivan](https://arxiv.org/search/cs?searchtype=author&query=Sullivan%2C+P), [Toshiko Shibano](https://arxiv.org/search/cs?searchtype=author&query=Shibano%2C+T), [Muhammad Abdul-Mageed](https://arxiv.org/search/cs?searchtype=author&query=Abdul-Mageed%2C+M)

> ASR systems designed for native English (L1) usually underperform on non-native English (L2). To address this performance gap, \textbf{(i)} we extend our previous work to investigate fine-tuning of a pre-trained wav2vec 2.0 model \cite{baevski2020wav2vec,xu2021self} under a rich set of L1 and L2 training conditions. We further \textbf{(ii)} incorporate language model decoding in the ASR system, along with the fine-tuning method. Quantifying gains acquired from each of these two approaches separately and an error analysis allows us to identify different sources of improvement within our models. We find that while the large self-trained wav2vec 2.0 may be internalizing sufficient decoding knowledge for clean L1 speech \cite{xu2021self}, this does not hold for L2 speech and accounts for the utility of employing language model decoding on L2 data.

| Comments: | arXiv admin note: substantial text overlap with [arXiv:2110.00678](https://arxiv.org/abs/2110.00678) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2202.05209](https://arxiv.org/abs/2202.05209) [cs.CL]** |
|           | (or **[arXiv:2202.05209v1](https://arxiv.org/abs/2202.05209v1) [cs.CL]** for this version) |





# 2022-02-10

[Return to Index](#Index)



<h2 id="2022-02-10-1">1. Machine Explanations and Human Understanding
</h2>

Title: [Machine Explanations and Human Understanding](https://arxiv.org/abs/2202.04092)

Authors: [Chacha Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+C), [Shi Feng](https://arxiv.org/search/cs?searchtype=author&query=Feng%2C+S), [Amit Sharma](https://arxiv.org/search/cs?searchtype=author&query=Sharma%2C+A), [Chenhao Tan](https://arxiv.org/search/cs?searchtype=author&query=Tan%2C+C)

> Explanations are hypothesized to improve human understanding of machine learning models and achieve a variety of desirable outcomes, ranging from model debugging to enhancing human decision making. However, empirical studies have found mixed and even negative results. An open question, therefore, is under what conditions explanations can improve human understanding and in what way. Using adapted causal diagrams, we provide a formal characterization of the interplay between machine explanations and human understanding, and show how human intuitions play a central role in enabling human understanding. Specifically, we identify three core concepts of interest that cover all existing quantitative measures of understanding in the context of human-AI decision making: task decision boundary, model decision boundary, and model error. Our key result is that without assumptions about task-specific intuitions, explanations may potentially improve human understanding of model decision boundary, but they cannot improve human understanding of task decision boundary or model error. To achieve complementary human-AI performance, we articulate possible ways on how explanations need to work with human intuitions. For instance, human intuitions about the relevance of features (e.g., education is more important than age in predicting a person's income) can be critical in detecting model error. We validate the importance of human intuitions in shaping the outcome of machine explanations with empirical human-subject studies. Overall, our work provides a general framework along with actionable implications for future algorithmic development and empirical experiments of machine explanations.

| Comments: | 26 pages, 13 figures                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Artificial Intelligence (cs.AI)**; Computation and Language (cs.CL); Computers and Society (cs.CY); Human-Computer Interaction (cs.HC) |
| Cite as:  | **[arXiv:2202.04092](https://arxiv.org/abs/2202.04092) [cs.AI]** |
|           | (or **[arXiv:2202.04092v1](https://arxiv.org/abs/2202.04092v1) [cs.AI]** for this version) |





<h2 id="2022-02-10-2">2. Image Difference Captioning with Pre-training and Contrastive Learning
</h2>

Title: [Image Difference Captioning with Pre-training and Contrastive Learning](https://arxiv.org/abs/2202.04298)

Authors: [Linli Yao](https://arxiv.org/search/cs?searchtype=author&query=Yao%2C+L), [Weiying Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+W), [Qin Jin](https://arxiv.org/search/cs?searchtype=author&query=Jin%2C+Q)

> The Image Difference Captioning (IDC) task aims to describe the visual differences between two similar images with natural language. The major challenges of this task lie in two aspects: 1) fine-grained visual differences that require learning stronger vision and language association and 2) high-cost of manual annotations that leads to limited supervised data. To address these challenges, we propose a new modeling framework following the pre-training-finetuning paradigm. Specifically, we design three self-supervised tasks and contrastive learning strategies to align visual differences and text descriptions at a fine-grained level. Moreover, we propose a data expansion strategy to utilize extra cross-task supervision information, such as data for fine-grained image classification, to alleviate the limitation of available supervised IDC data. Extensive experiments on two IDC benchmark datasets, CLEVR-Change and Birds-to-Words, demonstrate the effectiveness of the proposed modeling framework. The codes and models will be released at [this https URL](https://github.com/yaolinli/IDC).

| Comments: | Accepted to AAAI2022                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Multimedia (cs.MM)**; Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV) |
| Cite as:  | **[arXiv:2202.04298](https://arxiv.org/abs/2202.04298) [cs.MM]** |
|           | (or **[arXiv:2202.04298v1](https://arxiv.org/abs/2202.04298v1) [cs.MM]** for this version) |





<h2 id="2022-02-10-3">3. Exploring the Limits of Domain-Adaptive Training for Detoxifying Large-Scale Language Models
</h2>

Title: [Exploring the Limits of Domain-Adaptive Training for Detoxifying Large-Scale Language Models](https://arxiv.org/abs/2202.04173)

Authors: [Boxin Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+B), [Wei Ping](https://arxiv.org/search/cs?searchtype=author&query=Ping%2C+W), [Chaowei Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+C), [Peng Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+P), [Mostofa Patwary](https://arxiv.org/search/cs?searchtype=author&query=Patwary%2C+M), [Mohammad Shoeybi](https://arxiv.org/search/cs?searchtype=author&query=Shoeybi%2C+M), [Bo Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+B), [Anima Anandkumar](https://arxiv.org/search/cs?searchtype=author&query=Anandkumar%2C+A), [Bryan Catanzaro](https://arxiv.org/search/cs?searchtype=author&query=Catanzaro%2C+B)

> Pre-trained language models (LMs) are shown to easily generate toxic language. In this work, we systematically explore domain-adaptive training to reduce the toxicity of language models. We conduct this study on three dimensions: training corpus, model size, and parameter efficiency. For the training corpus, we propose to leverage the generative power of LMs and generate nontoxic datasets for domain-adaptive training, which mitigates the exposure bias and is shown to be more data-efficient than using a curated pre-training corpus. We demonstrate that the self-generation method consistently outperforms the existing baselines across various model sizes on both automatic and human evaluations, even when it uses a 1/3 smaller training corpus. We then comprehensively study detoxifying LMs with parameter sizes ranging from 126M up to 530B (3x larger than GPT-3), a scale that has never been studied before. We find that i) large LMs have similar toxicity levels as smaller ones given the same pre-training corpus, and ii) large LMs require more endeavor to detoxify. We also explore parameter-efficient training methods for detoxification. We demonstrate that adding and training adapter-only layers in LMs not only saves a lot of parameters but also achieves a better trade-off between toxicity and perplexity than whole model adaptation for the large-scale models.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Computers and Society (cs.CY); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2202.04173](https://arxiv.org/abs/2202.04173) [cs.CL]** |
|           | (or **[arXiv:2202.04173v1](https://arxiv.org/abs/2202.04173v1) [cs.CL]** for this version) |





<h2 id="2022-02-10-4">4. pNLP-Mixer: an Efficient all-MLP Architecture for Language
</h2>

Title: [pNLP-Mixer: an Efficient all-MLP Architecture for Language](https://arxiv.org/abs/2202.04350)

Authors: [Francesco Fusco](https://arxiv.org/search/cs?searchtype=author&query=Fusco%2C+F), [Damian Pascual](https://arxiv.org/search/cs?searchtype=author&query=Pascual%2C+D), [Peter Staar](https://arxiv.org/search/cs?searchtype=author&query=Staar%2C+P)

> Large pre-trained language models drastically changed the natural language processing(NLP) landscape. Nowadays, they represent the go-to framework to tackle diverse NLP tasks, even with a limited number of annotations. However, using those models in production, either in the cloud or at the edge, remains a challenge due to the memory footprint and/or inference costs. As an alternative, recent work on efficient NLP has shown that small weight-efficient models can reach competitive performance at a fraction of the costs. Here, we introduce pNLP-Mixer, an embbedding-free model based on the MLP-Mixer architecture that achieves high weight-efficiency thanks to a novel linguistically informed projection layer. We evaluate our model on two multi-lingual semantic parsing datasets, MTOP and multiATIS. On MTOP our pNLP-Mixer almost matches the performance of mBERT, which has 38 times more parameters, and outperforms the state-of-the-art of tiny models (pQRNN) with 3 times fewer parameters. On a long-sequence classification task (Hyperpartisan) our pNLP-Mixer without pretraining outperforms RoBERTa, which has 100 times more parameters, demonstrating the potential of this architecture.

| Comments: | Preprint                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2202.04350](https://arxiv.org/abs/2202.04350) [cs.CL]** |
|           | (or **[arXiv:2202.04350v1](https://arxiv.org/abs/2202.04350v1) [cs.CL]** for this version) |





<h2 id="2022-02-10-5">5. Generating Training Data with Language Models: Towards Zero-Shot Language Understanding
</h2>

Title: [Generating Training Data with Language Models: Towards Zero-Shot Language Understanding](https://arxiv.org/abs/2202.04538)

Authors: [Yu Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+Y), [Jiaxin Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+J), [Yu Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y), [Jiawei Han](https://arxiv.org/search/cs?searchtype=author&query=Han%2C+J)

> Pretrained language models (PLMs) have demonstrated remarkable performance in various natural language processing tasks: Unidirectional PLMs (e.g., GPT) are well known for their superior text generation capabilities; bidirectional PLMs (e.g., BERT) have been the prominent choice for natural language understanding (NLU) tasks. While both types of models have achieved promising few-shot learning performance, their potential for zero-shot learning has been underexplored. In this paper, we present a simple approach that uses both types of PLMs for fully zero-shot learning of NLU tasks without requiring any task-specific data: A unidirectional PLM generates class-conditioned texts guided by prompts, which are used as the training data for fine-tuning a bidirectional PLM. With quality training data selected based on the generation probability and regularization techniques (label smoothing and temporal ensembling) applied to the fine-tuning stage for better generalization and stability, our approach demonstrates strong performance across seven classification tasks of the GLUE benchmark (e.g., 72.3/73.8 on MNLI-m/mm and 92.8 on SST-2), significantly outperforming zero-shot prompting methods and achieving even comparable results to strong few-shot approaches using 32 training samples per class.

| Comments: | Code: [this https URL](https://github.com/yumeng5/SuperGen)  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2202.04538](https://arxiv.org/abs/2202.04538) [cs.CL]** |
|           | (or **[arXiv:2202.04538v1](https://arxiv.org/abs/2202.04538v1) [cs.CL]** for this version) |





# 2022-02-09

[Return to Index](#Index)



<h2 id="2022-02-08-1">1. DALL-Eval: Probing the Reasoning Skills and Social Biases of Text-to-Image Generative Transformers
</h2>

Title: [DALL-Eval: Probing the Reasoning Skills and Social Biases of Text-to-Image Generative Transformers](https://arxiv.org/abs/2202.04053)

Authors: [Jaemin Cho](https://arxiv.org/search/cs?searchtype=author&query=Cho%2C+J), [Abhay Zala](https://arxiv.org/search/cs?searchtype=author&query=Zala%2C+A), [Mohit Bansal](https://arxiv.org/search/cs?searchtype=author&query=Bansal%2C+M)

> Generating images from textual descriptions has gained a lot of attention. Recently, DALL-E, a multimodal transformer language model, and its variants have shown high-quality text-to-image generation capabilities with a simple architecture and training objective, powered by large-scale training data and computation. However, despite the interesting image generation results, there has not been a detailed analysis on how to evaluate such models. In this work, we investigate the reasoning capabilities and social biases of such text-to-image generative transformers in detail. First, we measure four visual reasoning skills: object recognition, object counting, color recognition, and spatial relation understanding. For this, we propose PaintSkills, a diagnostic dataset and evaluation toolkit that measures these four visual reasoning skills. Second, we measure the text alignment and quality of the generated images based on pretrained image captioning, image-text retrieval, and image classification models. Third, we assess social biases in the models. For this, we suggest evaluation of gender and racial biases of text-to-image generation models based on a pretrained image-text retrieval model and human evaluation. In our experiments, we show that recent text-to-image models perform better in recognizing and counting objects than recognizing colors and understanding spatial relations, while there exists a large gap between model performances and oracle accuracy on all skills. Next, we demonstrate that recent text-to-image models learn specific gender/racial biases from web image-text pairs. We also show that our automatic evaluations of visual reasoning skills and gender bias are highly correlated with human judgments. We hope our work will help guide future progress in improving text-to-image models on visual reasoning skills and social biases. Code and data at: [this https URL](https://github.com/j-min/DallEval)

| Comments: | 20 pages, 10 figures, 13 tables                              |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2202.04053](https://arxiv.org/abs/2202.04053) [cs.CV]** |
|           | (or **[arXiv:2202.04053v1](https://arxiv.org/abs/2202.04053v1) [cs.CV]** for this version) |







# 2022-02-08

[Return to Index](#Index)



<h2 id="2022-02-08-1">1. Machine Translation from Signed to Spoken Languages: State of the Art and Challenges
</h2>

Title: [Machine Translation from Signed to Spoken Languages: State of the Art and Challenges](https://arxiv.org/abs/2202.03086)

Authors: [Mathieu De Coster](https://arxiv.org/search/cs?searchtype=author&query=De+Coster%2C+M), [Dimitar Shterionov](https://arxiv.org/search/cs?searchtype=author&query=Shterionov%2C+D), [Mieke Van Herreweghe](https://arxiv.org/search/cs?searchtype=author&query=Van+Herreweghe%2C+M), [Joni Dambre](https://arxiv.org/search/cs?searchtype=author&query=Dambre%2C+J)

> Automatic translation from signed to spoken languages is an interdisciplinary research domain, lying on the intersection of computer vision, machine translation and linguistics. Nevertheless, research in this domain is performed mostly by computer scientists in isolation. As the domain is becoming increasingly popular - the majority of scientific papers on the topic of sign language translation have been published in the past three years - we provide an overview of the state of the art as well as some required background in the different related disciplines. We give a high-level introduction to sign language linguistics and machine translation to illustrate the requirements of automatic sign language translation. We present a systematic literature review to illustrate the state of the art in the domain and then, harking back to the requirements, lay out several challenges for future research. We find that significant advances have been made on the shoulders of spoken language machine translation research. However, current approaches are often not linguistically motivated or are not adapted to the different input modality of sign languages. We explore challenges related to the representation of sign language data, the collection of datasets, the need for interdisciplinary research and requirements for moving beyond research, towards applications. Based on our findings, we advocate for interdisciplinary research and to base future research on linguistic analysis of sign languages. Furthermore, the inclusion of deaf and hearing end users of sign language translation applications in use case identification, data collection and evaluation is of the utmost importance in the creation of useful sign language translation models. We recommend iterative, human-in-the-loop, design and development of sign language translation models.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2202.03086](https://arxiv.org/abs/2202.03086) [cs.CL]** |
|           | (or **[arXiv:2202.03086v1](https://arxiv.org/abs/2202.03086v1) [cs.CL]** for this version) |





<h2 id="2022-02-08-2">2. Efficient Adapter Transfer of Self-Supervised Speech Models for Automatic Speech Recognition
</h2>

Title: [Efficient Adapter Transfer of Self-Supervised Speech Models for Automatic Speech Recognition](https://arxiv.org/abs/2202.03218)

Authors: [Bethan Thomas](https://arxiv.org/search/cs?searchtype=author&query=Thomas%2C+B), [Samuel Kessler](https://arxiv.org/search/cs?searchtype=author&query=Kessler%2C+S), [Salah Karout](https://arxiv.org/search/cs?searchtype=author&query=Karout%2C+S)

> Self-supervised learning (SSL) is a powerful tool that allows learning of underlying representations from unlabeled data. Transformer based models such as wav2vec 2.0 and HuBERT are leading the field in the speech domain. Generally these models are fine-tuned on a small amount of labeled data for a downstream task such as Automatic Speech Recognition (ASR). This involves re-training the majority of the model for each task. Adapters are small lightweight modules which are commonly used in Natural Language Processing (NLP) to adapt pre-trained models to new tasks. In this paper we propose applying adapters to wav2vec 2.0 to reduce the number of parameters required for downstream ASR tasks, and increase scalability of the model to multiple tasks or languages. Using adapters we can perform ASR while training fewer than 10% of parameters per task compared to full fine-tuning with little degradation of performance. Ablations show that applying adapters into just the top few layers of the pre-trained network gives similar performance to full transfer, supporting the theory that higher pre-trained layers encode more phonemic information, and further optimizing efficiency.

| Comments: | 5 Pages, 4 figures. Accepted to ICASSP 2022                  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2202.03218](https://arxiv.org/abs/2202.03218) [cs.CL]** |
|           | (or **[arXiv:2202.03218v1](https://arxiv.org/abs/2202.03218v1) [cs.CL]** for this version) |





<h2 id="2022-02-08-3">3. Red Teaming Language Models with Language Models
</h2>

Title: [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286)

Authors: [Ethan Perez](https://arxiv.org/search/cs?searchtype=author&query=Perez%2C+E), [Saffron Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+S), [Francis Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+F), [Trevor Cai](https://arxiv.org/search/cs?searchtype=author&query=Cai%2C+T), [Roman Ring](https://arxiv.org/search/cs?searchtype=author&query=Ring%2C+R), [John Aslanides](https://arxiv.org/search/cs?searchtype=author&query=Aslanides%2C+J), [Amelia Glaese](https://arxiv.org/search/cs?searchtype=author&query=Glaese%2C+A), [Nat McAleese](https://arxiv.org/search/cs?searchtype=author&query=McAleese%2C+N), [Geoffrey Irving](https://arxiv.org/search/cs?searchtype=author&query=Irving%2C+G)

> Language Models (LMs) often cannot be deployed because of their potential to harm users in hard-to-predict ways. Prior work identifies harmful behaviors before deployment by using human annotators to hand-write test cases. However, human annotation is expensive, limiting the number and diversity of test cases. In this work, we automatically find cases where a target LM behaves in a harmful way, by generating test cases ("red teaming") using another LM. We evaluate the target LM's replies to generated test questions using a classifier trained to detect offensive content, uncovering tens of thousands of offensive replies in a 280B parameter LM chatbot. We explore several methods, from zero-shot generation to reinforcement learning, for generating test cases with varying levels of diversity and difficulty. Furthermore, we use prompt engineering to control LM-generated test cases to uncover a variety of other harms, automatically finding groups of people that the chatbot discusses in offensive ways, personal and hospital phone numbers generated as the chatbot's own contact info, leakage of private training data in generated text, and harms that occur over the course of a conversation. Overall, LM-based red teaming is one promising tool (among many needed) for finding and fixing diverse, undesirable LM behaviors before impacting users.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Cryptography and Security (cs.CR); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2202.03286](https://arxiv.org/abs/2202.03286) [cs.CL]** |
|           | (or **[arXiv:2202.03286v1](https://arxiv.org/abs/2202.03286v1) [cs.CL]** for this version) |







# 2022-02-07

[Return to Index](#Index)



<h2 id="2022-02-07-1">1. Data Scaling Laws in NMT: The Effect of Noise and Architecture
</h2>

Title: [Data Scaling Laws in NMT: The Effect of Noise and Architecture](https://arxiv.org/abs/2202.01994)

Authors: [Yamini Bansal](https://arxiv.org/search/cs?searchtype=author&query=Bansal%2C+Y), [Behrooz Ghorbani](https://arxiv.org/search/cs?searchtype=author&query=Ghorbani%2C+B), [Ankush Garg](https://arxiv.org/search/cs?searchtype=author&query=Garg%2C+A), [Biao Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+B), [Maxim Krikun](https://arxiv.org/search/cs?searchtype=author&query=Krikun%2C+M), [Colin Cherry](https://arxiv.org/search/cs?searchtype=author&query=Cherry%2C+C), [Behnam Neyshabur](https://arxiv.org/search/cs?searchtype=author&query=Neyshabur%2C+B), [Orhan Firat](https://arxiv.org/search/cs?searchtype=author&query=Firat%2C+O)

> In this work, we study the effect of varying the architecture and training data quality on the data scaling properties of Neural Machine Translation (NMT). First, we establish that the test loss of encoder-decoder transformer models scales as a power law in the number of training samples, with a dependence on the model size. Then, we systematically vary aspects of the training setup to understand how they impact the data scaling laws. In particular, we change the following (1) Architecture and task setup: We compare to a transformer-LSTM hybrid, and a decoder-only transformer with a language modeling loss (2) Noise level in the training distribution: We experiment with filtering, and adding iid synthetic noise. In all the above cases, we find that the data scaling exponents are minimally impacted, suggesting that marginally worse architectures or training data can be compensated for by adding more data. Lastly, we find that using back-translated data instead of parallel data, can significantly degrade the scaling exponent.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2202.01994](https://arxiv.org/abs/2202.01994) [cs.LG]** |
|           | (or **[arXiv:2202.01994v1](https://arxiv.org/abs/2202.01994v1) [cs.LG]** for this version) |





<h2 id="2022-02-07-2">2. Temporal Attention for Language Models
</h2>

Title: [Temporal Attention for Language Models](https://arxiv.org/abs/2202.02093)

Authors: [Guy D. Rosin](https://arxiv.org/search/cs?searchtype=author&query=Rosin%2C+G+D), [Kira Radinsky](https://arxiv.org/search/cs?searchtype=author&query=Radinsky%2C+K)

> Pretrained language models based on the transformer architecture have shown great success in NLP. Textual training data often comes from the web and is thus tagged with time-specific information, but most language models ignore this information. They are trained on the textual data alone, limiting their ability to generalize temporally. In this work, we extend the key component of the transformer architecture, i.e., the self-attention mechanism, and propose temporal attention - a time-aware self-attention mechanism. Temporal attention can be applied to any transformer model and requires the input texts to be accompanied with their relevant time points. It allows the transformer to capture this temporal information and create time-specific contextualized word representations. We leverage these representations for the task of semantic change detection; we apply our proposed mechanism to BERT and experiment on three datasets in different languages (English, German, and Latin) that also vary in time, size, and genre. Our proposed model achieves state-of-the-art results on all the datasets.

| Comments: | 8 pages                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2202.02093](https://arxiv.org/abs/2202.02093) [cs.CL]** |
|           | (or **[arXiv:2202.02093v1](https://arxiv.org/abs/2202.02093v1) [cs.CL]** for this version) |





<h2 id="2022-02-07-3">3. The Ecological Footprint of Neural Machine Translation Systems
</h2>

Title: [The Ecological Footprint of Neural Machine Translation Systems](https://arxiv.org/abs/2202.02170)

Authors: [Dimitar Sherionov](https://arxiv.org/search/cs?searchtype=author&query=Sherionov%2C+D), [Eva Vanmassenhove](https://arxiv.org/search/cs?searchtype=author&query=Vanmassenhove%2C+E)

> Over the past decade, deep learning (DL) has led to significant advancements in various fields of artificial intelligence, including machine translation (MT). These advancements would not be possible without the ever-growing volumes of data and the hardware that allows large DL models to be trained efficiently. Due to the large amount of computing cores as well as dedicated memory, graphics processing units (GPUs) are a more effective hardware solution for training and inference with DL models than central processing units (CPUs). However, the former is very power demanding. The electrical power consumption has economical as well as ecological implications. 
> This chapter focuses on the ecological footprint of neural MT systems. It starts from the power drain during the training of and the inference with neural MT models and moves towards the environment impact, in terms of carbon dioxide emissions. Different architectures (RNN and Transformer) and different GPUs (consumer-grate NVidia 1080Ti and workstation-grade NVidia P100) are compared. Then, the overall CO2 offload is calculated for Ireland and the Netherlands. The NMT models and their ecological impact are compared to common household appliances to draw a more clear picture. 
> The last part of this chapter analyses quantization, a technique for reducing the size and complexity of models, as a way to reduce power consumption. As quantized models can run on CPUs, they present a power-efficient inference solution without depending on a GPU.

| Comments: | 25 pages, 3 figures, 10 tables                               |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2202.02170](https://arxiv.org/abs/2202.02170) [cs.CL]** |
|           | (or **[arXiv:2202.02170v1](https://arxiv.org/abs/2202.02170v1) [cs.CL]** for this version) |



