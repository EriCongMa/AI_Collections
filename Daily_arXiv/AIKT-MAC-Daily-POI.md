# MA C.'s Daily Paper Of Interest - July a., 2022

# Index

- [2022-07-04](#2022-07-04)
  - [1. MultiViz: An Analysis Benchmark for Visualizing and Understanding Multimodal Models](#2022-07-04-1)

  - [2. Towards Human-Agent Communication via the Information Bottleneck Principle](#2022-07-04-2)
  
  - [3. VL-CheckList: Evaluating Pre-trained Vision-Language Models with Objects, Attributes and Relations](#2022-07-04-3)
  
- [2022-07-01](#2022-07-01)
  - [1. Building Multilingual Machine Translation Systems That Serve Arbitrary X-Y Translations](#2022-07-01-1)

  - [2. GSCLIP : A Framework for Explaining Distribution Shifts in Natural Language](#2022-07-01-2)

  - [3. FL-Tuning: Layer Tuning for Feed-Forward Network in Transformer](#2022-07-01-3)

- [2022-06-29](#2022-06-29)
  - [1. Wav2Vec-Aug: Improved self-supervised training with limited data](#2022-06-29-1)

- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



# 2022-07-04

[Return to Index](#Index)



<h2 id="2022-07-04-1">1. MultiViz: An Analysis Benchmark for Visualizing and Understanding Multimodal Models
</h2>

Title: [MultiViz: An Analysis Benchmark for Visualizing and Understanding Multimodal Models](https://arxiv.org/abs/2207.00056)
Authors: [Paul Pu Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+P+P), [Yiwei Lyu](https://arxiv.org/search/cs?searchtype=author&query=Lyu%2C+Y), [Gunjan Chhablani](https://arxiv.org/search/cs?searchtype=author&query=Chhablani%2C+G), [Nihal Jain](https://arxiv.org/search/cs?searchtype=author&query=Jain%2C+N), [Zihao Deng](https://arxiv.org/search/cs?searchtype=author&query=Deng%2C+Z), [Xingbo Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Louis-Philippe Morency](https://arxiv.org/search/cs?searchtype=author&query=Morency%2C+L), [Ruslan Salakhutdinov](https://arxiv.org/search/cs?searchtype=author&query=Salakhutdinov%2C+R)

> The promise of multimodal models for real-world applications has inspired research in visualizing and understanding their internal mechanics with the end goal of empowering stakeholders to visualize model behavior, perform model debugging, and promote trust in machine learning models. However, modern multimodal models are typically black-box neural networks, which makes it challenging to understand their internal mechanics. How can we visualize the internal modeling of multimodal interactions in these models? Our paper aims to fill this gap by proposing MultiViz, a method for analyzing the behavior of multimodal models by scaffolding the problem of interpretability into 4 stages: (1) unimodal importance: how each modality contributes towards downstream modeling and prediction, (2) cross-modal interactions: how different modalities relate with each other, (3) multimodal representations: how unimodal and cross-modal interactions are represented in decision-level features, and (4) multimodal prediction: how decision-level features are composed to make a prediction. MultiViz is designed to operate on diverse modalities, models, tasks, and research areas. Through experiments on 8 trained models across 6 real-world tasks, we show that the complementary stages in MultiViz together enable users to (1) simulate model predictions, (2) assign interpretable concepts to features, (3) perform error analysis on model misclassifications, and (4) use insights from error analysis to debug models. MultiViz is publicly available, will be regularly updated with new interpretation tools and metrics, and welcomes inputs from the community.

| Comments: | Code available at: [this https URL](https://github.com/pliang279/MultiViz). arXiv admin note: substantial text overlap with [arXiv:2107.07502](https://arxiv.org/abs/2107.07502) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV); Multimedia (cs.MM) |
| Cite as:  | **[arXiv:2207.00056](https://arxiv.org/abs/2207.00056) [cs.LG]** |
|           | (or **[arXiv:2207.00056v1](https://arxiv.org/abs/2207.00056v1) [cs.LG]** for this version) |
|           | https://doi.org/10.48550/arXiv.2207.00056Focus to learn more |





<h2 id="2022-07-04-2">2. Towards Human-Agent Communication via the Information Bottleneck Principle
</h2>

Title: [Towards Human-Agent Communication via the Information Bottleneck Principle](https://arxiv.org/abs/2207.00088)
Authors: [Mycal Tucker](https://arxiv.org/search/cs?searchtype=author&query=Tucker%2C+M), [Julie Shah](https://arxiv.org/search/cs?searchtype=author&query=Shah%2C+J), [Roger Levy](https://arxiv.org/search/cs?searchtype=author&query=Levy%2C+R), [Noga Zaslavsky](https://arxiv.org/search/cs?searchtype=author&query=Zaslavsky%2C+N)

> Emergent communication research often focuses on optimizing task-specific utility as a driver for communication. However, human languages appear to evolve under pressure to efficiently compress meanings into communication signals by optimizing the Information Bottleneck tradeoff between informativeness and complexity. In this work, we study how trading off these three factors -- utility, informativeness, and complexity -- shapes emergent communication, including compared to human communication. To this end, we propose Vector-Quantized Variational Information Bottleneck (VQ-VIB), a method for training neural agents to compress inputs into discrete signals embedded in a continuous space. We train agents via VQ-VIB and compare their performance to previously proposed neural architectures in grounded environments and in a Lewis reference game. Across all neural architectures and settings, taking into account communicative informativeness benefits communication convergence rates, and penalizing communicative complexity leads to human-like lexicon sizes while maintaining high utility. Additionally, we find that VQ-VIB outperforms other discrete communication methods. This work demonstrates how fundamental principles that are believed to characterize human language evolution may inform emergent communication in artificial agents.

| Subjects: | **Artificial Intelligence (cs.AI)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2207.00088](https://arxiv.org/abs/2207.00088) [cs.AI]** |
|           | (or **[arXiv:2207.00088v1](https://arxiv.org/abs/2207.00088v1) [cs.AI]** for this version) |
|           | https://doi.org/10.48550/arXiv.2207.00088Focus to learn more |





<h2 id="2022-07-04-3">3. VL-CheckList: Evaluating Pre-trained Vision-Language Models with Objects, Attributes and Relations
</h2>

Title: [VL-CheckList: Evaluating Pre-trained Vision-Language Models with Objects, Attributes and Relations](https://arxiv.org/abs/2207.00221)
Authors: [Tiancheng Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+T), [Tianqi Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+T), [Mingwei Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+M), [Haozhan Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+H), [Kyusong Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+K), [Xiaopeng Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+X), [Jianwei Yin](https://arxiv.org/search/cs?searchtype=author&query=Yin%2C+J)

> Vision-Language Pretraining (VLP) models have recently successfully facilitated many cross-modal downstream tasks. Most existing works evaluated their systems by comparing the fine-tuned downstream task performance. However, only average downstream task accuracy provides little information about the pros and cons of each VLP method, let alone provides insights on how the community can improve the systems in the future. Inspired by the CheckList for testing natural language processing, we introduce VL-CheckList, a novel framework to understand the capabilities of VLP models. The proposed method divides the image-texting ability of a VLP model into three categories: objects, attributes, and relations, and uses a novel taxonomy to further break down these three aspects. We conduct comprehensive studies to analyze seven recently popular VLP models via the proposed framework. Results confirm the effectiveness of the proposed method by revealing fine-grained differences among the compared models that were not visible from downstream task-only evaluation. Further results show promising research direction in building better VLP models. Data and Code: [this https URL](https://github.com/om-ai-lab/VL-CheckList)

| Comments: | 9 pages, preprint                                            |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2207.00221](https://arxiv.org/abs/2207.00221) [cs.CV]** |
|           | (or **[arXiv:2207.00221v1](https://arxiv.org/abs/2207.00221v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2207.00221Focus to learn more |






# 2022-07-01

[Return to Index](#Index)



<h2 id="2022-07-01-1">1. Building Multilingual Machine Translation Systems That Serve Arbitrary X-Y Translations
</h2>

Title: [Building Multilingual Machine Translation Systems That Serve Arbitrary X-Y Translations](https://arxiv.org/abs/2206.14982)

Authors: [Akiko Eriguchi](https://arxiv.org/search/cs?searchtype=author&query=Eriguchi%2C+A), [Shufang Xie](https://arxiv.org/search/cs?searchtype=author&query=Xie%2C+S), [Tao Qin](https://arxiv.org/search/cs?searchtype=author&query=Qin%2C+T), [Hany Hassan Awadalla](https://arxiv.org/search/cs?searchtype=author&query=Awadalla%2C+H+H)

> Multilingual Neural Machine Translation (MNMT) enables one system to translate sentences from multiple source languages to multiple target languages, greatly reducing deployment costs compared with conventional bilingual systems. The MNMT training benefit, however, is often limited to many-to-one directions. The model suffers from poor performance in one-to-many and many-to-many with zero-shot setup. To address this issue, this paper discusses how to practically build MNMT systems that serve arbitrary X-Y translation directions while leveraging multilinguality with a two-stage training strategy of pretraining and finetuning. Experimenting with the WMT'21 multilingual translation task, we demonstrate that our systems outperform the conventional baselines of direct bilingual models and pivot translation models for most directions, averagely giving +6.0 and +4.1 BLEU, without the need for architecture change or extra data collection. Moreover, we also examine our proposed approach in an extremely large-scale data setting to accommodate practical deployment scenarios.

| Comments: | NAACL 2022                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2206.14982](https://arxiv.org/abs/2206.14982) [cs.CL]** |
|           | (or **[arXiv:2206.14982v1](https://arxiv.org/abs/2206.14982v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.14982Focus to learn more |





<h2 id="2022-07-01-2">2. GSCLIP : A Framework for Explaining Distribution Shifts in Natural Language
</h2>

Title: [GSCLIP : A Framework for Explaining Distribution Shifts in Natural Language](https://arxiv.org/abs/2206.15007)

Authors: [Zhiying Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+Z), [Weixin Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+W), [James Zou](https://arxiv.org/search/cs?searchtype=author&query=Zou%2C+J)

> Helping end users comprehend the abstract distribution shifts can greatly facilitate AI deployment. Motivated by this, we propose a novel task, dataset explanation. Given two image data sets, dataset explanation aims to automatically point out their dataset-level distribution shifts with natural language. Current techniques for monitoring distribution shifts provide inadequate information to understand datasets with the goal of improving data quality. Therefore, we introduce GSCLIP, a training-free framework to solve the dataset explanation task. In GSCLIP, we propose the selector as the first quantitative evaluation method to identify explanations that are proper to summarize dataset shifts. Furthermore, we leverage this selector to demonstrate the superiority of a generator based on language model generation. Systematic evaluation on natural data shift verifies that GSCLIP, a combined system of a hybrid generator group and an efficient selector is not only easy-to-use but also powerful for dataset explanation at scale.

| Comments: | Accepted by ICML 2022 DataPerf                               |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2206.15007](https://arxiv.org/abs/2206.15007) [cs.CL]** |
|           | (or **[arXiv:2206.15007v1](https://arxiv.org/abs/2206.15007v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.15007Focus to learn more |





<h2 id="2022-07-01-3">3. FL-Tuning: Layer Tuning for Feed-Forward Network in Transformer
</h2>

Title: [FL-Tuning: Layer Tuning for Feed-Forward Network in Transformer](https://arxiv.org/abs/2206.15312)

Authors: [Jingping Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+J), [Yuqiu Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+Y), [Kui Xue](https://arxiv.org/search/cs?searchtype=author&query=Xue%2C+K), [Hongli Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+H), [Chao Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+C), [Lihan Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+L), [Haiyun Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+H), [Jiaqing Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+J), [Tong Ruan](https://arxiv.org/search/cs?searchtype=author&query=Ruan%2C+T)

> Prompt tuning is an emerging way of adapting pre-trained language models to downstream tasks. However, the existing studies are mainly to add prompts to the input sequence. This way would not work as expected due to the intermediate multi-head self-attention and feed-forward network computation, making model optimization not very smooth. Hence, we propose a novel tuning way called layer tuning, aiming to add learnable parameters in Transformer layers. Specifically, we focus on layer tuning for feed-forward network in the Transformer, namely FL-tuning. It introduces additional units into the hidden layer of each feed-forward network. We conduct extensive experiments on the public CLUE benchmark. The results show that: 1) Our FL-tuning outperforms prompt tuning methods under both full-data and few-shot settings in almost all cases. In particular, it improves accuracy by 17.93% (full-data setting) on WSC 1.0 and F1 by 16.142% (few-shot setting) on CLUENER over P-tuning v2. 2) Our FL-tuning is more stable and converges about 1.17 times faster than P-tuning v2. 3) With only about 3% of Transformer's parameters to be trained, FL-tuning is comparable with fine-tuning on most datasets, and significantly outperforms fine-tuning (e.g., accuracy improved by 12.9% on WSC 1.1) on several datasets. The source codes are available at [this https URL](https://github.com/genggui001/FL-Tuning).

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.15312](https://arxiv.org/abs/2206.15312) [cs.CL]** |
|           | (or **[arXiv:2206.15312v1](https://arxiv.org/abs/2206.15312v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.15312Focus to learn more |






# 2022-06-29

[Return to Index](#Index)



<h2 id="2022-06-29-1">1. Wav2Vec-Aug: Improved self-supervised training with limited data
</h2>

Title: [Wav2Vec-Aug: Improved self-supervised training with limited data](https://arxiv.org/abs/2206.13654)

Authors:  [Anuroop Sriram](https://arxiv.org/search/cs?searchtype=author&query=Sriram%2C+A), [Michael Auli](https://arxiv.org/search/cs?searchtype=author&query=Auli%2C+M), [Alexei Baevski](https://arxiv.org/search/cs?searchtype=author&query=Baevski%2C+A)

> Self-supervised learning (SSL) of speech representations has received much attention over the last few years but most work has focused on languages and domains with an abundance of unlabeled data. However, for many languages there is a shortage even in the unlabeled data which limits the effectiveness of SSL. In this work, we focus on the problem of applying SSL to domains with limited available data by leveraging data augmentation for Wav2Vec 2.0 pretraining. Further, we propose improvements to each component of the model which result in a combined relative word error rate (WER) improvement of up to 13% compared to Wav2Vec 2.0 on Librispeech test-clean / other.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.13654](https://arxiv.org/abs/2206.13654) [cs.CL]** |
|           | (or **[arXiv:2206.13654v1](https://arxiv.org/abs/2206.13654v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.13654Focus to learn more |



