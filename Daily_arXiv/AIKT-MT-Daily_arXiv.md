# Daily arXiv: Machine Translation - September, 2021

# Index


- [2021-09-01](#2021-09-01)

  - [1. SimulLR: Simultaneous Lip Reading Transducer with Attention-Guided Adaptive Memory](#2021-09-01-1)
  - [2. Want To Reduce Labeling Cost? GPT-3 Can Help](#2021-09-01-2)
  - [3. T3-Vis: a visual analytic framework for Training and fine-Tuning Transformers in NLP](#2021-09-01-3)
  - [4. Enjoy the Salience: Towards Better Transformer-based Faithful Explanations with Word Salience](#2021-09-01-4)
  - [5. Thermostat: A Large Collection of NLP Model Explanations and Analysis Tools](#2021-09-01-5)
- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MT-Daily_arXiv-index.md)



# 2021-09-01

[Return to Index](#Index)



<h2 id="2021-09-01-1">1. SimulLR: Simultaneous Lip Reading Transducer with Attention-Guided Adaptive Memory
</h2>


Title: [SimulLR: Simultaneous Lip Reading Transducer with Attention-Guided Adaptive Memory](https://arxiv.org/abs/2108.13630)

Authors: [Zhijie Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+Z), [Zhou Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+Z), [Haoyuan Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+H), [Jinglin Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+J), [Meng Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+M), [Xingshan Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+X), [Xiaofei He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+X)

> Lip reading, aiming to recognize spoken sentences according to the given video of lip movements without relying on the audio stream, has attracted great interest due to its application in many scenarios. Although prior works that explore lip reading have obtained salient achievements, they are all trained in a non-simultaneous manner where the predictions are generated requiring access to the full video. To breakthrough this constraint, we study the task of simultaneous lip reading and devise SimulLR, a simultaneous lip Reading transducer with attention-guided adaptive memory from three aspects: (1) To address the challenge of monotonic alignments while considering the syntactic structure of the generated sentences under simultaneous setting, we build a transducer-based model and design several effective training strategies including CTC pre-training, model warm-up and curriculum learning to promote the training of the lip reading transducer. (2) To learn better spatio-temporal representations for simultaneous encoder, we construct a truncated 3D convolution and time-restricted self-attention layer to perform the frame-to-frame interaction within a video segment containing fixed number of frames. (3) The history information is always limited due to the storage in real-time scenarios, especially for massive video data. Therefore, we devise a novel attention-guided adaptive memory to organize semantic information of history segments and enhance the visual representations with acceptable computation-aware latency. The experiments show that the SimulLR achieves the translation speedup 9.10× compared with the state-of-the-art non-simultaneous methods, and also obtains competitive results, which indicates the effectiveness of our proposed methods.

| Comments: | ACMMM 2021                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2108.13630](https://arxiv.org/abs/2108.13630) [cs.CV]** |
|           | (or **[arXiv:2108.13630v1](https://arxiv.org/abs/2108.13630v1) [cs.CV]** for this version) |





<h2 id="2021-09-01-2">2. Want To Reduce Labeling Cost? GPT-3 Can Help
</h2>


Title: [Want To Reduce Labeling Cost? GPT-3 Can Help](https://arxiv.org/abs/2108.13487)

Authors: [Shuohang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+S), [Yang Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y), [Yichong Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+Y), [Chenguang Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+C), [Michael Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+M)

> Data annotation is a time-consuming and labor-intensive process for many NLP tasks. Although there exist various methods to produce pseudo data labels, they are often task-specific and require a decent amount of labeled data to start with. Recently, the immense language model GPT-3 with 175 billion parameters has achieved tremendous improvement across many few-shot learning tasks. In this paper, we explore ways to leverage GPT-3 as a low-cost data labeler to train other models. We find that, to make the downstream model achieve the same performance on a variety of NLU and NLG tasks, it costs 50% to 96% less to use labels from GPT-3 than using labels from humans. Furthermore, we propose a novel framework of combining pseudo labels from GPT-3 with human labels, which leads to even better performance with limited labeling budget. These results present a cost-effective data labeling methodology that is generalizable to many practical applications.

| Comments: | Findings of EMNLP 2021, 11 pages                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2108.13487](https://arxiv.org/abs/2108.13487) [cs.CL]** |
|           | (or **[arXiv:2108.13487v1](https://arxiv.org/abs/2108.13487v1) [cs.CL]** for this version) |





<h2 id="2021-09-01-3">3. T3-Vis: a visual analytic framework for Training and fine-Tuning Transformers in NLP
</h2>


Title: [T3-Vis: a visual analytic framework for Training and fine-Tuning Transformers in NLP](https://arxiv.org/abs/2108.13587)

Authors: [Raymond Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+R) (1), [Wen Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+W) (1), [Lanjun Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L) (2), [Hyeju Jang](https://arxiv.org/search/cs?searchtype=author&query=Jang%2C+H) (1), [Giuseppe Carenini](https://arxiv.org/search/cs?searchtype=author&query=Carenini%2C+G) (1) ((1) University of British Columbia, (2) Huawei Cananda Technologies Co. Ltd.)

> Transformers are the dominant architecture in NLP, but their training and fine-tuning is still very challenging. In this paper, we present the design and implementation of a visual analytic framework for assisting researchers in such process, by providing them with valuable insights about the model's intrinsic properties and behaviours. Our framework offers an intuitive overview that allows the user to explore different facets of the model (e.g., hidden states, attention) through interactive visualization, and allows a suite of built-in algorithms that compute the importance of model components and different parts of the input sequence. Case studies and feedback from a user focus group indicate that the framework is useful, and suggest several improvements.

| Comments: | 10 pages, 4 figures, accepted to EMNLP 2021 System Demonstration |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Human-Computer Interaction (cs.HC) |
| Cite as:  | **[arXiv:2108.13587](https://arxiv.org/abs/2108.13587) [cs.CL]** |
|           | (or **[arXiv:2108.13587v1](https://arxiv.org/abs/2108.13587v1) [cs.CL]** for this version) |





<h2 id="2021-09-01-4">4. Enjoy the Salience: Towards Better Transformer-based Faithful Explanations with Word Salience
</h2>


Title: [Enjoy the Salience: Towards Better Transformer-based Faithful Explanations with Word Salience](https://arxiv.org/abs/2108.13759)

Authors: [George Chrysostomou](https://arxiv.org/search/cs?searchtype=author&query=Chrysostomou%2C+G), [Nikolaos Aletras](https://arxiv.org/search/cs?searchtype=author&query=Aletras%2C+N)

> Pretrained transformer-based models such as BERT have demonstrated state-of-the-art predictive performance when adapted into a range of natural language processing tasks. An open problem is how to improve the faithfulness of explanations (rationales) for the predictions of these models. In this paper, we hypothesize that salient information extracted a priori from the training data can complement the task-specific information learned by the model during fine-tuning on a downstream task. In this way, we aim to help BERT not to forget assigning importance to informative input tokens when making predictions by proposing SaLoss; an auxiliary loss function for guiding the multi-head attention mechanism during training to be close to salient information extracted a priori using TextRank. Experiments for explanation faithfulness across five datasets, show that models trained with SaLoss consistently provide more faithful explanations across four different feature attribution methods compared to vanilla BERT. Using the rationales extracted from vanilla BERT and SaLoss models to train inherently faithful classifiers, we further show that the latter result in higher predictive performance in downstream tasks.

| Comments: | EMNLP 2021 Pre-print                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2108.13759](https://arxiv.org/abs/2108.13759) [cs.CL]** |
|           | (or **[arXiv:2108.13759v1](https://arxiv.org/abs/2108.13759v1) [cs.CL]** for this version) |





<h2 id="2021-09-01-5">5. Thermostat: A Large Collection of NLP Model Explanations and Analysis Tools
</h2>


Title: [Thermostat: A Large Collection of NLP Model Explanations and Analysis Tools](https://arxiv.org/abs/2108.13961)

Authors: [Nils Feldhus](https://arxiv.org/search/cs?searchtype=author&query=Feldhus%2C+N), [Robert Schwarzenberg](https://arxiv.org/search/cs?searchtype=author&query=Schwarzenberg%2C+R), [Sebastian Möller](https://arxiv.org/search/cs?searchtype=author&query=Möller%2C+S)

> In the language domain, as in other domains, neural explainability takes an ever more important role, with feature attribution methods on the forefront. Many such methods require considerable computational resources and expert knowledge about implementation details and parameter choices. To facilitate research, we present Thermostat which consists of a large collection of model explanations and accompanying analysis tools. Thermostat allows easy access to over 200k explanations for the decisions of prominent state-of-the-art models spanning across different NLP tasks, generated with multiple explainers. The dataset took over 10k GPU hours (> one year) to compile; compute time that the community now saves. The accompanying software tools allow to analyse explanations instance-wise but also accumulatively on corpus level. Users can investigate and compare models, datasets and explainers without the need to orchestrate implementation details. Thermostat is fully open source, democratizes explainability research in the language domain, circumvents redundant computations and increases comparability and replicability.

| Comments: | Accepted to EMNLP 2021 System Demonstrations                 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2108.13961](https://arxiv.org/abs/2108.13961) [cs.CL]** |
|           | (or **[arXiv:2108.13961v1](https://arxiv.org/abs/2108.13961v1) [cs.CL]** for this version) |



