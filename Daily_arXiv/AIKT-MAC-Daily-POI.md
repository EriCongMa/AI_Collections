# MA C.'s Daily Paper Of Interest - March, 2022

# Index

- [2022-03-11](#2022-03-11)
  - [1. NLX-GPT: A Model for Natural Language Explanations in Vision and Vision-Language Tasks](#2022-03-11-1)
  - [2. Conditional Prompt Learning for Vision-Language Models](#2022-03-11-2)
  - [3. Faithfulness in Natural Language Generation: A Systematic Survey of Analysis, Evaluation and Optimization Methods](#2022-03-11-3)
  - [4. Look Backward and Forward: Self-Knowledge Distillation with Bidirectional Decoder for Neural Machine Translation](#2022-03-11-4)
  
- [2022-03-10](#2022-03-10)
  - [1. Efficient Sub-structured Knowledge Distillation](#2022-03-10-1)
  - [2. Model-Agnostic Multitask Fine-tuning for Few-shot Vision-Language Transfer Learning](#2022-03-10-2)
  - [3. Pose Guided Multi-person Image Generation From Text](#2022-03-10-3)
  - [4. Onception: Active Learning with Expert Advice for Real World Machine Translation](#2022-03-10-4)

- [2022-03-09](#2022-03-09)
  - [1. Multi-Modal Mixup for Robust Fine-tuning](#2022-03-09-1)
  - [2. IT5: Large-scale Text-to-text Pretraining for Italian Language Understanding and Generation](#2022-03-09-2)
  - [3. UniXcoder: Unified Cross-Modal Pre-training for Code Representation](#2022-03-09-3)
  - [4. HyperPELT: Unified Parameter-Efficient Language Model Tuning for Both Language and Vision-and-Language Tasks](#2022-03-09-4)
  - [5. Overcoming Catastrophic Forgetting beyond Continual Learning: Balanced Training for Neural Machine Translation](#2022-03-09-5)
  - [6. Adaptr: Objective-Centric Adaptation Framework for Language Models](#2022-03-09-6)
- [2022-03-08](#2022-03-08)
  - [1. OCR quality affects perceived usefulness of historical newspaper clippings -- a user study](#2022-03-08-1)
  - [2. Focus on the Target's Vocabulary: Masked Label Smoothing for Machine Translation](#2022-03-08-2)
  - [3. Conditional Bilingual Mutual Information Based Adaptive Training for Neural Machine Translation](#2022-03-08-3)
  - [4. Recent Advances in Neural Text Generation: A Task-Agnostic Survey](#2022-03-08-4)
  - [5. Input-Tuning: Adapting Unfamiliar Inputs to Frozen Pretrained Models](#2022-03-08-5)
  - [6. One Model, Multiple Tasks: Pathways for Natural Language Understanding](#2022-03-08-6)
- [2022-03-07](#2022-03-07)
  - [1. Overlap-based Vocabulary Generation Improves Cross-lingual Transfer Among Related Languages](#2022-03-07-1)
  - [2. Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning](#2022-03-07-2)
  - [3. EAG: Extract and Generate Multi-way Aligned Corpus for Complete Multi-lingual Neural Machine Translation](#2022-03-07-3)
  - [4. Comprehension of Subtitles from Re-Translating Simultaneous Speech Translation](#2022-03-07-4)
  - [5. From Simultaneous to Streaming Machine Translation by Leveraging Streaming History](#2022-03-07-5)
- [2022-03-04](#2022-03-04)
  - [1. Recent, rapid advancement in visual question answering architecture](#2022-03-04-1)
  - [2. Vision-Language Intelligence: Tasks, Representation Learning, and Large Models](#2022-03-04-2)
  - [3. UDAAN - Machine Learning based Post-Editing tool for Document Translation](#2022-03-04-3)
- [2022-03-03](#2022-03-03)
  - [1. HighMMT: Towards Modality and Task Generalization for High-Modality Representation Learning](#2022-03-03-1)
  - [2. Attend, Memorize and Generate: Towards Faithful Table-to-Text Generation in Few Shots](#2022-03-03-2)
  - [3. HyperPrompt: Prompt-based Task-Conditioning of Transformers](#2022-03-03-3)
  - [4. Do Prompts Solve NLP Tasks Using Natural Language?](#2022-03-03-4)
  - [5. Parameter-Efficient Mixture-of-Experts Architecture for Pre-trained Language Models](#2022-03-03-5)
- [2022-03-02](#2022-03-02)
  - [1. Exploring and Adapting Chinese GPT to Pinyin Input Method](#2022-03-02-1)
  - [2. TableFormer: Robust Transformer Modeling for Table-Text Encoding](#2022-03-02-2)
  - [3. DeepNet: Scaling Transformers to 1,000 Layers](#2022-03-02-3)
- [2022-03-01](#2022-03-01)
  - [1. Interactive Machine Learning for Image Captioning](#2022-03-01-1)
  - [2. Multi-Level Contrastive Learning for Cross-Lingual Alignment](#2022-03-01-2)
  - [3. OCR Improves Machine Translation for Low-Resource Languages](#2022-03-01-3)
  - [4. CINO: A Chinese Minority Pre-trained Language Model](#2022-03-01-4)
  - [5. LCP-dropout: Compression-based Multiple Subword Segmentation for Neural Machine Translation](#2022-03-01-5)
  - [6. MSCTD: A Multimodal Sentiment Chat Translation Dataset](#2022-03-01-6)
  - [7. Confidence Based Bidirectional Global Context Aware Training Framework for Neural Machine Translation](#2022-03-01-7)
  - [8. LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding](#2022-03-01-8)
- [2022-02-28](#2022-02-28)
  - [1. Screening Gender Transfer in Neural Machine Translation](#2022-02-28-1)
- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



# 2022-03-11

[Return to Index](#Index)



<h2 id="2022-03-11-1">1. NLX-GPT: A Model for Natural Language Explanations in Vision and Vision-Language Tasks
</h2>

Title: [NLX-GPT: A Model for Natural Language Explanations in Vision and Vision-Language Tasks](https://arxiv.org/abs/2203.05081)

Authors: [Fawaz Sammani](https://arxiv.org/search/cs?searchtype=author&query=Sammani%2C+F), [Tanmoy Mukherjee](https://arxiv.org/search/cs?searchtype=author&query=Mukherjee%2C+T), [Nikos Deligiannis](https://arxiv.org/search/cs?searchtype=author&query=Deligiannis%2C+N)

> Natural language explanation (NLE) models aim at explaining the decision-making process of a black box system via generating natural language sentences which are human-friendly, high-level and fine-grained. Current NLE models explain the decision-making process of a vision or vision-language model (a.k.a., task model), e.g., a VQA model, via a language model (a.k.a., explanation model), e.g., GPT. Other than the additional memory resources and inference time required by the task model, the task and explanation models are completely independent, which disassociates the explanation from the reasoning process made to predict the answer. We introduce NLX-GPT, a general, compact and faithful language model that can simultaneously predict an answer and explain it. We first conduct pre-training on large scale data of image-caption pairs for general understanding of images, and then formulate the answer as a text prediction task along with the explanation. Without region proposals nor a task model, our resulting overall framework attains better evaluation scores, contains much less parameters and is 15× faster than the current SoA model. We then address the problem of evaluating the explanations which can be in many times generic, data-biased and can come in several forms. We therefore design 2 new evaluation measures: (1) explain-predict and (2) retrieval-based attack, a self-evaluation framework that requires no labels. Code is at: [this https URL](https://github.com/fawazsammani/nlxgpt).

| Comments: | Accepted to CVPR 2022                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2203.05081](https://arxiv.org/abs/2203.05081) [cs.CV]** |
|           | (or **[arXiv:2203.05081v1](https://arxiv.org/abs/2203.05081v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.05081Focus to learn more |





<h2 id="2022-03-11-2">2. Conditional Prompt Learning for Vision-Language Models
</h2>

Title: [Conditional Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2203.05557)

Authors: [Kaiyang Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+K), [Jingkang Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+J), [Chen Change Loy](https://arxiv.org/search/cs?searchtype=author&query=Loy%2C+C+C), [Ziwei Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Z)

> With the rise of powerful pre-trained vision-language models like CLIP, it becomes essential to investigate ways to adapt these models to downstream datasets. A recently proposed method named Context Optimization (CoOp) introduces the concept of prompt learning -- a recent trend in NLP -- to the vision domain for adapting pre-trained vision-language models. Specifically, CoOp turns context words in a prompt into a set of learnable vectors and, with only a few labeled images for learning, can achieve huge improvements over intensively-tuned manual prompts. In our study we identify a critical problem of CoOp: the learned context is not generalizable to wider unseen classes within the same dataset, suggesting that CoOp overfits base classes observed during training. To address the problem, we propose Conditional Context Optimization (CoCoOp), which extends CoOp by further learning a lightweight neural network to generate for each image an input-conditional token (vector). Compared to CoOp's static prompts, our dynamic prompts adapt to each instance and are thus less sensitive to class shift. Extensive experiments show that CoCoOp generalizes much better than CoOp to unseen classes, even showing promising transferability beyond a single dataset; and yields stronger domain generalization performance as well. Code is available at [this https URL](https://github.com/KaiyangZhou/CoOp).

| Comments: | CVPR 2022. TL;DR: We propose a conditional prompt learning approach to solve the generalizability issue of static prompts |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2203.05557](https://arxiv.org/abs/2203.05557) [cs.CV]** |
|           | (or **[arXiv:2203.05557v1](https://arxiv.org/abs/2203.05557v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.05557Focus to learn more |





<h2 id="2022-03-11-3">3. Faithfulness in Natural Language Generation: A Systematic Survey of Analysis, Evaluation and Optimization Methods
</h2>

Title: [Faithfulness in Natural Language Generation: A Systematic Survey of Analysis, Evaluation and Optimization Methods](https://arxiv.org/abs/2203.05227)

Authors: [Wei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+W), [Wenhao Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+W), [Moye Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+M), [Jiachen Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+J), [Xinyan Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+X), [Hua Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+H)

> Natural Language Generation (NLG) has made great progress in recent years due to the development of deep learning techniques such as pre-trained language models. This advancement has resulted in more fluent, coherent and even properties controllable (e.g. stylistic, sentiment, length etc.) generation, naturally leading to development in downstream tasks such as abstractive summarization, dialogue generation, machine translation, and data-to-text generation. However, the faithfulness problem that the generated text usually contains unfaithful or non-factual information has become the biggest challenge, which makes the performance of text generation unsatisfactory for practical applications in many real-world scenarios. Many studies on analysis, evaluation, and optimization methods for faithfulness problems have been proposed for various tasks, but have not been organized, compared and discussed in a combined manner. In this survey, we provide a systematic overview of the research progress on the faithfulness problem of NLG, including problem analysis, evaluation metrics and optimization methods. We organize the evaluation and optimization methods for different tasks into a unified taxonomy to facilitate comparison and learning across tasks. Several research trends are discussed further.

| Comments: | The first version                                            |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2203.05227](https://arxiv.org/abs/2203.05227) [cs.CL]** |
|           | (or **[arXiv:2203.05227v1](https://arxiv.org/abs/2203.05227v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.05227Focus to learn more |





<h2 id="2022-03-11-4">4. Look Backward and Forward: Self-Knowledge Distillation with Bidirectional Decoder for Neural Machine Translation
</h2>

Title: [Look Backward and Forward: Self-Knowledge Distillation with Bidirectional Decoder for Neural Machine Translation](https://arxiv.org/abs/2203.05248)

Authors: [Xuanwei Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+X), [Libin Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+L), [Disheng Pan](https://arxiv.org/search/cs?searchtype=author&query=Pan%2C+D), [Liang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L), [Yanjun Miao](https://arxiv.org/search/cs?searchtype=author&query=Miao%2C+Y)

> Neural Machine Translation(NMT) models are usually trained via unidirectional decoder which corresponds to optimizing one-step-ahead prediction. However, this kind of unidirectional decoding framework may incline to focus on local structure rather than global coherence. To alleviate this problem, we propose a novel method, Self-Knowledge Distillation with Bidirectional Decoder for Neural Machine Translation(SBD-NMT). We deploy a backward decoder which can act as an effective regularization method to the forward decoder. By leveraging the backward decoder's information about the longer-term future, distilling knowledge learned in the backward decoder can encourage auto-regressive NMT models to plan ahead. Experiments show that our method is significantly better than the strong Transformer baselines on multiple machine translation data sets. Our codes will be released on github soon.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2203.05248](https://arxiv.org/abs/2203.05248) [cs.CL]** |
|           | (or **[arXiv:2203.05248v1](https://arxiv.org/abs/2203.05248v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.05248Focus to learn more |







# 2022-03-10

[Return to Index](#Index)



<h2 id="2022-03-10-1">1. Efficient Sub-structured Knowledge Distillation
</h2>

Title: [Efficient Sub-structured Knowledge Distillation](https://arxiv.org/abs/2203.04825)

Authors: [Wenye Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+W), [Yangming Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Lemao Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+L), [Shuming Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+S), [Hai-tao Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+H)

> Structured prediction models aim at solving a type of problem where the output is a complex structure, rather than a single variable. Performing knowledge distillation for such models is not trivial due to their exponentially large output space. In this work, we propose an approach that is much simpler in its formulation and far more efficient for training than existing approaches. Specifically, we transfer the knowledge from a teacher model to its student model by locally matching their predictions on all sub-structures, instead of the whole output space. In this manner, we avoid adopting some time-consuming techniques like dynamic programming (DP) for decoding output structures, which permits parallel computation and makes the training process even faster in practice. Besides, it encourages the student model to better mimic the internal behavior of the teacher model. Experiments on two structured prediction tasks demonstrate that our approach outperforms previous methods and halves the time cost for one training epoch.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2203.04825](https://arxiv.org/abs/2203.04825) [cs.LG]** |
|           | (or **[arXiv:2203.04825v1](https://arxiv.org/abs/2203.04825v1) [cs.LG]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.04825Focus to learn more |





<h2 id="2022-03-10-2">2. Model-Agnostic Multitask Fine-tuning for Few-shot Vision-Language Transfer Learning
</h2>

Title: [Model-Agnostic Multitask Fine-tuning for Few-shot Vision-Language Transfer Learning](https://arxiv.org/abs/2203.04904)

Authors: [Zhenhailong Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Z), [Hang Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+H), [Manling Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+M), [Han Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+H), [Heng Ji](https://arxiv.org/search/cs?searchtype=author&query=Ji%2C+H)

> Despite achieving state-of-the-art zero-shot performance, existing vision-language models, e.g., CLIP, still fall short of domain-specific classification tasks, e.g., Fungi Classification. In the context of few-shot transfer learning, traditional fine-tuning fails to prevent highly expressive model from exploiting spurious correlations in the training data. On the other hand, although model-agnostic meta-learning (MAML) presents as a natural alternative for transfer learning, the expensive computation due to implicit second-order optimization limits its use in large-scale models and datasets. In this work we aim to further improve the generalization of existing vision-language models on unseen tasks via a simple yet efficient fine-tuning strategy based on uniform task sampling. We term our method as Model-Agnostic Multitask Fine-tuning (MAMF). Compared with MAML, MAMF discards the bi-level optimization and uses only first-order gradients, which makes it easily scalable and computationally efficient. Due to the uniform task sampling procedure, MAMF consistently outperforms the classical fine-tuning method for few-shot transfer learning on five benchmark datasets. Empirically, we further discover that the effectiveness of first-order MAML is highly dependent on the zero-shot performance of the pretrained model, and our simple algorithm can outperform first-order MAML on more challenging datasets with low zero-shot performance.

| Comments: | 7 pages, 6 figures, under review                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Multimedia (cs.MM)**; Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV) |
| Cite as:  | **[arXiv:2203.04904](https://arxiv.org/abs/2203.04904) [cs.MM]** |
|           | (or **[arXiv:2203.04904v1](https://arxiv.org/abs/2203.04904v1) [cs.MM]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.04904Focus to learn more |





<h2 id="2022-03-10-3">3. Pose Guided Multi-person Image Generation From Text
</h2>

Title: [Pose Guided Multi-person Image Generation From Text](https://arxiv.org/abs/2203.04907)

Authors: [Soon Yau Cheong](https://arxiv.org/search/cs?searchtype=author&query=Cheong%2C+S+Y), [Armin Mustafa](https://arxiv.org/search/cs?searchtype=author&query=Mustafa%2C+A), [Andrew Gilbert](https://arxiv.org/search/cs?searchtype=author&query=Gilbert%2C+A)

> Transformers have recently been shown to generate high quality images from texts. However, existing methods struggle to create high fidelity full-body images, especially multiple people. A person's pose has a high degree of freedom that is difficult to describe using words only; this creates errors in the generated image, such as incorrect body proportions and pose. We propose a pose-guided text-to-image model, using pose as an additional input constraint. Using the proposed Keypoint Pose Encoding (KPE) to encode human pose into low dimensional representation, our model can generate novel multi-person images accurately representing the pose and text descriptions provided, with minimal errors. We demonstrate that KPE is invariant to changes in the target image domain and image resolution; we show results on the Deepfashion dataset and create a new multi-person Deepfashion dataset to demonstrate the multi-capabilities of our approach.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2203.04907](https://arxiv.org/abs/2203.04907) [cs.CV]** |
|           | (or **[arXiv:2203.04907v1](https://arxiv.org/abs/2203.04907v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.04907Focus to learn more |





<h2 id="2022-03-10-4">4. Onception: Active Learning with Expert Advice for Real World Machine Translation
</h2>

Title: [Onception: Active Learning with Expert Advice for Real World Machine Translation](https://arxiv.org/abs/2203.04507)

Authors: [Vânia Mendonça](https://arxiv.org/search/cs?searchtype=author&query=Mendonça%2C+V) (1 and 2), [Ricardo Rei](https://arxiv.org/search/cs?searchtype=author&query=Rei%2C+R) (1 and 2 and 3), [Luisa Coheur](https://arxiv.org/search/cs?searchtype=author&query=Coheur%2C+L) (1 and 2), [Alberto Sardinha](https://arxiv.org/search/cs?searchtype=author&query=Sardinha%2C+A) (1 and 2) ((1) INESC-ID Lisboa, (2) Instituto Superior Técnico, (3) Unbabel AI)

> Active learning can play an important role in low-resource settings (i.e., where annotated data is scarce), by selecting which instances may be more worthy to annotate. Most active learning approaches for Machine Translation assume the existence of a pool of sentences in a source language, and rely on human annotators to provide translations or post-edits, which can still be costly. In this paper, we assume a real world human-in-the-loop scenario in which: (i) the source sentences may not be readily available, but instead arrive in a stream; (ii) the automatic translations receive feedback in the form of a rating, instead of a correct/edited translation, since the human-in-the-loop might be a user looking for a translation, but not be able to provide one. To tackle the challenge of deciding whether each incoming pair source-translations is worthy to query for human feedback, we resort to a number of stream-based active learning query strategies. Moreover, since we not know in advance which query strategy will be the most adequate for a certain language pair and set of Machine Translation models, we propose to dynamically combine multiple strategies using prediction with expert advice. Our experiments show that using active learning allows to converge to the best Machine Translation systems with fewer human interactions. Furthermore, combining multiple strategies using prediction with expert advice often outperforms several individual active learning strategies with even fewer interactions.

| Comments: | Submitted to Machine Translation                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2203.04507](https://arxiv.org/abs/2203.04507) [cs.CL]** |
|           | (or **[arXiv:2203.04507v1](https://arxiv.org/abs/2203.04507v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.04507Focus to learn more |










# 2022-03-09

[Return to Index](#Index)



<h2 id="2022-03-09-1">1. Multi-Modal Mixup for Robust Fine-tuning
</h2>

Title: [Multi-Modal Mixup for Robust Fine-tuning](https://arxiv.org/abs/2203.03897)

Authors: [Junhyuk So](https://arxiv.org/search/cs?searchtype=author&query=So%2C+J), [Changdae Oh](https://arxiv.org/search/cs?searchtype=author&query=Oh%2C+C), [Minchul Shin](https://arxiv.org/search/cs?searchtype=author&query=Shin%2C+M), [Kyungwoo Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+K)

> Pre-trained large-scale models provide a transferable embedding, and they show comparable performance on the diverse downstream task. However, the transferability of multi-modal learning is restricted, and the analysis of learned embedding has not been explored well. This paper provides a perspective to understand the multi-modal embedding in terms of uniformity and alignment. We newly find that the representation learned by multi-modal learning models such as CLIP has a two separated representation space for each heterogeneous dataset with less alignment. Besides, there are unexplored large intermediate areas between two modalities with less uniformity. Less robust embedding might restrict the transferability of the representation for the downstream task. This paper provides a new end-to-end fine-tuning method for robust representation that encourages better uniformity and alignment score. First, we propose a multi-modal Mixup, m2-Mix that mixes the representation of image and text to generate the hard negative samples. Second, we fine-tune the multi-modal model on a hard negative sample as well as normal negative and positive samples with contrastive learning. Our multi-modal Mixup provides a robust representation, and we validate our methods on classification, retrieval, and structure-awareness task.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Information Retrieval (cs.IR); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2203.03897](https://arxiv.org/abs/2203.03897) [cs.CV]** |
|           | (or **[arXiv:2203.03897v1](https://arxiv.org/abs/2203.03897v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.03897Focus to learn more |







<h2 id="2022-03-09-2">2. IT5: Large-scale Text-to-text Pretraining for Italian Language Understanding and Generation
</h2>

Title: [IT5: Large-scale Text-to-text Pretraining for Italian Language Understanding and Generation](https://arxiv.org/abs/2203.03759)

Authors: [Gabriele Sarti](https://arxiv.org/search/cs?searchtype=author&query=Sarti%2C+G), [Malvina Nissim](https://arxiv.org/search/cs?searchtype=author&query=Nissim%2C+M)

> The T5 model and its unified text-to-text paradigm contributed in advancing the state-of-the-art for many natural language processing tasks. While some multilingual variants of the T5 model have recently been introduced, their performances were found to provide suboptimal performances for languages other than English if compared to monolingual variants. We are motivated by these findings to introduce IT5, the first family of encoder-decoder transformer models pretrained specifically on Italian. We perform a thorough cleaning of a web-crawled Italian corpus including more than 40 billion words and use it to pretrain three IT5 models of different sizes. The performance of IT5 models and their multilingual counterparts is then evaluated on a broad range of natural language understanding and generation benchmarks for Italian. We find the monolingual IT5 models to provide the best scale-to-performance ratio across tested models, consistently outperforming their multilingual counterparts and setting a new state-of-the-art for most Italian conditional language generation tasks.

| Comments: | 13 pages, 7 tables, 1 figure. Code and checkpoints available: [this https URL](https://github.com/gsarti/it5) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2203.03759](https://arxiv.org/abs/2203.03759) [cs.CL]** |
|           | (or **[arXiv:2203.03759v1](https://arxiv.org/abs/2203.03759v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.03759Focus to learn more |





<h2 id="2022-03-09-3">3. UniXcoder: Unified Cross-Modal Pre-training for Code Representation
</h2>

Title: [UniXcoder: Unified Cross-Modal Pre-training for Code Representation](https://arxiv.org/abs/2203.03850)

Authors: [Daya Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+D), [Shuai Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+S), [Nan Duan](https://arxiv.org/search/cs?searchtype=author&query=Duan%2C+N), [Yanlin Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Y), [Ming Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+M), [Jian Yin](https://arxiv.org/search/cs?searchtype=author&query=Yin%2C+J)

> Pre-trained models for programming languages have recently demonstrated great success on code intelligence. To support both code-related understanding and generation tasks, recent works attempt to pre-train unified encoder-decoder models. However, such encoder-decoder framework is sub-optimal for auto-regressive tasks, especially code completion that requires a decoder-only manner for efficient inference. In this paper, we present UniXcoder, a unified cross-modal pre-trained model for programming language. The model utilizes mask attention matrices with prefix adapters to control the behavior of the model and leverages cross-modal contents like AST and code comment to enhance code representation. To encode AST that is represented as a tree in parallel, we propose a one-to-one mapping method to transform AST in a sequence structure that retains all structural information from the tree. Furthermore, we propose to utilize multi-modal contents to learn representation of code fragment with contrastive learning, and then align representations among programming languages using a cross-modal generation task. We evaluate UniXcoder on five code-related tasks over nine datasets. To further evaluate the performance of code fragment representation, we also construct a dataset for a new task, called zero-shot code-to-code search. Results show that our model achieves state-of-the-art performance on most tasks and analysis reveals that comment and AST can both enhance UniXcoder.

| Comments: | Published in ACL 2022                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Programming Languages (cs.PL); Software Engineering (cs.SE) |
| Cite as:  | **[arXiv:2203.03850](https://arxiv.org/abs/2203.03850) [cs.CL]** |
|           | (or **[arXiv:2203.03850v1](https://arxiv.org/abs/2203.03850v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.03850Focus to learn more |





<h2 id="2022-03-09-4">4. HyperPELT: Unified Parameter-Efficient Language Model Tuning for Both Language and Vision-and-Language Tasks
</h2>

Title: [HyperPELT: Unified Parameter-Efficient Language Model Tuning for Both Language and Vision-and-Language Tasks](https://arxiv.org/abs/2203.03878)

Authors: [Zhengkun Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Z), [Wenya Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+W), [Xiaojun Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+X), [Yasheng Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Y), [Yadao Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Y), [Xin Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+X), [Qun Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Q), [Zhenglu Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Z)

> The workflow of pretraining and fine-tuning has emerged as a popular paradigm for solving various NLP and V&L (Vision-and-Language) downstream tasks. With the capacity of pretrained models growing rapidly, how to perform parameter-efficient fine-tuning has become fairly important for quick transfer learning and deployment. In this paper, we design a novel unified parameter-efficient transfer learning framework that works effectively on both pure language and V&L tasks. In particular, we use a shared hypernetwork that takes trainable hyper-embeddings as input, and outputs weights for fine-tuning different small modules in a pretrained language model, such as tuning the parameters inserted into multi-head attention blocks (i.e., prefix-tuning) and feed-forward blocks (i.e., adapter-tuning). We define a set of embeddings (e.g., layer, block, task and visual embeddings) as the key components to calculate hyper-embeddings, which thus can support both pure language and V&L tasks. Our proposed framework adds fewer trainable parameters in multi-task learning while achieving superior performances and transfer ability compared to state-of-the-art methods. Empirical results on the GLUE benchmark and multiple V&L tasks confirm the effectiveness of our framework on both textual and visual modalities.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2203.03878](https://arxiv.org/abs/2203.03878) [cs.CL]** |
|           | (or **[arXiv:2203.03878v1](https://arxiv.org/abs/2203.03878v1) [cs.CL]** for this version) |





<h2 id="2022-03-09-5">5. Overcoming Catastrophic Forgetting beyond Continual Learning: Balanced Training for Neural Machine Translation
</h2>

Title: [Overcoming Catastrophic Forgetting beyond Continual Learning: Balanced Training for Neural Machine Translation](https://arxiv.org/abs/2203.03910)

Authors: [Chenze Shao](https://arxiv.org/search/cs?searchtype=author&query=Shao%2C+C), [Yang Feng](https://arxiv.org/search/cs?searchtype=author&query=Feng%2C+Y)

> Neural networks tend to gradually forget the previously learned knowledge when learning multiple tasks sequentially from dynamic data distributions. This problem is called \textit{catastrophic forgetting}, which is a fundamental challenge in the continual learning of neural networks. In this work, we observe that catastrophic forgetting not only occurs in continual learning but also affects the traditional static training. Neural networks, especially neural machine translation models, suffer from catastrophic forgetting even if they learn from a static training set. To be specific, the final model pays imbalanced attention to training samples, where recently exposed samples attract more attention than earlier samples. The underlying cause is that training samples do not get balanced training in each model update, so we name this problem \textit{imbalanced training}. To alleviate this problem, we propose Complementary Online Knowledge Distillation (COKD), which uses dynamically updated teacher models trained on specific data orders to iteratively provide complementary knowledge to the student model. Experimental results on multiple machine translation tasks show that our method successfully alleviates the problem of imbalanced training and achieves substantial improvements over strong baseline systems.

| Comments:    | ACL 2022 main conference                                     |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| ACM classes: | I.2.7                                                        |
| Cite as:     | **[arXiv:2203.03910](https://arxiv.org/abs/2203.03910) [cs.CL]** |
|              | (or **[arXiv:2203.03910v1](https://arxiv.org/abs/2203.03910v1) [cs.CL]** for this version) |





<h2 id="2022-03-09-6">6. Adaptr: Objective-Centric Adaptation Framework for Language Models
</h2>

Title: [Adaptr: Objective-Centric Adaptation Framework for Language Models](https://arxiv.org/abs/2203.03989)

Authors: [Michal Štefánik](https://arxiv.org/search/cs?searchtype=author&query=Štefánik%2C+M), [Vít Novotný](https://arxiv.org/search/cs?searchtype=author&query=Novotný%2C+V), [Nikola Groverová](https://arxiv.org/search/cs?searchtype=author&query=Groverová%2C+N), [Petr Sojka](https://arxiv.org/search/cs?searchtype=author&query=Sojka%2C+P)

> Progress in natural language processing research is catalyzed by the possibilities given by the widespread software frameworks. This paper introduces Adaptor library that transposes the traditional model-centric approach composed of pre-training + fine-tuning steps to objective-centric approach, composing the training process by applications of selected objectives. We survey research directions that can benefit from enhanced objective-centric experimentation in multitask training, custom objectives development, dynamic training curricula, or domain adaptation. Adaptor aims to ease reproducibility of these research directions in practice. Finally, we demonstrate the practical applicability of Adaptor in selected unsupervised domain adaptation scenarios.

| Comments: | 60th Annual Meeting of the ACL (ACL 2022): System Demonstrations paper |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2203.03989](https://arxiv.org/abs/2203.03989) [cs.CL]** |
|           | (or **[arXiv:2203.03989v1](https://arxiv.org/abs/2203.03989v1) [cs.CL]** for this version) |



# 2022-03-08

[Return to Index](#Index)



<h2 id="2022-03-08-1">1. OCR quality affects perceived usefulness of historical newspaper clippings -- a user study
</h2>

Title: [OCR quality affects perceived usefulness of historical newspaper clippings -- a user study](https://arxiv.org/abs/2203.03557)
Authors: [Kimmo Kettunen](https://arxiv.org/search/cs?searchtype=author&query=Kettunen%2C+K), [Heikki Keskustalo](https://arxiv.org/search/cs?searchtype=author&query=Keskustalo%2C+H), [Sanna Kumpulainen](https://arxiv.org/search/cs?searchtype=author&query=Kumpulainen%2C+S), [Tuula Pääkkönen](https://arxiv.org/search/cs?searchtype=author&query=Pääkkönen%2C+T), [Juha Rautiainen](https://arxiv.org/search/cs?searchtype=author&query=Rautiainen%2C+J)

> Effects of Optical Character Recognition (OCR) quality on historical information retrieval have so far been studied in data-oriented scenarios regarding the effectiveness of retrieval results. Such studies have either focused on the effects of artificially degraded OCR quality (see, e.g., [1-2]) or utilized test collections containing texts based on authentic low quality OCR data (see, e.g., [3]). In this paper the effects of OCR quality are studied in a user-oriented information retrieval setting. Thirty-two users evaluated subjectively query results of six topics each (out of 30 topics) based on pre-formulated queries using a simulated work task setting. To the best of our knowledge our simulated work task experiment is the first one showing empirically that users' subjective relevance assessments of retrieved documents are affected by a change in the quality of optically read text. Users of historical newspaper collections have so far commented effects of OCR'ed data quality mainly in impressionistic ways, and controlled user environments for studying effects of OCR quality on users' relevance assessments of the retrieval results have so far been missing. To remedy this The National Library of Finland (NLF) set up an experimental query environment for the contents of one Finnish historical newspaper, Uusi Suometar 1869-1918, to be able to compare users' evaluation of search results of two different OCR qualities for digitized newspaper articles. The query interface was able to present the same underlying document for the user based on two alternatives: either based on the lower OCR quality, or based on the higher OCR quality, and the choice was randomized. The users did not know about quality differences in the article texts they evaluated. The main result of the study is that improved optical character recognition quality affects perceived usefulness of historical newspaper articles significantly. The mean average evaluation score for the improved OCR results was 7.94% higher than the mean average evaluation score of the old OCR results.

| Comments: | IRCDL2022                                                    |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Information Retrieval (cs.IR)**; Computation and Language (cs.CL); Digital Libraries (cs.DL) |
| Cite as:  | **[arXiv:2203.03557](https://arxiv.org/abs/2203.03557) [cs.IR]** |
|           | (or **[arXiv:2203.03557v1](https://arxiv.org/abs/2203.03557v1) [cs.IR]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.03557Focus to learn more |





<h2 id="2022-03-08-2">2. Focus on the Target's Vocabulary: Masked Label Smoothing for Machine Translation
</h2>

Title: [Focus on the Target's Vocabulary: Masked Label Smoothing for Machine Translation](https://arxiv.org/abs/2203.02889)
Authors: [Liang Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+L), [Runxin Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+R), [Baobao Chang](https://arxiv.org/search/cs?searchtype=author&query=Chang%2C+B)

> Label smoothing and vocabulary sharing are two widely used techniques in neural machine translation models. However, we argue that simply applying both techniques can be conflicting and even leads to sub-optimal performance. When allocating smoothed probability, original label smoothing treats the source-side words that would never appear in the target language equally to the real target-side words, which could bias the translation model. To address this issue, we propose Masked Label Smoothing (MLS), a new mechanism that masks the soft label probability of source-side words to zero. Simple yet effective, MLS manages to better integrate label smoothing with vocabulary sharing. Our extensive experiments show that MLS consistently yields improvement over original label smoothing on different datasets, including bilingual and multilingual translation from both translation quality and model's calibration. Our code is released at [this https URL](https://github.com/PKUnlp-icler/MLS)

| Comments: | ACL 2022 Main Conference, released at [this https URL](https://github.com/PKUnlp-icler/MLS) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2203.02889](https://arxiv.org/abs/2203.02889) [cs.CL]** |
|           | (or **[arXiv:2203.02889v1](https://arxiv.org/abs/2203.02889v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.02889Focus to learn more |





<h2 id="2022-03-08-3">3. Conditional Bilingual Mutual Information Based Adaptive Training for Neural Machine Translation
</h2>

Title: [Conditional Bilingual Mutual Information Based Adaptive Training for Neural Machine Translation](https://arxiv.org/abs/2203.02951)
Authors: [Songming Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+S), [Yijin Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y), [Fandong Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+F), [Yufeng Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Y), [Jinan Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+J), [Jian Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+J), [Jie Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+J)

> Token-level adaptive training approaches can alleviate the token imbalance problem and thus improve neural machine translation, through re-weighting the losses of different target tokens based on specific statistical metrics (e.g., token frequency or mutual information). Given that standard translation models make predictions on the condition of previous target contexts, we argue that the above statistical metrics ignore target context information and may assign inappropriate weights to target tokens. While one possible solution is to directly take target contexts into these statistical metrics, the target-context-aware statistical computing is extremely expensive, and the corresponding storage overhead is unrealistic. To solve the above issues, we propose a target-context-aware metric, named conditional bilingual mutual information (CBMI), which makes it feasible to supplement target context information for statistical metrics. Particularly, our CBMI can be formalized as the log quotient of the translation model probability and language model probability by decomposing the conditional joint distribution. Thus CBMI can be efficiently calculated during model training without any pre-specific statistical calculations and large storage overhead. Furthermore, we propose an effective adaptive training approach based on both the token- and sentence-level CBMI. Experimental results on WMT14 English-German and WMT19 Chinese-English tasks show our approach can significantly outperform the Transformer baseline and other related methods.

| Comments: | Accepted at ACL 2022 as a long paper of main conference. The code is available at: [this https URL](https://github.com/songmzhang/CBMI) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2203.02951](https://arxiv.org/abs/2203.02951) [cs.CL]** |
|           | (or **[arXiv:2203.02951v1](https://arxiv.org/abs/2203.02951v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.02951Focus to learn more |





<h2 id="2022-03-08-4">4. Recent Advances in Neural Text Generation: A Task-Agnostic Survey
</h2>

Title: [Recent Advances in Neural Text Generation: A Task-Agnostic Survey](https://arxiv.org/abs/2203.03047)
Authors: [Chen Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+C), [Frank Guerin](https://arxiv.org/search/cs?searchtype=author&query=Guerin%2C+F), [Yucheng Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Chenghua Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+C)

> In recent years much effort has been devoted to applying neural models to the task of natural language generation. The challenge is to generate natural human-like text, and to control the generation process. This paper presents a task-agnostic survey of recent advances in neural text generation. These advances have been achieved by numerous developments, which we group under the following four headings: data construction, neural frameworks, training and inference strategies, and evaluation metrics. Finally we discuss the future directions for the development of neural text generation including neural pipelines and exploiting back-ground knowledge.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2203.03047](https://arxiv.org/abs/2203.03047) [cs.CL]** |
|           | (or **[arXiv:2203.03047v1](https://arxiv.org/abs/2203.03047v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.03047Focus to learn more |





<h2 id="2022-03-08-5">5. Input-Tuning: Adapting Unfamiliar Inputs to Frozen Pretrained Models
</h2>

Title: [Input-Tuning: Adapting Unfamiliar Inputs to Frozen Pretrained Models](https://arxiv.org/abs/2203.03131)
Authors: [Shengnan An](https://arxiv.org/search/cs?searchtype=author&query=An%2C+S), [Yifei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Zeqi Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+Z), [Qian Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Q), [Bei Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+B), [Qiang Fu](https://arxiv.org/search/cs?searchtype=author&query=Fu%2C+Q), [Weizhu Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+W), [Nanning Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+N), [Jian-Guang Lou](https://arxiv.org/search/cs?searchtype=author&query=Lou%2C+J)

> Recently the prompt-tuning paradigm has attracted significant attention. By only tuning continuous prompts with a frozen pre-trained language model (PLM), prompt-tuning takes a step towards deploying a shared frozen PLM to serve numerous downstream tasks. Although prompt-tuning shows good performance on certain natural language understanding (NLU) tasks, its effectiveness on natural language generation (NLG) tasks is still under-explored. In this paper, we argue that one of the factors hindering the development of prompt-tuning on NLG tasks is the unfamiliar inputs (i.e., inputs are linguistically different from the pretraining corpus). For example, our preliminary exploration reveals a large performance gap between prompt-tuning and fine-tuning when unfamiliar inputs occur frequently in NLG tasks. This motivates us to propose input-tuning, which fine-tunes both the continuous prompts and the input representations, leading to a more effective way to adapt unfamiliar inputs to frozen PLMs. Our proposed input-tuning is conceptually simple and empirically powerful. Experimental results on seven NLG tasks demonstrate that input-tuning is significantly and consistently better than prompt-tuning. Furthermore, on three of these tasks, input-tuning can achieve a comparable or even better performance than fine-tuning.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2203.03131](https://arxiv.org/abs/2203.03131) [cs.CL]** |
|           | (or **[arXiv:2203.03131v1](https://arxiv.org/abs/2203.03131v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.03131Focus to learn more |





<h2 id="2022-03-08-6">6. One Model, Multiple Tasks: Pathways for Natural Language Understanding
</h2>

Title: [One Model, Multiple Tasks: Pathways for Natural Language Understanding](https://arxiv.org/abs/2203.03312)
Authors: [Duyu Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+D), [Fan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+F), [Yong Dai](https://arxiv.org/search/cs?searchtype=author&query=Dai%2C+Y), [Cong Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+C), [Shuangzhi Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+S), [Shuming Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+S)

> This paper presents a Pathways approach to handle many tasks at once. Our approach is general-purpose and sparse. Unlike prevailing single-purpose models that overspecialize at individual tasks and learn from scratch when being extended to new tasks, our approach is general-purpose with the ability of stitching together existing skills to learn new tasks more effectively. Different from traditional dense models that always activate all the model parameters, our approach is sparsely activated: only relevant parts of the model (like pathways through the network) are activated. 
> We take natural language understanding as a case study and define a set of skills like \textit{the skill of understanding the sentiment of text} and \textit{the skill of understanding natural language questions}. These skills can be reused and combined to support many different tasks and situations. We develop our system using Transformer as the backbone. For each skill, we implement skill-specific feed-forward networks, which are activated only if the skill is relevant to the task. An appealing feature of our model is that it not only supports sparsely activated fine-tuning, but also allows us to pretrain skills in the same sparse way with masked language modeling and next sentence prediction. We call this model \textbf{SkillNet}. 
> We have three major findings. First, with only one model checkpoint, SkillNet performs better than task-specific fine-tuning and two multi-task learning baselines (i.e., dense model and Mixture-of-Experts model) on six tasks. Second, sparsely activated pre-training further improves the overall performance. Third, SkillNet significantly outperforms baseline systems when being extended to new tasks.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2203.03312](https://arxiv.org/abs/2203.03312) [cs.CL]** |
|           | (or **[arXiv:2203.03312v1](https://arxiv.org/abs/2203.03312v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.03312Focus to learn more |







# 2022-03-07

[Return to Index](#Index)



<h2 id="2022-03-07-1">1. Overlap-based Vocabulary Generation Improves Cross-lingual Transfer Among Related Languages
</h2>

Title: [Overlap-based Vocabulary Generation Improves Cross-lingual Transfer Among Related Languages](https://arxiv.org/abs/2203.01976)

Authors: [Vaidehi Patil](https://arxiv.org/search/cs?searchtype=author&query=Patil%2C+V), [Partha Talukdar](https://arxiv.org/search/cs?searchtype=author&query=Talukdar%2C+P), [Sunita Sarawagi](https://arxiv.org/search/cs?searchtype=author&query=Sarawagi%2C+S)

> Pre-trained multilingual language models such as mBERT and XLM-R have demonstrated great potential for zero-shot cross-lingual transfer to low web-resource languages (LRL). However, due to limited model capacity, the large difference in the sizes of available monolingual corpora between high web-resource languages (HRL) and LRLs does not provide enough scope of co-embedding the LRL with the HRL, thereby affecting downstream task performance of LRLs. In this paper, we argue that relatedness among languages in a language family along the dimension of lexical overlap may be leveraged to overcome some of the corpora limitations of LRLs. We propose Overlap BPE (OBPE), a simple yet effective modification to the BPE vocabulary generation algorithm which enhances overlap across related languages. Through extensive experiments on multiple NLP tasks and datasets, we observe that OBPE generates a vocabulary that increases the representation of LRLs via tokens shared with HRLs. This results in improved zero-shot transfer from related HRLs to LRLs without reducing HRL representation and accuracy. Unlike previous studies that dismissed the importance of token-overlap, we show that in the low-resource related language setting, token overlap matters. Synthetically reducing the overlap to zero can cause as much as a four-fold drop in zero-shot transfer accuracy.

| Comments: | Accepted to appear at the ACL 2022 Main conference           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2203.01976](https://arxiv.org/abs/2203.01976) [cs.CL]** |
|           | (or **[arXiv:2203.01976v1](https://arxiv.org/abs/2203.01976v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.01976Focus to learn more |





<h2 id="2022-03-07-2">2. Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning
</h2>

Title: [Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning](https://arxiv.org/abs/2203.02053)

Authors: [Weixin Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+W), [Yuhui Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y), [Yongchan Kwon](https://arxiv.org/search/cs?searchtype=author&query=Kwon%2C+Y), [Serena Yeung](https://arxiv.org/search/cs?searchtype=author&query=Yeung%2C+S), [James Zou](https://arxiv.org/search/cs?searchtype=author&query=Zou%2C+J)

> We present modality gap, an intriguing geometric phenomenon of the representation space of multi-modal models. Specifically, we show that different data modalities (e.g. images and text) are embedded at arm's length in their shared representation in multi-modal models such as CLIP. Our systematic analysis demonstrates that this gap is caused by a combination of model initialization and contrastive learning optimization. In model initialization, we show empirically and theoretically that the representation of a common deep neural network is restricted to a narrow cone. As a consequence, in a multi-modal model with two encoders, the representations of the two modalities are clearly apart when the model is initialized. During optimization, contrastive learning keeps the different modalities separate by a certain distance, which is influenced by the temperature parameter in the loss function. Our experiments further demonstrate that varying the modality gap distance has a significant impact in improving the model's downstream zero-shot classification performance and fairness. Our code and data are available at [this https URL](https://modalitygap.readthedocs.io/)

| Comments: | Our code and data are available at [this https URL](https://modalitygap.readthedocs.io/) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG); Multimedia (cs.MM) |
| Cite as:  | **[arXiv:2203.02053](https://arxiv.org/abs/2203.02053) [cs.CL]** |
|           | (or **[arXiv:2203.02053v1](https://arxiv.org/abs/2203.02053v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.02053Focus to learn more |





<h2 id="2022-03-07-3">3. EAG: Extract and Generate Multi-way Aligned Corpus for Complete Multi-lingual Neural Machine Translation
</h2>

Title: [EAG: Extract and Generate Multi-way Aligned Corpus for Complete Multi-lingual Neural Machine Translation](https://arxiv.org/abs/2203.02180)

Authors: [Yulin Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+Y), [Zhen Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Z), [Fandong Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+F), [JieZhou](https://arxiv.org/search/cs?searchtype=author&query=JieZhou)

> Complete Multi-lingual Neural Machine Translation (C-MNMT) achieves superior performance against the conventional MNMT by constructing multi-way aligned corpus, i.e., aligning bilingual training examples from different language pairs when either their source or target sides are identical. However, since exactly identical sentences from different language pairs are scarce, the power of the multi-way aligned corpus is limited by its scale. To handle this problem, this paper proposes "Extract and Generate" (EAG), a two-step approach to construct large-scale and high-quality multi-way aligned corpus from bilingual data. Specifically, we first extract candidate aligned examples by pairing the bilingual examples from different language pairs with highly similar source or target sentences; and then generate the final aligned examples from the candidates with a well-trained generation model. With this two-step pipeline, EAG can construct a large-scale and multi-way aligned corpus whose diversity is almost identical to the original bilingual corpus. Experiments on two publicly available datasets i.e., WMT-5 and OPUS-100, show that the proposed method achieves significant improvements over strong baselines, with +1.1 and +1.4 BLEU points improvements on the two datasets respectively.

| Comments: | Accepted as a long paper at ACL 2022                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2203.02180](https://arxiv.org/abs/2203.02180) [cs.CL]** |
|           | (or **[arXiv:2203.02180v1](https://arxiv.org/abs/2203.02180v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.02180Focus to learn more |





<h2 id="2022-03-07-4">4. Comprehension of Subtitles from Re-Translating Simultaneous Speech Translation
</h2>

Title: [Comprehension of Subtitles from Re-Translating Simultaneous Speech Translation](https://arxiv.org/abs/2203.02458)

Authors: [Dávid Javorský](https://arxiv.org/search/cs?searchtype=author&query=Javorský%2C+D), [Dominik Macháček](https://arxiv.org/search/cs?searchtype=author&query=Macháček%2C+D), [Ondřej Bojar](https://arxiv.org/search/cs?searchtype=author&query=Bojar%2C+O)

> In simultaneous speech translation, one can vary the size of the output window, system latency and sometimes the allowed level of rewriting. The effect of these properties on readability and comprehensibility has not been tested with modern neural translation systems. In this work, we propose an evaluation method and investigate the effects on comprehension and user preferences. It is a pilot study with 14 users on 2 hours of German documentaries or speeches with online translations into Czech. We collect continuous feedback and answers on factual questions. Our results show that the subtitling layout or flicker have a little effect on comprehension, in contrast to machine translation itself and individual competence. Other results show that users with a limited knowledge of the source language have different preferences to stability and latency than the users with zero knowledge. The results are statistically insignificant, however, we show that our method works and can be reproduced in larger volume.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2203.02458](https://arxiv.org/abs/2203.02458) [cs.CL]** |
|           | (or **[arXiv:2203.02458v1](https://arxiv.org/abs/2203.02458v1) [cs.CL]** for this version) |





<h2 id="2022-03-07-5">5. From Simultaneous to Streaming Machine Translation by Leveraging Streaming History
</h2>

Title: [From Simultaneous to Streaming Machine Translation by Leveraging Streaming History](https://arxiv.org/abs/2203.02459)

Authors: [Javier Iranzo-Sánchez](https://arxiv.org/search/cs?searchtype=author&query=Iranzo-Sánchez%2C+J), [Jorge Civera](https://arxiv.org/search/cs?searchtype=author&query=Civera%2C+J), [Alfons Juan](https://arxiv.org/search/cs?searchtype=author&query=Juan%2C+A)

> Simultaneous Machine Translation is the task of incrementally translating an input sentence before it is fully available. Currently, simultaneous translation is carried out by translating each sentence independently of the previously translated text. More generally, Streaming MT can be understood as an extension of Simultaneous MT to the incremental translation of a continuous input text stream. In this work, a state-of-the-art simultaneous sentence-level MT system is extended to the streaming setup by leveraging the streaming history. Extensive empirical results are reported on IWSLT Translation Tasks, showing that leveraging the streaming history leads to significant quality gains. In particular, the proposed system proves to compare favorably to the best performing systems.

| Comments: | ACL 2022 - Camera ready                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2203.02459](https://arxiv.org/abs/2203.02459) [cs.CL]** |
|           | (or **[arXiv:2203.02459v1](https://arxiv.org/abs/2203.02459v1) [cs.CL]** for this version) |








# 2022-03-04

[Return to Index](#Index)



<h2 id="2022-03-04-1">1. Recent, rapid advancement in visual question answering architecture
</h2>

Title: [Recent, rapid advancement in visual question answering architecture](https://arxiv.org/abs/2203.01322)

Authors: [Venkat Kodali](https://arxiv.org/search/cs?searchtype=author&query=Kodali%2C+V), [Daniel Berleant](https://arxiv.org/search/cs?searchtype=author&query=Berleant%2C+D)

> Understanding visual question answering is going to be crucial for numerous human activities. However, it presents major challenges at the heart of the artificial intelligence endeavor. This paper presents an update on the rapid advancements in visual question answering using images that have occurred in the last couple of years. Tremendous growth in research on improving visual question answering system architecture has been published recently, showing the importance of multimodal architectures. Several points on the benefits of visual question answering are mentioned in the review paper by Manmadhan et al. (2020), on which the present article builds, including subsequent updates in the field.

| Comments: | 11 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Multimedia (cs.MM) |
| Cite as:  | **[arXiv:2203.01322](https://arxiv.org/abs/2203.01322) [cs.CV]** |
|           | (or **[arXiv:2203.01322v1](https://arxiv.org/abs/2203.01322v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.01322Focus to learn more |





<h2 id="2022-03-04-2">2. Vision-Language Intelligence: Tasks, Representation Learning, and Large Models
</h2>

Title: [Vision-Language Intelligence: Tasks, Representation Learning, and Large Models](https://arxiv.org/abs/2203.01922)

Authors: [Feng Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+F), [Hao Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+H), [Yi-Fan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y), [Shilong Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+S), [Jian Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+J), [Lionel M. Ni](https://arxiv.org/search/cs?searchtype=author&query=Ni%2C+L+M), [PengChuan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+P), [Lei Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+L)

> This paper presents a comprehensive survey of vision-language (VL) intelligence from the perspective of time. This survey is inspired by the remarkable progress in both computer vision and natural language processing, and recent trends shifting from single modality processing to multiple modality comprehension. We summarize the development in this field into three time periods, namely task-specific methods, vision-language pre-training (VLP) methods, and larger models empowered by large-scale weakly-labeled data. We first take some common VL tasks as examples to introduce the development of task-specific methods. Then we focus on VLP methods and comprehensively review key components of the model structures and training methods. After that, we show how recent work utilizes large-scale raw image-text data to learn language-aligned visual representations that generalize better on zero or few shot learning tasks. Finally, we discuss some potential future trends towards modality cooperation, unified representation, and knowledge incorporation. We believe that this review will be of help for researchers and practitioners of AI and ML, especially those interested in computer vision and natural language processing.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2203.01922](https://arxiv.org/abs/2203.01922) [cs.CV]** |
|           | (or **[arXiv:2203.01922v1](https://arxiv.org/abs/2203.01922v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.01922Focus to learn more |





<h2 id="2022-03-04-3">3. UDAAN - Machine Learning based Post-Editing tool for Document Translation
</h2>

Title: [UDAAN - Machine Learning based Post-Editing tool for Document Translation](https://arxiv.org/abs/2203.01644)

Authors: [Ayush Maheshwari](https://arxiv.org/search/cs?searchtype=author&query=Maheshwari%2C+A), [Ajay Ravindran](https://arxiv.org/search/cs?searchtype=author&query=Ravindran%2C+A), [Venkatapathy Subramanian](https://arxiv.org/search/cs?searchtype=author&query=Subramanian%2C+V), [Akshay Jalan](https://arxiv.org/search/cs?searchtype=author&query=Jalan%2C+A), [Ganesh Ramakrishnan](https://arxiv.org/search/cs?searchtype=author&query=Ramakrishnan%2C+G)

> We introduce UDAAN, an open-source post-editing tool that can reduce manual editing efforts to quickly produce publishable-standard documents in different languages. UDAAN has an end-to-end Machine Translation (MT) plus post-editing pipeline wherein users can upload a document to obtain raw MT output. Further, users can edit the raw translations using our tool. UDAAN offers several advantages: a) Domain-aware, vocabulary-based lexical constrained MT. b) source-target and target-target lexicon suggestions for users. Replacements are based on the source and target texts lexicon alignment. c) Suggestions for translations are based on logs created during user interaction. d) Source-target sentence alignment visualisation that reduces the cognitive load of users during editing. e) Translated outputs from our tool are available in multiple formats: docs, latex, and PDF. Although we limit our experiments to English-to-Hindi translation for the current study, our tool is independent of the source and target languages. Experimental results based on the usage of the tools and users feedback show that our tool speeds up the translation time approximately by a factor of three compared to the baseline method of translating documents from scratch.

| Comments: | system demonstration paper                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2203.01644](https://arxiv.org/abs/2203.01644) [cs.CL]** |
|           | (or **[arXiv:2203.01644v1](https://arxiv.org/abs/2203.01644v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.01644Focus to learn more |





# 2022-03-03

[Return to Index](#Index)



<h2 id="2022-03-03-1">1. HighMMT: Towards Modality and Task Generalization for High-Modality Representation Learning
</h2>

Title: [HighMMT: Towards Modality and Task Generalization for High-Modality Representation Learning](https://arxiv.org/abs/2203.01311)

Authors: [Paul Pu Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+P+P), [Yiwei Lyu](https://arxiv.org/search/cs?searchtype=author&query=Lyu%2C+Y), [Xiang Fan](https://arxiv.org/search/cs?searchtype=author&query=Fan%2C+X), [Shengtong Mo](https://arxiv.org/search/cs?searchtype=author&query=Mo%2C+S), [Dani Yogatama](https://arxiv.org/search/cs?searchtype=author&query=Yogatama%2C+D), [Louis-Philippe Morency](https://arxiv.org/search/cs?searchtype=author&query=Morency%2C+L), [Ruslan Salakhutdinov](https://arxiv.org/search/cs?searchtype=author&query=Salakhutdinov%2C+R)

> Learning multimodal representations involves discovering correspondences and integrating information from multiple heterogeneous sources of data. While recent research has begun to explore the design of more general-purpose multimodal models (contrary to prior focus on domain and modality-specific architectures), these methods are still largely focused on a small set of modalities in the language, vision, and audio space. In order to accelerate generalization towards diverse and understudied modalities, we investigate methods for high-modality (a large set of diverse modalities) and partially-observable (each task only defined on a small subset of modalities) scenarios. To tackle these challenges, we design a general multimodal model that enables multitask and transfer learning: multitask learning with shared parameters enables stable parameter counts (addressing scalability), and cross-modal transfer learning enables information sharing across modalities and tasks (addressing partial observability). Our resulting model generalizes across text, image, video, audio, time-series, sensors, tables, and set modalities from different research areas, improves the tradeoff between performance and efficiency, transfers to new modalities and tasks, and reveals surprising insights on the nature of information sharing in multitask models. We release our code and benchmarks which we hope will present a unified platform for subsequent theoretical and empirical analysis: [this https URL](https://github.com/pliang279/HighMMT).

| Comments: | Code available at [this https URL](https://github.com/pliang279/HighMMT) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV); Multimedia (cs.MM) |
| Cite as:  | **[arXiv:2203.01311](https://arxiv.org/abs/2203.01311) [cs.LG]** |
|           | (or **[arXiv:2203.01311v1](https://arxiv.org/abs/2203.01311v1) [cs.LG]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.01311Focus to learn more |





<h2 id="2022-03-03-2">2. Attend, Memorize and Generate: Towards Faithful Table-to-Text Generation in Few Shots
</h2>

Title: [Attend, Memorize and Generate: Towards Faithful Table-to-Text Generation in Few Shots](https://arxiv.org/abs/2203.00732)

Authors: [Wenting Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+W), [Ye Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y), [Yao Wan](https://arxiv.org/search/cs?searchtype=author&query=Wan%2C+Y), [Philip S. Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+P+S)

> Few-shot table-to-text generation is a task of composing fluent and faithful sentences to convey table content using limited data. Despite many efforts having been made towards generating impressive fluent sentences by fine-tuning powerful pre-trained language models, the faithfulness of generated content still needs to be improved. To this end, this paper proposes a novel approach Attend, Memorize and Generate (called AMG), inspired by the text generation process of humans. In particular, AMG (1) attends over the multi-granularity of context using a novel strategy based on table slot level and traditional token-by-token level attention to exploit both the table structure and natural linguistic information; (2) dynamically memorizes the table slot allocation states; and (3) generates faithful sentences according to both the context and memory allocation states. Comprehensive experiments with human evaluation on three domains (i.e., humans, songs, and books) of the Wiki dataset show that our model can generate higher qualified texts when compared with several state-of-the-art baselines, in both fluency and faithfulness.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2203.00732](https://arxiv.org/abs/2203.00732) [cs.CL]** |
|           | (or **[arXiv:2203.00732v1](https://arxiv.org/abs/2203.00732v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.00732Focus to learn more |





<h2 id="2022-03-03-3">3. HyperPrompt: Prompt-based Task-Conditioning of Transformers
</h2>

Title: [HyperPrompt: Prompt-based Task-Conditioning of Transformers](https://arxiv.org/abs/2203.00759)

Authors: [Yun He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+Y), [Huaixiu Steven Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+H+S), [Yi Tay](https://arxiv.org/search/cs?searchtype=author&query=Tay%2C+Y), [Jai Gupta](https://arxiv.org/search/cs?searchtype=author&query=Gupta%2C+J), [Yu Du](https://arxiv.org/search/cs?searchtype=author&query=Du%2C+Y), [Vamsi Aribandi](https://arxiv.org/search/cs?searchtype=author&query=Aribandi%2C+V), [Zhe Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+Z), [YaGuang Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Zhao Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Z), [Donald Metzler](https://arxiv.org/search/cs?searchtype=author&query=Metzler%2C+D), [Heng-Tze Cheng](https://arxiv.org/search/cs?searchtype=author&query=Cheng%2C+H), [Ed H. Chi](https://arxiv.org/search/cs?searchtype=author&query=Chi%2C+E+H)

> Prompt-Tuning is a new paradigm for finetuning pre-trained language models in a parameter-efficient way. Here, we explore the use of HyperNetworks to generate hyper-prompts: we propose HyperPrompt, a novel architecture for prompt-based task-conditioning of self-attention in Transformers. The hyper-prompts are end-to-end learnable via generation by a HyperNetwork. HyperPrompt allows the network to learn task-specific feature maps where the hyper-prompts serve as task global memories for the queries to attend to, at the same time enabling flexible information sharing among tasks. We show that HyperPrompt is competitive against strong multi-task learning baselines with as few as 0.14% of additional task-conditioning parameters, achieving great parameter and computational efficiency. Through extensive empirical experiments, we demonstrate that HyperPrompt can achieve superior performances over strong T5 multi-task learning baselines and parameter-efficient adapter variants including Prompt-Tuning and HyperFormer++ on Natural Language Understanding benchmarks of GLUE and SuperGLUE across many model sizes.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2203.00759](https://arxiv.org/abs/2203.00759) [cs.CL]** |
|           | (or **[arXiv:2203.00759v1](https://arxiv.org/abs/2203.00759v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.00759Focus to learn more |





<h2 id="2022-03-03-4">4. Do Prompts Solve NLP Tasks Using Natural Language?
</h2>

Title: [Do Prompts Solve NLP Tasks Using Natural Language?](https://arxiv.org/abs/2203.00902)

Authors: [Sen Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+S), [Yunchen Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y), [Leyang Cui](https://arxiv.org/search/cs?searchtype=author&query=Cui%2C+L), [Yue Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y)

> Thanks to the advanced improvement of large pre-trained language models, prompt-based fine-tuning is shown to be effective on a variety of downstream tasks. Though many prompting methods have been investigated, it remains unknown which type of prompts are the most effective among three types of prompts (i.e., human-designed prompts, schema prompts and null prompts). In this work, we empirically compare the three types of prompts under both few-shot and fully-supervised settings. Our experimental results show that schema prompts are the most effective in general. Besides, the performance gaps tend to diminish when the scale of training data grows large.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2203.00902](https://arxiv.org/abs/2203.00902) [cs.CL]** |
|           | (or **[arXiv:2203.00902v1](https://arxiv.org/abs/2203.00902v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.00902Focus to learn more |





<h2 id="2022-03-03-5">5. Parameter-Efficient Mixture-of-Experts Architecture for Pre-trained Language Models
</h2>

Title: [Parameter-Efficient Mixture-of-Experts Architecture for Pre-trained Language Models](https://arxiv.org/abs/2203.01104)

Authors: [Ze-Feng Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+Z), [Peiyu Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+P), [Wayne Xin Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+W+X), [Zhong-Yi Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+Z), [Ji-Rong Wen](https://arxiv.org/search/cs?searchtype=author&query=Wen%2C+J)

> The state-of-the-art Mixture-of-Experts (short as MoE) architecture has achieved several remarkable successes in terms of increasing model capacity. However, MoE has been hindered widespread adoption due to complexity, communication costs, and training instability. Here we present a novel MoE architecture based on matrix product operators (MPO) from quantum many-body physics. It can decompose an original matrix into central tensors (containing the core information) and auxiliary tensors (with only a small proportion of parameters). With the decomposed MPO structure, we can reduce the parameters of the original MoE architecture by sharing a global central tensor across experts and keeping expert-specific auxiliary tensors. We also design the gradient mask strategy for the tensor structure of MPO to alleviate the overfitting problem. Experiments on the three well-known downstream natural language datasets based on GPT2 show improved performance and efficiency in increasing model capacity (7.26x fewer parameters with the same amount of experts). We additionally demonstrate an improvement in the positive transfer effects of our approach for multi-task learning.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Quantum Physics (quant-ph) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2203.01104](https://arxiv.org/abs/2203.01104) [cs.CL]** |
|           | (or **[arXiv:2203.01104v1](https://arxiv.org/abs/2203.01104v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.01104Focus to learn more |





# 2022-03-02

[Return to Index](#Index)



<h2 id="2022-03-02-1">1. Exploring and Adapting Chinese GPT to Pinyin Input Method
</h2>

Title: [Exploring and Adapting Chinese GPT to Pinyin Input Method](https://arxiv.org/abs/2203.00249)

Authors: [Minghuan Tan](https://arxiv.org/search/cs?searchtype=author&query=Tan%2C+M), [Yong Dai](https://arxiv.org/search/cs?searchtype=author&query=Dai%2C+Y), [Duyu Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+D), [Zhangyin Feng](https://arxiv.org/search/cs?searchtype=author&query=Feng%2C+Z), [Guoping Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+G), [Jing Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+J), [Jiwei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J), [Shuming Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+S)

> While GPT has become the de-facto method for text generation tasks, its application to pinyin input method remains unexplored. In this work, we make the first exploration to leverage Chinese GPT for pinyin input method. We find that a frozen GPT achieves state-of-the-art performance on perfect pinyin. However, the performance drops dramatically when the input includes abbreviated pinyin. A reason is that an abbreviated pinyin can be mapped to many perfect pinyin, which links to even larger number of Chinese characters. We mitigate this issue with two strategies, including enriching the context with pinyin and optimizing the training process to help distinguish homophones. To further facilitate the evaluation of pinyin input method, we create a dataset consisting of 270K instances from 15 domains. Results show that our approach improves performance on abbreviated pinyin across all domains. Model analysis demonstrates that both strategies contribute to the performance boost.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2203.00249](https://arxiv.org/abs/2203.00249) [cs.CL]** |
|           | (or **[arXiv:2203.00249v1](https://arxiv.org/abs/2203.00249v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.00249Focus to learn more |





<h2 id="2022-03-02-2">2. TableFormer: Robust Transformer Modeling for Table-Text Encoding
</h22

Title: [TableFormer: Robust Transformer Modeling for Table-Text Encoding]()

Authors: [Jingfeng Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+J), [Aditya Gupta](https://arxiv.org/search/cs?searchtype=author&query=Gupta%2C+A), [Shyam Upadhyay](https://arxiv.org/search/cs?searchtype=author&query=Upadhyay%2C+S), [Luheng He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+L), [Rahul Goel](https://arxiv.org/search/cs?searchtype=author&query=Goel%2C+R), [Shachi Paul](https://arxiv.org/search/cs?searchtype=author&query=Paul%2C+S)

> Understanding tables is an important aspect of natural language understanding. Existing models for table understanding require linearization of the table structure, where row or column order is encoded as an unwanted bias. Such spurious biases make the model vulnerable to row and column order perturbations. Additionally, prior work has not thoroughly modeled the table structures or table-text alignments, hindering the table-text understanding ability. In this work, we propose a robust and structurally aware table-text encoding architecture TableFormer, where tabular structural biases are incorporated completely through learnable attention biases. TableFormer is (1) strictly invariant to row and column orders, and, (2) could understand tables better due to its tabular inductive biases. Our evaluations showed that TableFormer outperforms strong baselines in all settings on SQA, WTQ and TabFact table reasoning datasets, and achieves state-of-the-art performance on SQA, especially when facing answer-invariant row and column order perturbations (6% improvement over the best baseline), because previous SOTA models' performance drops by 4% - 6% when facing such perturbations while TableFormer is not affected.

| Comments: | ACL 2022, 10 pages                                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2203.00274](https://arxiv.org/abs/2203.00274) [cs.CL]** |
|           | (or **[arXiv:2203.00274v1](https://arxiv.org/abs/2203.00274v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.00274Focus to learn more |





<h2 id="2022-03-02-3">3. DeepNet: Scaling Transformers to 1,000 Layers
</h2>

Title: [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555)

Authors: [Hongyu Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+H), [Shuming Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+S), [Li Dong](https://arxiv.org/search/cs?searchtype=author&query=Dong%2C+L), [Shaohan Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+S), [Dongdong Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+D), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F)

> In this paper, we propose a simple yet effective method to stabilize extremely deep Transformers. Specifically, we introduce a new normalization function (DeepNorm) to modify the residual connection in Transformer, accompanying with theoretically derived initialization. In-depth theoretical analysis shows that model updates can be bounded in a stable way. The proposed method combines the best of two worlds, i.e., good performance of Post-LN and stable training of Pre-LN, making DeepNorm a preferred alternative. We successfully scale Transformers up to 1,000 layers (i.e., 2,500 attention and feed-forward network sublayers) without difficulty, which is one order of magnitude deeper than previous deep Transformers. Remarkably, on a multilingual benchmark with 7,482 translation directions, our 200-layer model with 3.2B parameters significantly outperforms the 48-layer state-of-the-art model with 12B parameters by 5 BLEU points, which indicates a promising scaling direction.

| Comments: | Work in progress                                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2203.00555](https://arxiv.org/abs/2203.00555) [cs.CL]** |
|           | (or **[arXiv:2203.00555v1](https://arxiv.org/abs/2203.00555v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.00555Focus to learn more |





# 2022-03-01

[Return to Index](#Index)



<h2 id="2022-03-01-1">1. Interactive Machine Learning for Image Captioning
</h2>

Title: [Interactive Machine Learning for Image Captioning](https://arxiv.org/abs/2202.13623)

Authors: [Mareike Hartmann](https://arxiv.org/search/cs?searchtype=author&query=Hartmann%2C+M), [Aliki Anagnostopoulou](https://arxiv.org/search/cs?searchtype=author&query=Anagnostopoulou%2C+A), [Daniel Sonntag](https://arxiv.org/search/cs?searchtype=author&query=Sonntag%2C+D)

> We propose an approach for interactive learning for an image captioning model. As human feedback is expensive and modern neural network based approaches often require large amounts of supervised data to be trained, we envision a system that exploits human feedback as good as possible by multiplying the feedback using data augmentation methods, and integrating the resulting training examples into the model in a smart way. This approach has three key components, for which we need to find suitable practical implementations: feedback collection, data augmentation, and model update. We outline our idea and review different possibilities to address these tasks.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2202.13623](https://arxiv.org/abs/2202.13623) [cs.CV]** |
|           | (or **[arXiv:2202.13623v1](https://arxiv.org/abs/2202.13623v1) [cs.CV]** for this version) |





<h2 id="2022-03-01-2">2. Multi-Level Contrastive Learning for Cross-Lingual Alignment
</h2>

Title: [Multi-Level Contrastive Learning for Cross-Lingual Alignment](https://arxiv.org/abs/2202.13083)

Authors: [Beiduo Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+B), [Wu Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+W), [Bin Gu](https://arxiv.org/search/cs?searchtype=author&query=Gu%2C+B), [Quan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Q), [Yongchao Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Y)

> Cross-language pre-trained models such as multilingual BERT (mBERT) have achieved significant performance in various cross-lingual downstream NLP tasks. This paper proposes a multi-level contrastive learning (ML-CTL) framework to further improve the cross-lingual ability of pre-trained models. The proposed method uses translated parallel data to encourage the model to generate similar semantic embeddings for different languages. However, unlike the sentence-level alignment used in most previous studies, in this paper, we explicitly integrate the word-level information of each pair of parallel sentences into contrastive learning. Moreover, cross-zero noise contrastive estimation (CZ-NCE) loss is proposed to alleviate the impact of the floating-point error in the training process with a small batch size. The proposed method significantly improves the cross-lingual transfer ability of our basic model (mBERT) and outperforms on multiple zero-shot cross-lingual downstream tasks compared to the same-size models in the Xtreme benchmark.

| Comments: | Accepted by ICASSP 2022                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2202.13083](https://arxiv.org/abs/2202.13083) [cs.CL]** |
|           | (or **[arXiv:2202.13083v1](https://arxiv.org/abs/2202.13083v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2202.13083Focus to learn more |





<h2 id="2022-03-01-3">3. OCR Improves Machine Translation for Low-Resource Languages
</h2>

Title: [OCR Improves Machine Translation for Low-Resource Languages](https://arxiv.org/abs/2202.13274)

Authors: [Oana Ignat](https://arxiv.org/search/cs?searchtype=author&query=Ignat%2C+O), [Jean Maillard](https://arxiv.org/search/cs?searchtype=author&query=Maillard%2C+J), [Vishrav Chaudhary](https://arxiv.org/search/cs?searchtype=author&query=Chaudhary%2C+V), [Francisco Guzmán](https://arxiv.org/search/cs?searchtype=author&query=Guzmán%2C+F)

> We aim to investigate the performance of current OCR systems on low resource languages and low resource scripts. We introduce and make publicly available a novel benchmark, \textsc{OCR4MT}, consisting of real and synthetic data, enriched with noise, for 60 low-resource languages in low resource scripts. We evaluate state-of-the-art OCR systems on our benchmark and analyse most common errors. We show that OCR monolingual data is a valuable resource that can increase performance of Machine Translation models, when used in backtranslation. We then perform an ablation study to investigate how OCR errors impact Machine Translation performance and determine what is the minimum level of OCR quality needed for the monolingual data to be useful for Machine Translation.

| Comments: | Accepted at ACL Findings 2022                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2202.13274](https://arxiv.org/abs/2202.13274) [cs.CL]** |
|           | (or **[arXiv:2202.13274v1](https://arxiv.org/abs/2202.13274v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2202.13274Focus to learn more |





<h2 id="2022-03-01-4">4. CINO: A Chinese Minority Pre-trained Language Model
</h2>

Title: [CINO: A Chinese Minority Pre-trained Language Model](https://arxiv.org/abs/2202.13558)

Authors: [Ziqing Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Z), [Zihang Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+Z), [Yiming Cui](https://arxiv.org/search/cs?searchtype=author&query=Cui%2C+Y), [Baoxin Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+B), [Min Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+M), [Dayong Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+D), [Zhigang Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Z)

> Multilingual pre-trained language models have shown impressive performance on cross-lingual tasks. It greatly facilitates the applications of natural language processing on low-resource languages. However, there are still some languages that the existing multilingual models do not perform well on. In this paper, we propose CINO (Chinese Minority Pre-trained Language Model), a multilingual pre-trained language model for Chinese minority languages. It covers Standard Chinese, Cantonese, and six other Chinese minority languages. To evaluate the cross-lingual ability of the multilingual models on the minority languages, we collect documents from Wikipedia and build a text classification dataset WCM (Wiki-Chinese-Minority). We test CINO on WCM and two other text classification tasks. Experiments show that CINO outperforms the baselines notably. The CINO model and the WCM dataset are available at [this http URL](http://cino.hfl-rc.com/).

| Comments: | 4 pages                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2202.13558](https://arxiv.org/abs/2202.13558) [cs.CL]** |
|           | (or **[arXiv:2202.13558v1](https://arxiv.org/abs/2202.13558v1) [cs.CL]** for this version) |





<h2 id="2022-03-01-5">5. LCP-dropout: Compression-based Multiple Subword Segmentation for Neural Machine Translation
</h2>

Title: [LCP-dropout: Compression-based Multiple Subword Segmentation for Neural Machine Translation](https://arxiv.org/abs/2202.13590)

Authors: [Keita Nonaka](https://arxiv.org/search/cs?searchtype=author&query=Nonaka%2C+K), [Kazutaka Yamanouchi](https://arxiv.org/search/cs?searchtype=author&query=Yamanouchi%2C+K), [Tomohiro I](https://arxiv.org/search/cs?searchtype=author&query=I%2C+T), [Tsuyoshi Okita](https://arxiv.org/search/cs?searchtype=author&query=Okita%2C+T), [Kazutaka Shimada](https://arxiv.org/search/cs?searchtype=author&query=Shimada%2C+K), [Hiroshi Sakamoto](https://arxiv.org/search/cs?searchtype=author&query=Sakamoto%2C+H)

> In this study, we propose a simple and effective preprocessing method for subword segmentation based on a data compression algorithm. Compression-based subword segmentation has recently attracted significant attention as a preprocessing method for training data in Neural Machine Translation. Among them, BPE/BPE-dropout is one of the fastest and most effective method compared to conventional approaches. However, compression-based approach has a drawback in that generating multiple segmentations is difficult due to the determinism. To overcome this difficulty, we focus on a probabilistic string algorithm, called locally-consistent parsing (LCP), that has been applied to achieve optimum compression. Employing the probabilistic mechanism of LCP, we propose LCP-dropout for multiple subword segmentation that improves BPE/BPE-dropout, and show that it outperforms various baselines in learning from especially small training data.

| Comments: | 10 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2202.13590](https://arxiv.org/abs/2202.13590) [cs.CL]** |
|           | (or **[arXiv:2202.13590v1](https://arxiv.org/abs/2202.13590v1) [cs.CL]** for this version) |





<h2 id="2022-03-01-6">6. MSCTD: A Multimodal Sentiment Chat Translation Dataset
</h2>

Title: [MSCTD: A Multimodal Sentiment Chat Translation Dataset](https://arxiv.org/abs/2202.13645)

Authors: [Yunlong Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+Y), [Fandong Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+F), [Jinan Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+J), [Yufeng Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Y), [Jie Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+J)

> Multimodal machine translation and textual chat translation have received considerable attention in recent years. Although the conversation in its natural form is usually multimodal, there still lacks work on multimodal machine translation in conversations. In this work, we introduce a new task named Multimodal Chat Translation (MCT), aiming to generate more accurate translations with the help of the associated dialogue history and visual context. To this end, we firstly construct a Multimodal Sentiment Chat Translation Dataset (MSCTD) containing 142,871 English-Chinese utterance pairs in 14,762 bilingual dialogues and 30,370 English-German utterance pairs in 3,079 bilingual dialogues. Each utterance pair, corresponding to the visual context that reflects the current conversational scene, is annotated with a sentiment label. Then, we benchmark the task by establishing multiple baseline systems that incorporate multimodal and sentiment features for MCT. Preliminary experiments on four language directions (English-Chinese and English-German) verify the potential of contextual and multimodal information fusion and the positive impact of sentiment on the MCT task. Additionally, as a by-product of the MSCTD, it also provides two new benchmarks on multimodal dialogue sentiment analysis. Our work can facilitate research on both multimodal chat translation and multimodal dialogue sentiment analysis.

| Comments: | Accepted at ACL 2022 as a long paper of main conference. Code and data: [this https URL](https://github.com/XL2248/MSCTD) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2202.13645](https://arxiv.org/abs/2202.13645) [cs.CL]** |
|           | (or **[arXiv:2202.13645v1](https://arxiv.org/abs/2202.13645v1) [cs.CL]** for this version) |





<h2 id="2022-03-01-7">7. Confidence Based Bidirectional Global Context Aware Training Framework for Neural Machine Translation
</h2>

Title: [Confidence Based Bidirectional Global Context Aware Training Framework for Neural Machine Translation](https://arxiv.org/abs/2202.13663)

Authors: [Chulun Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+C), [Fandong Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+F), [Jie Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+J), [Min Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+M), [Hongji Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+H), [Jinsong Su](https://arxiv.org/search/cs?searchtype=author&query=Su%2C+J)

> Most dominant neural machine translation (NMT) models are restricted to make predictions only according to the local context of preceding words in a left-to-right manner. Although many previous studies try to incorporate global information into NMT models, there still exist limitations on how to effectively exploit bidirectional global context. In this paper, we propose a Confidence Based Bidirectional Global Context Aware (CBBGCA) training framework for NMT, where the NMT model is jointly trained with an auxiliary conditional masked language model (CMLM). The training consists of two stages: (1) multi-task joint training; (2) confidence based knowledge distillation. At the first stage, by sharing encoder parameters, the NMT model is additionally supervised by the signal from the CMLM decoder that contains bidirectional global contexts. Moreover, at the second stage, using the CMLM as teacher, we further pertinently incorporate bidirectional global context to the NMT model on its unconfidently-predicted target words via knowledge distillation. Experimental results show that our proposed CBBGCA training framework significantly improves the NMT model by +1.02, +1.30 and +0.57 BLEU scores on three large-scale translation datasets, namely WMT'14 English-to-German, WMT'19 Chinese-to-English and WMT'14 English-to-French, respectively.

| Comments: | Pre-print version; Accepted at ACL 2022 as a long paper of main conference |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2202.13663](https://arxiv.org/abs/2202.13663) [cs.CL]** |
|           | (or **[arXiv:2202.13663v1](https://arxiv.org/abs/2202.13663v1) [cs.CL]** for this version) |





<h2 id="2022-03-01-8">8. LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding
</h2>

Title: [LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding](https://arxiv.org/abs/2202.13669)

Authors: [Jiapeng Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+J), [Lianwen Jin](https://arxiv.org/search/cs?searchtype=author&query=Jin%2C+L), [Kai Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+K)

> Structured document understanding has attracted considerable attention and made significant progress recently, owing to its crucial role in intelligent document processing. However, most existing related models can only deal with the document data of specific language(s) (typically English) included in the pre-training collection, which is extremely limited. To address this issue, we propose a simple yet effective Language-independent Layout Transformer (LiLT) for structured document understanding. LiLT can be pre-trained on the structured documents of a single language and then directly fine-tuned on other languages with the corresponding off-the-shelf monolingual/multilingual pre-trained textual models. Experimental results on eight languages have shown that LiLT can achieve competitive or even superior performance on diverse widely-used downstream benchmarks, which enables language-independent benefit from the pre-training of document layout structure. Code and model are publicly available at [this https URL](https://github.com/jpWang/LiLT).

| Comments: | ACL 2022 Main conference                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2202.13669](https://arxiv.org/abs/2202.13669) [cs.CL]** |
|           | (or **[arXiv:2202.13669v1](https://arxiv.org/abs/2202.13669v1) [cs.CL]** for this version) |





# 2022-02-28

[Return to Index](#Index)



<h2 id="2022-02-28-1">1. Screening Gender Transfer in Neural Machine Translation
</h2>

Title: [Screening Gender Transfer in Neural Machine Translation](https://arxiv.org/abs/2202.12568)

Authors: [Guillaume Wisniewski](https://arxiv.org/search/cs?searchtype=author&query=Wisniewski%2C+G), [Lichao Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+L), [Nicolas Ballier](https://arxiv.org/search/cs?searchtype=author&query=Ballier%2C+N), [François Yvon](https://arxiv.org/search/cs?searchtype=author&query=Yvon%2C+F)

> This paper aims at identifying the information flow in state-of-the-art machine translation systems, taking as example the transfer of gender when translating from French into English. Using a controlled set of examples, we experiment several ways to investigate how gender information circulates in a encoder-decoder architecture considering both probing techniques as well as interventions on the internal representations used in the MT system. Our results show that gender information can be found in all token representations built by the encoder and the decoder and lead us to conclude that there are multiple pathways for gender transfer.

| Comments:    | Accepted at BlackBoxNLP'2021                                 |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**                         |
| Cite as:     | **[arXiv:2202.12568](https://arxiv.org/abs/2202.12568) [cs.CL]** |
|              | (or **[arXiv:2202.12568v1](https://arxiv.org/abs/2202.12568v1) [cs.CL]** for this version) |
| Related DOI: | https://doi.org/10.18653/v1/2021.blackboxnlp-1.24Focus to learn more |



