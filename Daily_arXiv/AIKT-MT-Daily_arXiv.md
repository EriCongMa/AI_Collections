# Daily arXiv: Machine Translation - November, 2021

# Index


- [2021-11-05](#2021-11-05)

  - [1. Benchmarking Multimodal AutoML for Tabular Data with Text Fields](#2021-11-05-1)
  - [2. Lexically Aware Semi-Supervised Learning for OCR Post-Correction](#2021-11-05-2)
  - [3. Response Generation with Context-Aware Prompt Learning](#2021-11-05-3)
  - [4. A text autoencoder from transformer for fast encoding language representation](#2021-11-05-4)
  - [5. CoreLM: Coreference-aware Language Model Fine-Tuning](#2021-11-05-5)
- [2021-11-04](#2021-11-04)

  - [1. LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs](#2021-11-04-1)
  - [2. VLMo: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts](#2021-11-04-2)
  - [3. An Empirical Study of Training End-to-End Vision-and-Language Transformers](#2021-11-04-3)
  - [4. OpenPrompt: An Open-source Framework for Prompt-learning](#2021-11-04-4)
  - [5. Multilingual Machine Translation Systems from Microsoft for WMT21 Shared Task](#2021-11-04-5)
  - [6. Lingua Custodia's participation at the WMT 2021 Machine Translation using Terminologies shared task](#2021-11-04-6)
  - [7. BERT-DRE: BERT with Deep Recursive Encoder for Natural Language Sentence Matching](#2021-11-04-7)
- [2021-11-03](#2021-11-03)

  - [1. Recent Advances in End-to-End Automatic Speech Recognition](#2021-11-03-1)
  - [2. Recent Advances in Natural Language Processing via Large Pre-Trained Language Models: A Survey](#2021-11-03-2)
  - [3. Diverse Distributions of Self-Supervised Tasks for Meta-Learning in NLP](#2021-11-03-3)
  - [4. Adapting to the Long Tail: A Meta-Analysis of Transfer Learning Research for Language Understanding Tasks](#2021-11-03-4)
  - [5. System Combination for Grammatical Error Correction Based on Integer Programming](#2021-11-03-5)
  - [6. Zero-Shot Translation using Diffusion Models](#2021-11-03-6)
  - [7. HydraText: Multi-objective Optimization for Adversarial Textual Attack](#2021-11-03-7)
  - [8. LMdiff: A Visual Diff Tool to Compare Language Models](#2021-11-03-8)
- [2021-11-02](#2021-11-02)
  - [1. Introspective Distillation for Robust Question Answering](#2021-11-02-1)
  - [2. TransAug: Translate as Augmentation for Sentence Embeddings](#2021-11-02-2)
  - [3. How should human translation coexist with NMT? Efficient tool for building high quality parallel corpus](#2021-11-02-3)
  - [4. Visualization: the missing factor in Simultaneous Speech Translation](#2021-11-02-4)
  - [5. Quality Estimation Using Round-trip Translation with Sentence Embeddings](#2021-11-02-5)
  - [6. Unsupervised Domain Adaptation with Adapter](#2021-11-02-6)
  - [7. Interpretable contrastive word mover's embedding](#2021-11-02-7)
- [2021-11-01](#2021-11-01)
  - [1. Decision Attentive Regularization to Improve Simultaneous Speech Translation Systems](#2021-11-01-1)
  - [2. Analysing the Effect of Masking Length Distribution of MLM: An Evaluation Framework and Case Study on Chinese MRC Datasets](#2021-11-01-2)
  - [3. Analysing the Effect of Masking Length Distribution of MLM: An Evaluation Framework and Case Study on Chinese MRC Datasets](#2021-11-01-3)
  - [4. Batch-Softmax Contrastive Loss for Pairwise Sentence Scoring Tasks](#2021-11-01-4)
  - [5. BERMo: What can BERT learn from ELMo?](#2021-11-01-5)
  - [6. MetaICL: Learning to Learn In Context](#2021-11-01-6)
- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MT-Daily_arXiv-index.md)



# 2021-11-05

[Return to Index](#Index)



<h2 id="2021-11-05-1">1. Benchmarking Multimodal AutoML for Tabular Data with Text Fields
</h2>

Title: [Benchmarking Multimodal AutoML for Tabular Data with Text Fields](https://arxiv.org/abs/2111.02705)

Authors: [Xingjian Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+X), [Jonas Mueller](https://arxiv.org/search/cs?searchtype=author&query=Mueller%2C+J), [Nick Erickson](https://arxiv.org/search/cs?searchtype=author&query=Erickson%2C+N), [Mu Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+M), [Alexander J. Smola](https://arxiv.org/search/cs?searchtype=author&query=Smola%2C+A+J)

> We consider the use of automated supervised learning systems for data tables that not only contain numeric/categorical columns, but one or more text fields as well. Here we assemble 18 multimodal data tables that each contain some text fields and stem from a real business application. Our publicly-available benchmark enables researchers to comprehensively evaluate their own methods for supervised learning with numeric, categorical, and text features. To ensure that any single modeling strategy which performs well over all 18 datasets will serve as a practical foundation for multimodal text/tabular AutoML, the diverse datasets in our benchmark vary greatly in: sample size, problem types (a mix of classification and regression tasks), number of features (with the number of text columns ranging from 1 to 28 between datasets), as well as how the predictive signal is decomposed between text vs. numeric/categorical features (and predictive interactions thereof). Over this benchmark, we evaluate various straightforward pipelines to model such data, including standard two-stage approaches where NLP is used to featurize the text such that AutoML for tabular data can then be applied. Compared with human data science teams, the fully automated methodology that performed best on our benchmark (stack ensembling a multimodal Transformer with various tree models) also manages to rank 1st place when fit to the raw text/tabular data in two MachineHack prediction competitions and 2nd place (out of 2380 teams) in Kaggle's Mercari Price Suggestion Challenge.

| Comments: | Proceedings of the Neural Information Processing Systems (NeurIPS) Track on Datasets and Benchmarks 2021 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Machine Learning (stat.ML) |
| Cite as:  | **[arXiv:2111.02705](https://arxiv.org/abs/2111.02705) [cs.LG]** |
|           | (or **[arXiv:2111.02705v1](https://arxiv.org/abs/2111.02705v1) [cs.LG]** for this version) |





<h2 id="2021-11-05-2">2. Lexically Aware Semi-Supervised Learning for OCR Post-Correction
</h2>

Title: [Lexically Aware Semi-Supervised Learning for OCR Post-Correction](https://arxiv.org/abs/2111.02622)

Authors: [Shruti Rijhwani](https://arxiv.org/search/cs?searchtype=author&query=Rijhwani%2C+S), [Daisy Rosenblum](https://arxiv.org/search/cs?searchtype=author&query=Rosenblum%2C+D), [Antonios Anastasopoulos](https://arxiv.org/search/cs?searchtype=author&query=Anastasopoulos%2C+A), [Graham Neubig](https://arxiv.org/search/cs?searchtype=author&query=Neubig%2C+G)

> Much of the existing linguistic data in many languages of the world is locked away in non-digitized books and documents. Optical character recognition (OCR) can be used to produce digitized text, and previous work has demonstrated the utility of neural post-correction methods that improve the results of general-purpose OCR systems on recognition of less-well-resourced languages. However, these methods rely on manually curated post-correction data, which are relatively scarce compared to the non-annotated raw images that need to be digitized. 
> In this paper, we present a semi-supervised learning method that makes it possible to utilize these raw images to improve performance, specifically through the use of self-training, a technique where a model is iteratively trained on its own outputs. In addition, to enforce consistency in the recognized vocabulary, we introduce a lexically-aware decoding method that augments the neural post-correction model with a count-based language model constructed from the recognized texts, implemented using weighted finite-state automata (WFSA) for efficient and effective decoding. 
> Results on four endangered languages demonstrate the utility of the proposed method, with relative error reductions of 15-29%, where we find the combination of self-training and lexically-aware decoding essential for achieving consistent improvements. Data and code are available at [this https URL](https://shrutirij.github.io/ocr-el/).

| Comments: | Accepted to the Transactions of the Association for Computational Linguistics (TACL) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2111.02622](https://arxiv.org/abs/2111.02622) [cs.CL]** |
|           | (or **[arXiv:2111.02622v1](https://arxiv.org/abs/2111.02622v1) [cs.CL]** for this version) |





<h2 id="2021-11-05-3">3. Response Generation with Context-Aware Prompt Learning
</h2>

Title: [Response Generation with Context-Aware Prompt Learning](https://arxiv.org/abs/2111.02643)

Authors: [Xiaodong Gu](https://arxiv.org/search/cs?searchtype=author&query=Gu%2C+X), [Kang Min Yoo](https://arxiv.org/search/cs?searchtype=author&query=Yoo%2C+K+M), [Sang-Woo Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+S)

> Pre-trained language models (PLM) have marked a huge leap in neural dialogue modeling. While PLMs are pre-trained on large-scale text corpora, they are usually fine-tuned on scarce dialogue data with specific domain knowledge and dialogue styles. However, tailoring the language models while fully utilizing prior knowledge in large pre-trained models remains a challenge. In this paper, we present a novel approach for pre-trained dialogue modeling that casts the dialogue generation problem as a prompt-learning task. Instead of fine-tuning on limited dialogue data, our approach, DialogPrompt, learns continuous prompt embeddings optimized for dialogue contexts, which appropriately elicit knowledge from the large pre-trained model. To encourage the model to better utilize the prompt embeddings, the prompt encoders are designed to be conditioned on the input dialogue context. Experiments on popular conversation datasets show that our approach significantly outperforms the fine-tuning baseline and the generic prompt-learning methods. Furthermore, human evaluations strongly support the superiority of DialogPrompt in regard to response generation quality.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2111.02643](https://arxiv.org/abs/2111.02643) [cs.CL]** |
|           | (or **[arXiv:2111.02643v1](https://arxiv.org/abs/2111.02643v1) [cs.CL]** for this version) |





<h2 id="2021-11-05-4">4. A text autoencoder from transformer for fast encoding language representation
</h2>

Title: [A text autoencoder from transformer for fast encoding language representation](https://arxiv.org/abs/2111.02844)

Authors: [Tan Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+T)

> In recent years BERT shows apparent advantages and great potential in natural language processing tasks. However, both training and applying BERT requires intensive time and resources for computing contextual language representations, which hinders its universality and applicability. To overcome this bottleneck, we propose a deep bidirectional language model by using window masking mechanism at attention layer. This work computes contextual language representations without random masking as does in BERT and maintains the deep bidirectional architecture like BERT. To compute the same sentence representation, our method shows O(n) complexity less compared to other transformer-based models with O(n2). To further demonstrate its superiority, computing context language representations on CPU environments is conducted, by using the embeddings from the proposed method, logistic regression shows much higher accuracy in terms of SMS classification. Moverover, the proposed method also achieves significant higher performance in semantic similarity tasks.

| Comments: | 8 pages, 8 figures. arXiv admin note: text overlap with [arXiv:2004.08097](https://arxiv.org/abs/2004.08097) by other authors |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2111.02844](https://arxiv.org/abs/2111.02844) [cs.CL]** |
|           | (or **[arXiv:2111.02844v1](https://arxiv.org/abs/2111.02844v1) [cs.CL]** for this version) |





<h2 id="2021-11-05-5">5. CoreLM: Coreference-aware Language Model Fine-Tuning
</h2>

Title: [CoreLM: Coreference-aware Language Model Fine-Tuning](https://arxiv.org/abs/2111.02687)

Authors: [Nikolaos Stylianou](https://arxiv.org/search/cs?searchtype=author&query=Stylianou%2C+N), [Ioannis Vlahavas](https://arxiv.org/search/cs?searchtype=author&query=Vlahavas%2C+I)

> Language Models are the underpin of all modern Natural Language Processing (NLP) tasks. The introduction of the Transformers architecture has contributed significantly into making Language Modeling very effective across many NLP task, leading to significant advancements in the field. However, Transformers come with a big computational cost, which grows quadratically with respect to the input length. This presents a challenge as to understand long texts requires a lot of context. In this paper, we propose a Fine-Tuning framework, named CoreLM, that extends the architecture of current Pretrained Language Models so that they incorporate explicit entity information. By introducing entity representations, we make available information outside the contextual space of the model, which results in a better Language Model for a fraction of the computational cost. We implement our approach using GPT2 and compare the fine-tuned model to the original. Our proposed model achieves a lower Perplexity in GUMBY and LAMBDADA datasets when compared to GPT2 and a fine-tuned version of GPT2 without any changes. We also compare the models' performance in terms of Accuracy in LAMBADA and Children's Book Test, with and without the use of model-created coreference annotations.

| Comments: | 12 pages, 2 figures, Accepted at Fourth Workshop on Computational Models of Reference, Anaphora and Coreference |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2111.02687](https://arxiv.org/abs/2111.02687) [cs.CL]** |
|           | (or **[arXiv:2111.02687v1](https://arxiv.org/abs/2111.02687v1) [cs.CL]** for this version) |





# 2021-11-04

[Return to Index](#Index)



<h2 id="2021-11-04-1">1. LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs
</h2>

Title: [LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs](https://arxiv.org/abs/2111.02114)

Authors: [Christoph Schuhmann](https://arxiv.org/search/cs?searchtype=author&query=Schuhmann%2C+C), [Richard Vencu](https://arxiv.org/search/cs?searchtype=author&query=Vencu%2C+R), [Romain Beaumont](https://arxiv.org/search/cs?searchtype=author&query=Beaumont%2C+R), [Robert Kaczmarczyk](https://arxiv.org/search/cs?searchtype=author&query=Kaczmarczyk%2C+R), [Clayton Mullis](https://arxiv.org/search/cs?searchtype=author&query=Mullis%2C+C), [Aarush Katta](https://arxiv.org/search/cs?searchtype=author&query=Katta%2C+A), [Theo Coombes](https://arxiv.org/search/cs?searchtype=author&query=Coombes%2C+T), [Jenia Jitsev](https://arxiv.org/search/cs?searchtype=author&query=Jitsev%2C+J), [Aran Komatsuzaki](https://arxiv.org/search/cs?searchtype=author&query=Komatsuzaki%2C+A)

> Multi-modal language-vision models trained on hundreds of millions of image-text pairs (e.g. CLIP, DALL-E) gained a recent surge, showing remarkable capability to perform zero- or few-shot learning and transfer even in absence of per-sample labels on target image data. Despite this trend, to date there has been no publicly available datasets of sufficient scale for training such models from scratch. To address this issue, in a community effort we build and release for public LAION-400M, a dataset with CLIP-filtered 400 million image-text pairs, their CLIP embeddings and kNN indices that allow efficient similarity search.

| Comments: | Short version. Accepted at Data Centric AI NeurIPS Workshop 2021 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2111.02114](https://arxiv.org/abs/2111.02114) [cs.CV]** |
|           | (or **[arXiv:2111.02114v1](https://arxiv.org/abs/2111.02114v1) [cs.CV]** for this version) |





<h2 id="2021-11-04-2">2. VLMo: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts
</h2>

Title: [VLMo: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts](https://arxiv.org/abs/2111.02358)

Authors: [Wenhui Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+W), [Hangbo Bao](https://arxiv.org/search/cs?searchtype=author&query=Bao%2C+H), [Li Dong](https://arxiv.org/search/cs?searchtype=author&query=Dong%2C+L), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F)

> We present a unified Vision-Language pretrained Model (VLMo) that jointly learns a dual encoder and a fusion encoder with a modular Transformer network. Specifically, we introduce Mixture-of-Modality-Experts (MoME) Transformer, where each block contains a pool of modality-specific experts and a shared self-attention layer. Because of the modeling flexibility of MoME, pretrained VLMo can be fine-tuned as a fusion encoder for vision-language classification tasks, or used as a dual encoder for efficient image-text retrieval. Moreover, we propose a stagewise pre-training strategy, which effectively leverages large-scale image-only and text-only data besides image-text pairs. Experimental results show that VLMo achieves state-of-the-art results on various vision-language tasks, including VQA and NLVR2. The code and pretrained models are available at [this https URL](https://aka.ms/vlmo).

| Comments: | Work in progress                                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2111.02358](https://arxiv.org/abs/2111.02358) [cs.CV]** |
|           | (or **[arXiv:2111.02358v1](https://arxiv.org/abs/2111.02358v1) [cs.CV]** for this version) |



<h2 id="2021-11-04-3">3. An Empirical Study of Training End-to-End Vision-and-Language Transformers
</h2>

Title: [An Empirical Study of Training End-to-End Vision-and-Language Transformers](https://arxiv.org/abs/2111.02387)

Authors: [Zi-Yi Dou](https://arxiv.org/search/cs?searchtype=author&query=Dou%2C+Z), [Yichong Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+Y), [Zhe Gan](https://arxiv.org/search/cs?searchtype=author&query=Gan%2C+Z), [Jianfeng Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+J), [Shuohang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+S), [Lijuan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L), [Chenguang Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+C), [Nanyun](https://arxiv.org/search/cs?searchtype=author&query=Nanyun) (Violet)[Peng](https://arxiv.org/search/cs?searchtype=author&query=Peng), [Zicheng Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Z), [Michael Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+M)

> Vision-and-language (VL) pre-training has proven to be highly effective on various VL downstream tasks. While recent work has shown that fully transformer-based VL models can be more efficient than previous region-feature-based methods, their performance on downstream tasks are often degraded significantly. In this paper, we present METER~(\textbf{M}ultimodal \textbf{E}nd-to-end \textbf{T}ransform\textbf{ER}), through which we systematically investigate how to design and pre-train a fully transformer-based VL model in an end-to-end manner. Specifically, we dissect the model designs along multiple dimensions: vision encoders (e.g., CLIP-ViT, Swin transformer), text encoders (e.g., RoBERTa, DeBERTa), multimodal fusion (e.g., merged attention vs. co-attention), architecture design (e.g., encoder-only vs. encoder-decoder), and pre-training objectives (e.g., masked image modeling). We conduct comprehensive experiments on a wide range of VL tasks, and provide insights on how to train a performant VL transformer while maintaining fast inference speed. Notably, METER~achieves an accuracy of 77.64\% on the VQAv2 test-std set using only 4M images for pre-training, surpassing the state-of-the-art region-feature-based VinVL model by +1.04\%, and outperforming the previous best fully transformer-based ALBEF model by +1.6\%.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2111.02387](https://arxiv.org/abs/2111.02387) [cs.CV]** |
|           | (or **[arXiv:2111.02387v1](https://arxiv.org/abs/2111.02387v1) [cs.CV]** for this version) |





<h2 id="2021-11-04-4">4. OpenPrompt: An Open-source Framework for Prompt-learning
</h2>

Title: [OpenPrompt: An Open-source Framework for Prompt-learning](https://arxiv.org/abs/2111.01998)

Authors: [Ning Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+N), [Shengding Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+S), [Weilin Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+W), [Yulin Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Y), [Zhiyuan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Z), [Hai-Tao Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+H), [Maosong Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+M)

> Prompt-learning has become a new paradigm in modern natural language processing, which directly adapts pre-trained language models (PLMs) to cloze-style prediction, autoregressive modeling, or sequence to sequence generation, resulting in promising performances on various tasks. However, no standard implementation framework of prompt-learning is proposed yet, and most existing prompt-learning codebases, often unregulated, only provide limited implementations for specific scenarios. Since there are many details such as templating strategy, initializing strategy, and verbalizing strategy, etc. need to be considered in prompt-learning, practitioners face impediments to quickly adapting the desired prompt learning methods to their applications. In this paper, we present {OpenPrompt}, a unified easy-to-use toolkit to conduct prompt-learning over PLMs. OpenPrompt is a research-friendly framework that is equipped with efficiency, modularity, and extendibility, and its combinability allows the freedom to combine different PLMs, task formats, and prompting modules in a unified paradigm. Users could expediently deploy prompt-learning frameworks and evaluate the generalization of them on different NLP tasks without constraints. OpenPrompt is publicly released at {\url{ [this https URL](https://github.com/thunlp/OpenPrompt)}}.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2111.01998](https://arxiv.org/abs/2111.01998) [cs.CL]** |
|           | (or **[arXiv:2111.01998v1](https://arxiv.org/abs/2111.01998v1) [cs.CL]** for this version) |





<h2 id="2021-11-04-5">5. Multilingual Machine Translation Systems from Microsoft for WMT21 Shared Task
</h2>

Title: [Multilingual Machine Translation Systems from Microsoft for WMT21 Shared Task](https://arxiv.org/abs/2111.02086)

Authors: [Jian Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+J), [Shuming Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+S), [Haoyang Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+H), [Dongdong Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+D), [Li Dong](https://arxiv.org/search/cs?searchtype=author&query=Dong%2C+L), [Shaohan Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+S), [Alexandre Muzio](https://arxiv.org/search/cs?searchtype=author&query=Muzio%2C+A), [Saksham Singhal](https://arxiv.org/search/cs?searchtype=author&query=Singhal%2C+S), [Hany Hassan Awadalla](https://arxiv.org/search/cs?searchtype=author&query=Awadalla%2C+H+H), [Xia Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+X), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F)

> This report describes Microsoft's machine translation systems for the WMT21 shared task on large-scale multilingual machine translation. We participated in all three evaluation tracks including Large Track and two Small Tracks where the former one is unconstrained and the latter two are fully constrained. Our model submissions to the shared task were initialized with DeltaLM\footnote{\url{[this https URL](https://aka.ms/deltalm)}}, a generic pre-trained multilingual encoder-decoder model, and fine-tuned correspondingly with the vast collected parallel data and allowed data sources according to track settings, together with applying progressive learning and iterative back-translation approaches to further improve the performance. Our final submissions ranked first on three tracks in terms of the automatic evaluation metric.

| Comments: | WMT21                                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2111.02086](https://arxiv.org/abs/2111.02086) [cs.CL]** |
|           | (or **[arXiv:2111.02086v1](https://arxiv.org/abs/2111.02086v1) [cs.CL]** for this version) |





<h2 id="2021-11-04-6">6. Lingua Custodia's participation at the WMT 2021 Machine Translation using Terminologies shared task
</h2>

Title: [Lingua Custodia's participation at the WMT 2021 Machine Translation using Terminologies shared task](https://arxiv.org/abs/2111.02120)

Authors: [Melissa Ailem](https://arxiv.org/search/cs?searchtype=author&query=Ailem%2C+M), [Jinghsu Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+J), [Raheel Qader](https://arxiv.org/search/cs?searchtype=author&query=Qader%2C+R)

> This paper describes Lingua Custodia's submission to the WMT21 shared task on machine translation using terminologies. We consider three directions, namely English to French, Russian, and Chinese. We rely on a Transformer-based architecture as a building block, and we explore a method which introduces two main changes to the standard procedure to handle terminologies. The first one consists in augmenting the training data in such a way as to encourage the model to learn a copy behavior when it encounters terminology constraint terms. The second change is constraint token masking, whose purpose is to ease copy behavior learning and to improve model generalization. Empirical results show that our method satisfies most terminology constraints while maintaining high translation quality.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2111.02120](https://arxiv.org/abs/2111.02120) [cs.CL]** |
|           | (or **[arXiv:2111.02120v1](https://arxiv.org/abs/2111.02120v1) [cs.CL]** for this version) |





<h2 id="2021-11-04-7">7. BERT-DRE: BERT with Deep Recursive Encoder for Natural Language Sentence Matching
</h2>

Title: [BERT-DRE: BERT with Deep Recursive Encoder for Natural Language Sentence Matching](https://arxiv.org/abs/2111.02188)

Authors: [Ehsan Tavan](https://arxiv.org/search/cs?searchtype=author&query=Tavan%2C+E), [Ali Rahmati](https://arxiv.org/search/cs?searchtype=author&query=Rahmati%2C+A), [Maryam Najafi](https://arxiv.org/search/cs?searchtype=author&query=Najafi%2C+M), [Saeed Bibak](https://arxiv.org/search/cs?searchtype=author&query=Bibak%2C+S)

> This paper presents a deep neural architecture, for Natural Language Sentence Matching (NLSM) by adding a deep recursive encoder to BERT so called BERT with Deep Recursive Encoder (BERT-DRE). Our analysis of model behavior shows that BERT still does not capture the full complexity of text, so a deep recursive encoder is applied on top of BERT. Three Bi-LSTM layers with residual connection are used to design a recursive encoder and an attention module is used on top of this encoder. To obtain the final vector, a pooling layer consisting of average and maximum pooling is used. We experiment our model on four benchmarks, SNLI, FarsTail, MultiNLI, SciTail, and a novel Persian religious questions dataset. This paper focuses on improving the BERT results in the NLSM task. In this regard, comparisons between BERT-DRE and BERT are conducted, and it is shown that in all cases, BERT-DRE outperforms only BERT. The BERT algorithm on the religious dataset achieved an accuracy of 89.70%, and BERT-DRE architectures improved to 90.29% using the same dataset.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2111.02188](https://arxiv.org/abs/2111.02188) [cs.CL]** |
|           | (or **[arXiv:2111.02188v1](https://arxiv.org/abs/2111.02188v1) [cs.CL]** for this version) |







# 2021-11-03

[Return to Index](#Index)



<h2 id="2021-11-03-1">1. Recent Advances in End-to-End Automatic Speech Recognition
</h2>

Title: [Recent Advances in End-to-End Automatic Speech Recognition](https://arxiv.org/abs/2111.01690)

Authors: [Jinyu Li](https://arxiv.org/search/eess?searchtype=author&query=Li%2C+J)

> Recently, the speech community is seeing a significant trend of moving from deep neural network based hybrid modeling to end-to-end (E2E) modeling for automatic speech recognition (ASR). While E2E models achieve the state-of-the-art results in most benchmarks in terms of ASR accuracy, hybrid models are still used in a large proportion of commercial ASR systems at the current time. There are lots of practical factors that affect the production model deployment decision. Traditional hybrid models, being optimized for production for decades, are usually good at these factors. Without providing excellent solutions to all these factors, it is hard for E2E models to be widely commercialized. In this paper, we will overview the recent advances in E2E models, focusing on technologies addressing those challenges from the industry's perspective.

| Comments: | invited paper submitted to APSIPA Transactions on Signal and Information Processing |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Audio and Speech Processing (eess.AS)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD) |
| Cite as:  | **[arXiv:2111.01690](https://arxiv.org/abs/2111.01690) [eess.AS]** |
|           | (or **[arXiv:2111.01690v1](https://arxiv.org/abs/2111.01690v1) [eess.AS]** for this version) |





<h2 id="2021-11-03-2">2. Recent Advances in Natural Language Processing via Large Pre-Trained Language Models: A Survey
</h2>

Title: [Recent Advances in Natural Language Processing via Large Pre-Trained Language Models: A Survey](https://arxiv.org/abs/2111.01243)

Authors: [Bonan Min](https://arxiv.org/search/cs?searchtype=author&query=Min%2C+B), [Hayley Ross](https://arxiv.org/search/cs?searchtype=author&query=Ross%2C+H), [Elior Sulem](https://arxiv.org/search/cs?searchtype=author&query=Sulem%2C+E), [Amir Pouran Ben Veyseh](https://arxiv.org/search/cs?searchtype=author&query=Veyseh%2C+A+P+B), [Thien Huu Nguyen](https://arxiv.org/search/cs?searchtype=author&query=Nguyen%2C+T+H), [Oscar Sainz](https://arxiv.org/search/cs?searchtype=author&query=Sainz%2C+O), [Eneko Agirre](https://arxiv.org/search/cs?searchtype=author&query=Agirre%2C+E), [Ilana Heinz](https://arxiv.org/search/cs?searchtype=author&query=Heinz%2C+I), [Dan Roth](https://arxiv.org/search/cs?searchtype=author&query=Roth%2C+D)

> Large, pre-trained transformer-based language models such as BERT have drastically changed the Natural Language Processing (NLP) field. We present a survey of recent work that uses these large language models to solve NLP tasks via pre-training then fine-tuning, prompting, or text generation approaches. We also present approaches that use pre-trained language models to generate data for training augmentation or other purposes. We conclude with discussions on limitations and suggested directions for future research.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2111.01243](https://arxiv.org/abs/2111.01243) [cs.CL]** |
|           | (or **[arXiv:2111.01243v1](https://arxiv.org/abs/2111.01243v1) [cs.CL]** for this version) |





<h2 id="2021-11-03-3">3. Diverse Distributions of Self-Supervised Tasks for Meta-Learning in NLP
</h2>

Title: [Diverse Distributions of Self-Supervised Tasks for Meta-Learning in NLP](https://arxiv.org/abs/2111.01322)

Authors: Diverse Distributions of Self-Supervised Tasks for Meta-Learning in NLP

[Trapit Bansal](https://arxiv.org/search/cs?searchtype=author&query=Bansal%2C+T), [Karthick Gunasekaran](https://arxiv.org/search/cs?searchtype=author&query=Gunasekaran%2C+K), [Tong Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+T), [Tsendsuren Munkhdalai](https://arxiv.org/search/cs?searchtype=author&query=Munkhdalai%2C+T), [Andrew McCallum](https://arxiv.org/search/cs?searchtype=author&query=McCallum%2C+A)

> Meta-learning considers the problem of learning an efficient learning process that can leverage its past experience to accurately solve new tasks. However, the efficacy of meta-learning crucially depends on the distribution of tasks available for training, and this is often assumed to be known a priori or constructed from limited supervised datasets. In this work, we aim to provide task distributions for meta-learning by considering self-supervised tasks automatically proposed from unlabeled text, to enable large-scale meta-learning in NLP. We design multiple distributions of self-supervised tasks by considering important aspects of task diversity, difficulty, type, domain, and curriculum, and investigate how they affect meta-learning performance. Our analysis shows that all these factors meaningfully alter the task distribution, some inducing significant improvements in downstream few-shot accuracy of the meta-learned models. Empirically, results on 20 downstream tasks show significant improvements in few-shot learning -- adding up to +4.2% absolute accuracy (on average) to the previous unsupervised meta-learning method, and perform comparably to supervised methods on the FewRel 2.0 benchmark.

| Comments: | To appear at EMNLP 2021                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2111.01322](https://arxiv.org/abs/2111.01322) [cs.CL]** |
|           | (or **[arXiv:2111.01322v1](https://arxiv.org/abs/2111.01322v1) [cs.CL]** for this version) |





<h2 id="2021-11-03-4">4. Adapting to the Long Tail: A Meta-Analysis of Transfer Learning Research for Language Understanding Tasks
</h2>

Title: [Adapting to the Long Tail: A Meta-Analysis of Transfer Learning Research for Language Understanding Tasks](https://arxiv.org/abs/2111.01340)

Authors: [Aakanksha Naik](https://arxiv.org/search/cs?searchtype=author&query=Naik%2C+A), [Jill Lehman](https://arxiv.org/search/cs?searchtype=author&query=Lehman%2C+J), [Carolyn Rose](https://arxiv.org/search/cs?searchtype=author&query=Rose%2C+C)

> Natural language understanding (NLU) has made massive progress driven by large benchmarks, paired with research on transfer learning to broaden its impact. Benchmarks are dominated by a small set of frequent phenomena, leaving a long tail of infrequent phenomena underrepresented. In this work, we reflect on the question: have transfer learning methods sufficiently addressed performance of benchmark-trained models on the long tail? Since benchmarks do not list included/excluded phenomena, we conceptualize the long tail using macro-level dimensions such as underrepresented genres, topics, etc. We assess trends in transfer learning research through a qualitative meta-analysis of 100 representative papers on transfer learning for NLU. Our analysis asks three questions: (i) Which long tail dimensions do transfer learning studies target? (ii) Which properties help adaptation methods improve performance on the long tail? (iii) Which methodological gaps have greatest negative impact on long tail performance? Our answers to these questions highlight major avenues for future research in transfer learning for the long tail. Lastly, we present a case study comparing the performance of various adaptation methods on clinical narratives to show how systematically conducted meta-experiments can provide insights that enable us to make progress along these future avenues.

| Comments: | 14 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2111.01340](https://arxiv.org/abs/2111.01340) [cs.CL]** |
|           | (or **[arXiv:2111.01340v1](https://arxiv.org/abs/2111.01340v1) [cs.CL]** for this version) |





<h2 id="2021-11-03-5">5. System Combination for Grammatical Error Correction Based on Integer Programming
</h2>

Title: [System Combination for Grammatical Error Correction Based on Integer Programming](https://arxiv.org/abs/2111.01465)

Authors: [Ruixi Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+R), [Hwee Tou Ng](https://arxiv.org/search/cs?searchtype=author&query=Ng%2C+H+T)

> In this paper, we propose a system combination method for grammatical error correction (GEC), based on nonlinear integer programming (IP). Our method optimizes a novel F score objective based on error types, and combines multiple end-to-end GEC systems. The proposed IP approach optimizes the selection of a single best system for each grammatical error type present in the data. Experiments of the IP approach on combining state-of-the-art standalone GEC systems show that the combined system outperforms all standalone systems. It improves F0.5 score by 3.61% when combining the two best participating systems in the BEA 2019 shared task, and achieves F0.5 score of 73.08%. We also perform experiments to compare our IP approach with another state-of-the-art system combination method for GEC, demonstrating IP's competitive combination capability.

| Comments:          | Accepted for RANLP 2021                                      |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**                         |
| Journal reference: | RANLP (RECENT ADVANCES IN NATURAL LANGUAGE PROCESSING) (2021) |
| Cite as:           | **[arXiv:2111.01465](https://arxiv.org/abs/2111.01465) [cs.CL]** |
|                    | (or **[arXiv:2111.01465v1](https://arxiv.org/abs/2111.01465v1) [cs.CL]** for this version) |





<h2 id="2021-11-03-6">6. Zero-Shot Translation using Diffusion Models
</h2>

Title: [Zero-Shot Translation using Diffusion Models](https://arxiv.org/abs/2111.01471)

Authors: [Eliya Nachmani](https://arxiv.org/search/cs?searchtype=author&query=Nachmani%2C+E), [Shaked Dovrat](https://arxiv.org/search/cs?searchtype=author&query=Dovrat%2C+S)

> In this work, we show a novel method for neural machine translation (NMT), using a denoising diffusion probabilistic model (DDPM), adjusted for textual data, following recent advances in the field. We show that it's possible to translate sentences non-autoregressively using a diffusion model conditioned on the source sentence. We also show that our model is able to translate between pairs of languages unseen during training (zero-shot learning).

| Comments: | preprint                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2111.01471](https://arxiv.org/abs/2111.01471) [cs.CL]** |
|           | (or **[arXiv:2111.01471v1](https://arxiv.org/abs/2111.01471v1) [cs.CL]** for this version) |





<h2 id="2021-11-03-7">7. HydraText: Multi-objective Optimization for Adversarial Textual Attack
</h2>

Title: [HydraText: Multi-objective Optimization for Adversarial Textual Attack](https://arxiv.org/abs/2111.01528)

Authors: [Shengcai Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+S), [Ning Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+N), [Cheng Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+C), [Chao Qian](https://arxiv.org/search/cs?searchtype=author&query=Qian%2C+C), [Ke Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+K)

> The field of adversarial textual attack has significantly grown over the last years, where the commonly considered objective is to craft adversarial examples that can successfully fool the target models. However, the imperceptibility of attacks, which is also an essential objective, is often left out by previous studies. In this work, we advocate considering both objectives at the same time, and propose a novel multi-optimization approach (dubbed HydraText) with provable performance guarantee to achieve successful attacks with high imperceptibility. We demonstrate the efficacy of HydraText through extensive experiments under both score-based and decision-based settings, involving five modern NLP models across five benchmark datasets. In comparison to existing state-of-the-art attacks, HydraText consistently achieves simultaneously higher success rates, lower modification rates, and higher semantic similarity to the original texts. A human evaluation study shows that the adversarial examples crafted by HydraText maintain validity and naturality well. Finally, these examples also exhibit good transferability and can bring notable robustness improvement to the target models by adversarial training.

| Subjects: | **Computation and Language (cs.CL)**; Neural and Evolutionary Computing (cs.NE) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2111.01528](https://arxiv.org/abs/2111.01528) [cs.CL]** |
|           | (or **[arXiv:2111.01528v1](https://arxiv.org/abs/2111.01528v1) [cs.CL]** for this version) |





<h2 id="2021-11-03-8">8. LMdiff: A Visual Diff Tool to Compare Language Models
</h2>

Title: [LMdiff: A Visual Diff Tool to Compare Language Models](https://arxiv.org/abs/2111.01582)

Authors: [Hendrik Strobelt](https://arxiv.org/search/cs?searchtype=author&query=Strobelt%2C+H), [Benjamin Hoover](https://arxiv.org/search/cs?searchtype=author&query=Hoover%2C+B), [Arvind Satyanarayan](https://arxiv.org/search/cs?searchtype=author&query=Satyanarayan%2C+A), [Sebastian Gehrmann](https://arxiv.org/search/cs?searchtype=author&query=Gehrmann%2C+S)

> While different language models are ubiquitous in NLP, it is hard to contrast their outputs and identify which contexts one can handle better than the other. To address this question, we introduce LMdiff, a tool that visually compares probability distributions of two models that differ, e.g., through finetuning, distillation, or simply training with different parameter sizes. LMdiff allows the generation of hypotheses about model behavior by investigating text instances token by token and further assists in choosing these interesting text instances by identifying the most interesting phrases from large corpora. We showcase the applicability of LMdiff for hypothesis generation across multiple case studies. A demo is available at [this http URL](http://lmdiff.net/) .

| Comments: | EMNLP 2021 Demo Paper                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Human-Computer Interaction (cs.HC) |
| Cite as:  | **[arXiv:2111.01582](https://arxiv.org/abs/2111.01582) [cs.CL]** |
|           | (or **[arXiv:2111.01582v1](https://arxiv.org/abs/2111.01582v1) [cs.CL]** for this version) |








# 2021-11-02

[Return to Index](#Index)



<h2 id="2021-11-02-1">1. Introspective Distillation for Robust Question Answering
</h2>

Title: [Introspective Distillation for Robust Question Answering](https://arxiv.org/abs/2111.01026)

Authors: [Yulei Niu](https://arxiv.org/search/cs?searchtype=author&query=Niu%2C+Y), [Hanwang Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+H)

> Question answering (QA) models are well-known to exploit data bias, e.g., the language prior in visual QA and the position bias in reading comprehension. Recent debiasing methods achieve good out-of-distribution (OOD) generalizability with a considerable sacrifice of the in-distribution (ID) performance. Therefore, they are only applicable in domains where the test distribution is known in advance. In this paper, we present a novel debiasing method called Introspective Distillation (IntroD) to make the best of both worlds for QA. Our key technical contribution is to blend the inductive bias of OOD and ID by introspecting whether a training sample fits in the factual ID world or the counterfactual OOD one. Experiments on visual QA datasets VQA v2, VQA-CP, and reading comprehension dataset SQuAD demonstrate that our proposed IntroD maintains the competitive OOD performance compared to other debiasing methods, while sacrificing little or even achieving better ID performance compared to the non-debiasing ones.

| Comments: | Accepted by NeurIPS 2021                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2111.01026](https://arxiv.org/abs/2111.01026) [cs.CV]** |
|           | (or **[arXiv:2111.01026v1](https://arxiv.org/abs/2111.01026v1) [cs.CV]** for this version) |





<h2 id="2021-11-02-2">2. TransAug: Translate as Augmentation for Sentence Embeddings
</h2>

Title: [TransAug: Translate as Augmentation for Sentence Embeddings](https://arxiv.org/abs/2111.00157)

Authors: [Jue Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+J), [Haofan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+H), [Xing Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+X), [Chaochen Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+C), [Debing Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+D)

> While contrastive learning greatly advances the representation of sentence embeddings, it is still limited by the size of the existing sentence datasets. In this paper, we present TransAug (Translate as Augmentation), which provide the first exploration of utilizing translated sentence pairs as data augmentation for text, and introduce a two-stage paradigm to advances the state-of-the-art sentence embeddings. Instead of adopting an encoder trained in other languages setting, we first distill a Chinese encoder from a SimCSE encoder (pretrained in English), so that their embeddings are close in semantic space, which can be regraded as implicit data augmentation. Then, we only update the English encoder via cross-lingual contrastive learning and frozen the distilled Chinese encoder. Our approach achieves a new state-of-art on standard semantic textual similarity (STS), outperforming both SimCSE and Sentence-T5, and the best performance in corresponding tracks on transfer tasks evaluated by SentEval.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2111.00157](https://arxiv.org/abs/2111.00157) [cs.CL]** |
|           | (or **[arXiv:2111.00157v1](https://arxiv.org/abs/2111.00157v1) [cs.CL]** for this version) |





<h2 id="2021-11-02-3">3. How should human translation coexist with NMT? Efficient tool for building high quality parallel corpus
</h2>

Title: [How should human translation coexist with NMT? Efficient tool for building high quality parallel corpus](https://arxiv.org/abs/2111.00191)

Authors: [Chanjun Park](https://arxiv.org/search/cs?searchtype=author&query=Park%2C+C), [Seolhwa Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+S), [Hyeonseok Moon](https://arxiv.org/search/cs?searchtype=author&query=Moon%2C+H), [Sugyeong Eo](https://arxiv.org/search/cs?searchtype=author&query=Eo%2C+S), [Jaehyung Seo](https://arxiv.org/search/cs?searchtype=author&query=Seo%2C+J), [Heuiseok Lim](https://arxiv.org/search/cs?searchtype=author&query=Lim%2C+H)

> This paper proposes a tool for efficiently constructing high-quality parallel corpora with minimizing human labor and making this tool publicly available. Our proposed construction process is based on neural machine translation (NMT) to allow for it to not only coexist with human translation, but also improve its efficiency by combining data quality control with human translation in a data-centric approach.

| Comments: | Accepted for Data-centric AI workshop at NeurIPS 2021        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2111.00191](https://arxiv.org/abs/2111.00191) [cs.CL]** |
|           | (or **[arXiv:2111.00191v1](https://arxiv.org/abs/2111.00191v1) [cs.CL]** for this version) |





<h2 id="2021-11-02-4">4. Visualization: the missing factor in Simultaneous Speech Translation
</h2>

Title: [Visualization: the missing factor in Simultaneous Speech Translation](https://arxiv.org/abs/2111.00514)

Authors: [Sara Papi](https://arxiv.org/search/cs?searchtype=author&query=Papi%2C+S), [Matteo Negri](https://arxiv.org/search/cs?searchtype=author&query=Negri%2C+M), [Marco Turchi](https://arxiv.org/search/cs?searchtype=author&query=Turchi%2C+M)

> Simultaneous speech translation (SimulST) is the task in which output generation has to be performed on partial, incremental speech input. In recent years, SimulST has become popular due to the spread of cross-lingual application scenarios, like international live conferences and streaming lectures, in which on-the-fly speech translation can facilitate users' access to audio-visual content. In this paper, we analyze the characteristics of the SimulST systems developed so far, discussing their strengths and weaknesses. We then concentrate on the evaluation framework required to properly assess systems' effectiveness. To this end, we raise the need for a broader performance analysis, also including the user experience standpoint. SimulST systems, indeed, should be evaluated not only in terms of quality/latency measures, but also via task-oriented metrics accounting, for instance, for the visualization strategy adopted. In light of this, we highlight which are the goals achieved by the community and what is still missing.

| Comments: | Accepted at CLIC-it 2021                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2111.00514](https://arxiv.org/abs/2111.00514) [cs.CL]** |
|           | (or **[arXiv:2111.00514v1](https://arxiv.org/abs/2111.00514v1) [cs.CL]** for this version) |





<h2 id="2021-11-02-5">5. Quality Estimation Using Round-trip Translation with Sentence Embeddings
</h2>

Title: [Quality Estimation Using Round-trip Translation with Sentence Embeddings](https://arxiv.org/abs/2111.00554)

Authors: [Nathan Crone](https://arxiv.org/search/cs?searchtype=author&query=Crone%2C+N), [Adam Power](https://arxiv.org/search/cs?searchtype=author&query=Power%2C+A), [John Weldon](https://arxiv.org/search/cs?searchtype=author&query=Weldon%2C+J)

> Estimating the quality of machine translation systems has been an ongoing challenge for researchers in this field. Many previous attempts at using round-trip translation as a measure of quality have failed, and there is much disagreement as to whether it can be a viable method of quality estimation. In this paper, we revisit round-trip translation, proposing a system which aims to solve the previous pitfalls found with the approach. Our method makes use of recent advances in language representation learning to more accurately gauge the similarity between the original and round-trip translated sentences. Experiments show that while our approach does not reach the performance of current state of the art methods, it may still be an effective approach for some language pairs.

| Comments: | 10 pages, 5 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2111.00554](https://arxiv.org/abs/2111.00554) [cs.CL]** |
|           | (or **[arXiv:2111.00554v1](https://arxiv.org/abs/2111.00554v1) [cs.CL]** for this version) |





<h2 id="2021-11-02-6">6. Unsupervised Domain Adaptation with Adapter
</h2>

Title: [Unsupervised Domain Adaptation with Adapter](https://arxiv.org/abs/2111.00667)

Authors: [Rongsheng Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+R), [Yinhe Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+Y), [Xiaoxi Mao](https://arxiv.org/search/cs?searchtype=author&query=Mao%2C+X), [Minlie Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+M)

> Unsupervised domain adaptation (UDA) with pre-trained language models (PrLM) has achieved promising results since these pre-trained models embed generic knowledge learned from various domains. However, fine-tuning all the parameters of the PrLM on a small domain-specific corpus distort the learned generic knowledge, and it is also expensive to deployment a whole fine-tuned PrLM for each domain. This paper explores an adapter-based fine-tuning approach for unsupervised domain adaptation. Specifically, several trainable adapter modules are inserted in a PrLM, and the embedded generic knowledge is preserved by fixing the parameters of the original PrLM at fine-tuning. A domain-fusion scheme is introduced to train these adapters using a mix-domain corpus to better capture transferable features. Elaborated experiments on two benchmark datasets are carried out, and the results demonstrate that our approach is effective with different tasks, dataset sizes, and domain similarities.

| Comments: | Accepted by NeurIPS2021 workshop                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2111.00667](https://arxiv.org/abs/2111.00667) [cs.CL]** |
|           | (or **[arXiv:2111.00667v1](https://arxiv.org/abs/2111.00667v1) [cs.CL]** for this version) |





<h2 id="2021-11-02-7">7. Interpretable contrastive word mover's embedding
</h2>

Title: [Interpretable contrastive word mover's embedding](https://arxiv.org/abs/2111.01023)

Authors: [Ruijie Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+R), [Julia Gouvea](https://arxiv.org/search/cs?searchtype=author&query=Gouvea%2C+J), [Eric Miller](https://arxiv.org/search/cs?searchtype=author&query=Miller%2C+E), [David Hammer](https://arxiv.org/search/cs?searchtype=author&query=Hammer%2C+D), [Shuchin Aeron](https://arxiv.org/search/cs?searchtype=author&query=Aeron%2C+S)

> This paper shows that a popular approach to the supervised embedding of documents for classification, namely, contrastive Word Mover's Embedding, can be significantly enhanced by adding interpretability. This interpretability is achieved by incorporating a clustering promoting mechanism into the contrastive loss. On several public datasets, we show that our method improves significantly upon existing baselines while providing interpretation to the clusters via identifying a set of keywords that are the most representative of a particular class. Our approach was motivated in part by the need to develop Natural Language Processing (NLP) methods for the \textit{novel problem of assessing student work for scientific writing and thinking} - a problem that is central to the area of (educational) Learning Sciences (LS). In this context, we show that our approach leads to a meaningful assessment of the student work related to lab reports from a biology class and can help LS researchers gain insights into student understanding and assess evidence of scientific thought processes.

| Comments: | 8 pages, 4 figures                                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2111.01023](https://arxiv.org/abs/2111.01023) [cs.CL]** |
|           | (or **[arXiv:2111.01023v1](https://arxiv.org/abs/2111.01023v1) [cs.CL]** for this version) |





# 2021-11-01

[Return to Index](#Index)



<h2 id="2021-11-01-1">1. Decision Attentive Regularization to Improve Simultaneous Speech Translation Systems
</h2>

Title: [Decision Attentive Regularization to Improve Simultaneous Speech Translation Systems](https://arxiv.org/abs/2110.15729)

Authors: [Mohd Abbas Zaidi](https://arxiv.org/search/cs?searchtype=author&query=Zaidi%2C+M+A), [Beomseok Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+B), [Nikhil Kumar Lakumarapu](https://arxiv.org/search/cs?searchtype=author&query=Lakumarapu%2C+N+K), [Sangha Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+S), [Chanwoo Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+C)

> Simultaneous Speech-to-text Translation (SimulST) systems translate source speech in tandem with the speaker using partial input. Recent works have tried to leverage the text translation task to improve the performance of Speech Translation (ST) in the offline domain. Motivated by these improvements, we propose to add Decision Attentive Regularization (DAR) to Monotonic Multihead Attention (MMA) based SimulST systems. DAR improves the read/write decisions for speech using the Simultaneous text Translation (SimulMT) task. We also extend several techniques from the offline domain to the SimulST task. Our proposed system achieves significant performance improvements for the MuST-C English-German (EnDe) SimulST task, where we provide an average BLUE score improvement of around 4.57 points or 34.17% across different latencies. Further, the latency-quality tradeoffs establish that the proposed model achieves better results compared to the baseline.

| Comments: | 5 pages, 3 figures, 1 table                                  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Sound (cs.SD)**; Computation and Language (cs.CL); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2110.15729](https://arxiv.org/abs/2110.15729) [cs.SD]** |
|           | (or **[arXiv:2110.15729v1](https://arxiv.org/abs/2110.15729v1) [cs.SD]** for this version) |





<h2 id="2021-11-01-2">2. Analysing the Effect of Masking Length Distribution of MLM: An Evaluation Framework and Case Study on Chinese MRC Datasets
</h2>

Title: [Analysing the Effect of Masking Length Distribution of MLM: An Evaluation Framework and Case Study on Chinese MRC Datasets](https://arxiv.org/abs/2110.15712)

Authors: [Changchang. Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+C), [Shaobo. Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+S)

> Machine reading comprehension (MRC) is a challenging natural language processing (NLP) task. Recently, the emergence of pre-trained models (PTM) has brought this research field into a new era, in which the training objective plays a key role. The masked language model (MLM) is a self-supervised training objective that widely used in various PTMs. With the development of training objectives, many variants of MLM have been proposed, such as whole word masking, entity masking, phrase masking, span masking, and so on. In different MLM, the length of the masked tokens is different. Similarly, in different machine reading comprehension tasks, the length of the answer is also different, and the answer is often a word, phrase, or sentence. Thus, in MRC tasks with different answer lengths, whether the length of MLM is related to performance is a question worth studying. If this hypothesis is true, it can guide us how to pre-train the MLM model with a relatively suitable mask length distribution for MRC task. In this paper, we try to uncover how much of MLM's success in the machine reading comprehension tasks comes from the correlation between masking length distribution and answer length in MRC dataset. In order to address this issue, herein, (1) we propose four MRC tasks with different answer length distributions, namely short span extraction task, long span extraction task, short multiple-choice cloze task, long multiple-choice cloze task; (2) four Chinese MRC datasets are created for these tasks; (3) we also have pre-trained four masked language models according to the answer length distributions of these datasets; (4) ablation experiments are conducted on the datasets to verify our hypothesis. The experimental results demonstrate that our hypothesis is true.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2110.15712](https://arxiv.org/abs/2110.15712) [cs.CL]** |
|           | (or **[arXiv:2110.15712v1](https://arxiv.org/abs/2110.15712v1) [cs.CL]** for this version) |





<h2 id="2021-11-01-3">3. Building the Language Resource for a Cebuano-Filipino Neural Machine Translation System
</h2>

Title: [Building the Language Resource for a Cebuano-Filipino Neural Machine Translation System](https://arxiv.org/abs/2110.15716)

Authors: [Kristine Mae Adlaon](https://arxiv.org/search/cs?searchtype=author&query=Adlaon%2C+K+M), [Nelson Marcos](https://arxiv.org/search/cs?searchtype=author&query=Marcos%2C+N)

> Parallel corpus is a critical resource in machine learning-based translation. The task of collecting, extracting, and aligning texts in order to build an acceptable corpus for doing the translation is very tedious most especially for low-resource languages. In this paper, we present the efforts made to build a parallel corpus for Cebuano and Filipino from two different domains: biblical texts and the web. For the biblical resource, subword unit translation for verbs and copy-able approach for nouns were applied to correct inconsistencies in the translation. This correction mechanism was applied as a preprocessing technique. On the other hand, for Wikipedia being the main web resource, commonly occurring topic segments were extracted from both the source and the target languages. These observed topic segments are unique in 4 different categories. The identification of these topic segments may be used for the automatic extraction of sentences. A Recurrent Neural Network was used to implement the translation using OpenNMT sequence modeling tool in TensorFlow. The two different corpora were then evaluated by using them as two separate inputs in the neural network. Results have shown a difference in BLEU scores in both corpora.

| Comments:    | Published in the Proceedings of the 2019 3rd International Conference on Natural Language Processing and Information Retrieval. arXiv admin note: substantial text overlap with [arXiv:1902.07250](https://arxiv.org/abs/1902.07250) |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**                         |
| ACM classes: | A.2                                                          |
| DOI:         | [10.1145/3342827.3342833](https://arxiv.org/ct?url=https%3A%2F%2Fdx.doi.org%2F10.1145%2F3342827.3342833&v=b6a91262) |
| Cite as:     | **[arXiv:2110.15716](https://arxiv.org/abs/2110.15716) [cs.CL]** |
|              | (or **[arXiv:2110.15716v1](https://arxiv.org/abs/2110.15716v1) [cs.CL]** for this version) |





<h2 id="2021-11-01-4">4. Batch-Softmax Contrastive Loss for Pairwise Sentence Scoring Tasks
</h2>

Title: [Batch-Softmax Contrastive Loss for Pairwise Sentence Scoring Tasks](https://arxiv.org/abs/2110.15725)

Authors: [Anton Chernyavskiy](https://arxiv.org/search/cs?searchtype=author&query=Chernyavskiy%2C+A), [Dmitry Ilvovsky](https://arxiv.org/search/cs?searchtype=author&query=Ilvovsky%2C+D), [Pavel Kalinin](https://arxiv.org/search/cs?searchtype=author&query=Kalinin%2C+P), [Preslav Nakov](https://arxiv.org/search/cs?searchtype=author&query=Nakov%2C+P)

> The use of contrastive loss for representation learning has become prominent in computer vision, and it is now getting attention in Natural Language Processing (NLP). Here, we explore the idea of using a batch-softmax contrastive loss when fine-tuning large-scale pre-trained transformer models to learn better task-specific sentence embeddings for pairwise sentence scoring tasks. We introduce and study a number of variations in the calculation of the loss as well as in the overall training procedure; in particular, we find that data shuffling can be quite important. Our experimental results show sizable improvements on a number of datasets and pairwise sentence scoring tasks including classification, ranking, and regression. Finally, we offer detailed analysis and discussion, which should be useful for researchers aiming to explore the utility of contrastive loss in NLP.

| Comments:    | batch-softmax contrastive loss, pairwise sentence scoring, classification, ranking, and regression |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Information Retrieval (cs.IR); Machine Learning (cs.LG); Neural and Evolutionary Computing (cs.NE) |
| MSC classes: | 68T50                                                        |
| ACM classes: | F.2.2; I.2.7                                                 |
| Cite as:     | **[arXiv:2110.15725](https://arxiv.org/abs/2110.15725) [cs.CL]** |
|              | (or **[arXiv:2110.15725v1](https://arxiv.org/abs/2110.15725v1) [cs.CL]** for this version) |





<h2 id="2021-11-01-5">5. BERMo: What can BERT learn from ELMo?
</h2>

Title: [BERMo: What can BERT learn from ELMo?](https://arxiv.org/abs/2110.15802)

Authors: [Sangamesh Kodge](https://arxiv.org/search/cs?searchtype=author&query=Kodge%2C+S), [Kaushik Roy](https://arxiv.org/search/cs?searchtype=author&query=Roy%2C+K)

> We propose BERMo, an architectural modification to BERT, which makes predictions based on a hierarchy of surface, syntactic and semantic language features. We use linear combination scheme proposed in Embeddings from Language Models (ELMo) to combine the scaled internal representations from different network depths. Our approach has two-fold benefits: (1) improved gradient flow for the downstream task as every layer has a direct connection to the gradients of the loss function and (2) increased representative power as the model no longer needs to copy the features learned in the shallower layer which are necessary for the downstream task. Further, our model has a negligible parameter overhead as there is a single scalar parameter associated with each layer in the network. Experiments on the probing task from SentEval dataset show that our model performs up to 4.65% better in accuracy than the baseline with an average improvement of 2.67% on the semantic tasks. When subject to compression techniques, we find that our model enables stable pruning for compressing small datasets like SST-2, where the BERT model commonly diverges. We observe that our approach converges 1.67× and 1.15× faster than the baseline on MNLI and QQP tasks from GLUE dataset. Moreover, our results show that our approach can obtain better parameter efficiency for penalty based pruning approaches on QQP task.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2110.15802](https://arxiv.org/abs/2110.15802) [cs.CL]** |
|           | (or **[arXiv:2110.15802v1](https://arxiv.org/abs/2110.15802v1) [cs.CL]** for this version) |





<h2 id="2021-11-01-6">6. MetaICL: Learning to Learn In Context
</h2>

Title: [MetaICL: Learning to Learn In Context](https://arxiv.org/abs/2110.15943)

Authors: [Sewon Min](https://arxiv.org/search/cs?searchtype=author&query=Min%2C+S), [Mike Lewis](https://arxiv.org/search/cs?searchtype=author&query=Lewis%2C+M), [Luke Zettlemoyer](https://arxiv.org/search/cs?searchtype=author&query=Zettlemoyer%2C+L), [Hannaneh Hajishirzi](https://arxiv.org/search/cs?searchtype=author&query=Hajishirzi%2C+H)

> We introduce MetaICL (Meta-training for In-Context Learning), a new meta-training framework for few-shot learning where a pretrained language model is tuned to do in-context learn-ing on a large set of training tasks. This meta-training enables the model to more effectively learn a new task in context at test time, by simply conditioning on a few training examples with no parameter updates or task-specific templates. We experiment on a large, diverse collection of tasks consisting of 142 NLP datasets including classification, question answering, natural language inference, paraphrase detection and more, across seven different meta-training/target splits. MetaICL outperforms a range of baselines including in-context learning without meta-training and multi-task learning followed by zero-shot transfer. We find that the gains are particularly significant for target tasks that have domain shifts from the meta-training tasks, and that using a diverse set of the meta-training tasks is key to improvements. We also show that MetaICL approaches (and sometimes beats) the performance of models fully finetuned on the target task training data, and outperforms much bigger models with nearly 8x parameters.

| Comments: | 18 pages (9 pages for the main paper, 9 pages for references and appendices). 1 figure. Code available at [this https URL](https://github.com/facebookresearch/MetaICL) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2110.15943](https://arxiv.org/abs/2110.15943) [cs.CL]** |
|           | (or **[arXiv:2110.15943v1](https://arxiv.org/abs/2110.15943v1) [cs.CL]** for this version) |



