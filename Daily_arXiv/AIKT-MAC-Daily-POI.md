# MA C.'s Daily Paper Of Interest - April b., 2022

# Index

- [2022-04-27](#2022-04-27)
  - [1. Pretraining Chinese BERT for Detecting Word Insertion and Deletion Errors](#2022-04-27-1)
  
  - [2. When do Contrastive Word Alignments Improve Many-to-many Neural Machine Translation?](#2022-04-27-2)
  
  - [3. Flow-Adapter Architecture for Unsupervised Machine Translation](#2022-04-27-3)
  
- [2022-04-26](#2022-04-26)
  - [1. MCSE: Multimodal Contrastive Learning of Sentence Embeddings](#2022-04-26-1)

  - [2. MuCGEC: a Multi-Reference Multi-Source Evaluation Dataset for Chinese Grammatical Error Correction](#2022-04-26-2)

  - [3. Translation between Molecules and Natural Language](#2022-04-26-3)

- [2022-04-25](#2022-04-25)
  - [1. Multimodal Adaptive Distillation for Leveraging Unimodal Encoders for Vision-Language Tasks](#2022-04-25-1)

  - [2. KALA: Knowledge-Augmented Language Model Adaptation](#2022-04-25-2)

  - [3. LibriS2S: A German-English Speech-to-Speech Translation Corpus](#2022-04-25-3)

- [2022-04-22](#2022-04-22)
  - [1. A Masked Image Reconstruction Network for Document-level Relation Extraction](#2022-04-22-1)

  - [2. Standing on the Shoulders of Giant Frozen Language Models](#2022-04-22-2)

  - [3. Probing Script Knowledge from Pre-Trained Models](#2022-04-22-3)

  - [4. DiffCSE: Difference-based Contrastive Learning for Sentence Embeddings](#2022-04-22-4)

- [2022-04-21](#2022-04-21)
  - [1. DaLC: Domain Adaptation Learning Curve Prediction for Neural Machine Translation](#2022-04-21-1)

  - [2. A Survey on Non-Autoregressive Generation for Neural Machine Translation and Beyond](#2022-04-21-2)

  - [3. A Survey on Bias and Fairness in Natural Language Processing](#2022-04-21-3)

  - [4. Exploring Continuous Integrate-and-Fire for Efficient and Adaptive Simultaneous Speech Translation](#2022-04-21-4)

  - [5. Detecting Unintended Memorization in Language-Model-Fused ASR](#2022-04-21-5)

- [2022-04-20](#2022-04-20)
  - [1. Imagination-Augmented Natural Language Understanding](#2022-04-20-1)

  - [2. Blockwise Streaming Transformer for Spoken Language Understanding and Simultaneous Speech Translation](#2022-04-20-2)

  - [3. Feature Structure Distillation for BERT Transferring](#2022-04-20-3)

  - [4. On the Locality of Attention in Direct Speech Translation](#2022-04-20-4)

- [2022-04-19](#2022-04-19)
  - [1. mGPT: Few-Shot Learners Go Multilingual](#2022-04-19-1)

  - [2. MoEBERT: from BERT to Mixture-of-Experts via Importance-Guided Adaptation](#2022-04-19-2)

  - [3. Bridging Cross-Lingual Gaps During Leveraging the Multilingual Sequence-to-Sequence Pretraining for Text Generation](#2022-04-19-3)

  - [4. BLISS: Robust Sequence-to-Sequence Learning via Self-Supervised Input Representation](#2022-04-19-4)

  - [5. On Effectively Learning of Knowledge in Continual Pre-training](#2022-04-19-5)

  - [6. Dynamic Position Encoding for Transformers](#2022-04-19-6)

  - [7. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking](#2022-04-19-7)

  - [8. Exploring Dimensionality Reduction Techniques in Multilingual Transformers](#2022-04-19-8)

- [2022-04-18](#2022-04-18)
  - [1. Vision-and-Language Pretrained Models: A Survey](2022-04-18-1)

  - [2. COTS: Collaborative Two-Stream Vision-Language Pre-Training Model for Cross-Modal Retrieval](2022-04-18-2)

  - [3. XDBERT: Distilling Visual Information to BERT from Cross-Modal Systems to Improve Language Understanding](2022-04-18-3)

  - [4. LaMemo: Language Modeling with Look-Ahead Memory](2022-04-18-4)

  - [5. Text Revision by On-the-Fly Representation Optimization](2022-04-18-5)

  - [6. On the Role of Pre-trained Language Models in Word Ordering: A Case Study with BART](2022-04-18-6)

  - [7. Chinese Idiom Paraphrasing](2022-04-18-7)

- [2022-04-15](#2022-04-15)
  - [1. METRO: Efficient Denoising Pretraining of Large Scale Autoencoding Language Models with Model Generated Signals](#2022-04-15-1)

- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



# 2022-04-27

[Return to Index](#Index)



<h2 id="2022-04-27-1">1. Pretraining Chinese BERT for Detecting Word Insertion and Deletion Errors
</h2>

Title: [Pretraining Chinese BERT for Detecting Word Insertion and Deletion Errors](https://arxiv.org/abs/2204.12052)

Authors: [Cong Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+C), [Yong Dai](https://arxiv.org/search/cs?searchtype=author&query=Dai%2C+Y), [Duyu Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+D), [Enbo Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+E), [Zhangyin Feng](https://arxiv.org/search/cs?searchtype=author&query=Feng%2C+Z), [Li Kuang](https://arxiv.org/search/cs?searchtype=author&query=Kuang%2C+L), [Shuming Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+S)

> Chinese BERT models achieve remarkable progress in dealing with grammatical errors of word substitution. However, they fail to handle word insertion and deletion because BERT assumes the existence of a word at each position. To address this, we present a simple and effective Chinese pretrained model. The basic idea is to enable the model to determine whether a word exists at a particular position. We achieve this by introducing a special token \texttt{[null]}, the prediction of which stands for the non-existence of a word. In the training stage, we design pretraining tasks such that the model learns to predict \texttt{[null]} and real words jointly given the surrounding context. In the inference stage, the model readily detects whether a word should be inserted or deleted with the standard masked language modeling function. We further create an evaluation dataset to foster research on word insertion and deletion. It includes human-annotated corrections for 7,726 erroneous sentences. Results show that existing Chinese BERT performs poorly on detecting insertion and deletion errors. Our approach significantly improves the F1 scores from 24.1\% to 78.1\% for word insertion and from 26.5\% to 68.5\% for word deletion, respectively.

| Comments: | 12 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2204.12052](https://arxiv.org/abs/2204.12052) [cs.CL]** |
|           | (or **[arXiv:2204.12052v1](https://arxiv.org/abs/2204.12052v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.12052Focus to learn more |





<h2 id="2022-04-27-2">2. When do Contrastive Word Alignments Improve Many-to-many Neural Machine Translation?
</h2>

Title: [When do Contrastive Word Alignments Improve Many-to-many Neural Machine Translation?](https://arxiv.org/abs/2204.12165)

Authors: [Zhuoyuan Mao](https://arxiv.org/search/cs?searchtype=author&query=Mao%2C+Z), [Chenhui Chu](https://arxiv.org/search/cs?searchtype=author&query=Chu%2C+C), [Raj Dabre](https://arxiv.org/search/cs?searchtype=author&query=Dabre%2C+R), [Haiyue Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+H), [Zhen Wan](https://arxiv.org/search/cs?searchtype=author&query=Wan%2C+Z), [Sadao Kurohashi](https://arxiv.org/search/cs?searchtype=author&query=Kurohashi%2C+S)

> Word alignment has proven to benefit many-to-many neural machine translation (NMT). However, high-quality ground-truth bilingual dictionaries were used for pre-editing in previous methods, which are unavailable for most language pairs. Meanwhile, the contrastive objective can implicitly utilize automatically learned word alignment, which has not been explored in many-to-many NMT. This work proposes a word-level contrastive objective to leverage word alignments for many-to-many NMT. Empirical results show that this leads to 0.8 BLEU gains for several language pairs. Analyses reveal that in many-to-many NMT, the encoder's sentence retrieval performance highly correlates with the translation quality, which explains when the proposed method impacts translation. This motivates future exploration for many-to-many NMT to improve the encoder's sentence retrieval performance.

| Comments: | NAACL 2022 findings                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2204.12165](https://arxiv.org/abs/2204.12165) [cs.CL]** |
|           | (or **[arXiv:2204.12165v1](https://arxiv.org/abs/2204.12165v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.12165Focus to learn more |





<h2 id="2022-04-27-3">3. Flow-Adapter Architecture for Unsupervised Machine Translation
</h2>

Title: [Flow-Adapter Architecture for Unsupervised Machine Translation](https://arxiv.org/abs/2204.12225)

Authors: [Yihong Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y), [Haris Jabbar](https://arxiv.org/search/cs?searchtype=author&query=Jabbar%2C+H), [Hinrich Schütze](https://arxiv.org/search/cs?searchtype=author&query=Schütze%2C+H)

> In this work, we propose a flow-adapter architecture for unsupervised NMT. It leverages normalizing flows to explicitly model the distributions of sentence-level latent representations, which are subsequently used in conjunction with the attention mechanism for the translation task. The primary novelties of our model are: (a) capturing language-specific sentence representations separately for each language using normalizing flows and (b) using a simple transformation of these latent representations for translating from one language to another. This architecture allows for unsupervised training of each language independently. While there is prior work on latent variables for supervised MT, to the best of our knowledge, this is the first work that uses latent variables and normalizing flows for unsupervised MT. We obtain competitive results on several unsupervised MT benchmarks.

| Comments: | ACL 2022                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2204.12225](https://arxiv.org/abs/2204.12225) [cs.CL]** |
|           | (or **[arXiv:2204.12225v1](https://arxiv.org/abs/2204.12225v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.12225Focus to learn more |







# 2022-04-26

[Return to Index](#Index)



<h2 id="2022-04-26-1">1. MCSE: Multimodal Contrastive Learning of Sentence Embeddings
</h2>

Title: [MCSE: Multimodal Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2204.10931)

Authors: [Miaoran Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+M), [Marius Mosbach](https://arxiv.org/search/cs?searchtype=author&query=Mosbach%2C+M), [David Ifeoluwa Adelani](https://arxiv.org/search/cs?searchtype=author&query=Adelani%2C+D+I), [Michael A. Hedderich](https://arxiv.org/search/cs?searchtype=author&query=Hedderich%2C+M+A), [Dietrich Klakow](https://arxiv.org/search/cs?searchtype=author&query=Klakow%2C+D)

> Learning semantically meaningful sentence embeddings is an open problem in natural language processing. In this work, we propose a sentence embedding learning approach that exploits both visual and textual information via a multimodal contrastive objective. Through experiments on a variety of semantic textual similarity tasks, we demonstrate that our approach consistently improves the performance across various datasets and pre-trained encoders. In particular, combining a small amount of multimodal data with a large text-only corpus, we improve the state-of-the-art average Spearman's correlation by 1.7%. By analyzing the properties of the textual embedding space, we show that our model excels in aligning semantically similar sentences, providing an explanation for its improved performance.

| Comments: | Accepted by NAACL 2022 main conference (short paper), 11 pages |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2204.10931](https://arxiv.org/abs/2204.10931) [cs.CL]** |
|           | (or **[arXiv:2204.10931v1](https://arxiv.org/abs/2204.10931v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.10931Focus to learn more |





<h2 id="2022-04-26-2">2. MuCGEC: a Multi-Reference Multi-Source Evaluation Dataset for Chinese Grammatical Error Correction
</h2>

Title: [MuCGEC: a Multi-Reference Multi-Source Evaluation Dataset for Chinese Grammatical Error Correction](https://arxiv.org/abs/2204.10994)

Authors: [Yue Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y), [Zhenghua Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Z), [Zuyi Bao](https://arxiv.org/search/cs?searchtype=author&query=Bao%2C+Z), [Jiacheng Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J), [Bo Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+B), [Chen Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+C), [Fei Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+F), [Min Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+M)

> This paper presents MuCGEC, a multi-reference multi-source evaluation dataset for Chinese Grammatical Error Correction (CGEC), consisting of 7,063 sentences collected from three different Chinese-as-a-Second-Language (CSL) learner sources. Each sentence has been corrected by three annotators, and their corrections are meticulously reviewed by an expert, resulting in 2.3 references per sentence. We conduct experiments with two mainstream CGEC models, i.e., the sequence-to-sequence (Seq2Seq) model and the sequence-to-edit (Seq2Edit) model, both enhanced with large pretrained language models (PLMs), achieving competitive benchmark performance on previous and our datasets. We also discuss CGEC evaluation methodologies, including the effect of multiple references and using a char-based metric. Our annotation guidelines, data, and code are available at \url{[this https URL](https://github.com/HillZhang1999/MuCGEC)}.

| Comments: | Accepted by NAACL2022 (main conference)                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2204.10994](https://arxiv.org/abs/2204.10994) [cs.CL]** |
|           | (or **[arXiv:2204.10994v1](https://arxiv.org/abs/2204.10994v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.10994Focus to learn more |





<h2 id="2022-04-26-3">3. Translation between Molecules and Natural Language
</h2>

Title: [Translation between Molecules and Natural Language](https://arxiv.org/abs/2204.11817)

Authors: [Carl Edwards](https://arxiv.org/search/cs?searchtype=author&query=Edwards%2C+C), [Tuan Lai](https://arxiv.org/search/cs?searchtype=author&query=Lai%2C+T), [Kevin Ros](https://arxiv.org/search/cs?searchtype=author&query=Ros%2C+K), [Garrett Honke](https://arxiv.org/search/cs?searchtype=author&query=Honke%2C+G), [Heng Ji](https://arxiv.org/search/cs?searchtype=author&query=Ji%2C+H)

> Joint representations between images and text have been deeply investigated in the literature. In computer vision, the benefits of incorporating natural language have become clear for enabling semantic-level control of images. In this work, we present **MolT5**−a self-supervised learning framework for pretraining models on a vast amount of unlabeled natural language text and molecule strings. **MolT5** allows for new, useful, and challenging analogs of traditional vision-language tasks, such as molecule captioning and text-based de novo molecule generation (altogether: translation between molecules and language), which we explore for the first time. Furthermore, since **MolT5** pretrains models on single-modal data, it helps overcome the chemistry domain shortcoming of data scarcity. Additionally, we consider several metrics, including a new cross-modal embedding-based metric, to evaluate the tasks of molecule captioning and text-based molecule generation. By interfacing molecules with natural language, we enable a higher semantic level of control over molecule discovery and understanding--a critical task for scientific domains such as drug discovery and material design. Our results show that **MolT5**-based models are able to generate outputs, both molecule and text, which in many cases are high quality and match the input modality. On molecule generation, our best model achieves 30% exact matching test accuracy (i.e., it generates the correct structure for about one-third of the captions in our held-out test set).

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2204.11817](https://arxiv.org/abs/2204.11817) [cs.CL]** |
|           | (or **[arXiv:2204.11817v1](https://arxiv.org/abs/2204.11817v1) [cs.CL]** for this version) |





# 2022-04-25

[Return to Index](#Index)



<h2 id="2022-04-25-1">1. Multimodal Adaptive Distillation for Leveraging Unimodal Encoders for Vision-Language Tasks
</h2>

Title: [Multimodal Adaptive Distillation for Leveraging Unimodal Encoders for Vision-Language Tasks](https://arxiv.org/abs/2204.10496)

Authors: [Zhecan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Z), [Noel Codella](https://arxiv.org/search/cs?searchtype=author&query=Codella%2C+N), [Yen-Chun Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Y), [Luowei Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+L), [Xiyang Dai](https://arxiv.org/search/cs?searchtype=author&query=Dai%2C+X), [Bin Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+B), [Jianwei Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+J), [Haoxuan You](https://arxiv.org/search/cs?searchtype=author&query=You%2C+H), [Kai-Wei Chang](https://arxiv.org/search/cs?searchtype=author&query=Chang%2C+K), [Shih-fu Chang](https://arxiv.org/search/cs?searchtype=author&query=Chang%2C+S), [Lu Yuan](https://arxiv.org/search/cs?searchtype=author&query=Yuan%2C+L)

> Cross-modal encoders for vision-language (VL) tasks are often pretrained with carefully curated vision-language datasets. While these datasets reach an order of 10 million samples, the labor cost is prohibitive to scale further. Conversely, unimodal encoders are pretrained with simpler annotations that are less cost-prohibitive, achieving scales of hundreds of millions to billions. As a result, unimodal encoders have achieved state-of-art (SOTA) on many downstream tasks. However, challenges remain when applying to VL tasks. The pretraining data is not optimal for cross-modal architectures and requires heavy computational resources. In addition, unimodal architectures lack cross-modal interactions that have demonstrated significant benefits for VL tasks. Therefore, how to best leverage pretrained unimodal encoders for VL tasks is still an area of active research. In this work, we propose a method to leverage unimodal vision and text encoders for VL tasks that augment existing VL approaches while conserving computational complexity. Specifically, we propose Multimodal Adaptive Distillation (MAD), which adaptively distills useful knowledge from pretrained encoders to cross-modal VL encoders. Second, to better capture nuanced impacts on VL task performance, we introduce an evaluation protocol that includes Visual Commonsense Reasoning (VCR), Visual Entailment (SNLI-VE), and Visual Question Answering (VQA), across a variety of data constraints and conditions of domain shift. Experiments demonstrate that MAD leads to consistent gains in the low-shot, domain-shifted, and fully-supervised conditions on VCR, SNLI-VE, and VQA, achieving SOTA performance on VCR compared to other single models pretrained with image-text data. Finally, MAD outperforms concurrent works utilizing pretrained vision encoder from CLIP. Code will be made available.

| Comments: | arXiv admin note: substantial text overlap with [arXiv:2201.05729](https://arxiv.org/abs/2201.05729) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG); Multimedia (cs.MM) |
| Cite as:  | **[arXiv:2204.10496](https://arxiv.org/abs/2204.10496) [cs.CV]** |
|           | (or **[arXiv:2204.10496v1](https://arxiv.org/abs/2204.10496v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.10496Focus to learn more |





<h2 id="2022-04-25-2">2. KALA: Knowledge-Augmented Language Model Adaptation
</h2>

Title: [KALA: Knowledge-Augmented Language Model Adaptation](https://arxiv.org/abs/2204.10555)

Authors: [Minki Kang](https://arxiv.org/search/cs?searchtype=author&query=Kang%2C+M), [Jinheon Baek](https://arxiv.org/search/cs?searchtype=author&query=Baek%2C+J), [Sung Ju Hwang](https://arxiv.org/search/cs?searchtype=author&query=Hwang%2C+S+J)

> Pre-trained language models (PLMs) have achieved remarkable success on various natural language understanding tasks. Simple fine-tuning of PLMs, on the other hand, might be suboptimal for domain-specific tasks because they cannot possibly cover knowledge from all domains. While adaptive pre-training of PLMs can help them obtain domain-specific knowledge, it requires a large training cost. Moreover, adaptive pre-training can harm the PLM's performance on the downstream task by causing catastrophic forgetting of its general knowledge. To overcome such limitations of adaptive pre-training for PLM adaption, we propose a novel domain adaption framework for PLMs coined as Knowledge-Augmented Language model Adaptation (KALA), which modulates the intermediate hidden representations of PLMs with domain knowledge, consisting of entities and their relational facts. We validate the performance of our KALA on question answering and named entity recognition tasks on multiple datasets across various domains. The results show that, despite being computationally efficient, our KALA largely outperforms adaptive pre-training. Code is available at: [this https URL](https://github.com/Nardien/KALA/).

| Comments: | NAACL 2022                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2204.10555](https://arxiv.org/abs/2204.10555) [cs.CL]** |
|           | (or **[arXiv:2204.10555v1](https://arxiv.org/abs/2204.10555v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.10555Focus to learn more |





<h2 id="2022-04-25-3">3. LibriS2S: A German-English Speech-to-Speech Translation Corpus
</h2>

Title: [LibriS2S: A German-English Speech-to-Speech Translation Corpus](https://arxiv.org/abs/2204.10593)

Authors: [Pedro Jeuris](https://arxiv.org/search/cs?searchtype=author&query=Jeuris%2C+P), [Jan Niehues](https://arxiv.org/search/cs?searchtype=author&query=Niehues%2C+J)

> Recently, we have seen an increasing interest in the area of speech-to-text translation. This has led to astonishing improvements in this area. In contrast, the activities in the area of speech-to-speech translation is still limited, although it is essential to overcome the language barrier. We believe that one of the limiting factors is the availability of appropriate training data. We address this issue by creating LibriS2S, to our knowledge the first publicly available speech-to-speech training corpus between German and English. For this corpus, we used independently created audio for German and English leading to an unbiased pronunciation of the text in both languages. This allows the creation of a new text-to-speech and speech-to-speech translation model that directly learns to generate the speech signal based on the pronunciation of the source language. Using this created corpus, we propose Text-to-Speech models based on the example of the recently proposed FastSpeech 2 model that integrates source language information. We do this by adapting the model to take information such as the pitch, energy or transcript from the source speech as additional input.

| Comments: | Accepted to LREC 2022                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2204.10593](https://arxiv.org/abs/2204.10593) [cs.CL]** |
|           | (or **[arXiv:2204.10593v1](https://arxiv.org/abs/2204.10593v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.10593Focus to learn more |








# 2022-04-22

[Return to Index](#Index)



<h2 id="2022-04-22-1">1. A Masked Image Reconstruction Network for Document-level Relation Extraction
</h2>

Title: [A Masked Image Reconstruction Network for Document-level Relation Extraction](https://arxiv.org/abs/2204.09851)

Authors: [Liang Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+L), [Yidong Cheng](https://arxiv.org/search/cs?searchtype=author&query=Cheng%2C+Y)

> Document-level relation extraction aims to extract relations among entities within a document. Compared with its sentence-level counterpart, Document-level relation extraction requires inference over multiple sentences to extract complex relational triples. Previous research normally complete reasoning through information propagation on the mention-level or entity-level document-graphs, regardless of the correlations between the relationships. In this paper, we propose a novel Document-level Relation Extraction model based on a Masked Image Reconstruction network (DRE-MIR), which models inference as a masked image reconstruction problem to capture the correlations between relationships. Specifically, we first leverage an encoder module to get the features of entities and construct the entity-pair matrix based on the features. After that, we look on the entity-pair matrix as an image and then randomly mask it and restore it through an inference module to capture the correlations between the relationships. We evaluate our model on three public document-level relation extraction datasets, i.e. DocRED, CDR, and GDA. Experimental results demonstrate that our model achieves state-of-the-art performance on these three datasets and has excellent robustness against the noises during the inference process.

| Comments: | arXiv admin note: text overlap with [arXiv:2204.00255](https://arxiv.org/abs/2204.00255) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2204.09851](https://arxiv.org/abs/2204.09851) [cs.CL]** |
|           | (or **[arXiv:2204.09851v1](https://arxiv.org/abs/2204.09851v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.09851Focus to learn more |





<h2 id="2022-04-22-2">2. Standing on the Shoulders of Giant Frozen Language Models
</h2>

Title: [Standing on the Shoulders of Giant Frozen Language Models](https://arxiv.org/abs/2204.10019)

Authors: [Yoav Levine](https://arxiv.org/search/cs?searchtype=author&query=Levine%2C+Y), [Itay Dalmedigos](https://arxiv.org/search/cs?searchtype=author&query=Dalmedigos%2C+I), [Ori Ram](https://arxiv.org/search/cs?searchtype=author&query=Ram%2C+O), [Yoel Zeldes](https://arxiv.org/search/cs?searchtype=author&query=Zeldes%2C+Y), [Daniel Jannai](https://arxiv.org/search/cs?searchtype=author&query=Jannai%2C+D), [Dor Muhlgay](https://arxiv.org/search/cs?searchtype=author&query=Muhlgay%2C+D), [Yoni Osin](https://arxiv.org/search/cs?searchtype=author&query=Osin%2C+Y), [Opher Lieber](https://arxiv.org/search/cs?searchtype=author&query=Lieber%2C+O), [Barak Lenz](https://arxiv.org/search/cs?searchtype=author&query=Lenz%2C+B), [Shai Shalev-Shwartz](https://arxiv.org/search/cs?searchtype=author&query=Shalev-Shwartz%2C+S), [Amnon Shashua](https://arxiv.org/search/cs?searchtype=author&query=Shashua%2C+A), [Kevin Leyton-Brown](https://arxiv.org/search/cs?searchtype=author&query=Leyton-Brown%2C+K), [Yoav Shoham](https://arxiv.org/search/cs?searchtype=author&query=Shoham%2C+Y)

> Huge pretrained language models (LMs) have demonstrated surprisingly good zero-shot capabilities on a wide variety of tasks. This gives rise to the appealing vision of a single, versatile model with a wide range of functionalities across disparate applications. However, current leading techniques for leveraging a "frozen" LM -- i.e., leaving its weights untouched -- still often underperform fine-tuning approaches which modify these weights in a task-dependent way. Those, in turn, suffer forgetfulness and compromise versatility, suggesting a tradeoff between performance and versatility. The main message of this paper is that current frozen-model techniques such as prompt tuning are only the tip of the iceberg, and more powerful methods for leveraging frozen LMs can do just as well as fine tuning in challenging domains without sacrificing the underlying model's versatility. To demonstrate this, we introduce three novel methods for leveraging frozen models: input-dependent prompt tuning, frozen readers, and recursive LMs, each of which vastly improves on current frozen-model approaches. Indeed, some of our methods even outperform fine-tuning approaches in domains currently dominated by the latter. The computational cost of each method is higher than that of existing frozen model methods, but still negligible relative to a single pass through a huge frozen LM. Each of these methods constitutes a meaningful contribution in its own right, but by presenting these contributions together we aim to convince the reader of a broader message that goes beyond the details of any given method: that frozen models have untapped potential and that fine-tuning is often unnecessary.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2204.10019](https://arxiv.org/abs/2204.10019) [cs.CL]** |
|           | (or **[arXiv:2204.10019v1](https://arxiv.org/abs/2204.10019v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.10019Focus to learn more |





<h2 id="2022-04-22-3">3. Probing Script Knowledge from Pre-Trained Models
</h2>

Title: [Probing Script Knowledge from Pre-Trained Models](https://arxiv.org/abs/2204.10176)

Authors: [Zijian Jin](https://arxiv.org/search/cs?searchtype=author&query=Jin%2C+Z), [Xingyu Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+X), [Mo Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+M), [Lifu Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+L)

> Script knowledge is critical for humans to understand the broad daily tasks and routine activities in the world. Recently researchers have explored the large-scale pre-trained language models (PLMs) to perform various script related tasks, such as story generation, temporal ordering of event, future event prediction and so on. However, it's still not well studied in terms of how well the PLMs capture the script knowledge. To answer this question, we design three probing tasks: inclusive sub-event selection, starting sub-event selection and temporal ordering to investigate the capabilities of PLMs with and without fine-tuning. The three probing tasks can be further used to automatically induce a script for each main event given all the possible sub-events. Taking BERT as a case study, by analyzing its performance on script induction as well as each individual probing task, we conclude that the stereotypical temporal knowledge among the sub-events is well captured in BERT, however the inclusive or starting sub-event knowledge is barely encoded.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2204.10176](https://arxiv.org/abs/2204.10176) [cs.CL]** |
|           | (or **[arXiv:2204.10176v1](https://arxiv.org/abs/2204.10176v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.10176Focus to learn more |





<h2 id="2022-04-22-4">4. DiffCSE: Difference-based Contrastive Learning for Sentence Embeddings
</h2>

Title: [DiffCSE: Difference-based Contrastive Learning for Sentence Embeddings](https://arxiv.org/abs/2204.10298)

Authors: [Yung-Sung Chuang](https://arxiv.org/search/cs?searchtype=author&query=Chuang%2C+Y), [Rumen Dangovski](https://arxiv.org/search/cs?searchtype=author&query=Dangovski%2C+R), [Hongyin Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+H), [Yang Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y), [Shiyu Chang](https://arxiv.org/search/cs?searchtype=author&query=Chang%2C+S), [Marin Soljačić](https://arxiv.org/search/cs?searchtype=author&query=Soljačić%2C+M), [Shang-Wen Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+S), [Wen-tau Yih](https://arxiv.org/search/cs?searchtype=author&query=Yih%2C+W), [Yoon Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+Y), [James Glass](https://arxiv.org/search/cs?searchtype=author&query=Glass%2C+J)

> We propose DiffCSE, an unsupervised contrastive learning framework for learning sentence embeddings. DiffCSE learns sentence embeddings that are sensitive to the difference between the original sentence and an edited sentence, where the edited sentence is obtained by stochastically masking out the original sentence and then sampling from a masked language model. We show that DiffSCE is an instance of equivariant contrastive learning (Dangovski et al., 2021), which generalizes contrastive learning and learns representations that are insensitive to certain types of augmentations and sensitive to other "harmful" types of augmentations. Our experiments show that DiffCSE achieves state-of-the-art results among unsupervised sentence representation learning methods, outperforming unsupervised SimCSE by 2.3 absolute points on semantic textual similarity tasks.

| Comments: | NAACL 2022 main conference (Long paper). Pretrained models and code are available at [this https URL](https://github.com/voidism/DiffCSE) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2204.10298](https://arxiv.org/abs/2204.10298) [cs.CL]** |
|           | (or **[arXiv:2204.10298v1](https://arxiv.org/abs/2204.10298v1) [cs.CL]** for this version) |





# 2022-04-21

[Return to Index](#Index)



<h2 id="2022-04-21-1">1. DaLC: Domain Adaptation Learning Curve Prediction for Neural Machine Translation
</h2>

Title: [DaLC: Domain Adaptation Learning Curve Prediction for Neural Machine Translation](https://arxiv.org/abs/2204.09259)

Authors: [Cheonbok Park](https://arxiv.org/search/cs?searchtype=author&query=Park%2C+C), [Hantae Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+H), [Ioan Calapodescu](https://arxiv.org/search/cs?searchtype=author&query=Calapodescu%2C+I), [Hyunchang Cho](https://arxiv.org/search/cs?searchtype=author&query=Cho%2C+H), [Vassilina Nikoulina](https://arxiv.org/search/cs?searchtype=author&query=Nikoulina%2C+V)

> Domain Adaptation (DA) of Neural Machine Translation (NMT) model often relies on a pre-trained general NMT model which is adapted to the new domain on a sample of in-domain parallel data. Without parallel data, there is no way to estimate the potential benefit of DA, nor the amount of parallel samples it would require. It is however a desirable functionality that could help MT practitioners to make an informed decision before investing resources in dataset creation. We propose a Domain adaptation Learning Curve prediction (DaLC) model that predicts prospective DA performance based on in-domain monolingual samples in the source language. Our model relies on the NMT encoder representations combined with various instance and corpus-level features. We demonstrate that instance-level is better able to distinguish between different domains compared to corpus-level frameworks proposed in previous studies. Finally, we perform in-depth analyses of the results highlighting the limitations of our approach, and provide directions for future research.

| Comments: | to be published in ACL2021                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2204.09259](https://arxiv.org/abs/2204.09259) [cs.CL]** |
|           | (or **[arXiv:2204.09259v1](https://arxiv.org/abs/2204.09259v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.09259Focus to learn more |





<h2 id="2022-04-21-2">2. A Survey on Non-Autoregressive Generation for Neural Machine Translation and Beyond
</h2>

Title: [A Survey on Non-Autoregressive Generation for Neural Machine Translation and Beyond](https://arxiv.org/abs/2204.09269)

Authors: [Yisheng Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+Y), [Lijun Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+L), [Junliang Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+J), [Juntao Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J), [Min Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+M), [Tao Qin](https://arxiv.org/search/cs?searchtype=author&query=Qin%2C+T), [Tie-yan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+T)

> Non-autoregressive (NAR) generation, which is first proposed in neural machine translation (NMT) to speed up inference, has attracted much attention in both machine learning and natural language processing communities. While NAR generation can significantly accelerate inference speed for machine translation, the speedup comes at the cost of sacrificed translation accuracy compared to its counterpart, auto-regressive (AR) generation. In recent years, many new models and algorithms have been designed/proposed to bridge the accuracy gap between NAR generation and AR generation. In this paper, we conduct a systematic survey with comparisons and discussions of various non-autoregressive translation (NAT) models from different aspects. Specifically, we categorize the efforts of NAT into several groups, including data manipulation, modeling methods, training criterion, decoding algorithms, and the benefit from pre-trained models. Furthermore, we briefly review other applications of NAR models beyond machine translation, such as dialogue generation, text summarization, grammar error correction, semantic parsing, speech synthesis, and automatic speech recognition. In addition, we also discuss potential directions for future exploration, including releasing the dependency of KD, dynamic length prediction, pre-training for NAR, and wider applications, etc. We hope this survey can help researchers capture the latest progress in NAR generation, inspire the design of advanced NAR models and algorithms, and enable industry practitioners to choose appropriate solutions for their applications. The web page of this survey is at \url{[this https URL](https://github.com/LitterBrother-Xiao/Overview-of-Non-autoregressive-Applications)}.

| Comments: | 25 pages, 11 figures, 4 tables                               |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2204.09269](https://arxiv.org/abs/2204.09269) [cs.CL]** |
|           | (or **[arXiv:2204.09269v1](https://arxiv.org/abs/2204.09269v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.09269Focus to learn more |





<h2 id="2022-04-21-3">3. A Survey on Bias and Fairness in Natural Language Processing
</h2>

Title: [A Survey on Bias and Fairness in Natural Language Processing](https://arxiv.org/abs/2204.09591)

Authors: [Rajas Bansal](https://arxiv.org/search/cs?searchtype=author&query=Bansal%2C+R)

> As NLP models become more integrated with the everyday lives of people, it becomes important to examine the social effect that the usage of these systems has. While these models understand language and have increased accuracy on difficult downstream tasks, there is evidence that these models amplify gender, racial and cultural stereotypes and lead to a vicious cycle in many settings. In this survey, we analyze the origins of biases, the definitions of fairness, and how different subfields of NLP mitigate bias. We finally discuss how future studies can work towards eradicating pernicious biases from NLP algorithms.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2204.09591](https://arxiv.org/abs/2204.09591) [cs.CL]** |
|           | (or **[arXiv:2204.09591v1](https://arxiv.org/abs/2204.09591v1) [cs.CL]** for this version) |





<h2 id="2022-04-21-4">4. Exploring Continuous Integrate-and-Fire for Efficient and Adaptive Simultaneous Speech Translation
</h2>

Title: [Exploring Continuous Integrate-and-Fire for Efficient and Adaptive Simultaneous Speech Translation](https://arxiv.org/abs/2204.09595)

Authors: [Chih-Chiang Chang](https://arxiv.org/search/cs?searchtype=author&query=Chang%2C+C), [Hung-yi Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+H)

> Simultaneous speech translation (SimulST) is a challenging task that aims to directly translate streaming speech before the complete input is observed. A SimulST system generally includes two important components: the pre-decision that aggregates the speech information, and the policy that decides read or write. While recent works had proposed a variety of strategies to improve the pre-decision, they mostly adopt the fixed wait-k policy. The adaptive policies are rarely explored. We propose to model the adaptive policy using the Continuous Integrate-and-Fire (CIF). In our proposed model, the CIF is not only responsible for aggregating speech information, but also deciding when to read or write. To adapt the CIF to SimulST task, we propose two modifications: a token-level quantity loss or an infinite lookback attention. We show that our model can learn an adaptive policy effectively, achieving comparable or superior performance to MMA at lower latency, while being more efficient to train.

| Comments: | Submitted to INTERSPEECH 2022                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2204.09595](https://arxiv.org/abs/2204.09595) [cs.CL]** |
|           | (or **[arXiv:2204.09595v1](https://arxiv.org/abs/2204.09595v1) [cs.CL]** for this version) |





<h2 id="2022-04-21-5">5. Detecting Unintended Memorization in Language-Model-Fused ASR
</h2>

Title: [Detecting Unintended Memorization in Language-Model-Fused ASR](https://arxiv.org/abs/2204.09606)

Authors: [W. Ronny Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+W+R), [Steve Chien](https://arxiv.org/search/cs?searchtype=author&query=Chien%2C+S), [Om Thakkar](https://arxiv.org/search/cs?searchtype=author&query=Thakkar%2C+O), [Rajiv Mathews](https://arxiv.org/search/cs?searchtype=author&query=Mathews%2C+R)

> End-to-end (E2E) models are often being accompanied by language models (LMs) via shallow fusion for boosting their overall quality as well as recognition of rare words. At the same time, several prior works show that LMs are susceptible to unintentionally memorizing rare or unique sequences in the training data. In this work, we design a framework for detecting memorization of random textual sequences (which we call canaries) in the LM training data when one has only black-box (query) access to LM-fused speech recognizer, as opposed to direct access to the LM. On a production-grade Conformer RNN-T E2E model fused with a Transformer LM, we show that detecting memorization of singly-occurring canaries from the LM training data of 300M examples is possible. Motivated to protect privacy, we also show that such memorization gets significantly reduced by per-example gradient-clipped LM training without compromising overall quality.

| Subjects: | **Computation and Language (cs.CL)**; Cryptography and Security (cs.CR); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2204.09606](https://arxiv.org/abs/2204.09606) [cs.CL]** |
|           | (or **[arXiv:2204.09606v1](https://arxiv.org/abs/2204.09606v1) [cs.CL]** for this version) |









# 2022-04-20

[Return to Index](#Index)



<h2 id="2022-04-20-1">1. Imagination-Augmented Natural Language Understanding
</h2>

Title: [Imagination-Augmented Natural Language Understanding](https://arxiv.org/abs/2204.08535)
Authors: [Yujie Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+Y), [Wanrong Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+W), [Xin Eric Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X+E), [Miguel Eckstein](https://arxiv.org/search/cs?searchtype=author&query=Eckstein%2C+M), [William Yang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+W+Y)

> Human brains integrate linguistic and perceptual information simultaneously to understand natural language, and hold the critical ability to render imaginations. Such abilities enable us to construct new abstract concepts or concrete objects, and are essential in involving practical knowledge to solve problems in low-resource scenarios. However, most existing methods for Natural Language Understanding (NLU) are mainly focused on textual signals. They do not simulate human visual imagination ability, which hinders models from inferring and learning efficiently from limited data samples. Therefore, we introduce an Imagination-Augmented Cross-modal Encoder (iACE) to solve natural language understanding tasks from a novel learning perspective -- imagination-augmented cross-modal understanding. iACE enables visual imagination with external knowledge transferred from the powerful generative and pre-trained vision-and-language models. Extensive experiments on GLUE and SWAG show that iACE achieves consistent improvement over visually-supervised pre-trained models. More importantly, results in extreme and normal few-shot settings validate the effectiveness of iACE in low-resource natural language understanding circumstances.

| Comments: | 11 pages, 4 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2204.08535](https://arxiv.org/abs/2204.08535) [cs.CL]** |
|           | (or **[arXiv:2204.08535v1](https://arxiv.org/abs/2204.08535v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.08535Focus to learn more |





<h2 id="2022-04-20-2">2. Blockwise Streaming Transformer for Spoken Language Understanding and Simultaneous Speech Translation
</h2>

Title: [Blockwise Streaming Transformer for Spoken Language Understanding and Simultaneous Speech Translation](https://arxiv.org/abs/2204.08920)
Authors: [Keqi Deng](https://arxiv.org/search/cs?searchtype=author&query=Deng%2C+K), [Shinji Watanabe](https://arxiv.org/search/cs?searchtype=author&query=Watanabe%2C+S), [Jiatong Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+J), [Siddhant Arora](https://arxiv.org/search/cs?searchtype=author&query=Arora%2C+S)

> Although Transformers have gained success in several speech processing tasks like spoken language understanding (SLU) and speech translation (ST), achieving online processing while keeping competitive performance is still essential for real-world interaction. In this paper, we take the first step on streaming SLU and simultaneous ST using a blockwise streaming Transformer, which is based on contextual block processing and blockwise synchronous beam search. Furthermore, we design an automatic speech recognition (ASR)-based intermediate loss regularization for the streaming SLU task to improve the classification performance further. As for the simultaneous ST task, we propose a cross-lingual encoding method, which employs a CTC branch optimized with target language translations. In addition, the CTC translation output is also used to refine the search space with CTC prefix score, achieving joint CTC/attention simultaneous translation for the first time. Experiments for SLU are conducted on FSC and SLURP corpora, while the ST task is evaluated on Fisher-CallHome Spanish and MuST-C En-De corpora. Experimental results show that the blockwise streaming Transformer achieves competitive results compared to offline models, especially with our proposed methods that further yield a 2.4% accuracy gain on the SLU task and a 4.3 BLEU gain on the ST task over streaming baselines.

| Comments: | Submitted to Interspeech2022                                 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2204.08920](https://arxiv.org/abs/2204.08920) [cs.CL]** |
|           | (or **[arXiv:2204.08920v1](https://arxiv.org/abs/2204.08920v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.08920Focus to learn more |





<h2 id="2022-04-20-3">3. Feature Structure Distillation for BERT Transferring
</h2>

Title: [Feature Structure Distillation for BERT Transferring](https://arxiv.org/abs/2204.08922)
Authors: [Hee-Jun Jung](https://arxiv.org/search/cs?searchtype=author&query=Jung%2C+H), [Doyeon Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+D), [Seung-Hoon Na](https://arxiv.org/search/cs?searchtype=author&query=Na%2C+S), [Kangil Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+K)

> Knowledge distillation is an approach to transfer information on representations from a teacher to a student by reducing their difference. A challenge of this approach is to reduce the flexibility of the student's representations inducing inaccurate learning of the teacher's knowledge. To resolve it in BERT transferring, we investigate distillation of structures of representations specified to three types: intra-feature, local inter-feature, global inter-feature structures. To transfer them, we introduce \textit{feature structure distillation} methods based on the Centered Kernel Alignment, which assigns a consistent value to similar features structures and reveals more informative relations. In particular, a memory-augmented transfer method with clustering is implemented for the global structures. In the experiments on the nine tasks for language understanding of the GLUE dataset, the proposed methods effectively transfer the three types of structures and improve performance compared to state-of-the-art distillation methods. Indeed, the code for the methods is available in [this https URL](https://github.com/maroo-sky/FSD)

| Comments: | This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2204.08922](https://arxiv.org/abs/2204.08922) [cs.CL]** |
|           | (or **[arXiv:2204.08922v1](https://arxiv.org/abs/2204.08922v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.08922Focus to learn more |





<h2 id="2022-04-20-4">4. On the Locality of Attention in Direct Speech Translation
</h2>

Title: [On the Locality of Attention in Direct Speech Translation](https://arxiv.org/abs/2204.09028)
Authors: [Belen Alastruey](https://arxiv.org/search/cs?searchtype=author&query=Alastruey%2C+B), [Javier Ferrando](https://arxiv.org/search/cs?searchtype=author&query=Ferrando%2C+J), [Gerard I. Gállego](https://arxiv.org/search/cs?searchtype=author&query=Gállego%2C+G+I), [Marta R. Costa-jussà](https://arxiv.org/search/cs?searchtype=author&query=Costa-jussà%2C+M+R)

> Transformers have achieved state-of-the-art results across multiple NLP tasks. However, the self-attention mechanism complexity scales quadratically with the sequence length, creating an obstacle for tasks involving long sequences, like in the speech domain. In this paper, we discuss the usefulness of self-attention for Direct Speech Translation. First, we analyze the layer-wise token contributions in the self-attention of the encoder, unveiling local diagonal patterns. To prove that some attention weights are avoidable, we propose to substitute the standard self-attention with a local efficient one, setting the amount of context used based on the results of the analysis. With this approach, our model matches the baseline performance, and improves the efficiency by skipping the computation of those weights that standard attention discards.

| Comments: | ACL-SRW 2022. Equal contribution between Belen Alastruey and Javier Ferrando |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2204.09028](https://arxiv.org/abs/2204.09028) [cs.CL]** |
|           | (or **[arXiv:2204.09028v1](https://arxiv.org/abs/2204.09028v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.09028Focus to learn more |





# 2022-04-19

[Return to Index](#Index)



<h2 id="2022-04-19-1">1. mGPT: Few-Shot Learners Go Multilingual
</h2>

Title: [mGPT: Few-Shot Learners Go Multilingual](https://arxiv.org/abs/2204.07580)

Authors: [Oleh Shliazhko](https://arxiv.org/search/cs?searchtype=author&query=Shliazhko%2C+O), [Alena Fenogenova](https://arxiv.org/search/cs?searchtype=author&query=Fenogenova%2C+A), [Maria Tikhonova](https://arxiv.org/search/cs?searchtype=author&query=Tikhonova%2C+M), [Vladislav Mikhailov](https://arxiv.org/search/cs?searchtype=author&query=Mikhailov%2C+V), [Anastasia Kozlova](https://arxiv.org/search/cs?searchtype=author&query=Kozlova%2C+A), [Tatiana Shavrina](https://arxiv.org/search/cs?searchtype=author&query=Shavrina%2C+T)

> Recent studies report that autoregressive language models can successfully solve many NLP tasks via zero- and few-shot learning paradigms, which opens up new possibilities for using the pre-trained language models. This paper introduces two autoregressive GPT-like models with 1.3 billion and 13 billion parameters trained on 60 languages from 25 language families using Wikipedia and Colossal Clean Crawled Corpus. We reproduce the GPT-3 architecture using GPT-2 sources and the sparse attention mechanism; Deepspeed and Megatron frameworks allow us to parallelize the training and inference steps effectively. The resulting models show performance on par with the recently released XGLM models by Facebook, covering more languages and enhancing NLP possibilities for low resource languages of CIS countries and Russian small nations. We detail the motivation for the choices of the architecture design, thoroughly describe the data preparation pipeline, and train five small versions of the model to choose the most optimal multilingual tokenization strategy. We measure the model perplexity in all covered languages and evaluate it on the wide spectre of multilingual tasks, including classification, generative, sequence labeling and knowledge probing. The models were evaluated with the zero-shot and few-shot methods. Furthermore, we compared the classification tasks with the state-of-the-art multilingual model XGLM. source code and the mGPT XL model are publicly released.

| Subjects:    | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| ------------ | ------------------------------------------------------------ |
| MSC classes: | 68-06, 68-04, 68T50, 68T01                                   |
| ACM classes: | I.2; I.2.7                                                   |
| Cite as:     | **[arXiv:2204.07580](https://arxiv.org/abs/2204.07580) [cs.CL]** |
|              | (or **[arXiv:2204.07580v1](https://arxiv.org/abs/2204.07580v1) [cs.CL]** for this version) |
|              | https://doi.org/10.48550/arXiv.2204.07580Focus to learn more |





<h2 id="2022-04-19-2">2. MoEBERT: from BERT to Mixture-of-Experts via Importance-Guided Adaptation
</h2>

Title: [MoEBERT: from BERT to Mixture-of-Experts via Importance-Guided Adaptation](https://arxiv.org/abs/2204.07675)

Authors: [Simiao Zuo](https://arxiv.org/search/cs?searchtype=author&query=Zuo%2C+S), [Qingru Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Q), [Chen Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+C), [Pengcheng He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+P), [Tuo Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+T), [Weizhu Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+W)

> Pre-trained language models have demonstrated superior performance in various natural language processing tasks. However, these models usually contain hundreds of millions of parameters, which limits their practicality because of latency requirements in real-world applications. Existing methods train small compressed models via knowledge distillation. However, performance of these small models drops significantly compared with the pre-trained models due to their reduced model capacity. We propose MoEBERT, which uses a Mixture-of-Experts structure to increase model capacity and inference speed. We initialize MoEBERT by adapting the feed-forward neural networks in a pre-trained model into multiple experts. As such, representation power of the pre-trained model is largely retained. During inference, only one of the experts is activated, such that speed can be improved. We also propose a layer-wise distillation method to train MoEBERT. We validate the efficiency and effectiveness of MoEBERT on natural language understanding and question answering tasks. Results show that the proposed method outperforms existing task-specific distillation algorithms. For example, our method outperforms previous approaches by over 2% on the MNLI (mismatched) dataset. Our code is publicly available at [this https URL](https://github.com/SimiaoZuo/MoEBERT).

| Comments: | NAACL 2022                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2204.07675](https://arxiv.org/abs/2204.07675) [cs.CL]** |
|           | (or **[arXiv:2204.07675v1](https://arxiv.org/abs/2204.07675v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.07675Focus to learn more |





<h2 id="2022-04-19-3">3. Bridging Cross-Lingual Gaps During Leveraging the Multilingual Sequence-to-Sequence Pretraining for Text Generation
</h2>

Title: [Bridging Cross-Lingual Gaps During Leveraging the Multilingual Sequence-to-Sequence Pretraining for Text Generation](https://arxiv.org/abs/2204.07834)

Authors: [Changtong Zan](https://arxiv.org/search/cs?searchtype=author&query=Zan%2C+C), [Liang Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+L), [Li Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+L), [Yu Cao](https://arxiv.org/search/cs?searchtype=author&query=Cao%2C+Y), [Weifeng Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+W), [Dacheng Tao](https://arxiv.org/search/cs?searchtype=author&query=Tao%2C+D)

> For multilingual sequence-to-sequence pretrained language models (multilingual Seq2Seq PLMs), e.g. mBART, the self-supervised pretraining task is trained on a wide range of monolingual languages, e.g. 25 languages from commoncrawl, while the downstream cross-lingual tasks generally progress on a bilingual language subset, e.g. English-German, making there exists the cross-lingual data discrepancy, namely \textit{domain discrepancy}, and cross-lingual learning objective discrepancy, namely \textit{task discrepancy}, between the pretrain and finetune stages. To bridge the above cross-lingual domain and task gaps, we extend the vanilla pretrain-finetune pipeline with extra code-switching restore task. Specifically, the first stage employs the self-supervised code-switching restore task as a pretext task, allowing the multilingual Seq2Seq PLM to acquire some in-domain alignment information. And for the second stage, we continuously fine-tune the model on labeled data normally. Experiments on a variety of cross-lingual NLG tasks, including 12 bilingual translation tasks, 36 zero-shot translation tasks, and cross-lingual summarization tasks show our model outperforms the strong baseline mBART consistently. Comprehensive analyses indicate our approach could narrow the cross-lingual sentence representation distance and improve low-frequency word translation with trivial computational cost.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2204.07834](https://arxiv.org/abs/2204.07834) [cs.CL]** |
|           | (or **[arXiv:2204.07834v1](https://arxiv.org/abs/2204.07834v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.07834Focus to learn more |





<h2 id="2022-04-19-4">4. BLISS: Robust Sequence-to-Sequence Learning via Self-Supervised Input Representation
</h2>

Title: [BLISS: Robust Sequence-to-Sequence Learning via Self-Supervised Input Representation](https://arxiv.org/abs/2204.07837)

Authors: [Zheng Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Z), [Liang Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+L), [Dazhao Cheng](https://arxiv.org/search/cs?searchtype=author&query=Cheng%2C+D), [Xuebo Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+X), [Min Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+M), [Dacheng Tao](https://arxiv.org/search/cs?searchtype=author&query=Tao%2C+D)

> Data augmentations (DA) are the cores to achieving robust sequence-to-sequence learning on various natural language processing (NLP) tasks. However, most of the DA approaches force the decoder to make predictions conditioned on the perturbed input representation, underutilizing supervised information provided by perturbed input. In this work, we propose a framework-level robust sequence-to-sequence learning approach, named BLISS, via self-supervised input representation, which has the great potential to complement the data-level augmentation approaches. The key idea is to supervise the sequence-to-sequence framework with both the \textit{supervised} ("input→output") and \textit{self-supervised} ("perturbed input→input") information. We conduct comprehensive experiments to validate the effectiveness of BLISS on various tasks, including machine translation, grammatical error correction, and text summarization. The results show that BLISS outperforms significantly the vanilla Transformer and consistently works well across tasks than the other five contrastive baselines. Extensive analyses reveal that BLISS learns robust representations and rich linguistic knowledge, confirming our claim. Source code will be released upon publication.

| Comments: | arXiv admin note: text overlap with [arXiv:1904.03092](https://arxiv.org/abs/1904.03092), [arXiv:1904.03100](https://arxiv.org/abs/1904.03100) by other authors |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2204.07837](https://arxiv.org/abs/2204.07837) [cs.CL]** |
|           | (or **[arXiv:2204.07837v1](https://arxiv.org/abs/2204.07837v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.07837Focus to learn more |





<h2 id="2022-04-19-5">5. On Effectively Learning of Knowledge in Continual Pre-training
</h2>

Title: [On Effectively Learning of Knowledge in Continual Pre-training](https://arxiv.org/abs/2204.07994)

Authors: [Cunxiang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+C), [Fuli Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+F), [Yanyang Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Runxin Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+R), [Fei Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+F), [Yue Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y)

> Pre-trained language models (PLMs) like BERT have made significant progress in various downstream NLP tasks. However, by asking models to do cloze-style tests, recent work finds that PLMs are short in acquiring knowledge from unstructured text. To understand the internal behaviour of PLMs in retrieving knowledge, we first define knowledge-baring (K-B) tokens and knowledge-free (K-F) tokens for unstructured text and ask professional annotators to label some samples manually. Then, we find that PLMs are more likely to give wrong predictions on K-B tokens and attend less attention to those tokens inside the self-attention module. Based on these observations, we develop two solutions to help the model learn more knowledge from unstructured text in a fully self-supervised manner. Experiments on knowledge-intensive tasks show the effectiveness of the proposed methods. To our best knowledge, we are the first to explore fully self-supervised learning of knowledge in continual pre-training.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2204.07994](https://arxiv.org/abs/2204.07994) [cs.CL]** |
|           | (or **[arXiv:2204.07994v1](https://arxiv.org/abs/2204.07994v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.07994Focus to learn more |





<h2 id="2022-04-19-6">6. Dynamic Position Encoding for Transformers
</h2>

Title: [Dynamic Position Encoding for Transformers](https://arxiv.org/abs/2204.08142)

Authors: [Joyce Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+J), [Mehdi Rezagholizadeh](https://arxiv.org/search/cs?searchtype=author&query=Rezagholizadeh%2C+M), [Peyman Passban](https://arxiv.org/search/cs?searchtype=author&query=Passban%2C+P)

> Recurrent models have been dominating the field of neural machine translation (NMT) for the past few years. Transformers \citep{vaswani2017attention}, have radically changed it by proposing a novel architecture that relies on a feed-forward backbone and self-attention mechanism. Although Transformers are powerful, they could fail to properly encode sequential/positional information due to their non-recurrent nature. To solve this problem, position embeddings are defined exclusively for each time step to enrich word information. However, such embeddings are fixed after training regardless of the task and the word ordering system of the source or target language. 
> In this paper, we propose a novel architecture with new position embeddings depending on the input text to address this shortcoming by taking the order of target words into consideration. Instead of using predefined position embeddings, our solution \textit{generates} new embeddings to refine each word's position information. Since we do not dictate the position of source tokens and learn them in an end-to-end fashion, we refer to our method as \textit{dynamic} position encoding (DPE). We evaluated the impact of our model on multiple datasets to translate from English into German, French, and Italian and observed meaningful improvements in comparison to the original Transformer.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2204.08142](https://arxiv.org/abs/2204.08142) [cs.CL]** |
|           | (or **[arXiv:2204.08142v1](https://arxiv.org/abs/2204.08142v1) [cs.CL]** for this version) |





<h2 id="2022-04-19-7">7. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking
</h2>

Title: [LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking](https://arxiv.org/abs/2204.08387)

Authors: [Yupan Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+Y), [Tengchao Lv](https://arxiv.org/search/cs?searchtype=author&query=Lv%2C+T), [Lei Cui](https://arxiv.org/search/cs?searchtype=author&query=Cui%2C+L), [Yutong Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+Y), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F)

> Self-supervised pre-training techniques have achieved remarkable progress in Document AI. Most multimodal pre-trained models use a masked language modeling objective to learn bidirectional representations on the text modality, but they differ in pre-training objectives for the image modality. This discrepancy adds difficulty to multimodal representation learning. In this paper, we propose LayoutLMv3 to pre-train multimodal Transformers for Document AI with unified text and image masking. Additionally, LayoutLMv3 is pre-trained with a word-patch alignment objective to learn cross-modal alignment by predicting whether the corresponding image patch of a text word is masked. The simple unified architecture and training objectives make LayoutLMv3 a general-purpose pre-trained model for both text-centric and image-centric Document AI tasks. Experimental results show that LayoutLMv3 achieves state-of-the-art performance not only in text-centric tasks, including form understanding, receipt understanding, and document visual question answering, but also in image-centric tasks such as document image classification and document layout analysis. The code and models are publicly available at [this https URL](https://aka.ms/layoutlmv3).

| Comments: | Work in Progress                                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Computer Vision and Pattern Recognition (cs.CV) |
| Cite as:  | **[arXiv:2204.08387](https://arxiv.org/abs/2204.08387) [cs.CL]** |
|           | (or **[arXiv:2204.08387v1](https://arxiv.org/abs/2204.08387v1) [cs.CL]** for this version) |





<h2 id="2022-04-19-8">8. Exploring Dimensionality Reduction Techniques in Multilingual Transformers
</h2>

Title: [Exploring Dimensionality Reduction Techniques in Multilingual Transformers](https://arxiv.org/abs/2204.08415)

Authors: [Álvaro Huertas-García](https://arxiv.org/search/cs?searchtype=author&query=Huertas-García%2C+Á), [Alejandro Martín](https://arxiv.org/search/cs?searchtype=author&query=Martín%2C+A), [Javier Huertas-Tato](https://arxiv.org/search/cs?searchtype=author&query=Huertas-Tato%2C+J), [David Camacho](https://arxiv.org/search/cs?searchtype=author&query=Camacho%2C+D)

> Both in scientific literature and in industry,, Semantic and context-aware Natural Language Processing-based solutions have been gaining importance in recent years. The possibilities and performance shown by these models when dealing with complex Language Understanding tasks is unquestionable, from conversational agents to the fight against disinformation in social networks. In addition, considerable attention is also being paid to developing multilingual models to tackle the language bottleneck. The growing need to provide more complex models implementing all these features has been accompanied by an increase in their size, without being conservative in the number of dimensions required. This paper aims to give a comprehensive account of the impact of a wide variety of dimensional reduction techniques on the performance of different state-of-the-art multilingual Siamese Transformers, including unsupervised dimensional reduction techniques such as linear and nonlinear feature extraction, feature selection, and manifold techniques. In order to evaluate the effects of these techniques, we considered the multilingual extended version of Semantic Textual Similarity Benchmark (mSTSb) and two different baseline approaches, one using the pre-trained version of several models and another using their fine-tuned STS version. The results evidence that it is possible to achieve an average reduction in the number of dimensions of 91.58%±2.59% and 54.65%±32.20%, respectively. This work has also considered the consequences of dimensionality reduction for visualization purposes. The results of this study will significantly contribute to the understanding of how different tuning approaches affect performance on semantic-aware tasks and how dimensional reduction techniques deal with the high-dimensional embeddings computed for the STS task and their potential for highly demanding NLP tasks

| Comments: | 22 pages, 4 figures and 8 tables                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2204.08415](https://arxiv.org/abs/2204.08415) [cs.CL]** |
|           | (or **[arXiv:2204.08415v1](https://arxiv.org/abs/2204.08415v1) [cs.CL]** for this version) |





# 2022-04-18

[Return to Index](#Index)



<h2 id="2022-04-18-1">1. Vision-and-Language Pretrained Models: A Survey
</h2>

Title: [Vision-and-Language Pretrained Models: A Survey](https://arxiv.org/abs/2204.07356)

Authors: [Siqu Long](https://arxiv.org/search/cs?searchtype=author&query=Long%2C+S), [Feiqi Cao](https://arxiv.org/search/cs?searchtype=author&query=Cao%2C+F), [Soyeon Caren Han](https://arxiv.org/search/cs?searchtype=author&query=Han%2C+S+C), [Haiqing Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+H)

> Pretrained models have produced great success in both Computer Vision (CV) and Natural Language Processing (NLP). This progress leads to learning joint representations of vision and language pretraining by feeding visual and linguistic contents into a multi-layer transformer, Visual-Language Pretrained Models (VLPMs). In this paper, we present an overview of the major advances achieved in VLPMs for producing joint representations of vision and language. As the preliminaries, we briefly describe the general task definition and genetic architecture of VLPMs. We first discuss the language and vision data encoding methods and then present the mainstream VLPM structure as the core content. We further summarise several essential pretraining and fine-tuning strategies. Finally, we highlight three future directions for both CV and NLP researchers to provide insightful guidance.

| Comments: | Accepted in IJCAI 2022                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2204.07356](https://arxiv.org/abs/2204.07356) [cs.CV]** |
|           | (or **[arXiv:2204.07356v1](https://arxiv.org/abs/2204.07356v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.07356Focus to learn more |





<h2 id="2022-04-18-2">2. COTS: Collaborative Two-Stream Vision-Language Pre-Training Model for Cross-Modal Retrieval
</h2>

Title: [COTS: Collaborative Two-Stream Vision-Language Pre-Training Model for Cross-Modal Retrieval](https://arxiv.org/abs/2204.07441)

Authors: [Haoyu Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+H), [Nanyi Fei](https://arxiv.org/search/cs?searchtype=author&query=Fei%2C+N), [Yuqi Huo](https://arxiv.org/search/cs?searchtype=author&query=Huo%2C+Y), [Yizhao Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+Y), [Zhiwu Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+Z), [Ji-Rong Wen](https://arxiv.org/search/cs?searchtype=author&query=Wen%2C+J)

> Large-scale single-stream pre-training has shown dramatic performance in image-text retrieval. Regrettably, it faces low inference efficiency due to heavy attention layers. Recently, two-stream methods like CLIP and ALIGN with high inference efficiency have also shown promising performance, however, they only consider instance-level alignment between the two streams (thus there is still room for improvement). To overcome these limitations, we propose a novel COllaborative Two-Stream vision-language pretraining model termed COTS for image-text retrieval by enhancing cross-modal interaction. In addition to instance level alignment via momentum contrastive learning, we leverage two extra levels of cross-modal interactions in our COTS: (1) Token-level interaction - a masked visionlanguage modeling (MVLM) learning objective is devised without using a cross-stream network module, where variational autoencoder is imposed on the visual encoder to generate visual tokens for each image. (2) Task-level interaction - a KL-alignment learning objective is devised between text-to-image and image-to-text retrieval tasks, where the probability distribution per task is computed with the negative queues in momentum contrastive learning. Under a fair comparison setting, our COTS achieves the highest performance among all two-stream methods and comparable performance (but with 10,800X faster in inference) w.r.t. the latest single-stream methods. Importantly, our COTS is also applicable to text-to-video retrieval, yielding new state-ofthe-art on the widely-used MSR-VTT dataset.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Information Retrieval (cs.IR) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2204.07441](https://arxiv.org/abs/2204.07441) [cs.CV]** |
|           | (or **[arXiv:2204.07441v1](https://arxiv.org/abs/2204.07441v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.07441Focus to learn more |





<h2 id="2022-04-18-3">3. XDBERT: Distilling Visual Information to BERT from Cross-Modal Systems to Improve Language Understanding
</h2>

Title: [XDBERT: Distilling Visual Information to BERT from Cross-Modal Systems to Improve Language Understanding](https://arxiv.org/abs/2204.07316)

Authors: [Chan-Jan Hsu](https://arxiv.org/search/cs?searchtype=author&query=Hsu%2C+C), [Hung-yi Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+H), [Yu Tsao](https://arxiv.org/search/cs?searchtype=author&query=Tsao%2C+Y)

> Transformer-based models are widely used in natural language understanding (NLU) tasks, and multimodal transformers have been effective in visual-language tasks. This study explores distilling visual information from pretrained multimodal transformers to pretrained language encoders. Our framework is inspired by cross-modal encoders' success in visual-language tasks while we alter the learning objective to cater to the language-heavy characteristics of NLU. After training with a small number of extra adapting steps and finetuned, the proposed XDBERT (cross-modal distilled BERT) outperforms pretrained-BERT in general language understanding evaluation (GLUE), situations with adversarial generations (SWAG) benchmarks, and readability benchmarks. We analyze the performance of XDBERT on GLUE to show that the improvement is likely visually grounded.

| Comments: | ACL 2022                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2204.07316](https://arxiv.org/abs/2204.07316) [cs.CL]** |
|           | (or **[arXiv:2204.07316v1](https://arxiv.org/abs/2204.07316v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.07316Focus to learn more |





<h2 id="2022-04-18-4">4. LaMemo: Language Modeling with Look-Ahead Memory
</h2>

Title: [LaMemo: Language Modeling with Look-Ahead Memory](https://arxiv.org/abs/2204.07341)

Authors: [Haozhe Ji](https://arxiv.org/search/cs?searchtype=author&query=Ji%2C+H), [Rongsheng Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+R), [Zhenyu Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Z), [Zhipeng Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+Z), [Minlie Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+M)

> Although Transformers with fully connected self-attentions are powerful to model long-term dependencies, they are struggling to scale to long texts with thousands of words in language modeling. One of the solutions is to equip the model with a recurrence memory. However, existing approaches directly reuse hidden states from the previous segment that encodes contexts in a uni-directional way. As a result, this prohibits the memory to dynamically interact with the current context that provides up-to-date information for token prediction. To remedy this issue, we propose Look-Ahead Memory (LaMemo) that enhances the recurrence memory by incrementally attending to the right-side tokens, and interpolating with the old memory states to maintain long-term information in the history. LaMemo embraces bi-directional attention and segment recurrence with an additional computation overhead only linearly proportional to the memory length. Experiments on widely used language modeling benchmarks demonstrate its superiority over the baselines equipped with different types of memory.

| Comments: | Accepted by NAACL 2022                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2204.07341](https://arxiv.org/abs/2204.07341) [cs.CL]** |
|           | (or **[arXiv:2204.07341v1](https://arxiv.org/abs/2204.07341v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.07341Focus to learn more |





<h2 id="2022-04-18-5">5. Text Revision by On-the-Fly Representation Optimization
</h2>

Title: [Text Revision by On-the-Fly Representation Optimization](https://arxiv.org/abs/2204.07359)

Authors: [Jingjing Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J), [Zichao Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Z), [Tao Ge](https://arxiv.org/search/cs?searchtype=author&query=Ge%2C+T), [Irwin King](https://arxiv.org/search/cs?searchtype=author&query=King%2C+I), [Michael R. Lyu](https://arxiv.org/search/cs?searchtype=author&query=Lyu%2C+M+R)

> Text revision refers to a family of natural language generation tasks, where the source and target sequences share moderate resemblance in surface form but differentiate in attributes, such as text formality and simplicity. Current state-of-the-art methods formulate these tasks as sequence-to-sequence learning problems, which rely on large-scale parallel training corpus. In this paper, we present an iterative in-place editing approach for text revision, which requires no parallel data. In this approach, we simply fine-tune a pre-trained Transformer with masked language modeling and attribute classification. During inference, the editing at each iteration is realized by two-step span replacement. At the first step, the distributed representation of the text optimizes on the fly towards an attribute function. At the second step, a text span is masked and another new one is proposed conditioned on the optimized representation. The empirical experiments on two typical and important text revision tasks, text formalization and text simplification, show the effectiveness of our approach. It achieves competitive and even better performance than state-of-the-art supervised methods on text simplification, and gains better performance than strong unsupervised methods on text formalization \footnote{Code and model are available at \url{[this https URL](https://github.com/jingjingli01/OREO)}}.

| Comments: | AAAI 2022                                                    |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2204.07359](https://arxiv.org/abs/2204.07359) [cs.CL]** |
|           | (or **[arXiv:2204.07359v1](https://arxiv.org/abs/2204.07359v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.07359Focus to learn more |





<h2 id="2022-04-18-6">6. On the Role of Pre-trained Language Models in Word Ordering: A Case Study with BART
</h2>

Title: [On the Role of Pre-trained Language Models in Word Ordering: A Case Study with BART](https://arxiv.org/abs/2204.07367)

Authors: [Zebin Ou](https://arxiv.org/search/cs?searchtype=author&query=Ou%2C+Z), [Meishan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+M), [Yue Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y)

> Word ordering is a constrained language generation task taking unordered words as input. Existing work uses linear models and neural networks for the task, yet pre-trained language models have not been studied in word ordering, let alone why they help. We use BART as an instance and show its effectiveness in the task. To explain why BART helps word ordering, we extend analysis with probing and empirically identify that syntactic dependency knowledge in BART is a reliable explanation. We also report performance gains with BART in the related partial tree linearization task, which readily extends our analysis.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2204.07367](https://arxiv.org/abs/2204.07367) [cs.CL]** |
|           | (or **[arXiv:2204.07367v1](https://arxiv.org/abs/2204.07367v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.07367Focus to learn more |





<h2 id="2022-04-18-7">7. Chinese Idiom Paraphrasing
</h2>

Title: [Chinese Idiom Paraphrasing](https://arxiv.org/abs/2204.07555)

Authors: [Jipeng Qiang](https://arxiv.org/search/cs?searchtype=author&query=Qiang%2C+J), [Yang Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Chaowei Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+C), [Yun Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Yunhao Yuan](https://arxiv.org/search/cs?searchtype=author&query=Yuan%2C+Y), [Yi Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+Y), [Xindong Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+X)

> Idioms, are a kind of idiomatic expression in Chinese, most of which consist of four Chinese characters. Due to the properties of non-compositionality and metaphorical meaning, Chinese Idioms are hard to be understood by children and non-native speakers. This study proposes a novel task, denoted as Chinese Idiom Paraphrasing (CIP). CIP aims to rephrase idioms-included sentences to non-idiomatic ones under the premise of preserving the original sentence's meaning. Since the sentences without idioms are easier handled by Chinese NLP systems, CIP can be used to pre-process Chinese datasets, thereby facilitating and improving the performance of Chinese NLP tasks, e.g., machine translation system, Chinese idiom cloze, and Chinese idiom embeddings. In this study, CIP task is treated as a special paraphrase generation task. To circumvent difficulties in acquiring annotations, we first establish a large-scale CIP dataset based on human and machine collaboration, which consists of 115,530 sentence pairs. We further deploy three baselines and two novel CIP approaches to deal with CIP problems. The results show that the proposed methods have better performances than the baselines based on the established CIP dataset.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2204.07555](https://arxiv.org/abs/2204.07555) [cs.CL]** |
|           | (or **[arXiv:2204.07555v1](https://arxiv.org/abs/2204.07555v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.07555Focus to learn more |



# 2022-04-15

[Return to Index](#Index)



<h2 id="2022-04-15-1">1. METRO: Efficient Denoising Pretraining of Large Scale Autoencoding Language Models with Model Generated Signals
</h2>

Title: [METRO: Efficient Denoising Pretraining of Large Scale Autoencoding Language Models with Model Generated Signals](https://arxiv.org/abs/2204.06644)

Authors: [Payal Bajaj](https://arxiv.org/search/cs?searchtype=author&query=Bajaj%2C+P), [Chenyan Xiong](https://arxiv.org/search/cs?searchtype=author&query=Xiong%2C+C), [Guolin Ke](https://arxiv.org/search/cs?searchtype=author&query=Ke%2C+G), [Xiaodong Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+X), [Di He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+D), [Saurabh Tiwary](https://arxiv.org/search/cs?searchtype=author&query=Tiwary%2C+S), [Tie-Yan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+T), [Paul Bennett](https://arxiv.org/search/cs?searchtype=author&query=Bennett%2C+P), [Xia Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+X), [Jianfeng Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+J)

> We present an efficient method of pretraining large-scale autoencoding language models using training signals generated by an auxiliary model. Originated in ELECTRA, this training strategy has demonstrated sample-efficiency to pretrain models at the scale of hundreds of millions of parameters. In this work, we conduct a comprehensive empirical study, and propose a recipe, namely "Model generated dEnoising TRaining Objective" (METRO), which incorporates some of the best modeling techniques developed recently to speed up, stabilize, and enhance pretrained language models without compromising model effectiveness. The resultant models, METRO-LM, consisting of up to 5.4 billion parameters, achieve new state-of-the-art on the GLUE, SuperGLUE, and SQuAD benchmarks. More importantly, METRO-LM are efficient in that they often outperform previous large models with significantly smaller model sizes and lower pretraining cost.

| Subjects: | **Machine Learning (cs.LG)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2204.06644](https://arxiv.org/abs/2204.06644) [cs.LG]** |
|           | (or **[arXiv:2204.06644v1](https://arxiv.org/abs/2204.06644v1) [cs.LG]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.06644Focus to learn more |

