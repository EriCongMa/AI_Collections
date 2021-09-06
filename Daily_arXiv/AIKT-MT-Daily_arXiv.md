# Daily arXiv: Machine Translation - September, 2021

# Index


- [2021-09-06](#2021-09-06)

  - [1. Ranking Scientific Papers Using Preference Learning](#2021-09-06-1)
  - [2. Establishing Interlingua in Multilingual Language Models](#2021-09-06-2)
  - [3. Quantifying Reproducibility in NLP and ML](#2021-09-06-3)
  - [4. Multimodal Conditionality for Natural Language Generation](#2021-09-06-4)
  - [5. Do Prompt-Based Models Really Understand the Meaning of their Prompts?](#2021-09-06-5)
  - [6. Language Modeling, Lexical Translation, Reordering: The Training Process of NMT through the Lens of Classical SMT](#2021-09-06-6)
  - [7. Finetuned Language Models Are Zero-Shot Learners](#2021-09-06-7)
- [2021-09-03](#2021-09-03)
  - [1. Skim-Attention: Learning to Focus via Document Layout](#2021-09-03-1)
  - [2. How Suitable Are Subword Segmentation Strategies for Translating Non-Concatenative Morphology?](#2021-09-03-2)
  - [3. Sequence-to-Sequence Learning with Latent Neural Grammars](#2021-09-03-3)
  - [4. Knowledge Perceived Multi-modal Pretraining in E-commerce](#2021-09-03-4)
  - [5. Improving Multimodal fusion via Mutual Dependency Maximisation](#2021-09-03-5)
  - [6. Towards Improving Adversarial Training of NLP Models](#2021-09-03-6)
  - [7. Point-of-Interest Type Prediction using Text and Images](#2021-09-03-7)
  - [8. Towards Making the Most of Dialogue Characteristics for Neural Chat Translation](#2021-09-03-8)
- [2021-09-02](#2021-09-02)

  - [1. Sentence Bottleneck Autoencoders from Transformer Language Models](#2021-09-02-1)
  - [2. It's not Rocket Science : Interpreting Figurative Language in Narratives](#2021-09-02-2)
  - [3. Aligning Cross-lingual Sentence Representations with Dual Momentum Contrast](#2021-09-02-3)
  - [4. Discovering Representation Sprachbund For Multilingual Pre-Training](#2021-09-02-4)
  - [5. ∞-former: Infinite Memory Transformer](#2021-09-02-5)
  - [6. Masked Adversarial Generation for Neural Machine Translation](#2021-09-02-6)
  - [7. Position Masking for Improved Layout-Aware Document Understanding](#2021-09-02-7)
  - [8. Survey of Low-Resource Machine Translation](#2021-09-02-8)
- [2021-09-01](#2021-09-01)
  - [1. SimulLR: Simultaneous Lip Reading Transducer with Attention-Guided Adaptive Memory](#2021-09-01-1)
  - [2. Want To Reduce Labeling Cost? GPT-3 Can Help](#2021-09-01-2)
  - [3. T3-Vis: a visual analytic framework for Training and fine-Tuning Transformers in NLP](#2021-09-01-3)
  - [4. Enjoy the Salience: Towards Better Transformer-based Faithful Explanations with Word Salience](#2021-09-01-4)
  - [5. Thermostat: A Large Collection of NLP Model Explanations and Analysis Tools](#2021-09-01-5)
- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MT-Daily_arXiv-index.md)



# 2021-09-06

[Return to Index](#Index)



<h2 id="2021-09-06-1">1. Ranking Scientific Papers Using Preference Learning
</h2>

Title: [Ranking Scientific Papers Using Preference Learning](https://arxiv.org/abs/2109.01190)

Authors: [Nils Dycke](https://arxiv.org/search/cs?searchtype=author&query=Dycke%2C+N), [Edwin Simpson](https://arxiv.org/search/cs?searchtype=author&query=Simpson%2C+E), [Ilia Kuznetsov](https://arxiv.org/search/cs?searchtype=author&query=Kuznetsov%2C+I), [Iryna Gurevych](https://arxiv.org/search/cs?searchtype=author&query=Gurevych%2C+I)

> Peer review is the main quality control mechanism in academia. Quality of scientific work has many dimensions; coupled with the subjective nature of the reviewing task, this makes final decision making based on the reviews and scores therein very difficult and time-consuming. To assist with this important task, we cast it as a paper ranking problem based on peer review texts and reviewer scores. We introduce a novel, multi-faceted generic evaluation framework for making final decisions based on peer reviews that takes into account effectiveness, efficiency and fairness of the evaluated system. We propose a novel approach to paper ranking based on Gaussian Process Preference Learning (GPPL) and evaluate it on peer review data from the ACL-2018 conference. Our experiments demonstrate the superiority of our GPPL-based approach over prior work, while highlighting the importance of using both texts and review scores for paper ranking during peer review aggregation.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2109.01190](https://arxiv.org/abs/2109.01190) [cs.CL]** |
|           | (or **[arXiv:2109.01190v1](https://arxiv.org/abs/2109.01190v1) [cs.CL]** for this version) |





<h2 id="2021-09-06-2">2. Establishing Interlingua in Multilingual Language Models
</h2>

Title: [Establishing Interlingua in Multilingual Language Models](https://arxiv.org/abs/2109.01207)

Authors: [Maksym Del](https://arxiv.org/search/cs?searchtype=author&query=Del%2C+M), [Mark Fishel](https://arxiv.org/search/cs?searchtype=author&query=Fishel%2C+M)

> Large multilingual language models show remarkable zero-shot cross-lingual transfer performance on a range of tasks. Follow-up works hypothesized that these models internally project representations of different languages into a shared interlingual space. However, they produced contradictory results. In this paper, we correct %one of the previous works the famous prior work claiming that "BERT is not an Interlingua" and show that with the proper choice of sentence representation different languages actually do converge to a shared space in such language models. Furthermore, we demonstrate that this convergence pattern is robust across four measures of correlation similarity and six mBERT-like models. We then extend our analysis to 28 diverse languages and find that the interlingual space exhibits a particular structure similar to the linguistic relatedness of languages. We also highlight a few outlier languages that seem to fail to converge to the shared space. The code for replicating our results is available at the following URL: [this https URL](https://github.com/maksym-del/interlingua).

| Comments:    | 8 pages, 10 figures                                          |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**                         |
| ACM classes: | I.2.7; I.2.6                                                 |
| Cite as:     | **[arXiv:2109.01207](https://arxiv.org/abs/2109.01207) [cs.CL]** |
|              | (or **[arXiv:2109.01207v1](https://arxiv.org/abs/2109.01207v1) [cs.CL]** for this version) |





<h2 id="2021-09-06-3">3. Quantifying Reproducibility in NLP and ML
</h2>

Title: [Quantifying Reproducibility in NLP and ML](https://arxiv.org/abs/2109.01211)

Authors: [Anya Belz](https://arxiv.org/search/cs?searchtype=author&query=Belz%2C+A)

> Reproducibility has become an intensely debated topic in NLP and ML over recent years, but no commonly accepted way of assessing reproducibility, let alone quantifying it, has so far emerged. The assumption has been that wider scientific reproducibility terminology and definitions are not applicable to NLP/ML, with the result that many different terms and definitions have been proposed, some diametrically opposed. In this paper, we test this assumption, by taking the standard terminology and definitions from metrology and applying them directly to NLP/ML. We find that we are able to straightforwardly derive a practical framework for assessing reproducibility which has the desirable property of yielding a quantified degree of reproducibility that is comparable across different reproduction studies.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2109.01211](https://arxiv.org/abs/2109.01211) [cs.CL]** |
|           | (or **[arXiv:2109.01211v1](https://arxiv.org/abs/2109.01211v1) [cs.CL]** for this version) |





<h2 id="2021-09-06-4">4. Multimodal Conditionality for Natural Language Generation
</h2>

Title: [Multimodal Conditionality for Natural Language Generation](https://arxiv.org/abs/2109.01229)

Authors: [Michael Sollami](https://arxiv.org/search/cs?searchtype=author&query=Sollami%2C+M), [Aashish Jain](https://arxiv.org/search/cs?searchtype=author&query=Jain%2C+A)

> Large scale pretrained language models have demonstrated state-of-the-art performance in language understanding tasks. Their application has recently expanded into multimodality learning, leading to improved representations combining vision and language. However, progress in adapting language models towards conditional Natural Language Generation (NLG) has been limited to a single modality, generally text. We propose MAnTiS, Multimodal Adaptation for Text Synthesis, a general approach for multimodal conditionality in transformer-based NLG models. In this method, we pass inputs from each modality through modality-specific encoders, project to textual token space, and finally join to form a conditionality prefix. We fine-tune the pretrained language model and encoders with the conditionality prefix guiding the generation. We apply MAnTiS to the task of product description generation, conditioning a network on both product images and titles to generate descriptive text. We demonstrate that MAnTiS outperforms strong baseline approaches on standard NLG scoring metrics. Furthermore, qualitative assessments demonstrate that MAnTiS can generate human quality descriptions consistent with given multimodal inputs.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2109.01229](https://arxiv.org/abs/2109.01229) [cs.CL]** |
|           | (or **[arXiv:2109.01229v1](https://arxiv.org/abs/2109.01229v1) [cs.CL]** for this version) |





<h2 id="2021-09-06-5">5. Do Prompt-Based Models Really Understand the Meaning of their Prompts?
</h2>

Title: [Do Prompt-Based Models Really Understand the Meaning of their Prompts?](https://arxiv.org/abs/2109.01247)

Authors: [Albert Webson](https://arxiv.org/search/cs?searchtype=author&query=Webson%2C+A), [Ellie Pavlick](https://arxiv.org/search/cs?searchtype=author&query=Pavlick%2C+E)

> Recently, a boom of papers have shown extraordinary progress in few-shot learning with various prompt-based models. Such success can give the impression that prompts help models to learn faster in the same way that humans learn faster when provided with task instructions expressed in natural language. In this study, we experiment with over 30 prompts manually written for natural language inference (NLI). We find that models learn just as fast with many prompts that are intentionally irrelevant or even pathologically misleading as they do with instructively "good" prompts. Additionally, we find that model performance is more dependent on the choice of the LM target words (a.k.a. the "verbalizer" that converts LM vocabulary prediction to class labels) than on the text of the prompt itself. In sum, we find little evidence that suggests existing prompt-based models truly understand the meaning of their given prompts.

| Comments: | Code available at [this https URL](https://github.com/awebson/prompt_semantics) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2109.01247](https://arxiv.org/abs/2109.01247) [cs.CL]** |
|           | (or **[arXiv:2109.01247v1](https://arxiv.org/abs/2109.01247v1) [cs.CL]** for this version) |





<h2 id="2021-09-06-6">6. Language Modeling, Lexical Translation, Reordering: The Training Process of NMT through the Lens of Classical SMT
</h2>

Title: [Language Modeling, Lexical Translation, Reordering: The Training Process of NMT through the Lens of Classical SMT](https://arxiv.org/abs/2109.01396)

Authors: [Elena Voita](https://arxiv.org/search/cs?searchtype=author&query=Voita%2C+E), [Rico Sennrich](https://arxiv.org/search/cs?searchtype=author&query=Sennrich%2C+R), [Ivan Titov](https://arxiv.org/search/cs?searchtype=author&query=Titov%2C+I)

> Differently from the traditional statistical MT that decomposes the translation task into distinct separately learned components, neural machine translation uses a single neural network to model the entire translation process. Despite neural machine translation being de-facto standard, it is still not clear how NMT models acquire different competences over the course of training, and how this mirrors the different models in traditional SMT. In this work, we look at the competences related to three core SMT components and find that during training, NMT first focuses on learning target-side language modeling, then improves translation quality approaching word-by-word translation, and finally learns more complicated reordering patterns. We show that this behavior holds for several models and language pairs. Additionally, we explain how such an understanding of the training process can be useful in practice and, as an example, show how it can be used to improve vanilla non-autoregressive neural machine translation by guiding teacher model selection.

| Comments: | EMNLP 2021                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2109.01396](https://arxiv.org/abs/2109.01396) [cs.CL]** |
|           | (or **[arXiv:2109.01396v1](https://arxiv.org/abs/2109.01396v1) [cs.CL]** for this version) |





<h2 id="2021-09-06-7">7. Finetuned Language Models Are Zero-Shot Learners
</h2>

Title: [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)

Authors: [Jason Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+J), [Maarten Bosma](https://arxiv.org/search/cs?searchtype=author&query=Bosma%2C+M), [Vincent Y. Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+V+Y), [Kelvin Guu](https://arxiv.org/search/cs?searchtype=author&query=Guu%2C+K), [Adams Wei Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+A+W), [Brian Lester](https://arxiv.org/search/cs?searchtype=author&query=Lester%2C+B), [Nan Du](https://arxiv.org/search/cs?searchtype=author&query=Du%2C+N), [Andrew M. Dai](https://arxiv.org/search/cs?searchtype=author&query=Dai%2C+A+M), [Quoc V. Le](https://arxiv.org/search/cs?searchtype=author&query=Le%2C+Q+V)

> This paper explores a simple method for improving the zero-shot learning abilities of language models. We show that instruction tuning -- finetuning language models on a collection of tasks described via instructions -- substantially boosts zero-shot performance on unseen tasks.
> We take a 137B parameter pretrained language model and instruction-tune it on over 60 NLP tasks verbalized via natural language instruction templates. We evaluate this instruction-tuned model, which we call FLAN, on unseen task types. FLAN substantially improves the performance of its unmodified counterpart and surpasses zero-shot 175B GPT-3 on 19 of 25 tasks that we evaluate. FLAN even outperforms few-shot GPT-3 by a large margin on ANLI, RTE, BoolQ, AI2-ARC, OpenbookQA, and StoryCloze. Ablation studies reveal that number of tasks and model scale are key components to the success of instruction tuning.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2109.01652](https://arxiv.org/abs/2109.01652) [cs.CL]** |
|           | (or **[arXiv:2109.01652v1](https://arxiv.org/abs/2109.01652v1) [cs.CL]** for this version) |








# 2021-09-03

[Return to Index](#Index)



<h2 id="2021-09-03-1">1. Skim-Attention: Learning to Focus via Document Layout
</h2>

Title: [Skim-Attention: Learning to Focus via Document Layout](https://arxiv.org/abs/2109.01078)

Authors: [Laura Nguyen](https://arxiv.org/search/cs?searchtype=author&query=Nguyen%2C+L), [Thomas Scialom](https://arxiv.org/search/cs?searchtype=author&query=Scialom%2C+T), [Jacopo Staiano](https://arxiv.org/search/cs?searchtype=author&query=Staiano%2C+J), [Benjamin Piwowarski](https://arxiv.org/search/cs?searchtype=author&query=Piwowarski%2C+B)

> Transformer-based pre-training techniques of text and layout have proven effective in a number of document understanding tasks. Despite this success, multimodal pre-training models suffer from very high computational and memory costs. Motivated by human reading strategies, this paper presents Skim-Attention, a new attention mechanism that takes advantage of the structure of the document and its layout. Skim-Attention only attends to the 2-dimensional position of the words in a document. Our experiments show that Skim-Attention obtains a lower perplexity than prior works, while being more computationally efficient. Skim-Attention can be further combined with long-range Transformers to efficiently process long documents. We also show how Skim-Attention can be used off-the-shelf as a mask for any Pre-trained Language Model, allowing to improve their performance while restricting attention. Finally, we show the emergence of a document structure representation in Skim-Attention.

| Comments: | 15 pages, 6 figures, to be published in EMNLP 2021 Findings  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2109.01078](https://arxiv.org/abs/2109.01078) [cs.CL]** |
|           | (or **[arXiv:2109.01078v1](https://arxiv.org/abs/2109.01078v1) [cs.CL]** for this version) |





<h2 id="2021-09-03-2">2. How Suitable Are Subword Segmentation Strategies for Translating Non-Concatenative Morphology?
</h2>

Title: [How Suitable Are Subword Segmentation Strategies for Translating Non-Concatenative Morphology?](https://arxiv.org/abs/2109.01100)

Authors: [Chantal Amrhein](https://arxiv.org/search/cs?searchtype=author&query=Amrhein%2C+C), [Rico Sennrich](https://arxiv.org/search/cs?searchtype=author&query=Sennrich%2C+R)

> Data-driven subword segmentation has become the default strategy for open-vocabulary machine translation and other NLP tasks, but may not be sufficiently generic for optimal learning of non-concatenative morphology. We design a test suite to evaluate segmentation strategies on different types of morphological phenomena in a controlled, semi-synthetic setting. In our experiments, we compare how well machine translation models trained on subword- and character-level can translate these morphological phenomena. We find that learning to analyse and generate morphologically complex surface representations is still challenging, especially for non-concatenative morphological phenomena like reduplication or vowel harmony and for rare word stems. Based on our results, we recommend that novel text representation strategies be tested on a range of typologically diverse languages to minimise the risk of adopting a strategy that inadvertently disadvantages certain languages.

| Comments:    | Findings of EMNLP 2021                                       |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**                         |
| ACM classes: | I.2.7                                                        |
| Cite as:     | **[arXiv:2109.01100](https://arxiv.org/abs/2109.01100) [cs.CL]** |
|              | (or **[arXiv:2109.01100v1](https://arxiv.org/abs/2109.01100v1) [cs.CL]** for this version) |





<h2 id="2021-09-03-3">3. Sequence-to-Sequence Learning with Latent Neural Grammars
</h2>

Title: [Sequence-to-Sequence Learning with Latent Neural Grammars](https://arxiv.org/abs/2109.01135)

Authors: [Yoon Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+Y)

> Sequence-to-sequence learning with neural networks has become the de facto standard for sequence prediction tasks. This approach typically models the local distribution over the next word with a powerful neural network that can condition on arbitrary context. While flexible and performant, these models often require large datasets for training and can fail spectacularly on benchmarks designed to test for compositional generalization. This work explores an alternative, hierarchical approach to sequence-to-sequence learning with quasi-synchronous grammars, where each node in the target tree is transduced by a node in the source tree. Both the source and target trees are treated as latent and induced during training. We develop a neural parameterization of the grammar which enables parameter sharing over the combinatorial space of derivation rules without the need for manual feature engineering. We apply this latent neural grammar to various domains -- a diagnostic language navigation task designed to test for compositional generalization (SCAN), style transfer, and small-scale machine translation -- and find that it performs respectably compared to standard baselines.

| Comments: | Preprint                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2109.01135](https://arxiv.org/abs/2109.01135) [cs.CL]** |
|           | (or **[arXiv:2109.01135v1](https://arxiv.org/abs/2109.01135v1) [cs.CL]** for this version) |





<h2 id="2021-09-03-4">4. Knowledge Perceived Multi-modal Pretraining in E-commerce
</h2>

Title: [Knowledge Perceived Multi-modal Pretraining in E-commerce](https://arxiv.org/abs/2109.00895)

Authors: [Yushan Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+Y), [Huaixiao Tou](https://arxiv.org/search/cs?searchtype=author&query=Tou%2C+H), [Wen Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+W), [Ganqiang Ye](https://arxiv.org/search/cs?searchtype=author&query=Ye%2C+G), [Hui Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+H), [Ningyu Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+N), [Huajun Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+H)

> In this paper, we address multi-modal pretraining of product data in the field of E-commerce. Current multi-modal pretraining methods proposed for image and text modalities lack robustness in the face of modality-missing and modality-noise, which are two pervasive problems of multi-modal product data in real E-commerce scenarios. To this end, we propose a novel method, K3M, which introduces knowledge modality in multi-modal pretraining to correct the noise and supplement the missing of image and text modalities. The modal-encoding layer extracts the features of each modality. The modal-interaction layer is capable of effectively modeling the interaction of multiple modalities, where an initial-interactive feature fusion model is designed to maintain the independence of image modality and text modality, and a structure aggregation module is designed to fuse the information of image, text, and knowledge modalities. We pretrain K3M with three pretraining tasks, including masked object modeling (MOM), masked language modeling (MLM), and link prediction modeling (LPM). Experimental results on a real-world E-commerce dataset and a series of product-based downstream tasks demonstrate that K3M achieves significant improvements in performances than the baseline and state-of-the-art methods when modality-noise or modality-missing exists.

| Comments: | Accepted to ACM MM 2021                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL) |
| DOI:      | [10.1145/3474085.3475648](https://arxiv.org/ct?url=https%3A%2F%2Fdx.doi.org%2F10.1145%2F3474085.3475648&v=072100ff) |
| Cite as:  | **[arXiv:2109.00895](https://arxiv.org/abs/2109.00895) [cs.CV]** |
|           | (or **[arXiv:2109.00895v1](https://arxiv.org/abs/2109.00895v1) [cs.CV]** for this version) |





<h2 id="2021-09-03-5">5. Improving Multimodal fusion via Mutual Dependency Maximisation
</h2>

Title: [Improving Multimodal fusion via Mutual Dependency Maximisation](https://arxiv.org/abs/2109.00922)

Authors: [Pierre Colombo](https://arxiv.org/search/cs?searchtype=author&query=Colombo%2C+P), [Emile Chapuis](https://arxiv.org/search/cs?searchtype=author&query=Chapuis%2C+E), [Matthieu Labeau](https://arxiv.org/search/cs?searchtype=author&query=Labeau%2C+M), [Chloe Clavel](https://arxiv.org/search/cs?searchtype=author&query=Clavel%2C+C)

> Multimodal sentiment analysis is a trending area of research, and the multimodal fusion is one of its most active topic. Acknowledging humans communicate through a variety of channels (i.e visual, acoustic, linguistic), multimodal systems aim at integrating different unimodal representations into a synthetic one. So far, a consequent effort has been made on developing complex architectures allowing the fusion of these modalities. However, such systems are mainly trained by minimising simple losses such as L1 or cross-entropy. In this work, we investigate unexplored penalties and propose a set of new objectives that measure the dependency between modalities. We demonstrate that our new penalties lead to a consistent improvement (up to 4.3 on accuracy) across a large variety of state-of-the-art models on two well-known sentiment analysis datasets: \texttt{CMU-MOSI} and \texttt{CMU-MOSEI}. Our method not only achieves a new SOTA on both datasets but also produces representations that are more robust to modality drops. Finally, a by-product of our methods includes a statistical network which can be used to interpret the high dimensional representations learnt by the model.

| Subjects:          | **Machine Learning (cs.LG)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL) |
| ------------------ | ------------------------------------------------------------ |
| Journal reference: | EMNLP 2021                                                   |
| Cite as:           | **[arXiv:2109.00922](https://arxiv.org/abs/2109.00922) [cs.LG]** |
|                    | (or **[arXiv:2109.00922v1](https://arxiv.org/abs/2109.00922v1) [cs.LG]** for this version) |





<h2 id="2021-09-03-6">6. Towards Improving Adversarial Training of NLP Models
</h2>

Title: [Towards Improving Adversarial Training of NLP Models](https://arxiv.org/abs/2109.00544)

Authors: [Jin Yong Yoo](https://arxiv.org/search/cs?searchtype=author&query=Yoo%2C+J+Y), [Yanjun Qi](https://arxiv.org/search/cs?searchtype=author&query=Qi%2C+Y)

> Adversarial training, a method for learning robust deep neural networks, constructs adversarial examples during training. However, recent methods for generating NLP adversarial examples involve combinatorial search and expensive sentence encoders for constraining the generated instances. As a result, it remains challenging to use vanilla adversarial training to improve NLP models' performance, and the benefits are mainly uninvestigated. This paper proposes a simple and improved vanilla adversarial training process for NLP, which we name Attacking to Training (𝙰𝟸𝚃). The core part of 𝙰𝟸𝚃 is a new and cheaper word substitution attack optimized for vanilla adversarial training. We use 𝙰𝟸𝚃 to train BERT and RoBERTa models on IMDB, Rotten Tomatoes, Yelp, and SNLI datasets. Our results show that it is possible to train empirically robust NLP models using a much cheaper adversary. We demonstrate that vanilla adversarial training with 𝙰𝟸𝚃 can improve an NLP model's robustness to the attack it was originally trained with and also defend the model against other types of attacks. Furthermore, we show that 𝙰𝟸𝚃 can improve NLP models' standard accuracy, cross-domain generalization, and interpretability. Code is available at [this http URL](http://github.com/jinyongyoo/A2T) .

| Comments: | EMNLP Findings 2021                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2109.00544](https://arxiv.org/abs/2109.00544) [cs.CL]** |
|           | (or **[arXiv:2109.00544v1](https://arxiv.org/abs/2109.00544v1) [cs.CL]** for this version) |





<h2 id="2021-09-03-7">7. Point-of-Interest Type Prediction using Text and Images
</h2>

Title: [Point-of-Interest Type Prediction using Text and Images](https://arxiv.org/abs/2109.00602)

Authors: [Danae Sánchez Villegas](https://arxiv.org/search/cs?searchtype=author&query=Villegas%2C+D+S), [Nikolaos Aletras](https://arxiv.org/search/cs?searchtype=author&query=Aletras%2C+N)

> Point-of-interest (POI) type prediction is the task of inferring the type of a place from where a social media post was shared. Inferring a POI's type is useful for studies in computational social science including sociolinguistics, geosemiotics, and cultural geography, and has applications in geosocial networking technologies such as recommendation and visualization systems. Prior efforts in POI type prediction focus solely on text, without taking visual information into account. However in reality, the variety of modalities, as well as their semiotic relationships with one another, shape communication and interactions in social media. This paper presents a study on POI type prediction using multimodal information from text and images available at posting time. For that purpose, we enrich a currently available data set for POI type prediction with the images that accompany the text messages. Our proposed method extracts relevant information from each modality to effectively capture interactions between text and image achieving a macro F1 of 47.21 across eight categories significantly outperforming the state-of-the-art method for POI type prediction based on text-only methods. Finally, we provide a detailed analysis to shed light on cross-modal interactions and the limitations of our best performing model.

| Comments: | Accepted at EMNLP 2021                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2109.00602](https://arxiv.org/abs/2109.00602) [cs.CL]** |
|           | (or **[arXiv:2109.00602v1](https://arxiv.org/abs/2109.00602v1) [cs.CL]** for this version) |





<h2 id="2021-09-03-8">8. Towards Making the Most of Dialogue Characteristics for Neural Chat Translation
</h2>

Title: [Towards Making the Most of Dialogue Characteristics for Neural Chat Translation](https://arxiv.org/abs/2109.00668)

Authors: [Yunlong Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+Y), [Chulun Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+C), [Fandong Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+F), [Jinan Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+J), [Yufeng Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Y), [Jinsong Su](https://arxiv.org/search/cs?searchtype=author&query=Su%2C+J), [Jie Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+J)

> Neural Chat Translation (NCT) aims to translate conversational text between speakers of different languages. Despite the promising performance of sentence-level and context-aware neural machine translation models, there still remain limitations in current NCT models because the inherent dialogue characteristics of chat, such as dialogue coherence and speaker personality, are neglected. In this paper, we propose to promote the chat translation by introducing the modeling of dialogue characteristics into the NCT model. To this end, we design four auxiliary tasks including monolingual response generation, cross-lingual response generation, next utterance discrimination, and speaker identification. Together with the main chat translation task, we optimize the NCT model through the training objectives of all these tasks. By this means, the NCT model can be enhanced by capturing the inherent dialogue characteristics, thus generating more coherent and speaker-relevant translations. Comprehensive experiments on four language directions (English-German and English-Chinese) verify the effectiveness and superiority of the proposed approach.

| Comments: | Accepted as a long paper at EMNLP 2021 main conference. The first two authors contributed equally. Code: [this https URL](https://github.com/XL2248/CSA-NCT) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2109.00668](https://arxiv.org/abs/2109.00668) [cs.CL]** |
|           | (or **[arXiv:2109.00668v1](https://arxiv.org/abs/2109.00668v1) [cs.CL]** for this version) |





# 2021-09-02

[Return to Index](#Index)



<h2 id="2021-09-02-1">1. Sentence Bottleneck Autoencoders from Transformer Language Models
</h2>

Title: [Sentence Bottleneck Autoencoders from Transformer Language Models](https://arxiv.org/abs/2109.00055)

Authors: [Ivan Montero](https://arxiv.org/search/cs?searchtype=author&query=Montero%2C+I), [Nikolaos Pappas](https://arxiv.org/search/cs?searchtype=author&query=Pappas%2C+N), [Noah A. Smith](https://arxiv.org/search/cs?searchtype=author&query=Smith%2C+N+A)

> Representation learning for text via pretraining a language model on a large corpus has become a standard starting point for building NLP systems. This approach stands in contrast to autoencoders, also trained on raw text, but with the objective of learning to encode each input as a vector that allows full reconstruction. Autoencoders are attractive because of their latent space structure and generative properties. We therefore explore the construction of a sentence-level autoencoder from a pretrained, frozen transformer language model. We adapt the masked language modeling objective as a generative, denoising one, while only training a sentence bottleneck and a single-layer modified transformer decoder. We demonstrate that the sentence representations discovered by our model achieve better quality than previous methods that extract representations from pretrained transformers on text similarity tasks, style transfer (an example of controlled generation), and single-sentence classification tasks in the GLUE benchmark, while using fewer parameters than large pretrained models.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2109.00055](https://arxiv.org/abs/2109.00055) [cs.CL]** |
|           | (or **[arXiv:2109.00055v1](https://arxiv.org/abs/2109.00055v1) [cs.CL]** for this version) |





<h2 id="2021-09-02-2">2. It's not Rocket Science : Interpreting Figurative Language in Narratives
</h2>

Title: [It's not Rocket Science : Interpreting Figurative Language in Narratives](https://arxiv.org/abs/2109.00087)

Authors: [Tuhin Chakrabarty](https://arxiv.org/search/cs?searchtype=author&query=Chakrabarty%2C+T), [Yejin Choi](https://arxiv.org/search/cs?searchtype=author&query=Choi%2C+Y), [Vered Shwartz](https://arxiv.org/search/cs?searchtype=author&query=Shwartz%2C+V)

> Figurative language is ubiquitous in English. Yet, the vast majority of NLP research focuses on literal language. Existing text representations by design rely on compositionality, while figurative language is often non-compositional. In this paper, we study the interpretation of two non-compositional figurative languages (idioms and similes). We collected datasets of fictional narratives containing a figurative expression along with crowd-sourced plausible and implausible continuations relying on the correct interpretation of the expression. We then trained models to choose or generate the plausible continuation. Our experiments show that models based solely on pre-trained language models perform substantially worse than humans on these tasks. We additionally propose knowledge-enhanced models, adopting human strategies for interpreting figurative language: inferring meaning from the context and relying on the constituent word's literal meanings. The knowledge-enhanced models improve the performance on both the discriminative and generative tasks, further bridging the gap from human performance.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2109.00087](https://arxiv.org/abs/2109.00087) [cs.CL]** |
|           | (or **[arXiv:2109.00087v1](https://arxiv.org/abs/2109.00087v1) [cs.CL]** for this version) |





<h2 id="2021-09-02-3">3. Aligning Cross-lingual Sentence Representations with Dual Momentum Contrast
</h2>

Title: [Aligning Cross-lingual Sentence Representations with Dual Momentum Contrast](https://arxiv.org/abs/2109.00253)

Authors: [Liang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L), [Wei Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+W), [Jingming Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+J)

> In this paper, we propose to align sentence representations from different languages into a unified embedding space, where semantic similarities (both cross-lingual and monolingual) can be computed with a simple dot product. Pre-trained language models are fine-tuned with the translation ranking task. Existing work (Feng et al., 2020) uses sentences within the same batch as negatives, which can suffer from the issue of easy negatives. We adapt MoCo (He et al., 2020) to further improve the quality of alignment. As the experimental results show, the sentence representations produced by our model achieve the new state-of-the-art on several tasks, including Tatoeba en-zh similarity search (Artetxe and Schwenk, 2019b), BUCC en-zh bitext mining, and semantic textual similarity on 7 datasets.

| Comments: | Accepted to EMNLP 2021 main conference                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Information Retrieval (cs.IR) |
| Cite as:  | **[arXiv:2109.00253](https://arxiv.org/abs/2109.00253) [cs.CL]** |
|           | (or **[arXiv:2109.00253v1](https://arxiv.org/abs/2109.00253v1) [cs.CL]** for this version) |





<h2 id="2021-09-02-4">4. Discovering Representation Sprachbund For Multilingual Pre-Training
</h2>

Title: [Discovering Representation Sprachbund For Multilingual Pre-Training](https://arxiv.org/abs/2109.00271)

Authors: [Yimin Fan](https://arxiv.org/search/cs?searchtype=author&query=Fan%2C+Y), [Yaobo Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+Y), [Alexandre Muzio](https://arxiv.org/search/cs?searchtype=author&query=Muzio%2C+A), [Hany Hassan](https://arxiv.org/search/cs?searchtype=author&query=Hassan%2C+H), [Houqiang Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+H), [Ming Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+M), [Nan Duan](https://arxiv.org/search/cs?searchtype=author&query=Duan%2C+N)

> Multilingual pre-trained models have demonstrated their effectiveness in many multilingual NLP tasks and enabled zero-shot or few-shot transfer from high-resource languages to low resource ones. However, due to significant typological differences and contradictions between some languages, such models usually perform poorly on many languages and cross-lingual settings, which shows the difficulty of learning a single model to handle massive diverse languages well at the same time. To alleviate this issue, we present a new multilingual pre-training pipeline. We propose to generate language representation from multilingual pre-trained models and conduct linguistic analysis to show that language representation similarity reflect linguistic similarity from multiple perspectives, including language family, geographical sprachbund, lexicostatistics and syntax. Then we cluster all the target languages into multiple groups and name each group as a representation sprachbund. Thus, languages in the same representation sprachbund are supposed to boost each other in both pre-training and fine-tuning as they share rich linguistic similarity. We pre-train one multilingual model for each representation sprachbund. Experiments are conducted on cross-lingual benchmarks and significant improvements are achieved compared to strong baselines.

| Comments: | To Appear at the Findings of EMNLP2021                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2109.00271](https://arxiv.org/abs/2109.00271) [cs.CL]** |
|           | (or **[arXiv:2109.00271v1](https://arxiv.org/abs/2109.00271v1) [cs.CL]** for this version) |





<h2 id="2021-09-02-5">5. ∞-former: Infinite Memory Transformer
</h2>

Title: [∞-former: Infinite Memory Transformer](https://arxiv.org/abs/2109.00301)

Authors: [Pedro Henrique Martins](https://arxiv.org/search/cs?searchtype=author&query=Martins%2C+P+H), [Zita Marinho](https://arxiv.org/search/cs?searchtype=author&query=Marinho%2C+Z), [André F. T. Martins](https://arxiv.org/search/cs?searchtype=author&query=Martins%2C+A+F+T)

> Transformers struggle when attending to long contexts, since the amount of computation grows with the context length, and therefore they cannot model long-term memories effectively. Several variations have been proposed to alleviate this problem, but they all have a finite memory capacity, being forced to drop old information. In this paper, we propose the ∞-former, which extends the vanilla transformer with an unbounded long-term memory. By making use of a continuous-space attention mechanism to attend over the long-term memory, the ∞-former's attention complexity becomes independent of the context length. Thus, it is able to model arbitrarily long contexts and maintain "sticky memories" while keeping a fixed computation budget. Experiments on a synthetic sorting task demonstrate the ability of the ∞-former to retain information from long sequences. We also perform experiments on language modeling, by training a model from scratch and by fine-tuning a pre-trained language model, which show benefits of unbounded long-term memories.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2109.00301](https://arxiv.org/abs/2109.00301) [cs.CL]** |
|           | (or **[arXiv:2109.00301v1](https://arxiv.org/abs/2109.00301v1) [cs.CL]** for this version) |





<h2 id="2021-09-02-6">6. Masked Adversarial Generation for Neural Machine Translation
</h2>

Title: [Masked Adversarial Generation for Neural Machine Translation](https://arxiv.org/abs/2109.00417)

Authors: [Badr Youbi Idrissi](https://arxiv.org/search/cs?searchtype=author&query=Idrissi%2C+B+Y), [Stéphane Clinchant](https://arxiv.org/search/cs?searchtype=author&query=Clinchant%2C+S)

> Attacking Neural Machine Translation models is an inherently combinatorial task on discrete sequences, solved with approximate heuristics. Most methods use the gradient to attack the model on each sample independently. Instead of mechanically applying the gradient, could we learn to produce meaningful adversarial attacks ? In contrast to existing approaches, we learn to attack a model by training an adversarial generator based on a language model. We propose the Masked Adversarial Generation (MAG) model, that learns to perturb the translation model throughout the training process. The experiments show that it improves the robustness of machine translation models, while being faster than competing methods.

| Comments: | 5 pages                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2109.00417](https://arxiv.org/abs/2109.00417) [cs.CL]** |
|           | (or **[arXiv:2109.00417v1](https://arxiv.org/abs/2109.00417v1) [cs.CL]** for this version) |





<h2 id="2021-09-02-7">7. Position Masking for Improved Layout-Aware Document Understanding
</h2>

Title: [Position Masking for Improved Layout-Aware Document Understanding](https://arxiv.org/abs/2109.00442)

Authors: [Anik Saha](https://arxiv.org/search/cs?searchtype=author&query=Saha%2C+A), [Catherine Finegan-Dollak](https://arxiv.org/search/cs?searchtype=author&query=Finegan-Dollak%2C+C), [Ashish Verma](https://arxiv.org/search/cs?searchtype=author&query=Verma%2C+A)

> Natural language processing for document scans and PDFs has the potential to enormously improve the efficiency of business processes. Layout-aware word embeddings such as LayoutLM have shown promise for classification of and information extraction from such documents. This paper proposes a new pre-training task called that can improve performance of layout-aware word embeddings that incorporate 2-D position embeddings. We compare models pre-trained with only language masking against models pre-trained with both language masking and position masking, and we find that position masking improves performance by over 5% on a form understanding task.

| Comments: | Document Intelligence Workshop at KDD, 2021                  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2109.00442](https://arxiv.org/abs/2109.00442) [cs.CL]** |
|           | (or **[arXiv:2109.00442v1](https://arxiv.org/abs/2109.00442v1) [cs.CL]** for this version) |





<h2 id="2021-09-02-8">8. Survey of Low-Resource Machine Translation
</h2>

Title: [Survey of Low-Resource Machine Translation](https://arxiv.org/abs/2109.00486)

Authors: [Barry Haddow](https://arxiv.org/search/cs?searchtype=author&query=Haddow%2C+B), [Rachel Bawden](https://arxiv.org/search/cs?searchtype=author&query=Bawden%2C+R), [Antonio Valerio Miceli Barone](https://arxiv.org/search/cs?searchtype=author&query=Barone%2C+A+V+M), [Jindřich Helcl](https://arxiv.org/search/cs?searchtype=author&query=Helcl%2C+J), [Alexandra Birch](https://arxiv.org/search/cs?searchtype=author&query=Birch%2C+A)

> We present a survey covering the state of the art in low-resource machine translation. There are currently around 7000 languages spoken in the world and almost all language pairs lack significant resources for training machine translation models. There has been increasing interest in research addressing the challenge of producing useful translation models when very little translated training data is available. We present a high level summary of this topical field and provide an overview of best practices.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2109.00486](https://arxiv.org/abs/2109.00486) [cs.CL]** |
|           | (or **[arXiv:2109.00486v1](https://arxiv.org/abs/2109.00486v1) [cs.CL]** for this version) |








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



