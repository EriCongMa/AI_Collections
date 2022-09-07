# MA C.'s Daily Paper Of Interest - August a., 2022

# Index

- [2022-08-30](#2022-08-30)
  - [1. Contrastive Audio-Language Learning for Music](#2022-08-30-1)

  - [2. Shortcut Learning of Large Language Models in Natural Language Understanding: A Survey](#2022-08-30-2)
  
  - [3. On Reality and the Limits of Language Data](#2022-08-30-3)
  
  - [4. Training a T5 Using Lab-sized Resources](#2022-08-30-4)
  
- [2022-08-24](#2022-08-24)
  - [1. MATra: A Multilingual Attentive Transliteration System for Indian Scripts](#2022-08-24-1)

  - [2. Learning Better Masking for Better Language Model Pre-training](#2022-08-24-2)

  - [3. CLOWER: A Pre-trained Language Model with Contrastive Learning over Word and Character Representations](#2022-08-24-3)

- [2022-08-12](#2022-08-12)
  - [1. Comparison and Analysis of New Curriculum Criteria for End-to-End ASR](#2022-08-12-1)

  - [2. Language Tokens: A Frustratingly Simple Approach Improves Zero-Shot Performance of Multilingual Translation](#2022-08-12-2)

  - [3. Domain-Specific Text Generation for Machine Translation](#2022-08-12-3)

- [2022-08-10](#2022-08-10)
  - [1. Thai Wav2Vec2.0 with CommonVoice V8](#2022-08-10-1)

- [2022-08-09](#2022-08-09)
  - [1. Creating Reverse Bilingual Dictionaries](#2022-08-09-1)

- [2022-08-08](#2022-08-08)
  - [1. Phrase translation using a bilingual dictionary and n-gram data: A case study from Vietnamese to English](#2022-08-08-1)

- [2022-08-04](#2022-08-04)
  - [1. Masked Vision and Language Modeling for Multi-modal Representation Learning](#2022-08-04-1)

- [2022-07-20](#2022-07-20)
  - [1. Multilingual Transformer Encoders: a Word-Level Task-Agnostic Evaluation](#2022-07-20-1)

- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



# 2022-08-30

[Return to Index](#Index)



<h2 id="2022-08-30-1">1. Contrastive Audio-Language Learning for Music
</h2>

Title: [Contrastive Audio-Language Learning for Music](https://arxiv.org/abs/2208.12208)

Authors: [Ilaria Manco](https://arxiv.org/search/cs?searchtype=author&query=Manco%2C+I), [Emmanouil Benetos](https://arxiv.org/search/cs?searchtype=author&query=Benetos%2C+E), [Elio Quinton](https://arxiv.org/search/cs?searchtype=author&query=Quinton%2C+E), [György Fazekas](https://arxiv.org/search/cs?searchtype=author&query=Fazekas%2C+G)

> As one of the most intuitive interfaces known to humans, natural language has the potential to mediate many tasks that involve human-computer interaction, especially in application-focused fields like Music Information Retrieval. In this work, we explore cross-modal learning in an attempt to bridge audio and language in the music domain. To this end, we propose MusCALL, a framework for Music Contrastive Audio-Language Learning. Our approach consists of a dual-encoder architecture that learns the alignment between pairs of music audio and descriptive sentences, producing multimodal embeddings that can be used for text-to-audio and audio-to-text retrieval out-of-the-box. Thanks to this property, MusCALL can be transferred to virtually any task that can be cast as text-based retrieval. Our experiments show that our method performs significantly better than the baselines at retrieving audio that matches a textual description and, conversely, text that matches an audio query. We also demonstrate that the multimodal alignment capability of our model can be successfully extended to the zero-shot transfer scenario for genre classification and auto-tagging on two public datasets.

| Comments: | Accepted to ISMIR 2022                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Sound (cs.SD)**; Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2208.12208](https://arxiv.org/abs/2208.12208) [cs.SD]** |
|           | (or **[arXiv:2208.12208v1](https://arxiv.org/abs/2208.12208v1) [cs.SD]** for this version) |
|           | https://doi.org/10.48550/arXiv.2208.12208Focus to learn more |





<h2 id="2022-08-30-2">2. Shortcut Learning of Large Language Models in Natural Language Understanding: A Survey
</h2>

Title: [Shortcut Learning of Large Language Models in Natural Language Understanding: A Survey](https://arxiv.org/abs/2208.11857)

Authors: [Mengnan Du](https://arxiv.org/search/cs?searchtype=author&query=Du%2C+M), [Fengxiang He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+F), [Na Zou](https://arxiv.org/search/cs?searchtype=author&query=Zou%2C+N), [Dacheng Tao](https://arxiv.org/search/cs?searchtype=author&query=Tao%2C+D), [Xia Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+X)

> Large language models (LLMs) have achieved state-of-the-art performance on a series of natural language understanding tasks. However, these LLMs might rely on dataset bias and artifacts as shortcuts for prediction. This has significantly hurt their Out-of-Distribution (OOD) generalization and adversarial robustness. In this paper, we provide a review of recent developments that address the robustness challenge of LLMs. We first introduce the concepts and robustness challenge of LLMs. We then introduce methods to identify shortcut learning behavior in LLMs, characterize the reasons for shortcut learning, as well as introduce mitigation solutions. Finally, we identify key challenges and introduce the connections of this line of research to other directions.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2208.11857](https://arxiv.org/abs/2208.11857) [cs.CL]** |
|           | (or **[arXiv:2208.11857v1](https://arxiv.org/abs/2208.11857v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2208.11857Focus to learn more |





<h2 id="2022-08-30-3">3. On Reality and the Limits of Language Data
</h2>

Title: [On Reality and the Limits of Language Data](https://arxiv.org/abs/2208.11981)

Authors: [Nigel H. Collier](https://arxiv.org/search/cs?searchtype=author&query=Collier%2C+N+H), [Fangyu Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+F), [Ehsan Shareghi](https://arxiv.org/search/cs?searchtype=author&query=Shareghi%2C+E)

> Recent advances in neural network language models have shown that it is possible to derive expressive meaning representations by leveraging linguistic associations in large-scale natural language data. These potentially Gestalt representations have enabled state-of-the-art performance for many practical applications. It would appear that we are on a pathway to empirically deriving a robust and expressive computable semantics. A key question that arises is how far can language data alone enable computers to understand the necessary truth about the physical world? Attention to this question is warranted because our future interactions with intelligent machines depends on how well our techniques correctly represent and process the concepts (objects, properties, and processes) that humans commonly observe to be true. After reviewing existing protocols, the objective of this work is to explore this question using a novel and tightly controlled reasoning test and to highlight what models might learn directly from pure linguistic data.

| Comments: | 14 pages; data available, see [this https URL](https://sites.google.com/site/nhcollier/projects/art) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2208.11981](https://arxiv.org/abs/2208.11981) [cs.CL]** |
|           | (or **[arXiv:2208.11981v1](https://arxiv.org/abs/2208.11981v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2208.11981Focus to learn more |





<h2 id="2022-08-30-4">4. Training a T5 Using Lab-sized Resources
</h2>

Title: [Training a T5 Using Lab-sized Resources](https://arxiv.org/abs/2208.12097)

Authors: [Manuel R. Ciosici](https://arxiv.org/search/cs?searchtype=author&query=Ciosici%2C+M+R), [Leon Derczynski](https://arxiv.org/search/cs?searchtype=author&query=Derczynski%2C+L)

> Training large neural language models on large datasets is resource- and time-intensive. These requirements create a barrier to entry, where those with fewer resources cannot build competitive models. This paper presents various techniques for making it possible to (a) train a large language model using resources that a modest research lab might have, and (b) train it in a reasonable amount of time. We provide concrete recommendations for practitioners, which we illustrate with a case study: a T5 model for Danish, the first for this language.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2208.12097](https://arxiv.org/abs/2208.12097) [cs.CL]** |
|           | (or **[arXiv:2208.12097v1](https://arxiv.org/abs/2208.12097v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2208.12097Focus to learn more |






# 2022-08-24

[Return to Index](#Index)



<h2 id="2022-08-24-1">1. MATra: A Multilingual Attentive Transliteration System for Indian Scripts
</h2>

Title: [MATra: A Multilingual Attentive Transliteration System for Indian Scripts](https://arxiv.org/abs/2208.10801)

Authors: [Yash Raj](https://arxiv.org/search/cs?searchtype=author&query=Raj%2C+Y), [Bhavesh Laddagiri](https://arxiv.org/search/cs?searchtype=author&query=Laddagiri%2C+B)

> Transliteration is a task in the domain of NLP where the output word is a similar-sounding word written using the letters of any foreign language. Today this system has been developed for several language pairs that involve English as either the source or target word and deployed in several places like Google Translate and chatbots. However, there is very little research done in the field of Indic languages transliterated to other Indic languages. This paper demonstrates a multilingual model based on transformers (with some modifications) that can give noticeably higher performance and accuracy than all existing models in this domain and get much better results than state-of-the-art models. This paper shows a model that can perform transliteration between any pair among the following five languages - English, Hindi, Bengali, Kannada and Tamil. It is applicable in scenarios where language is a barrier to communication in any written task. The model beats the state-of-the-art (for all pairs among the five mentioned languages - English, Hindi, Bengali, Kannada, and Tamil) and achieves a top-1 accuracy score of 80.7%, about 29.5% higher than the best current results. Furthermore, the model achieves 93.5% in terms of Phonetic Accuracy (transliteration is primarily a phonetic/sound-based task).

| Comments: | 10 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2208.10801](https://arxiv.org/abs/2208.10801) [cs.CL]** |
|           | (or **[arXiv:2208.10801v1](https://arxiv.org/abs/2208.10801v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2208.10801Focus to learn more |





<h2 id="2022-08-24-2">2. Learning Better Masking for Better Language Model Pre-training
</h2>

Title: [Learning Better Masking for Better Language Model Pre-training](https://arxiv.org/abs/2208.10806)

Authors: [Dongjie Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+D), [Zhuosheng Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Z), [Hai Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+H)

> Masked Language Modeling (MLM) has been widely used as the denoising objective in pre-training language models (PrLMs). Existing PrLMs commonly adopt a random-token masking strategy where a fixed masking ratio is applied and different contents are masked by an equal probability throughout the entire training. However, the model may receive complicated impact from pre-training status, which changes accordingly as training time goes on. In this paper, we show that such time-invariant MLM settings on masking ratio and masked content are unlikely to deliver an optimal outcome, which motivates us to explore the influence of time-variant MLM settings. We propose two scheduled masking approaches that adaptively tune the masking ratio and contents in different training stages, which improves the pre-training efficiency and effectiveness verified on the downstream tasks. Our work is a pioneer study on time-variant masking strategy on ratio and contents and gives a better understanding of how masking ratio and masked content influence the MLM pre-training.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2208.10806](https://arxiv.org/abs/2208.10806) [cs.CL]** |
|           | (or **[arXiv:2208.10806v1](https://arxiv.org/abs/2208.10806v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2208.10806Focus to learn more |





<h2 id="2022-08-24-3">3. CLOWER: A Pre-trained Language Model with Contrastive Learning over Word and Character Representations
</h2>

Title: [CLOWER: A Pre-trained Language Model with Contrastive Learning over Word and Character Representations](https://arxiv.org/abs/2208.10844)

Authors: [Borun Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+B), [Hongyin Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+H), [Jingang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+J), [Qifan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Q), [Hai-Tao Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+H), [Wei Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+W), [Liqian Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+L)

> Pre-trained Language Models (PLMs) have achieved remarkable performance gains across numerous downstream tasks in natural language understanding. Various Chinese PLMs have been successively proposed for learning better Chinese language representation. However, most current models use Chinese characters as inputs and are not able to encode semantic information contained in Chinese words. While recent pre-trained models incorporate both words and characters simultaneously, they usually suffer from deficient semantic interactions and fail to capture the semantic relation between words and characters. To address the above issues, we propose a simple yet effective PLM CLOWER, which adopts the Contrastive Learning Over Word and charactER representations. In particular, CLOWER implicitly encodes the coarse-grained information (i.e., words) into the fine-grained representations (i.e., characters) through contrastive learning on multi-grained information. CLOWER is of great value in realistic scenarios since it can be easily incorporated into any existing fine-grained based PLMs without modifying the production pipelines.Extensive experiments conducted on a range of downstream tasks demonstrate the superior performance of CLOWER over several state-of-the-art baselines.

| Comments: | Accepted in COLING 2022                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2208.10844](https://arxiv.org/abs/2208.10844) [cs.CL]** |
|           | (or **[arXiv:2208.10844v1](https://arxiv.org/abs/2208.10844v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2208.10844Focus to learn more |






# 2022-08-12

[Return to Index](#Index)



<h2 id="2022-08-12-1">1. Comparison and Analysis of New Curriculum Criteria for End-to-End ASR
</h2>

Title: [Comparison and Analysis of New Curriculum Criteria for End-to-End ASR](https://arxiv.org/abs/2208.05782)

Authors: [Georgios Karakasidis](https://arxiv.org/search/eess?searchtype=author&query=Karakasidis%2C+G), [Tamás Grósz](https://arxiv.org/search/eess?searchtype=author&query=Grósz%2C+T), [Mikko Kurimo](https://arxiv.org/search/eess?searchtype=author&query=Kurimo%2C+M)

> It is common knowledge that the quantity and quality of the training data play a significant role in the creation of a good machine learning model. In this paper, we take it one step further and demonstrate that the way the training examples are arranged is also of crucial importance. Curriculum Learning is built on the observation that organized and structured assimilation of knowledge has the ability to enable faster training and better comprehension. When humans learn to speak, they first try to utter basic phones and then gradually move towards more complex structures such as words and sentences. This methodology is known as Curriculum Learning, and we employ it in the context of Automatic Speech Recognition. We hypothesize that end-to-end models can achieve better performance when provided with an organized training set consisting of examples that exhibit an increasing level of difficulty (i.e. a curriculum). To impose structure on the training set and to define the notion of an easy example, we explored multiple scoring functions that either use feedback from an external neural network or incorporate feedback from the model itself. Empirical results show that with different curriculums we can balance the training times and the network's performance.

| Comments:    | 5 pages, 2 figures, in Proceedings Interspeech 2022          |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Audio and Speech Processing (eess.AS)**; Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD) |
| ACM classes: | I.2.7; I.2.0                                                 |
| Cite as:     | **[arXiv:2208.05782](https://arxiv.org/abs/2208.05782) [eess.AS]** |
|              | (or **[arXiv:2208.05782v1](https://arxiv.org/abs/2208.05782v1) [eess.AS]** for this version) |
|              | https://doi.org/10.48550/arXiv.2208.05782Focus to learn more |





<h2 id="2022-08-12-2">2. Language Tokens: A Frustratingly Simple Approach Improves Zero-Shot Performance of Multilingual Translation
</h2>

Title: [Language Tokens: A Frustratingly Simple Approach Improves Zero-Shot Performance of Multilingual Translation](https://arxiv.org/abs/2208.05852)

Authors: [Muhammad ElNokrashy](https://arxiv.org/search/cs?searchtype=author&query=ElNokrashy%2C+M) (1), [Amr Hendy](https://arxiv.org/search/cs?searchtype=author&query=Hendy%2C+A) (1), [Mohamed Maher](https://arxiv.org/search/cs?searchtype=author&query=Maher%2C+M) (1), [Mohamed Afify](https://arxiv.org/search/cs?searchtype=author&query=Afify%2C+M) (1), [Hany Hassan Awadalla](https://arxiv.org/search/cs?searchtype=author&query=Awadalla%2C+H+H) (2) ((1) Microsoft ATL Cairo, (2) Microsoft Redmond)

> This paper proposes a simple yet effective method to improve direct (X-to-Y) translation for both cases: zero-shot and when direct data is available. We modify the input tokens at both the encoder and decoder to include signals for the source and target languages. We show a performance gain when training from scratch, or finetuning a pretrained model with the proposed setup. In the experiments, our method shows nearly 10.0 BLEU points gain on in-house datasets depending on the checkpoint selection criteria. In a WMT evaluation campaign, From-English performance improves by 4.17 and 2.87 BLEU points, in the zero-shot setting, and when direct data is available for training, respectively. While X-to-Y improves by 1.29 BLEU over the zero-shot baseline, and 0.44 over the many-to-many baseline. In the low-resource setting, we see a 1.5~1.7 point improvement when finetuning on X-to-Y domain data.

| Comments: | 10 pages, accepted at AMTA-2022 (Association for Machine Translation in the Americas Conference) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2208.05852](https://arxiv.org/abs/2208.05852) [cs.CL]** |
|           | (or **[arXiv:2208.05852v1](https://arxiv.org/abs/2208.05852v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2208.05852Focus to learn more |





<h2 id="2022-08-12-3">3. Domain-Specific Text Generation for Machine Translation
</h2>

Title: [Domain-Specific Text Generation for Machine Translation](https://arxiv.org/abs/2208.05909)

Authors: [Yasmin Moslem](https://arxiv.org/search/cs?searchtype=author&query=Moslem%2C+Y), [Rejwanul Haque](https://arxiv.org/search/cs?searchtype=author&query=Haque%2C+R), [John D. Kelleher](https://arxiv.org/search/cs?searchtype=author&query=Kelleher%2C+J+D), [Andy Way](https://arxiv.org/search/cs?searchtype=author&query=Way%2C+A)

> Preservation of domain knowledge from the source to target is crucial in any translation workflow. It is common in the translation industry to receive highly specialized projects, where there is hardly any parallel in-domain data. In such scenarios where there is insufficient in-domain data to fine-tune Machine Translation (MT) models, producing translations that are consistent with the relevant context is challenging. In this work, we propose a novel approach to domain adaptation leveraging state-of-the-art pretrained language models (LMs) for domain-specific data augmentation for MT, simulating the domain characteristics of either (a) a small bilingual dataset, or (b) the monolingual source text to be translated. Combining this idea with back-translation, we can generate huge amounts of synthetic bilingual in-domain data for both use cases. For our investigation, we use the state-of-the-art Transformer architecture. We employ mixed fine-tuning to train models that significantly improve translation of in-domain texts. More specifically, in both scenarios, our proposed methods achieve improvements of approximately 5-6 BLEU and 2-3 BLEU, respectively, on the Arabic-to-English and English-to-Arabic language pairs. Furthermore, the outcome of human evaluation corroborates the automatic evaluation results.

| Comments: | AMTA 2022 - MT Research Track                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2208.05909](https://arxiv.org/abs/2208.05909) [cs.CL]** |
|           | (or **[arXiv:2208.05909v1](https://arxiv.org/abs/2208.05909v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2208.05909Focus to learn more |






# 2022-08-10

[Return to Index](#Index)



<h2 id="2022-08-10-1">1. Thai Wav2Vec2.0 with CommonVoice V8
</h2>

Title: [Thai Wav2Vec2.0 with CommonVoice V8](https://arxiv.org/abs/2208.04799)

Authors: [Wannaphong Phatthiyaphaibun](https://arxiv.org/search/cs?searchtype=author&query=Phatthiyaphaibun%2C+W), [Chompakorn Chaksangchaichot](https://arxiv.org/search/cs?searchtype=author&query=Chaksangchaichot%2C+C), [Peerat Limkonchotiwat](https://arxiv.org/search/cs?searchtype=author&query=Limkonchotiwat%2C+P), [Ekapol Chuangsuwanich](https://arxiv.org/search/cs?searchtype=author&query=Chuangsuwanich%2C+E), [Sarana Nutanong](https://arxiv.org/search/cs?searchtype=author&query=Nutanong%2C+S)

> Recently, Automatic Speech Recognition (ASR), a system that converts audio into text, has caught a lot of attention in the machine learning community. Thus, a lot of publicly available models were released in HuggingFace. However, most of these ASR models are available in English; only a minority of the models are available in Thai. Additionally, most of the Thai ASR models are closed-sourced, and the performance of existing open-sourced models lacks robustness. To address this problem, we train a new ASR model on a pre-trained XLSR-Wav2Vec model with the Thai CommonVoice corpus V8 and train a trigram language model to boost the performance of our ASR model. We hope that our models will be beneficial to individuals and the ASR community in Thailand.

| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2208.04799](https://arxiv.org/abs/2208.04799) [cs.CL]** |
|           | (or **[arXiv:2208.04799v1](https://arxiv.org/abs/2208.04799v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2208.04799Focus to learn more |





# 2022-08-09

[Return to Index](#Index)



<h2 id="2022-08-09-1">1. Creating Reverse Bilingual Dictionaries
</h2>

Title: [Creating Reverse Bilingual Dictionaries](https://arxiv.org/abs/2208.03863)

Authors: [Khang Nhut Lam](https://arxiv.org/search/cs?searchtype=author&query=Lam%2C+K+N), [Jugal Kalita](https://arxiv.org/search/cs?searchtype=author&query=Kalita%2C+J)

> Bilingual dictionaries are expensive resources and not many are available when one of the languages is resource-poor. In this paper, we propose algorithms for creation of new reverse bilingual dictionaries from existing bilingual dictionaries in which English is one of the two languages. Our algorithms exploit the similarity between word-concept pairs using the English Wordnet to produce reverse dictionary entries. Since our algorithms rely on available bilingual dictionaries, they are applicable to any bilingual dictionary as long as one of the two languages has Wordnet type lexical ontology.

| Comments:          | 5 pages                                                      |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**                         |
| Cite as:           | **[arXiv:2208.03863](https://arxiv.org/abs/2208.03863) [cs.CL]** |
|                    | (or **[arXiv:2208.03863v1](https://arxiv.org/abs/2208.03863v1) [cs.CL]** for this version) |
|                    | https://doi.org/10.48550/arXiv.2208.03863Focus to learn more |
| Journal reference: | Proceedings of the 2013 conference of the North American chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 524-528. 2013 |







# 2022-08-08

[Return to Index](#Index)



<h2 id="2022-08-08-1">1. Phrase translation using a bilingual dictionary and n-gram data: A case study from Vietnamese to English
</h2>

Title: [Phrase translation using a bilingual dictionary and n-gram data: A case study from Vietnamese to English](https://arxiv.org/abs/2208.03018)

Authors: [Khang Nhut Lam](https://arxiv.org/search/cs?searchtype=author&query=Lam%2C+K+N), [Feras Al Tarouti](https://arxiv.org/search/cs?searchtype=author&query=Tarouti%2C+F+A), [Jugal Kalita](https://arxiv.org/search/cs?searchtype=author&query=Kalita%2C+J)

> Past approaches to translate a phrase in a language L1 to a language L2 using a dictionary-based approach require grammar rules to restructure initial translations. This paper introduces a novel method without using any grammar rules to translate a given phrase in L1, which does not exist in the dictionary, to L2. We require at least one L1-L2 bilingual dictionary and n-gram data in L2. The average manual evaluation score of our translations is 4.29/5.00, which implies very high quality.

| Comments:          | 5 pages                                                      |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**                         |
| Cite as:           | **[arXiv:2208.03018](https://arxiv.org/abs/2208.03018) [cs.CL]** |
|                    | (or **[arXiv:2208.03018v1](https://arxiv.org/abs/2208.03018v1) [cs.CL]** for this version) |
|                    | https://doi.org/10.48550/arXiv.2208.03018Focus to learn more |
| Journal reference: | In Proceedings of the 11th Workshop on Multiword Expressions, pp. 65-69. 2015 |
| Related DOI:       | https://doi.org/10.3115/v1/w15-0911Focus to learn more       |










# 2022-08-04

[Return to Index](#Index)



<h2 id="2022-08-04-1">1. Masked Vision and Language Modeling for Multi-modal Representation Learning
</h2>

Title: [Masked Vision and Language Modeling for Multi-modal Representation Learning](https://arxiv.org/abs/2208.02131)

Authors: [Gukyeong Kwon](https://arxiv.org/search/cs?searchtype=author&query=Kwon%2C+G), [Zhaowei Cai](https://arxiv.org/search/cs?searchtype=author&query=Cai%2C+Z), [Avinash Ravichandran](https://arxiv.org/search/cs?searchtype=author&query=Ravichandran%2C+A), [Erhan Bas](https://arxiv.org/search/cs?searchtype=author&query=Bas%2C+E), [Rahul Bhotika](https://arxiv.org/search/cs?searchtype=author&query=Bhotika%2C+R), [Stefano Soatto](https://arxiv.org/search/cs?searchtype=author&query=Soatto%2C+S)

> In this paper, we study how to use masked signal modeling in vision and language (V+L) representation learning. Instead of developing masked language modeling (MLM) and masked image modeling (MIM) independently, we propose to build joint masked vision and language modeling, where the masked signal of one modality is reconstructed with the help from another modality. This is motivated by the nature of image-text paired data that both of the image and the text convey almost the same information but in different formats. The masked signal reconstruction of one modality conditioned on another modality can also implicitly learn cross-modal alignment between language tokens and image patches. Our experiments on various V+L tasks show that the proposed method not only achieves state-of-the-art performances by using a large amount of data, but also outperforms the other competitors by a significant margin in the regimes of limited training data.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2208.02131](https://arxiv.org/abs/2208.02131) [cs.CV]** |
|           | (or **[arXiv:2208.02131v1](https://arxiv.org/abs/2208.02131v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2208.02131Focus to learn more |






# 2022-07-20

[Return to Index](#Index)



<h2 id="2022-07-19-1">1. Multilingual Transformer Encoders: a Word-Level Task-Agnostic Evaluation
</h2>

Title: [Multilingual Transformer Encoders: a Word-Level Task-Agnostic Evaluation](https://arxiv.org/abs/2207.09076)

Authors: [Félix Gaschi](https://arxiv.org/search/cs?searchtype=author&query=Gaschi%2C+F), [François Plesse](https://arxiv.org/search/cs?searchtype=author&query=Plesse%2C+F), [Parisa Rastin](https://arxiv.org/search/cs?searchtype=author&query=Rastin%2C+P), [Yannick Toussaint](https://arxiv.org/search/cs?searchtype=author&query=Toussaint%2C+Y)

> Some Transformer-based models can perform cross-lingual transfer learning: those models can be trained on a specific task in one language and give relatively good results on the same task in another language, despite having been pre-trained on monolingual tasks only. But, there is no consensus yet on whether those transformer-based models learn universal patterns across languages. We propose a word-level task-agnostic method to evaluate the alignment of contextualized representations built by such models. We show that our method provides more accurate translated word pairs than previous methods to evaluate word-level alignment. And our results show that some inner layers of multilingual Transformer-based models outperform other explicitly aligned representations, and even more so according to a stricter definition of multilingual alignment.

| Comments: | accepted at IJCNN 2022                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2207.09076](https://arxiv.org/abs/2207.09076) [cs.CL]** |
|           | (or **[arXiv:2207.09076v1](https://arxiv.org/abs/2207.09076v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2207.09076Focus to learn more |



