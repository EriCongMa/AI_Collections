# MA C.'s Daily Paper Of Interest - August a., 2022

# Index

- [2022-09-07](#2022-09-07)
  - [1. Informative Language Representation Learning for Massively Multilingual Neural Machine Translation](#2022-09-07-1)
  - [2. Distilling the Knowledge of BERT for CTC-based ASR](#2022-09-07-2)
  - [3. Rare but Severe Neural Machine Translation Errors Induced by Minimal Deletion: An Empirical Study on Chinese and English](#2022-09-07-3)
  - [4. Analyzing Transformers in Embedding Space](#2022-09-07-4)
  
- [2022-09-05](#2022-09-05)
  - [1. Extend and Explain: Interpreting Very Long Language Models](#2022-09-05-1)
- [2022-08-30](#2022-08-30)
  - [1. Contrastive Audio-Language Learning for Music](#2022-08-30-1)


- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



# 2022-09-07

[Return to Index](#Index)



<h2 id="2022-09-07-1">1. Informative Language Representation Learning for Massively Multilingual Neural Machine Translation
</h2>

Title: [Informative Language Representation Learning for Massively Multilingual Neural Machine Translation](https://arxiv.org/abs/2209.01530)

Authors: [Renren Jin](https://arxiv.org/search/cs?searchtype=author&query=Jin%2C+R), [Deyi Xiong](https://arxiv.org/search/cs?searchtype=author&query=Xiong%2C+D)

> In a multilingual neural machine translation model that fully shares parameters across all languages, an artificial language token is usually used to guide translation into the desired target language. However, recent studies show that prepending language tokens sometimes fails to navigate the multilingual neural machine translation models into right translation directions, especially on zero-shot translation. To mitigate this issue, we propose two methods, language embedding embodiment and language-aware multi-head attention, to learn informative language representations to channel translation into right directions. The former embodies language embeddings into different critical switching points along the information flow from the source to the target, aiming at amplifying translation direction guiding signals. The latter exploits a matrix, instead of a vector, to represent a language in the continuous space. The matrix is chunked into multiple heads so as to learn language representations in multiple subspaces. Experiment results on two datasets for massively multilingual neural machine translation demonstrate that language-aware multi-head attention benefits both supervised and zero-shot translation and significantly alleviates the off-target translation issue. Further linguistic typology prediction experiments show that matrix-based language representations learned by our methods are capable of capturing rich linguistic typology features.

| Comments: | Accepted by COLING 2022                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2209.01530](https://arxiv.org/abs/2209.01530) [cs.CL]** |
|           | (or **[arXiv:2209.01530v1](https://arxiv.org/abs/2209.01530v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2209.01530Focus to learn more |





<h2 id="2022-09-07-2">2. Distilling the Knowledge of BERT for CTC-based ASR
</h2>

Title: [Distilling the Knowledge of BERT for CTC-based ASR](https://arxiv.org/abs/2209.02030)

Authors: [Hayato Futami](https://arxiv.org/search/cs?searchtype=author&query=Futami%2C+H), [Hirofumi Inaguma](https://arxiv.org/search/cs?searchtype=author&query=Inaguma%2C+H), [Masato Mimura](https://arxiv.org/search/cs?searchtype=author&query=Mimura%2C+M), [Shinsuke Sakai](https://arxiv.org/search/cs?searchtype=author&query=Sakai%2C+S), [Tatsuya Kawahara](https://arxiv.org/search/cs?searchtype=author&query=Kawahara%2C+T)

> Connectionist temporal classification (CTC) -based models are attractive because of their fast inference in automatic speech recognition (ASR). Language model (LM) integration approaches such as shallow fusion and rescoring can improve the recognition accuracy of CTC-based ASR by taking advantage of the knowledge in text corpora. However, they significantly slow down the inference of CTC. In this study, we propose to distill the knowledge of BERT for CTC-based ASR, extending our previous study for attention-based ASR. CTC-based ASR learns the knowledge of BERT during training and does not use BERT during testing, which maintains the fast inference of CTC. Different from attention-based models, CTC-based models make frame-level predictions, so they need to be aligned with token-level predictions of BERT for distillation. We propose to obtain alignments by calculating the most plausible CTC paths. Experimental evaluations on the Corpus of Spontaneous Japanese (CSJ) and TED-LIUM2 show that our method improves the performance of CTC-based ASR without the cost of inference speed.

| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2209.02030](https://arxiv.org/abs/2209.02030) [cs.CL]** |
|           | (or **[arXiv:2209.02030v1](https://arxiv.org/abs/2209.02030v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2209.02030Focus to learn more |





<h2 id="2022-09-07-3">3. Rare but Severe Neural Machine Translation Errors Induced by Minimal Deletion: An Empirical Study on Chinese and English
</h2>

Title: [Rare but Severe Neural Machine Translation Errors Induced by Minimal Deletion: An Empirical Study on Chinese and English](https://arxiv.org/abs/2209.02145)

Authors: [Ruikang Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+R), [Alvin Grissom II](https://arxiv.org/search/cs?searchtype=author&query=II%2C+A+G), [Duc Minh Trinh](https://arxiv.org/search/cs?searchtype=author&query=Trinh%2C+D+M)

> We examine the inducement of rare but severe errors in English-Chinese and Chinese-English in-domain neural machine translation by minimal deletion of the source text with character-based models. By deleting a single character, we find that we can induce severe errors in the translation. We categorize these errors and compare the results of deleting single characters and single words. We also examine the effect of training data size on the number and types of pathological cases induced by these minimal perturbations, finding significant variation.

| Comments: | Accepted to COLING 2022                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2209.02145](https://arxiv.org/abs/2209.02145) [cs.CL]** |
|           | (or **[arXiv:2209.02145v1](https://arxiv.org/abs/2209.02145v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2209.02145Focus to learn more |





<h2 id="2022-09-07-4">4. Analyzing Transformers in Embedding Space
</h2>

Title: [Analyzing Transformers in Embedding Space](https://arxiv.org/abs/2209.02535)

Authors: [Guy Dar](https://arxiv.org/search/cs?searchtype=author&query=Dar%2C+G), [Mor Geva](https://arxiv.org/search/cs?searchtype=author&query=Geva%2C+M), [Ankit Gupta](https://arxiv.org/search/cs?searchtype=author&query=Gupta%2C+A), [Jonathan Berant](https://arxiv.org/search/cs?searchtype=author&query=Berant%2C+J)

> Understanding Transformer-based models has attracted significant attention, as they lie at the heart of recent technological advances across machine learning. While most interpretability methods rely on running models over inputs, recent work has shown that a zero-pass approach, where parameters are interpreted directly without a forward/backward pass is feasible for some Transformer parameters, and for two-layer attention networks. In this work, we present a theoretical analysis where all parameters of a trained Transformer are interpreted by projecting them into the embedding space, that is, the space of vocabulary items they operate on. We derive a simple theoretical framework to support our arguments and provide ample evidence for its validity. First, an empirical analysis showing that parameters of both pretrained and fine-tuned models can be interpreted in embedding space. Second, we present two applications of our framework: (a) aligning the parameters of different models that share a vocabulary, and (b) constructing a classifier without training by ``translating'' the parameters of a fine-tuned classifier to parameters of a different model that was only pretrained. Overall, our findings open the door to interpretation methods that, at least in part, abstract away from model specifics and operate in the embedding space only.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2209.02535](https://arxiv.org/abs/2209.02535) [cs.CL]** |
|           | (or **[arXiv:2209.02535v1](https://arxiv.org/abs/2209.02535v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2209.02535Focus to learn more |






# 2022-09-05

[Return to Index](#Index)



<h2 id="2022-09-05-1">1. Extend and Explain: Interpreting Very Long Language Models
</h2>

Title: [Extend and Explain: Interpreting Very Long Language Models](https://arxiv.org/abs/2209.01174)

Authors: [Joel Stremmel](https://arxiv.org/search/cs?searchtype=author&query=Stremmel%2C+J), [Brian L. Hill](https://arxiv.org/search/cs?searchtype=author&query=Hill%2C+B+L), [Jeffrey Hertzberg](https://arxiv.org/search/cs?searchtype=author&query=Hertzberg%2C+J), [Jaime Murillo](https://arxiv.org/search/cs?searchtype=author&query=Murillo%2C+J), [Llewelyn Allotey](https://arxiv.org/search/cs?searchtype=author&query=Allotey%2C+L), [Eran Halperin](https://arxiv.org/search/cs?searchtype=author&query=Halperin%2C+E)

> While Transformer language models (LMs) are state-of-the-art for information extraction, long text introduces computational challenges requiring suboptimal preprocessing steps or alternative model architectures. Sparse-attention LMs can represent longer sequences, overcoming performance hurdles. However, it remains unclear how to explain predictions from these models, as not all tokens attend to each other in the self-attention layers, and long sequences pose computational challenges for explainability algorithms when runtime depends on document length. These challenges are severe in the medical context where documents can be very long, and machine learning (ML) models must be auditable and trustworthy. We introduce a novel Masked Sampling Procedure (MSP) to identify the text blocks that contribute to a prediction, apply MSP in the context of predicting diagnoses from medical text, and validate our approach with a blind review by two clinicians. Our method identifies about 1.7x more clinically informative text blocks than the previous state-of-the-art, runs up to 100x faster, and is tractable for generating important phrase pairs. MSP is particularly well-suited to long LMs but can be applied to any text classifier. We provide a general implementation of MSP.

| Comments:    | 10 pages                                                     |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**                         |
| MSC classes: | I.2.7                                                        |
| Cite as:     | **[arXiv:2209.01174](https://arxiv.org/abs/2209.01174) [cs.CL]** |
|              | (or **[arXiv:2209.01174v1](https://arxiv.org/abs/2209.01174v1) [cs.CL]** for this version) |
|              | https://doi.org/10.48550/arXiv.2209.01174Focus to learn more |






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



