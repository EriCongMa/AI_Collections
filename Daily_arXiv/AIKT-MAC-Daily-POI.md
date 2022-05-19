# MA C.'s Daily Paper Of Interest - May b., 2022

# Index

- [2022-05-19](#2022-05-19)
  - [1. Geographical Distance Is The New Hyperparameter: A Case Study Of Finding The Optimal Pre-trained Language For English-isiZulu Machine Translation](#2022-05-19-1)
  - [2. Data Augmentation to Address Out-of-Vocabulary Problem in Low-Resource Sinhala-English Neural Machine Translation](#2022-05-19-2)
  - [3. Leveraging Pseudo-labeled Data to Improve Direct Speech-to-Speech Translation](#2022-05-19-3)

- [2022-05-18](#2022-05-18)
  - [1. Towards Debiasing Translation Artifacts](#2022-05-18-1)
  - [2. When to Use Multi-Task Learning vs Intermediate Fine-Tuning for Pre-Trained Encoder Transfer Learning](#2022-05-18-2)
  - [3. Consistent Human Evaluation of Machine Translation across Language Pairs](#2022-05-18-3)

- [2022-05-17](#2022-05-17)
  - [1. Improving Neural Machine Translation of Indigenous Languages with Multilingual Transfer Learning](#2022-05-17-1)
  - [2. Multiformer: A Head-Configurable Transformer-Based Model for Direct Speech Translation](#2022-05-17-2)
  - [3. Directed Acyclic Transformer for Non-Autoregressive Machine Translation](#2022-05-17-3)

- [2022-05-16](#2022-05-16)
  - [1. An empirical study of CTC based models for OCR of Indian languages](#2022-05-16-1)
  - [2. The Devil is in the Details: On the Pitfalls of Vocabulary Selection in Neural Machine Translation](#2022-05-16-2)
  - [3. Controlling Translation Formality Using Pre-trained Multilingual Language Models](#2022-05-16-3)
  - [4. Who Are We Talking About? Handling Person Names in Speech Translation](#2022-05-16-4)

- [2022-05-13](#2022-05-13)
  - [1. Some Grammatical Errors are Frequent, Others are Important](#2022-05-13-1)



- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



# 2022-05-19

[Return to Index](#Index)



<h2 id="2022-05-19-1">1. Geographical Distance Is The New Hyperparameter: A Case Study Of Finding The Optimal Pre-trained Language For English-isiZulu Machine Translation
</h2>

Title: [Geographical Distance Is The New Hyperparameter: A Case Study Of Finding The Optimal Pre-trained Language For English-isiZulu Machine Translation](https://arxiv.org/abs/2205.08621)

Authors: [Muhammad Umair Nasir](https://arxiv.org/search/cs?searchtype=author&query=Nasir%2C+M+U), [Innocent Amos Mchechesi](https://arxiv.org/search/cs?searchtype=author&query=Mchechesi%2C+I+A)

> Stemming from the limited availability of datasets and textual resources for low-resource languages such as isiZulu, there is a significant need to be able to harness knowledge from pre-trained models to improve low resource machine translation. Moreover, a lack of techniques to handle the complexities of morphologically rich languages has compounded the unequal development of translation models, with many widely spoken African languages being left behind. This study explores the potential benefits of transfer learning in an English-isiZulu translation framework. The results indicate the value of transfer learning from closely related languages to enhance the performance of low-resource translation models, thus providing a key strategy for low-resource translation going forward. We gathered results from 8 different language corpora, including one multi-lingual corpus, and saw that isiXhosa-isiZulu outperformed all languages, with a BLEU score of 8.56 on the test set which was better from the multi-lingual corpora pre-trained model by 2.73. We also derived a new coefficient, Nasir's Geographical Distance Coefficient (NGDC) which provides an easy selection of languages for the pre-trained models. NGDC also indicated that isiXhosa should be selected as the language for the pre-trained model.

| Comments: | Accepted at NAACL 2022 Workshop MIA                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2205.08621](https://arxiv.org/abs/2205.08621) [cs.CL]** |
|           | (or **[arXiv:2205.08621v1](https://arxiv.org/abs/2205.08621v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.08621Focus to learn more |





<h2 id="2022-05-19-2">2. Data Augmentation to Address Out-of-Vocabulary Problem in Low-Resource Sinhala-English Neural Machine Translation
</h2>

Title: [Data Augmentation to Address Out-of-Vocabulary Problem in Low-Resource Sinhala-English Neural Machine Translation](https://arxiv.org/abs/2205.08722)

Authors: [Aloka Fernando](https://arxiv.org/search/cs?searchtype=author&query=Fernando%2C+A), [Surangika Ranathunga](https://arxiv.org/search/cs?searchtype=author&query=Ranathunga%2C+S)

> Out-of-Vocabulary (OOV) is a problem for Neural Machine Translation (NMT). OOV refers to words with a low occurrence in the training data, or to those that are absent from the training data. To alleviate this, word or phrase-based Data Augmentation (DA) techniques have been used. However, existing DA techniques have addressed only one of these OOV types and limit to considering either syntactic constraints or semantic constraints. We present a word and phrase replacement-based DA technique that consider both types of OOV, by augmenting (1) rare words in the existing parallel corpus, and (2) new words from a bilingual dictionary. During augmentation, we consider both syntactic and semantic properties of the words to guarantee fluency in the synthetic sentences. This technique was experimented with low resource Sinhala-English language pair. We observe with only semantic constraints in the DA, the results are comparable with the scores obtained considering syntactic constraints, and is favourable for low-resourced languages that lacks linguistic tool support. Additionally, results can be further improved by considering both syntactic and semantic constraints.

| Subjects:          | **Computation and Language (cs.CL)**                         |
| ------------------ | ------------------------------------------------------------ |
| Cite as:           | **[arXiv:2205.08722](https://arxiv.org/abs/2205.08722) [cs.CL]** |
|                    | (or **[arXiv:2205.08722v1](https://arxiv.org/abs/2205.08722v1) [cs.CL]** for this version) |
|                    | https://doi.org/10.48550/arXiv.2205.08722Focus to learn more |
| Journal reference: | Proceedings of the 35th Pacific Asia Conference on Language, Information and Computation (2021) 61-70 |





<h2 id="2022-05-19-3">3. Leveraging Pseudo-labeled Data to Improve Direct Speech-to-Speech Translation
</h2>

Title: [Leveraging Pseudo-labeled Data to Improve Direct Speech-to-Speech Translation](https://arxiv.org/abs/2205.08993)

Authors: [Qianqian Dong](https://arxiv.org/search/cs?searchtype=author&query=Dong%2C+Q), [Fengpeng Yue](https://arxiv.org/search/cs?searchtype=author&query=Yue%2C+F), [Tom Ko](https://arxiv.org/search/cs?searchtype=author&query=Ko%2C+T), [Mingxuan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+M), [Qibing Bai](https://arxiv.org/search/cs?searchtype=author&query=Bai%2C+Q), [Yu Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y)

> Direct Speech-to-speech translation (S2ST) has drawn more and more attention recently. The task is very challenging due to data scarcity and complex speech-to-speech mapping. In this paper, we report our recent achievements in S2ST. Firstly, we build a S2ST Transformer baseline which outperforms the original Translatotron. Secondly, we utilize the external data by pseudo-labeling and obtain a new state-of-the-art result on the Fisher English-to-Spanish test set. Indeed, we exploit the pseudo data with a combination of popular techniques which are not trivial when applied to S2ST. Moreover, we evaluate our approach on both syntactically similar (Spanish-English) and distant (English-Chinese) language pairs. Our implementation is available at [this https URL](https://github.com/fengpeng-yue/speech-to-speech-translation).

| Comments: | Submitted to INTERSPEECH 2022                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2205.08993](https://arxiv.org/abs/2205.08993) [cs.CL]** |
|           | (or **[arXiv:2205.08993v1](https://arxiv.org/abs/2205.08993v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.08993Focus to learn more |







# 2022-05-18

[Return to Index](#Index)



<h2 id="2022-05-18-1">1. Towards Debiasing Translation Artifacts
</h2>

Title: [Towards Debiasing Translation Artifacts](https://arxiv.org/abs/2205.08001)

Authors: [Koel Dutta Chowdhury](https://arxiv.org/search/cs?searchtype=author&query=Chowdhury%2C+K+D), [Rricha Jalota](https://arxiv.org/search/cs?searchtype=author&query=Jalota%2C+R), [Cristina España-Bonet](https://arxiv.org/search/cs?searchtype=author&query=España-Bonet%2C+C), [Josef van Genabith](https://arxiv.org/search/cs?searchtype=author&query=van+Genabith%2C+J)

> Cross-lingual natural language processing relies on translation, either by humans or machines, at different levels, from translating training data to translating test sets. However, compared to original texts in the same language, translations possess distinct qualities referred to as translationese. Previous research has shown that these translation artifacts influence the performance of a variety of cross-lingual tasks. In this work, we propose a novel approach to reducing translationese by extending an established bias-removal technique. We use the Iterative Null-space Projection (INLP) algorithm, and show by measuring classification accuracy before and after debiasing, that translationese is reduced at both sentence and word level. We evaluate the utility of debiasing translationese on a natural language inference (NLI) task, and show that by reducing this bias, NLI accuracy improves. To the best of our knowledge, this is the first study to debias translationese as represented in latent embedding space.

| Comments: | Accepted to NAACL 2022, Main Conference                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.08001](https://arxiv.org/abs/2205.08001) [cs.CL]** |
|           | (or **[arXiv:2205.08001v1](https://arxiv.org/abs/2205.08001v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.08001Focus to learn more |





<h2 id="2022-05-18-2">2. When to Use Multi-Task Learning vs Intermediate Fine-Tuning for Pre-Trained Encoder Transfer Learning
</h2>

Title: [When to Use Multi-Task Learning vs Intermediate Fine-Tuning for Pre-Trained Encoder Transfer Learning](https://arxiv.org/abs/2205.08124)

Authors: [Orion Weller](https://arxiv.org/search/cs?searchtype=author&query=Weller%2C+O), [Kevin Seppi](https://arxiv.org/search/cs?searchtype=author&query=Seppi%2C+K), [Matt Gardner](https://arxiv.org/search/cs?searchtype=author&query=Gardner%2C+M)

> Transfer learning (TL) in natural language processing (NLP) has seen a surge of interest in recent years, as pre-trained models have shown an impressive ability to transfer to novel tasks. Three main strategies have emerged for making use of multiple supervised datasets during fine-tuning: training on an intermediate task before training on the target task (STILTs), using multi-task learning (MTL) to train jointly on a supplementary task and the target task (pairwise MTL), or simply using MTL to train jointly on all available datasets (MTL-ALL). In this work, we compare all three TL methods in a comprehensive analysis on the GLUE dataset suite. We find that there is a simple heuristic for when to use one of these techniques over the other: pairwise MTL is better than STILTs when the target task has fewer instances than the supporting task and vice versa. We show that this holds true in more than 92% of applicable cases on the GLUE dataset and validate this hypothesis with experiments varying dataset size. The simplicity and effectiveness of this heuristic is surprising and warrants additional exploration by the TL community. Furthermore, we find that MTL-ALL is worse than the pairwise methods in almost every case. We hope this study will aid others as they choose between TL methods for NLP tasks.

| Comments: | ACL 2022                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.08124](https://arxiv.org/abs/2205.08124) [cs.CL]** |
|           | (or **[arXiv:2205.08124v1](https://arxiv.org/abs/2205.08124v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.08124Focus to learn more |





<h2 id="2022-05-18-3">3. Consistent Human Evaluation of Machine Translation across Language Pairs
</h2>

Title: [Consistent Human Evaluation of Machine Translation across Language Pairs](https://arxiv.org/abs/2205.08533)

Authors: [Daniel Licht](https://arxiv.org/search/cs?searchtype=author&query=Licht%2C+D), [Cynthia Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+C), [Janice Lam](https://arxiv.org/search/cs?searchtype=author&query=Lam%2C+J), [Francisco Guzman](https://arxiv.org/search/cs?searchtype=author&query=Guzman%2C+F), [Mona Diab](https://arxiv.org/search/cs?searchtype=author&query=Diab%2C+M), [Philipp Koehn](https://arxiv.org/search/cs?searchtype=author&query=Koehn%2C+P)

> Obtaining meaningful quality scores for machine translation systems through human evaluation remains a challenge given the high variability between human evaluators, partly due to subjective expectations for translation quality for different language pairs. We propose a new metric called XSTS that is more focused on semantic equivalence and a cross-lingual calibration method that enables more consistent assessment. We demonstrate the effectiveness of these novel contributions in large scale evaluation studies across up to 14 language pairs, with translation both into and out of English.

| Comments: | 10 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.08533](https://arxiv.org/abs/2205.08533) [cs.CL]** |
|           | (or **[arXiv:2205.08533v1](https://arxiv.org/abs/2205.08533v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.08533Focus to learn more |







# 2022-05-17

[Return to Index](#Index)



<h2 id="2022-05-17-1">1. Improving Neural Machine Translation of Indigenous Languages with Multilingual Transfer Learning
</h2>

Title: [Improving Neural Machine Translation of Indigenous Languages with Multilingual Transfer Learning](https://arxiv.org/abs/2205.06993)

Authors: [Wei-Rui Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+W), [Muhammad Abdul-Mageed](https://arxiv.org/search/cs?searchtype=author&query=Abdul-Mageed%2C+M)

> Machine translation (MT) involving Indigenous languages, including those possibly endangered, is challenging due to lack of sufficient parallel data. We describe an approach exploiting bilingual and multilingual pretrained MT models in a transfer learning setting to translate from Spanish to ten South American Indigenous languages. Our models set new SOTA on five out of the ten language pairs we consider, even doubling performance on one of these five pairs. Unlike previous SOTA that perform data augmentation to enlarge the train sets, we retain the low-resource setting to test the effectiveness of our models under such a constraint. In spite of the rarity of linguistic information available about the Indigenous languages, we offer a number of quantitative and qualitative analyses (e.g., as to morphology, tokenization, and orthography) to contextualize our results.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.06993](https://arxiv.org/abs/2205.06993) [cs.CL]** |
|           | (or **[arXiv:2205.06993v1](https://arxiv.org/abs/2205.06993v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.06993Focus to learn more |





<h2 id="2022-05-17-2">2. Multiformer: A Head-Configurable Transformer-Based Model for Direct Speech Translation
</h2>

Title: [Multiformer: A Head-Configurable Transformer-Based Model for Direct Speech Translation](https://arxiv.org/abs/2205.07100)

Authors: [Gerard Sant](https://arxiv.org/search/cs?searchtype=author&query=Sant%2C+G), [Gerard I. Gállego](https://arxiv.org/search/cs?searchtype=author&query=Gállego%2C+G+I), [Belen Alastruey](https://arxiv.org/search/cs?searchtype=author&query=Alastruey%2C+B), [Marta R. Costa-Jussà](https://arxiv.org/search/cs?searchtype=author&query=Costa-Jussà%2C+M+R)

> Transformer-based models have been achieving state-of-the-art results in several fields of Natural Language Processing. However, its direct application to speech tasks is not trivial. The nature of this sequences carries problems such as long sequence lengths and redundancy between adjacent tokens. Therefore, we believe that regular self-attention mechanism might not be well suited for it. 
> Different approaches have been proposed to overcome these problems, such as the use of efficient attention mechanisms. However, the use of these methods usually comes with a cost, which is a performance reduction caused by information loss. In this study, we present the Multiformer, a Transformer-based model which allows the use of different attention mechanisms on each head. By doing this, the model is able to bias the self-attention towards the extraction of more diverse token interactions, and the information loss is reduced. Finally, we perform an analysis of the head contributions, and we observe that those architectures where all heads relevance is uniformly distributed obtain better results. Our results show that mixing attention patterns along the different heads and layers outperforms our baseline by up to 0.7 BLEU.

| Comments: | NAACL-SRW 2022                                               |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Multimedia (cs.MM); Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2205.07100](https://arxiv.org/abs/2205.07100) [cs.CL]** |
|           | (or **[arXiv:2205.07100v1](https://arxiv.org/abs/2205.07100v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.07100Focus to learn more |







<h2 id="2022-05-17-3">3. Directed Acyclic Transformer for Non-Autoregressive Machine Translation
</h2>

Title: [Directed Acyclic Transformer for Non-Autoregressive Machine Translation](https://arxiv.org/abs/2205.07459)

Authors: [Fei Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+F), [Hao Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+H), [Yang Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y), [Hang Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+H), [Minlie Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+M)

> Non-autoregressive Transformers (NATs) significantly reduce the decoding latency by generating all tokens in parallel. However, such independent predictions prevent NATs from capturing the dependencies between the tokens for generating multiple possible translations. In this paper, we propose Directed Acyclic Transfomer (DA-Transformer), which represents the hidden states in a Directed Acyclic Graph (DAG), where each path of the DAG corresponds to a specific translation. The whole DAG simultaneously captures multiple translations and facilitates fast predictions in a non-autoregressive fashion. Experiments on the raw training data of WMT benchmark show that DA-Transformer substantially outperforms previous NATs by about 3 BLEU on average, which is the first NAT model that achieves competitive results with autoregressive Transformers without relying on knowledge distillation.

| Comments: | accepted at ICML2022                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.07459](https://arxiv.org/abs/2205.07459) [cs.CL]** |
|           | (or **[arXiv:2205.07459v1](https://arxiv.org/abs/2205.07459v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.07459Focus to learn more |





# 2022-05-16

[Return to Index](#Index)



<h2 id="2022-05-16-1">1. An empirical study of CTC based models for OCR of Indian languages
</h2>

Title: [An empirical study of CTC based models for OCR of Indian languages](https://arxiv.org/abs/2205.06740)

Authors: [Minesh Mathew](https://arxiv.org/search/cs?searchtype=author&query=Mathew%2C+M), [CV Jawahar](https://arxiv.org/search/cs?searchtype=author&query=Jawahar%2C+C)

> Recognition of text on word or line images, without the need for sub-word segmentation has become the mainstream of research and development of text recognition for Indian languages. Modelling unsegmented sequences using Connectionist Temporal Classification (CTC) is the most commonly used approach for segmentation-free OCR. In this work we present a comprehensive empirical study of various neural network models that uses CTC for transcribing step-wise predictions in the neural network output to a Unicode sequence. The study is conducted for 13 Indian languages, using an internal dataset that has around 1000 pages per language. We study the choice of line vs word as the recognition unit, and use of synthetic data to train the models. We compare our models with popular publicly available OCR tools for end-to-end document image recognition. Our end-to-end pipeline that employ our recognition models and existing text segmentation tools outperform these public OCR tools for 8 out of the 13 languages. We also introduce a new public dataset called Mozhi for word and line recognition in Indian language. The dataset contains more than 1.2 million annotated word images (120 thousand text lines) across 13 Indian languages. Our code, trained models and the Mozhi dataset will be made available at [this http URL](http://cvit.iiit.ac.in/research/projects/cvit-projects/)

| Comments: | work in progress                                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2205.06740](https://arxiv.org/abs/2205.06740) [cs.CV]** |
|           | (or **[arXiv:2205.06740v1](https://arxiv.org/abs/2205.06740v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.06740Focus to learn more |





<h2 id="2022-05-16-2">2. The Devil is in the Details: On the Pitfalls of Vocabulary Selection in Neural Machine Translation
</h2>

Title: [The Devil is in the Details: On the Pitfalls of Vocabulary Selection in Neural Machine Translation](https://arxiv.org/abs/2205.06618)

Authors: [Tobias Domhan](https://arxiv.org/search/cs?searchtype=author&query=Domhan%2C+T), [Eva Hasler](https://arxiv.org/search/cs?searchtype=author&query=Hasler%2C+E), [Ke Tran](https://arxiv.org/search/cs?searchtype=author&query=Tran%2C+K), [Sony Trenous](https://arxiv.org/search/cs?searchtype=author&query=Trenous%2C+S), [Bill Byrne](https://arxiv.org/search/cs?searchtype=author&query=Byrne%2C+B), [Felix Hieber](https://arxiv.org/search/cs?searchtype=author&query=Hieber%2C+F)

> Vocabulary selection, or lexical shortlisting, is a well-known technique to improve latency of Neural Machine Translation models by constraining the set of allowed output words during inference. The chosen set is typically determined by separately trained alignment model parameters, independent of the source-sentence context at inference time. While vocabulary selection appears competitive with respect to automatic quality metrics in prior work, we show that it can fail to select the right set of output words, particularly for semantically non-compositional linguistic phenomena such as idiomatic expressions, leading to reduced translation quality as perceived by humans. Trading off latency for quality by increasing the size of the allowed set is often not an option in real-world scenarios. We propose a model of vocabulary selection, integrated into the neural translation model, that predicts the set of allowed output words from contextualized encoder representations. This restores translation quality of an unconstrained system, as measured by human evaluations on WMT newstest2020 and idiomatic expressions, at an inference latency competitive with alignment-based selection using aggressive thresholds, thereby removing the dependency on separately trained alignment models.

| Comments: | NAACL 2022                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2205.06618](https://arxiv.org/abs/2205.06618) [cs.CL]** |
|           | (or **[arXiv:2205.06618v1](https://arxiv.org/abs/2205.06618v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.06618Focus to learn more |





<h2 id="2022-05-16-3">3. Controlling Translation Formality Using Pre-trained Multilingual Language Models
</h2>

Title: [Controlling Translation Formality Using Pre-trained Multilingual Language Models](https://arxiv.org/abs/2205.06644)

Authors: [Elijah Rippeth](https://arxiv.org/search/cs?searchtype=author&query=Rippeth%2C+E), [Sweta Agrawal](https://arxiv.org/search/cs?searchtype=author&query=Agrawal%2C+S), [Marine Carpuat](https://arxiv.org/search/cs?searchtype=author&query=Carpuat%2C+M)

> This paper describes the University of Maryland's submission to the Special Task on Formality Control for Spoken Language Translation at \iwslt, which evaluates translation from English into 6 languages with diverse grammatical formality markers. We investigate to what extent this problem can be addressed with a \textit{single multilingual model}, simultaneously controlling its output for target language and formality. Results show that this strategy can approach the translation quality and formality control achieved by dedicated translation models. However, the nature of the underlying pre-trained language model and of the finetuning samples greatly impact results.

| Comments: | 9 pages, 2 figures, IWSLT22 camera-ready (system paper @ ACL-IWSLT Shared Task on Formality Control for Spoken Language Translation) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.06644](https://arxiv.org/abs/2205.06644) [cs.CL]** |
|           | (or **[arXiv:2205.06644v1](https://arxiv.org/abs/2205.06644v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.06644Focus to learn more |





<h2 id="2022-05-16-4">4. Who Are We Talking About? Handling Person Names in Speech Translation
</h2>

Title: [Who Are We Talking About? Handling Person Names in Speech Translation](https://arxiv.org/abs/2205.06755)

Authors: [Marco Gaido](https://arxiv.org/search/cs?searchtype=author&query=Gaido%2C+M), [Matteo Negri](https://arxiv.org/search/cs?searchtype=author&query=Negri%2C+M), [Marco Turchi](https://arxiv.org/search/cs?searchtype=author&query=Turchi%2C+M)

> Recent work has shown that systems for speech translation (ST) -- similarly to automatic speech recognition (ASR) -- poorly handle person names. This shortcoming does not only lead to errors that can seriously distort the meaning of the input, but also hinders the adoption of such systems in application scenarios (like computer-assisted interpreting) where the translation of named entities, like person names, is crucial. In this paper, we first analyse the outputs of ASR/ST systems to identify the reasons of failures in person name transcription/translation. Besides the frequency in the training data, we pinpoint the nationality of the referred person as a key factor. We then mitigate the problem by creating multilingual models, and further improve our ST systems by forcing them to jointly generate transcripts and translations, prioritising the former over the latter. Overall, our solutions result in a relative improvement in token-level person name accuracy by 47.8% on average for three language pairs (en->es,fr,it).

| Comments: | Accepted at IWSLT2022                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.06755](https://arxiv.org/abs/2205.06755) [cs.CL]** |
|           | (or **[arXiv:2205.06755v1](https://arxiv.org/abs/2205.06755v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.06755Focus to learn more |






# 2022-05-13

[Return to Index](#Index)



<h2 id="2022-05-13-1">1. Some Grammatical Errors are Frequent, Others are Important
</h2>

Title: [Some Grammatical Errors are Frequent, Others are Important](https://arxiv.org/abs/2205.05730)

Authors: [Leshem Choshen](https://arxiv.org/search/cs?searchtype=author&query=Choshen%2C+L), [Ofir Shifman](https://arxiv.org/search/cs?searchtype=author&query=Shifman%2C+O), [Omri Abend](https://arxiv.org/search/cs?searchtype=author&query=Abend%2C+O)

> In Grammatical Error Correction, systems are evaluated by the number of errors they correct. However, no one has assessed whether all error types are equally important. We provide and apply a method to quantify the importance of different grammatical error types to humans. We show that some rare errors are considered disturbing while other common ones are not. This affects possible directions to improve both systems and their evaluation.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Computers and Society (cs.CY) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.05730](https://arxiv.org/abs/2205.05730) [cs.CL]** |
|           | (or **[arXiv:2205.05730v1](https://arxiv.org/abs/2205.05730v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.05730Focus to learn more |

