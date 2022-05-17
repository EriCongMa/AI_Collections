# MA C.'s Daily Paper Of Interest - May b., 2022

# Index

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

