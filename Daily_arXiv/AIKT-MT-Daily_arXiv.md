# Daily arXiv: Machine Translation - October, 2021

# Index


- [2021-10-06](#2021-10-06)

  - [1. OPAD: An Optimized Policy-based Active Learning Framework for Document Content Analysis](#2021-10-06-1)
  - [2. Rerunning OCR -- A Machine Learning Approach to Quality Assessment and Enhancement Prediction](#2021-10-06-2)
  - [3. On the Complementarity between Pre-Training and Back-Translation for Neural Machine Translation](#2021-10-06-3)
  - [4. Data Augmentation Approaches in Natural Language Processing: A Survey](#2021-10-06-4)
  - [5. Sicilian Translator: A Recipe for Low-Resource NMT](#2021-10-06-5)
  - [6. Transfer Learning for Multi-lingual Tasks -- a Survey](#2021-10-06-6)
  - [7. Structured Prediction in NLP -- A survey](#2021-10-06-7)
  - [8. Interactively Generating Explanations for Transformer-based Language Models](#2021-10-06-8)
- [2021-10-05](#2021-10-05)

  - [1. Improving Zero-shot Multilingual Neural Machine Translation for Low-Resource Languages](#2021-10-05-1)
- [2021-10-04](#2021-10-04)

  - [1. Improving Punctuation Restoration for Speech Transcripts via External Data](#2021-10-04-1)
  - [2. A Survey of Knowledge Enhanced Pre-trained Models](#2021-10-04-2)
  - [3. Attention based Sequence to Sequence Learning for Machine Translation of Low Resourced Indic Languages -- A case of Sanskrit to Hindi](#2021-10-04-3)
- [2021-10-01](#2021-10-01)
  - [1. Phonetic Word Embeddings](#2021-10-01-1)
  - [2. Improved statistical machine translation using monolingual paraphrases](#2021-10-01-2)
- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MT-Daily_arXiv-index.md)



# 2021-10-06

[Return to Index](#Index)



<h2 id="2021-10-06-1">1. OPAD: An Optimized Policy-based Active Learning Framework for Document Content Analysis
</h2>

Title: [OPAD: An Optimized Policy-based Active Learning Framework for Document Content Analysis](https://arxiv.org/abs/2110.02069)

Authors: [Sumit Shekhar](https://arxiv.org/search/cs?searchtype=author&query=Shekhar%2C+S), [Bhanu Prakash Reddy Guda](https://arxiv.org/search/cs?searchtype=author&query=Guda%2C+B+P+R), [Ashutosh Chaubey](https://arxiv.org/search/cs?searchtype=author&query=Chaubey%2C+A), [Ishan Jindal](https://arxiv.org/search/cs?searchtype=author&query=Jindal%2C+I), [Avanish Jain](https://arxiv.org/search/cs?searchtype=author&query=Jain%2C+A)

> Documents are central to many business systems, and include forms, reports, contracts, invoices or purchase orders. The information in documents is typically in natural language, but can be organized in various layouts and formats. There have been recent spurt of interest in understanding document content with novel deep learning architectures. However, document understanding tasks need dense information annotations, which are costly to scale and generalize. Several active learning techniques have been proposed to reduce the overall budget of annotation while maintaining the performance of the underlying deep learning model. However, most of these techniques work only for classification problems. But content detection is a more complex task, and has been scarcely explored in active learning literature. In this paper, we propose \textit{OPAD}, a novel framework using reinforcement policy for active learning in content detection tasks for documents. The proposed framework learns the acquisition function to decide the samples to be selected while optimizing performance metrics that the tasks typically have. Furthermore, we extend to weak labelling scenarios to further reduce the cost of annotation significantly. We propose novel rewards to account for class imbalance and user feedback in the annotation interface, to improve the active learning method. We show superior performance of the proposed \textit{OPAD} framework for active learning for various tasks related to document understanding like layout parsing, object detection and named entity recognition. Ablation studies for human feedback and class imbalance rewards are presented, along with a comparison of annotation times for different approaches.

| Subjects: | **Information Retrieval (cs.IR)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2110.02069](https://arxiv.org/abs/2110.02069) [cs.IR]** |
|           | (or **[arXiv:2110.02069v1](https://arxiv.org/abs/2110.02069v1) [cs.IR]** for this version) |





<h2 id="2021-10-06-2">2. Rerunning OCR -- A Machine Learning Approach to Quality Assessment and Enhancement Prediction
</h2>

Title: [Rerunning OCR -- A Machine Learning Approach to Quality Assessment and Enhancement Prediction](https://arxiv.org/abs/2110.01661)

Authors: [Pit Schneider](https://arxiv.org/search/cs?searchtype=author&query=Schneider%2C+P)

> Iterating with new and improved OCR solutions enforces decisions to be taken when it comes to targeting the right reprocessing candidates. This especially applies when the underlying data collection is of considerable size and rather diverse in terms of fonts, languages, periods of publication and consequently OCR quality. This article captures the efforts of the National Library of Luxembourg to support those exact decisions. They are crucial in order to guarantee low computational overhead and reduced quality degradation risks, combined with a more quantifiable OCR improvement. In particular, this work explains the methodology of the library with respect to text block level quality assessment. As an extension of this technique, another contribution comes in the form of a regression model that takes the enhancement potential of a new OCR engine into account. They both mark promising approaches, especially for cultural institutions dealing with historic data of lower quality.

| Comments:    | Journal of Data Mining and Digital Humanities                |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| ACM classes: | I.2.7                                                        |
| Cite as:     | **[arXiv:2110.01661](https://arxiv.org/abs/2110.01661) [cs.CL]** |
|              | (or **[arXiv:2110.01661v1](https://arxiv.org/abs/2110.01661v1) [cs.CL]** for this version) |





<h2 id="2021-10-06-3">3. On the Complementarity between Pre-Training and Back-Translation for Neural Machine Translation
</h2>

Title: [On the Complementarity between Pre-Training and Back-Translation for Neural Machine Translation](https://arxiv.org/abs/2110.01811)

Authors: [Xuebo Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+X), [Longyue Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L), [Derek F. Wong](https://arxiv.org/search/cs?searchtype=author&query=Wong%2C+D+F), [Liang Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+L), [Lidia S. Chao](https://arxiv.org/search/cs?searchtype=author&query=Chao%2C+L+S), [Shuming Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+S), [Zhaopeng Tu](https://arxiv.org/search/cs?searchtype=author&query=Tu%2C+Z)

> Pre-training (PT) and back-translation (BT) are two simple and powerful methods to utilize monolingual data for improving the model performance of neural machine translation (NMT). This paper takes the first step to investigate the complementarity between PT and BT. We introduce two probing tasks for PT and BT respectively and find that PT mainly contributes to the encoder module while BT brings more benefits to the decoder. Experimental results show that PT and BT are nicely complementary to each other, establishing state-of-the-art performances on the WMT16 English-Romanian and English-Russian benchmarks. Through extensive analyses on sentence originality and word frequency, we also demonstrate that combining Tagged BT with PT is more helpful to their complementarity, leading to better translation quality. Source code is freely available at [this https URL](https://github.com/SunbowLiu/PTvsBT).

| Comments: | Accepted to Findings of EMNLP 2021                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2110.01811](https://arxiv.org/abs/2110.01811) [cs.CL]** |
|           | (or **[arXiv:2110.01811v1](https://arxiv.org/abs/2110.01811v1) [cs.CL]** for this version) |





<h2 id="2021-10-06-4">4. Data Augmentation Approaches in Natural Language Processing: A Survey
</h2>

Title: [Data Augmentation Approaches in Natural Language Processing: A Survey](https://arxiv.org/abs/2110.01852)

Authors: [Bohan Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+B), [Yutai Hou](https://arxiv.org/search/cs?searchtype=author&query=Hou%2C+Y), [Wanxiang Che](https://arxiv.org/search/cs?searchtype=author&query=Che%2C+W)

> As an effective strategy, data augmentation (DA) alleviates data scarcity scenarios where deep learning techniques may fail. It is widely applied in computer vision then introduced to natural language processing and achieves improvements in many tasks. One of the main focuses of the DA methods is to improve the diversity of training data, thereby helping the model to better generalize to unseen testing data. In this survey, we frame DA methods into three categories based on the diversity of augmented data, including paraphrasing, noising, and sampling. Our paper sets out to analyze DA methods in detail according to the above categories. Further, we also introduce their applications in NLP tasks as well as the challenges.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2110.01852](https://arxiv.org/abs/2110.01852) [cs.CL]** |
|           | (or **[arXiv:2110.01852v1](https://arxiv.org/abs/2110.01852v1) [cs.CL]** for this version) |





<h2 id="2021-10-06-5">5. Sicilian Translator: A Recipe for Low-Resource NMT
</h2>

Title: [Sicilian Translator: A Recipe for Low-Resource NMT](https://arxiv.org/abs/2110.01938)

Authors: [Eryk Wdowiak](https://arxiv.org/search/cs?searchtype=author&query=Wdowiak%2C+E)

> With 17,000 pairs of Sicilian-English translated sentences, Arba Sicula developed the first neural machine translator for the Sicilian language. Using small subword vocabularies, we trained small Transformer models with high dropout parameters and achieved BLEU scores in the upper 20s. Then we supplemented our dataset with backtranslation and multilingual translation and pushed our scores into the mid 30s. We also attribute our success to incorporating theoretical information in our dataset. Prior to training, we biased the subword vocabulary towards the desinences one finds in a textbook. And we included textbook exercises in our dataset.

| Comments:    | 7 pages, 2 tables                                            |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**                         |
| ACM classes: | I.2.7                                                        |
| Cite as:     | **[arXiv:2110.01938](https://arxiv.org/abs/2110.01938) [cs.CL]** |
|              | (or **[arXiv:2110.01938v1](https://arxiv.org/abs/2110.01938v1) [cs.CL]** for this version) |





<h2 id="2021-10-06-6">6. Transfer Learning for Multi-lingual Tasks -- a Survey
</h2>

Title: [Transfer Learning for Multi-lingual Tasks -- a Survey](https://arxiv.org/abs/2110.02052)

Authors: [Amir Reza Jafari](https://arxiv.org/search/cs?searchtype=author&query=Jafari%2C+A+R), [Behnam Heidary](https://arxiv.org/search/cs?searchtype=author&query=Heidary%2C+B), [Reza Farahbakhsh](https://arxiv.org/search/cs?searchtype=author&query=Farahbakhsh%2C+R), [Mostafa Salehi](https://arxiv.org/search/cs?searchtype=author&query=Salehi%2C+M), [Mahdi Jalili](https://arxiv.org/search/cs?searchtype=author&query=Jalili%2C+M)

> These days different platforms such as social media provide their clients from different backgrounds and languages the possibility to connect and exchange information. It is not surprising anymore to see comments from different languages in posts published by international celebrities or data providers. In this era, understanding cross languages content and multilingualism in natural language processing (NLP) are hot topics, and multiple efforts have tried to leverage existing technologies in NLP to tackle this challenging research problem. In this survey, we provide a comprehensive overview of the existing literature with a focus on transfer learning techniques in multilingual tasks. We also identify potential opportunities for further research in this domain.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2110.02052](https://arxiv.org/abs/2110.02052) [cs.CL]** |
|           | (or **[arXiv:2110.02052v1](https://arxiv.org/abs/2110.02052v1) [cs.CL]** for this version) |





<h2 id="2021-10-06-7">7. Structured Prediction in NLP -- A survey
</h2>

Title: [Structured Prediction in NLP -- A survey](https://arxiv.org/abs/2110.02057)

Authors: [Chauhan Dev](https://arxiv.org/search/cs?searchtype=author&query=Dev%2C+C), [Naman Biyani](https://arxiv.org/search/cs?searchtype=author&query=Biyani%2C+N), [Nirmal P. Suthar](https://arxiv.org/search/cs?searchtype=author&query=Suthar%2C+N+P), [Prashant Kumar](https://arxiv.org/search/cs?searchtype=author&query=Kumar%2C+P), [Priyanshu Agarwal](https://arxiv.org/search/cs?searchtype=author&query=Agarwal%2C+P)

> Over the last several years, the field of Structured prediction in NLP has had seen huge advancements with sophisticated probabilistic graphical models, energy-based networks, and its combination with deep learning-based approaches. This survey provides a brief of major techniques in structured prediction and its applications in the NLP domains like parsing, sequence labeling, text generation, and sequence to sequence tasks. We also deep-dived into energy-based and attention-based techniques in structured prediction, identified some relevant open issues and gaps in the current state-of-the-art research, and have come up with some detailed ideas for future research in these fields.

| Comments: | 6 pages, 0 figures                                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2110.02057](https://arxiv.org/abs/2110.02057) [cs.CL]** |
|           | (or **[arXiv:2110.02057v1](https://arxiv.org/abs/2110.02057v1) [cs.CL]** for this version) |





<h2 id="2021-10-06-8">8. Interactively Generating Explanations for Transformer-based Language Models
</h2>

Title: [Interactively Generating Explanations for Transformer-based Language Models](https://arxiv.org/abs/2110.02058)

Authors: [Patrick Schramowski](https://arxiv.org/search/cs?searchtype=author&query=Schramowski%2C+P), [Felix Friedrich](https://arxiv.org/search/cs?searchtype=author&query=Friedrich%2C+F), [Christopher Tauchmann](https://arxiv.org/search/cs?searchtype=author&query=Tauchmann%2C+C), [Kristian Kersting](https://arxiv.org/search/cs?searchtype=author&query=Kersting%2C+K)

> Transformer language models are state-of-the-art in a multitude of NLP tasks. Despite these successes, their opaqueness remains problematic. Recent methods aiming to provide interpretability and explainability to black-box models primarily focus on post-hoc explanations of (sometimes spurious) input-output correlations. Instead, we emphasize using prototype networks directly incorporated into the model architecture and hence explain the reasoning process behind the network's decisions. Moreover, while our architecture performs on par with several language models, it enables one to learn from user interactions. This not only offers a better understanding of language models, but uses human capabilities to incorporate knowledge outside of the rigid range of purely data-driven approaches.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2110.02058](https://arxiv.org/abs/2110.02058) [cs.CL]** |
|           | (or **[arXiv:2110.02058v1](https://arxiv.org/abs/2110.02058v1) [cs.CL]** for this version) |









# 2021-10-05

[Return to Index](#Index)



<h2 id="2021-10-05-1">1. Improving Zero-shot Multilingual Neural Machine Translation for Low-Resource Languages
</h2>

Title: [Improving Zero-shot Multilingual Neural Machine Translation for Low-Resource Languages](https://arxiv.org/abs/2110.00712)

Authors: [Chenyang Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+C), [Gongxu Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+G)

> Although the multilingual Neural Machine Translation(NMT), which extends Google's multilingual NMT, has ability to perform zero-shot translation and the iterative self-learning algorithm can improve the quality of zero-shot translation, it confronts with two problems: the multilingual NMT model is prone to generate wrong target language when implementing zero-shot translation; the self-learning algorithm, which uses beam search to generate synthetic parallel data, demolishes the diversity of the generated source language and amplifies the impact of the same noise during the iterative learning process. In this paper, we propose the tagged-multilingual NMT model and improve the self-learning algorithm to handle these two problems. Firstly, we extend the Google's multilingual NMT model and add target tokens to the target languages, which associates the start tag with the target language to ensure that the source language can be translated to the required target language. Secondly, we improve the self-learning algorithm by replacing beam search with random sample to increases the diversity of the generated data and makes it properly cover the true data distribution. Experimental results on IWSLT show that the adjusted tagged-multilingual NMT separately obtains 9.41 and 7.85 BLEU scores over the multilingual NMT on 2010 and 2017 Romanian-Italian test sets. Similarly, it obtains 9.08 and 7.99 BLEU scores on Italian-Romanian zero-shot translation. Furthermore, the improved self-learning algorithm shows its superiorities over the conventional self-learning algorithm on zero-shot translations.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2110.00712](https://arxiv.org/abs/2110.00712) [cs.CL]** |
|           | (or **[arXiv:2110.00712v1](https://arxiv.org/abs/2110.00712v1) [cs.CL]** for this version) |









# 2021-10-04

[Return to Index](#Index)



<h2 id="2021-10-04-1">1. Improving Punctuation Restoration for Speech Transcripts via External Data
</h2>

Title: [Improving Punctuation Restoration for Speech Transcripts via External Data](https://arxiv.org/abs/2110.00560)

Authors:[Xue-Yong Fu](https://arxiv.org/search/cs?searchtype=author&query=Fu%2C+X), [Cheng Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+C), [Md Tahmid Rahman Laskar](https://arxiv.org/search/cs?searchtype=author&query=Laskar%2C+M+T+R), [Shashi Bhushan TN](https://arxiv.org/search/cs?searchtype=author&query=TN%2C+S+B), [Simon Corston-Oliver](https://arxiv.org/search/cs?searchtype=author&query=Corston-Oliver%2C+S)

> Automatic Speech Recognition (ASR) systems generally do not produce punctuated transcripts. To make transcripts more readable and follow the expected input format for downstream language models, it is necessary to add punctuation marks. In this paper, we tackle the punctuation restoration problem specifically for the noisy text (e.g., phone conversation scenarios). To leverage the available written text datasets, we introduce a data sampling technique based on an n-gram language model to sample more training data that are similar to our in-domain data. Moreover, we propose a two-stage fine-tuning approach that utilizes the sampled external data as well as our in-domain dataset for models based on BERT. Extensive experiments show that the proposed approach outperforms the baseline with an improvement of 1:12% F1 score.

| Comments: | Accepted by W-NUT at EMNLP 2021                              |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2110.00560](https://arxiv.org/abs/2110.00560) [cs.CL]** |
|           | (or **[arXiv:2110.00560v1](https://arxiv.org/abs/2110.00560v1) [cs.CL]** for this version) |





<h2 id="2021-10-04-2">2. A Survey of Knowledge Enhanced Pre-trained Models
</h2>

Title: [A Survey of Knowledge Enhanced Pre-trained Models](https://arxiv.org/abs/2110.00269)

Authors:[Jian Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+J), [Gang Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+G), [Yulong Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+Y), [Wei Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+W), [Xinyu Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+X), [Ying Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y), [Jinghui Peng](https://arxiv.org/search/cs?searchtype=author&query=Peng%2C+J)

> Pre-trained models learn contextualized word representations on large-scale text corpus through a self-supervised learning method, which has achieved promising performance after fine-tuning. These models, however, suffer from poor robustness and lack of interpretability. Pre-trained models with knowledge injection, which we call knowledge enhanced pre-trained models (KEPTMs), possess deep understanding and logical reasoning and introduce interpretability to some extent. In this survey, we provide a comprehensive overview of KEPTMs for natural language processing. We first introduce the progress of pre-trained models and knowledge representation learning. Then we systematically categorize existing KEPTMs from three different perspectives. Finally, we outline some potential directions of KEPTMs for future research.

| Comments: | 16 pages, 11 figures                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2110.00269](https://arxiv.org/abs/2110.00269) [cs.CL]** |
|           | (or **[arXiv:2110.00269v1](https://arxiv.org/abs/2110.00269v1) [cs.CL]** for this version) |





<h2 id="2021-10-04-3">3. Attention based Sequence to Sequence Learning for Machine Translation of Low Resourced Indic Languages -- A case of Sanskrit to Hindi
</h2>

Title: [Attention based Sequence to Sequence Learning for Machine Translation of Low Resourced Indic Languages -- A case of Sanskrit to Hindi](https://arxiv.org/abs/2110.00435)

Authors:[Vishvajit Bakarola](https://arxiv.org/search/cs?searchtype=author&query=Bakarola%2C+V), [Jitendra Nasriwala](https://arxiv.org/search/cs?searchtype=author&query=Nasriwala%2C+J)

> Deep Learning techniques are powerful in mimicking humans in a particular set of problems. They have achieved a remarkable performance in complex learning tasks. Deep learning inspired Neural Machine Translation (NMT) is a proficient technique that outperforms traditional machine translation. Performing machine-aided translation on Indic languages has always been a challenging task considering their rich and diverse grammar. The neural machine translation has shown quality results compared to the traditional machine translation approaches. The fully automatic machine translation becomes problematic when it comes to low-resourced languages, especially with Sanskrit. This paper presents attention mechanism based neural machine translation by selectively focusing on a particular part of language sentences during translation. The work shows the construction of Sanskrit to Hindi bilingual parallel corpus with nearly 10K samples and having 178,000 tokens. The neural translation model equipped with an attention mechanism has been trained on Sanskrit to Hindi parallel corpus. The approach has shown the significance of attention mechanisms to overcome long-term dependencies, primarily associated with low resources Indic languages. The paper shows the attention plots on testing data to demonstrate the alignment between source and translated words. For the evaluation of the translated sentences, manual score based human evaluation and automatic evaluation metric based techniques have been adopted. The attention mechanism based neural translation has achieved 88% accuracy in human evaluation and a BLEU score of 0.92 on Sanskrit to Hindi translation.

| Comments:          | Published with International Journal of Engineering Trends and Technology (IJETT) |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Journal reference: | International Journal of Engineering Trends and Technology (IJETT) 69.9(2021):230-235 |
| DOI:               | [10.14445/22315381/IJETT-V69I9P227](https://arxiv.org/ct?url=https%3A%2F%2Fdx.doi.org%2F10.14445%2F22315381%2FIJETT-V69I9P227&v=cf20c5d7) |
| Cite as:           | **[arXiv:2110.00435](https://arxiv.org/abs/2110.00435) [cs.CL]** |
|                    | (or **[arXiv:2110.00435v1](https://arxiv.org/abs/2110.00435v1) [cs.CL]** for this version) |





# 2021-10-01

[Return to Index](#Index)



<h2 id="2021-10-01-1">1. Phonetic Word Embeddings
</h2>

Title: [Phonetic Word Embeddings](https://arxiv.org/abs/2109.14796)

Authors: [Rahul Sharma](https://arxiv.org/search/cs?searchtype=author&query=Sharma%2C+R), [Kunal Dhawan](https://arxiv.org/search/cs?searchtype=author&query=Dhawan%2C+K), [Balakrishna Pailla](https://arxiv.org/search/cs?searchtype=author&query=Pailla%2C+B)

> This work presents a novel methodology for calculating the phonetic similarity between words taking motivation from the human perception of sounds. This metric is employed to learn a continuous vector embedding space that groups similar sounding words together and can be used for various downstream computational phonology tasks. The efficacy of the method is presented for two different languages (English, Hindi) and performance gains over previous reported works are discussed on established tests for predicting phonetic similarity. To address limited benchmarking mechanisms in this field, we also introduce a heterographic pun dataset based evaluation methodology to compare the effectiveness of acoustic similarity algorithms. Further, a visualization of the embedding space is presented with a discussion on the various possible use-cases of this novel algorithm. An open-source implementation is also shared to aid reproducibility and enable adoption in related tasks.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2109.14796](https://arxiv.org/abs/2109.14796) [cs.CL]** |
|           | (or **[arXiv:2109.14796v1](https://arxiv.org/abs/2109.14796v1) [cs.CL]** for this version) |





<h2 id="2021-10-01-2">2. Improved statistical machine translation using monolingual paraphrases
</h2>

Title: [Improved statistical machine translation using monolingual paraphrases](https://arxiv.org/abs/2109.15119)

Authors: [Preslav Nakov](https://arxiv.org/search/cs?searchtype=author&query=Nakov%2C+P)

> We propose a novel monolingual sentence paraphrasing method for augmenting the training data for statistical machine translation systems "for free" -- by creating it from data that is already available rather than having to create more aligned data. Starting with a syntactic tree, we recursively generate new sentence variants where noun compounds are paraphrased using suitable prepositions, and vice-versa -- preposition-containing noun phrases are turned into noun compounds. The evaluation shows an improvement equivalent to 33%-50% of that of doubling the amount of training data.

| Comments:          | machine translation, SMT, paraphrasing, data augmentation. arXiv admin note: substantial text overlap with [arXiv:1912.01113](https://arxiv.org/abs/1912.01113) |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| MSC classes:       | 68T50                                                        |
| ACM classes:       | F.2.2; I.2.7                                                 |
| Journal reference: | ECAI-2008                                                    |
| Cite as:           | **[arXiv:2109.15119](https://arxiv.org/abs/2109.15119) [cs.CL]** |
|                    | (or **[arXiv:2109.15119v1](https://arxiv.org/abs/2109.15119v1) [cs.CL]** for this version) |

