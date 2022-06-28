# MA C.'s Daily Paper Of Interest - June b., 2022

# Index

- [2022-06-28](#2022-06-28)
  - [1. Bi-VLDoc: Bidirectional Vision-Language Modeling for Visually-Rich Document Understanding](#2022-06-28-1)

  - [2. Probing Causes of Hallucinations in Neural Machine Translations](#2022-06-28-2)
  
  - [3. Language Models as Knowledge Embeddings](#2022-06-28-3)
  
  - [4. Distilling a Pretrained Language Model to a Multilingual ASR Model](#2022-06-28-4)
  
  - [5. Protoformer: Embedding Prototypes for Transformers](#2022-06-28-5)
  
- [2022-06-27](#2022-06-27)
  - [1. Robustness of Explanation Methods for NLP Models](#2022-06-27-1)

- [2022-06-24](#2022-06-24)
  - [1. Lifelong Learning Natural Language Processing Approach for Multilingual Data Classification](#2022-06-24-1)

- [2022-06-23](#2022-06-23)
  - [1. Generalizing Multimodal Pre-training into Multilingual via Language Acquisition](#2022-06-23-1)

  - [2. reStructured Pre-training](#2022-06-23-2)

  - [3. Understanding the Properties of Generated Corpora](#2022-06-23-3)

  - [4. GEMv2: Multilingual NLG Benchmarking in a Single Line of Code](#2022-06-23-4)

- [2022-06-22](#2022-06-22)
  - [1. BN-HTRd: A Benchmark Dataset for Document Level Offline Bangla Handwritten Text Recognition (HTR) and Line Segmentation](#2022-06-22-1)

  - [2. Towards Adversarial Attack on Vision-Language Pre-training Models](#2022-06-22-2)

  - [3. CLiMB: A Continual Learning Benchmark for Vision-and-Language Tasks](#2022-06-22-3)

  - [4. Learning Multiscale Transformer Models for Sequence Generation](#2022-06-22-4)

  - [5. LayoutXLM vs. GNN: An Empirical Evaluation of Relation Extraction for Documents](#2022-06-22-5)

  - [6. Plug and Play Counterfactual Text Generation for Model Robustness](#2022-06-22-6)

- [2022-06-20](#2022-06-20)
  - [1. VLMbench: A Compositional Benchmark for Vision-and-Language Manipulation](#2022-06-20-1)

  - [2. Bridge-Tower: Building Bridges Between Encoders in Vision-Language Representation Learning](#2022-06-20-2)

  - [3. Automatic Correction of Human Translations](#2022-06-20-3)

  - [4. Language with Vision: a Study on Grounded Word and Sentence Embeddings](#2022-06-20-4)

- [2022-06-17](#2022-06-17)
  - [1. How Adults Understand What Young Children Say](#2022-06-17-1)

  - [2. Alexa Teacher Model: Pretraining and Distilling Multi-Billion-Parameter Encoders for Natural Language Understanding Systems](#2022-06-17-2)

  - [3. TransDrift: Modeling Word-Embedding Drift using Transformer](#2022-06-17-3)

  - [4. Self-Generated In-Context Learning: Leveraging Auto-regressive Language Models as a Demonstration Generator](#2022-06-17-4)

  - [5. Deep Learning Architecture for Automatic Essay Scoring](#2022-06-17-5)

- [2022-06-16](#2022-06-16)
  - [1. Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone](#2022-06-16-1)

  - [2. A Unified Sequence Interface for Vision Tasks](#2022-06-16-2)

  - [3. Prefix Language Models are Unified Modal Learners](#2022-06-16-3)

  - [4. Human Heuristics for AI-Generated Language Are Flawed](#2022-06-16-4)

  - [5. MPI: Evaluating and Inducing Personality in Pre-trained Language Models](#2022-06-16-5)

  - [6. Emergent Abilities of Large Language Models](#2022-06-16-6)

- [2022-06-15](#2022-06-15)
  - [1. LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning](#2022-06-15-1)

- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



# 2022-06-28

[Return to Index](#Index)



<h2 id="2022-06-28-1">1. Bi-VLDoc: Bidirectional Vision-Language Modeling for Visually-Rich Document Understanding
</h2>

Title: [Bi-VLDoc: Bidirectional Vision-Language Modeling for Visually-Rich Document Understanding](https://arxiv.org/abs/2206.13155)

Authors: [Chuwei Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+C), [Guozhi Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+G), [Qi Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+Q), [Cong Yao](https://arxiv.org/search/cs?searchtype=author&query=Yao%2C+C), [Lianwen Jin](https://arxiv.org/search/cs?searchtype=author&query=Jin%2C+L), [Chenliang Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+C), [Yang Xue](https://arxiv.org/search/cs?searchtype=author&query=Xue%2C+Y), [Luo Si](https://arxiv.org/search/cs?searchtype=author&query=Si%2C+L)

> Multi-modal document pre-trained models have proven to be very effective in a variety of visually-rich document understanding (VrDU) tasks. Though existing document pre-trained models have achieved excellent performance on standard benchmarks for VrDU, the way they model and exploit the interactions between vision and language on documents has hindered them from better generalization ability and higher accuracy. In this work, we investigate the problem of vision-language joint representation learning for VrDU mainly from the perspective of supervisory signals. Specifically, a pre-training paradigm called Bi-VLDoc is proposed, in which a bidirectional vision-language supervision strategy and a vision-language hybrid-attention mechanism are devised to fully explore and utilize the interactions between these two modalities, to learn stronger cross-modal document representations with richer semantics. Benefiting from the learned informative cross-modal document representations, Bi-VLDoc significantly advances the state-of-the-art performance on three widely-used document understanding benchmarks, including Form Understanding (from 85.14% to 93.44%), Receipt Information Extraction (from 96.01% to 97.84%), and Document Classification (from 96.08% to 97.12%). On Document Visual QA, Bi-VLDoc achieves the state-of-the-art performance compared to previous single model methods.

| Comments: | Under review                                                 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Multimedia (cs.MM) |
| Cite as:  | **[arXiv:2206.13155](https://arxiv.org/abs/2206.13155) [cs.CV]** |
|           | (or **[arXiv:2206.13155v1](https://arxiv.org/abs/2206.13155v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.13155Focus to learn more |





<h2 id="2022-06-28-2">2. Probing Causes of Hallucinations in Neural Machine Translations
</h2>

Title: [Probing Causes of Hallucinations in Neural Machine Translations](https://arxiv.org/abs/2206.12529)

Authors: [Jianhao Yan](https://arxiv.org/search/cs?searchtype=author&query=Yan%2C+J), [Fandong Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+F), [Jie Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+J)

> Hallucination, one kind of pathological translations that bothers Neural Machine Translation, has recently drawn much attention. In simple terms, hallucinated translations are fluent sentences but barely related to source inputs. Arguably, it remains an open problem how hallucination occurs. In this paper, we propose to use probing methods to investigate the causes of hallucinations from the perspective of model architecture, aiming to avoid such problems in future architecture designs. By conducting experiments over various NMT datasets, we find that hallucination is often accompanied by the deficient encoder, especially embeddings, and vulnerable cross-attentions, while, interestingly, cross-attention mitigates some errors caused by the encoder.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.12529](https://arxiv.org/abs/2206.12529) [cs.CL]** |
|           | (or **[arXiv:2206.12529v1](https://arxiv.org/abs/2206.12529v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.12529Focus to learn more |





<h2 id="2022-06-28-3">3. Language Models as Knowledge Embeddings
</h2>

Title: [Language Models as Knowledge Embeddings](https://arxiv.org/abs/2206.12617)

Authors: [Xintao Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Qianyu He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+Q), [Jiaqing Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+J), [Yanghua Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+Y)

> Knowledge embeddings (KE) represent a knowledge graph (KG) by embedding entities and relations into continuous vector spaces. Existing methods are mainly structure-based or description-based. Structure-based methods learn representations that preserve the inherent structure of KGs. They cannot well represent abundant long-tail entities in real-world KGs with limited structural information. Description-based methods leverage textual information and language models. Prior approaches in this direction barely outperform structure-based ones, and suffer from problems like expensive negative sampling and restrictive description demand. In this paper, we propose LMKE, which adopts Language Models to derive Knowledge Embeddings, aiming at both enriching representations of long-tail entities and solving problems of prior description-based methods. We formulate description-based KE learning with a contrastive learning framework to improve efficiency in training and evaluation. Experimental results show that LMKE achieves state-of-the-art performance on KE benchmarks of link prediction and triple classification, especially for long-tail entities.

| Comments: | Accepted to IJCAI 2022                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2206.12617](https://arxiv.org/abs/2206.12617) [cs.CL]** |
|           | (or **[arXiv:2206.12617v1](https://arxiv.org/abs/2206.12617v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.12617Focus to learn more |





<h2 id="2022-06-28-4">4. Distilling a Pretrained Language Model to a Multilingual ASR Model
</h2>

Title: [Distilling a Pretrained Language Model to a Multilingual ASR Model](https://arxiv.org/abs/2206.12638)

Authors: [Kwanghee Choi](https://arxiv.org/search/cs?searchtype=author&query=Choi%2C+K), [Hyung-Min Park](https://arxiv.org/search/cs?searchtype=author&query=Park%2C+H)

> Multilingual speech data often suffer from long-tailed language distribution, resulting in performance degradation. However, multilingual text data is much easier to obtain, yielding a more useful general language model. Hence, we are motivated to distill the rich knowledge embedded inside a well-trained teacher text model to the student speech model. We propose a novel method called the Distilling a Language model to a Speech model (Distill-L2S), which aligns the latent representations of two different modalities. The subtle differences are handled by the shrinking mechanism, nearest-neighbor interpolation, and a learnable linear projection layer. We demonstrate the effectiveness of our distillation method by applying it to the multilingual automatic speech recognition (ASR) task. We distill the transformer-based cross-lingual language model (InfoXLM) while fine-tuning the large-scale multilingual ASR model (XLSR-wav2vec 2.0) for each language. We show the superiority of our method on 20 low-resource languages of the CommonVoice dataset with less than 100 hours of speech data.

| Comments: | Accepted to Interspeech 2022. Official implementation provided in [this https URL](https://github.com/juice500ml/xlm_to_xlsr) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2206.12638](https://arxiv.org/abs/2206.12638) [cs.CL]** |
|           | (or **[arXiv:2206.12638v1](https://arxiv.org/abs/2206.12638v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.12638Focus to learn more |





<h2 id="2022-06-28-5">5. Protoformer: Embedding Prototypes for Transformers
</h2>

Title: [Protoformer: Embedding Prototypes for Transformers](https://arxiv.org/abs/2206.12710)

Authors: [Ashkan Farhangi](https://arxiv.org/search/cs?searchtype=author&query=Farhangi%2C+A), [Ning Sui](https://arxiv.org/search/cs?searchtype=author&query=Sui%2C+N), [Nan Hua](https://arxiv.org/search/cs?searchtype=author&query=Hua%2C+N), [Haiyan Bai](https://arxiv.org/search/cs?searchtype=author&query=Bai%2C+H), [Arthur Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+A), [Zhishan Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+Z)

> Transformers have been widely applied in text classification. Unfortunately, real-world data contain anomalies and noisy labels that cause challenges for state-of-art Transformers. This paper proposes Protoformer, a novel self-learning framework for Transformers that can leverage problematic samples for text classification. Protoformer features a selection mechanism for embedding samples that allows us to efficiently extract and utilize anomalies prototypes and difficult class prototypes. We demonstrated such capabilities on datasets with diverse textual structures (e.g., Twitter, IMDB, ArXiv). We also applied the framework to several models. The results indicate that Protoformer can improve current Transformers in various empirical settings.

| Comments:          | Advances in Knowledge Discovery and Data Mining (PAKDD 2022) |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:           | **[arXiv:2206.12710](https://arxiv.org/abs/2206.12710) [cs.CL]** |
|                    | (or **[arXiv:2206.12710v1](https://arxiv.org/abs/2206.12710v1) [cs.CL]** for this version) |
|                    | https://doi.org/10.48550/arXiv.2206.12710Focus to learn more |
| Journal reference: | Advances in Knowledge Discovery and Data Mining: 26th Pacific-Asia Conference, PAKDD 2022 |
| Related DOI:       | https://doi.org/10.1007/978-3-031-05933-9_35Focus to learn more |






# 2022-06-27

[Return to Index](#Index)



<h2 id="2022-06-27-1">1. Robustness of Explanation Methods for NLP Models
</h2>

Title: [Robustness of Explanation Methods for NLP Models](Robustness of Explanation Methods for NLP Models)

Authors: [Shriya Atmakuri](https://arxiv.org/search/cs?searchtype=author&query=Atmakuri%2C+S), [Tejas Chheda](https://arxiv.org/search/cs?searchtype=author&query=Chheda%2C+T), [Dinesh Kandula](https://arxiv.org/search/cs?searchtype=author&query=Kandula%2C+D), [Nishant Yadav](https://arxiv.org/search/cs?searchtype=author&query=Yadav%2C+N), [Taesung Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+T), [Hessel Tuinhof](https://arxiv.org/search/cs?searchtype=author&query=Tuinhof%2C+H)

> Explanation methods have emerged as an important tool to highlight the features responsible for the predictions of neural networks. There is mounting evidence that many explanation methods are rather unreliable and susceptible to malicious manipulations. In this paper, we particularly aim to understand the robustness of explanation methods in the context of text modality. We provide initial insights and results towards devising a successful adversarial attack against text explanations. To our knowledge, this is the first attempt to evaluate the adversarial robustness of an explanation method. Our experiments show the explanation method can be largely disturbed for up to 86% of the tested samples with small changes in the input sentence and its semantics.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.12284](https://arxiv.org/abs/2206.12284) [cs.CL]** |
|           | (or **[arXiv:2206.12284v1](https://arxiv.org/abs/2206.12284v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.12284Focus to learn more |






# 2022-06-24

[Return to Index](#Index)



<h2 id="2022-06-24-1">1. Lifelong Learning Natural Language Processing Approach for Multilingual Data Classification
</h2>

Title: [Lifelong Learning Natural Language Processing Approach for Multilingual Data Classification](https://arxiv.org/abs/2206.11867)

Authors: [Jędrzej Kozal](https://arxiv.org/search/cs?searchtype=author&query=Kozal%2C+J), [Michał Leś](https://arxiv.org/search/cs?searchtype=author&query=Leś%2C+M), [Paweł Zyblewski](https://arxiv.org/search/cs?searchtype=author&query=Zyblewski%2C+P), [Paweł Ksieniewicz](https://arxiv.org/search/cs?searchtype=author&query=Ksieniewicz%2C+P), [Michał Woźniak](https://arxiv.org/search/cs?searchtype=author&query=Woźniak%2C+M)

> The abundance of information in digital media, which in today's world is the main source of knowledge about current events for the masses, makes it possible to spread disinformation on a larger scale than ever before. Consequently, there is a need to develop novel fake news detection approaches capable of adapting to changing factual contexts and generalizing previously or concurrently acquired knowledge. To deal with this problem, we propose a lifelong learning-inspired approach, which allows for fake news detection in multiple languages and the mutual transfer of knowledge acquired in each of them. Both classical feature extractors, such as Term frequency-inverse document frequency or Latent Dirichlet Allocation, and integrated deep NLP (Natural Language Processing) BERT (Bidirectional Encoder Representations from Transformers) models paired with MLP (Multilayer Perceptron) classifier, were employed. The results of experiments conducted on two datasets dedicated to the fake news classification task (in English and Spanish, respectively), supported by statistical analysis, confirmed that utilization of additional languages could improve performance for traditional methods. Also, in some cases supplementing the deep learning method with classical ones can positively impact obtained results. The ability of models to generalize the knowledge acquired between the analyzed languages was also observed.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.11867](https://arxiv.org/abs/2206.11867) [cs.CL]** |
|           | (or **[arXiv:2206.11867v1](https://arxiv.org/abs/2206.11867v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.11867Focus to learn more |









# 2022-06-23

[Return to Index](#Index)



<h2 id="2022-06-23-1">1. Generalizing Multimodal Pre-training into Multilingual via Language Acquisition
</h2>

Title: [Generalizing Multimodal Pre-training into Multilingual via Language Acquisition](https://arxiv.org/abs/2206.11091)

Authors: [Liang Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+L), [Anwen Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+A), [Qin Jin](https://arxiv.org/search/cs?searchtype=author&query=Jin%2C+Q)

> English-based Vision-Language Pre-training (VLP) has achieved great success in various downstream tasks. Some efforts have been taken to generalize this success to non-English languages through Multilingual Vision-Language Pre-training (M-VLP). However, due to the large number of languages, M-VLP models often require huge computing resources and cannot be flexibly extended to new languages. In this work, we propose a \textbf{M}ulti\textbf{L}ingual \textbf{A}cquisition (MLA) framework that can easily generalize a monolingual Vision-Language Pre-training model into multilingual. Specifically, we design a lightweight language acquisition encoder based on state-of-the-art monolingual VLP models. We further propose a two-stage training strategy to optimize the language acquisition encoder, namely the Native Language Transfer stage and the Language Exposure stage. With much less multilingual training data and computing resources, our model achieves state-of-the-art performance on multilingual image-text and video-text retrieval benchmarks.

| Comments: | 14 pages, 5 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2206.11091](https://arxiv.org/abs/2206.11091) [cs.CL]** |
|           | (or **[arXiv:2206.11091v1](https://arxiv.org/abs/2206.11091v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.11091Focus to learn more |





<h2 id="2022-06-23-2">2. reStructured Pre-training
</h2>

Title: [reStructured Pre-training](https://arxiv.org/abs/2206.11147)

Authors: [Weizhe Yuan](https://arxiv.org/search/cs?searchtype=author&query=Yuan%2C+W), [Pengfei Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+P)

> In this work, we try to decipher the internal connection of NLP technology development in the past decades, searching for essence, which rewards us with a (potential) new learning paradigm for NLP tasks, dubbed as reStructured Pre-training (RST). In such a paradigm, the role of data will be re-emphasized, and model pre-training and fine-tuning of downstream tasks are viewed as a process of data storing and accessing. Based on that, we operationalize the simple principle that a good storage mechanism should not only have the ability to cache a large amount of data but also consider the ease of access. We achieve this by pre-training models over restructured data that consist of a variety of valuable information instead of raw data after overcoming several engineering challenges. Experimentally, RST models not only surpass strong competitors (e.g., T0) on 52/55 popular datasets from a variety of NLP tasks, but also achieve superior performance in National College Entrance Examination - English (Gaokao-English),the most authoritative examination in China. Specifically, the proposed system Qin achieves 40 points higher than the average scores made by students and 15 points higher than GPT3 with 1/16 parameters. In particular, Qin gets a high score of 138.5 (the full mark is 150) in the 2018 English exam (national paper III). We have released the Gaokao Benchmark with an online submission platform. 
> In addition, we test our model in the 2022 College Entrance Examination English that happened a few days ago (2022.06.08), and it gets a total score of 134 (v.s. GPT3's 108).

| Comments: | A gift for NLPers :)                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2206.11147](https://arxiv.org/abs/2206.11147) [cs.CL]** |
|           | (or **[arXiv:2206.11147v1](https://arxiv.org/abs/2206.11147v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.11147Focus to learn more |





<h2 id="2022-06-23-3">3. Understanding the Properties of Generated Corpora
</h2>

Title: [Understanding the Properties of Generated Corpora](https://arxiv.org/abs/2206.11219)

Authors: [Naama Zwerdling](https://arxiv.org/search/cs?searchtype=author&query=Zwerdling%2C+N), [Segev Shlomov](https://arxiv.org/search/cs?searchtype=author&query=Shlomov%2C+S), [Esther Goldbraich](https://arxiv.org/search/cs?searchtype=author&query=Goldbraich%2C+E), [George Kour](https://arxiv.org/search/cs?searchtype=author&query=Kour%2C+G), [Boaz Carmeli](https://arxiv.org/search/cs?searchtype=author&query=Carmeli%2C+B), [Naama Tepper](https://arxiv.org/search/cs?searchtype=author&query=Tepper%2C+N), [Inbal Ronen](https://arxiv.org/search/cs?searchtype=author&query=Ronen%2C+I), [Vitaly Zabershinsky](https://arxiv.org/search/cs?searchtype=author&query=Zabershinsky%2C+V), [Ateret Anaby-Tavor](https://arxiv.org/search/cs?searchtype=author&query=Anaby-Tavor%2C+A)

> Models for text generation have become focal for many research tasks and especially for the generation of sentence corpora. However, understanding the properties of an automatically generated text corpus remains challenging. We propose a set of tools that examine the properties of generated text corpora. Applying these tools on various generated corpora allowed us to gain new insights into the properties of the generative models. As part of our characterization process, we found remarkable differences in the corpora generated by two leading generative technologies.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.11219](https://arxiv.org/abs/2206.11219) [cs.CL]** |
|           | (or **[arXiv:2206.11219v1](https://arxiv.org/abs/2206.11219v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.11219Focus to learn more |





<h2 id="2022-06-23-4">4. GEMv2: Multilingual NLG Benchmarking in a Single Line of Code
</h2>

Title: [GEMv2: Multilingual NLG Benchmarking in a Single Line of Code](https://arxiv.org/abs/2206.11249)

Authors: [Sebastian Gehrmann](https://arxiv.org/search/cs?searchtype=author&query=Gehrmann%2C+S), [Abhik Bhattacharjee](https://arxiv.org/search/cs?searchtype=author&query=Bhattacharjee%2C+A), [Abinaya Mahendiran](https://arxiv.org/search/cs?searchtype=author&query=Mahendiran%2C+A), [Alex Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+A), [Alexandros Papangelis](https://arxiv.org/search/cs?searchtype=author&query=Papangelis%2C+A), [Aman Madaan](https://arxiv.org/search/cs?searchtype=author&query=Madaan%2C+A), [Angelina McMillan-Major](https://arxiv.org/search/cs?searchtype=author&query=McMillan-Major%2C+A), [Anna Shvets](https://arxiv.org/search/cs?searchtype=author&query=Shvets%2C+A), [Ashish Upadhyay](https://arxiv.org/search/cs?searchtype=author&query=Upadhyay%2C+A), [Bingsheng Yao](https://arxiv.org/search/cs?searchtype=author&query=Yao%2C+B), [Bryan Wilie](https://arxiv.org/search/cs?searchtype=author&query=Wilie%2C+B), [Chandra Bhagavatula](https://arxiv.org/search/cs?searchtype=author&query=Bhagavatula%2C+C), [Chaobin You](https://arxiv.org/search/cs?searchtype=author&query=You%2C+C), [Craig Thomson](https://arxiv.org/search/cs?searchtype=author&query=Thomson%2C+C), [Cristina Garbacea](https://arxiv.org/search/cs?searchtype=author&query=Garbacea%2C+C), [Dakuo Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+D), [Daniel Deutsch](https://arxiv.org/search/cs?searchtype=author&query=Deutsch%2C+D), [Deyi Xiong](https://arxiv.org/search/cs?searchtype=author&query=Xiong%2C+D), [Di Jin](https://arxiv.org/search/cs?searchtype=author&query=Jin%2C+D), [Dimitra Gkatzia](https://arxiv.org/search/cs?searchtype=author&query=Gkatzia%2C+D), [Dragomir Radev](https://arxiv.org/search/cs?searchtype=author&query=Radev%2C+D), [Elizabeth Clark](https://arxiv.org/search/cs?searchtype=author&query=Clark%2C+E), [Esin Durmus](https://arxiv.org/search/cs?searchtype=author&query=Durmus%2C+E), [Faisal Ladhak](https://arxiv.org/search/cs?searchtype=author&query=Ladhak%2C+F), [Filip Ginter](https://arxiv.org/search/cs?searchtype=author&query=Ginter%2C+F), [Genta Indra Winata](https://arxiv.org/search/cs?searchtype=author&query=Winata%2C+G+I), [Hendrik Strobelt](https://arxiv.org/search/cs?searchtype=author&query=Strobelt%2C+H), [Hiroaki Hayashi](https://arxiv.org/search/cs?searchtype=author&query=Hayashi%2C+H), [Jekaterina Novikova](https://arxiv.org/search/cs?searchtype=author&query=Novikova%2C+J), [Jenna Kanerva](https://arxiv.org/search/cs?searchtype=author&query=Kanerva%2C+J), [Jenny Chim](https://arxiv.org/search/cs?searchtype=author&query=Chim%2C+J), [Jiawei Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+J), [Jordan Clive](https://arxiv.org/search/cs?searchtype=author&query=Clive%2C+J), [Joshua Maynez](https://arxiv.org/search/cs?searchtype=author&query=Maynez%2C+J), [João Sedoc](https://arxiv.org/search/cs?searchtype=author&query=Sedoc%2C+J), [Juraj Juraska](https://arxiv.org/search/cs?searchtype=author&query=Juraska%2C+J), [Kaustubh Dhole](https://arxiv.org/search/cs?searchtype=author&query=Dhole%2C+K), [Khyathi Raghavi Chandu](https://arxiv.org/search/cs?searchtype=author&query=Chandu%2C+K+R), [Leonardo F. R. Ribeiro](https://arxiv.org/search/cs?searchtype=author&query=Ribeiro%2C+L+F+R), [Lewis Tunstall](https://arxiv.org/search/cs?searchtype=author&query=Tunstall%2C+L), [Li Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+L), [Mahima Pushkarna](https://arxiv.org/search/cs?searchtype=author&query=Pushkarna%2C+M), [Mathias Creutz](https://arxiv.org/search/cs?searchtype=author&query=Creutz%2C+M), [Michael White](https://arxiv.org/search/cs?searchtype=author&query=White%2C+M), [Mihir Sanjay Kale](https://arxiv.org/search/cs?searchtype=author&query=Kale%2C+M+S), [Moussa Kamal Eddine](https://arxiv.org/search/cs?searchtype=author&query=Eddine%2C+M+K), [Nico Daheim](https://arxiv.org/search/cs?searchtype=author&query=Daheim%2C+N), [Nishant Subramani](https://arxiv.org/search/cs?searchtype=author&query=Subramani%2C+N), [Ondrej Dusek](https://arxiv.org/search/cs?searchtype=author&query=Dusek%2C+O), [Paul Pu Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+P+P), [Pawan Sasanka Ammanamanchi](https://arxiv.org/search/cs?searchtype=author&query=Ammanamanchi%2C+P+S), [Qi Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+Q), [Ratish Puduppully](https://arxiv.org/search/cs?searchtype=author&query=Puduppully%2C+R), [Reno Kriz](https://arxiv.org/search/cs?searchtype=author&query=Kriz%2C+R), [Rifat Shahriyar](https://arxiv.org/search/cs?searchtype=author&query=Shahriyar%2C+R), [Ronald Cardenas](https://arxiv.org/search/cs?searchtype=author&query=Cardenas%2C+R), [Saad Mahamood](https://arxiv.org/search/cs?searchtype=author&query=Mahamood%2C+S), [Salomey Osei](https://arxiv.org/search/cs?searchtype=author&query=Osei%2C+S), [Samuel Cahyawijaya](https://arxiv.org/search/cs?searchtype=author&query=Cahyawijaya%2C+S), [Sanja Štajner](https://arxiv.org/search/cs?searchtype=author&query=Štajner%2C+S), [Sebastien Montella](https://arxiv.org/search/cs?searchtype=author&query=Montella%2C+S), [Shailza](https://arxiv.org/search/cs?searchtype=author&query=Shailza), [Shailza Jolly](https://arxiv.org/search/cs?searchtype=author&query=Jolly%2C+S), [Simon Mille](https://arxiv.org/search/cs?searchtype=author&query=Mille%2C+S), [Tahmid Hasan](https://arxiv.org/search/cs?searchtype=author&query=Hasan%2C+T), [Tianhao Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+T), [Tosin Adewumi](https://arxiv.org/search/cs?searchtype=author&query=Adewumi%2C+T), [Vikas Raunak](https://arxiv.org/search/cs?searchtype=author&query=Raunak%2C+V), [Vipul Raheja](https://arxiv.org/search/cs?searchtype=author&query=Raheja%2C+V), [Vitaly Nikolaev](https://arxiv.org/search/cs?searchtype=author&query=Nikolaev%2C+V), [Vivian Tsai](https://arxiv.org/search/cs?searchtype=author&query=Tsai%2C+V), [Yacine Jernite](https://arxiv.org/search/cs?searchtype=author&query=Jernite%2C+Y), [Ying Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+Y), [Yisi Sang](https://arxiv.org/search/cs?searchtype=author&query=Sang%2C+Y), [Yixin Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y), [Yufang Hou](https://arxiv.org/search/cs?searchtype=author&query=Hou%2C+Y)

> Evaluation in machine learning is usually informed by past choices, for example which datasets or metrics to use. This standardization enables the comparison on equal footing using leaderboards, but the evaluation choices become sub-optimal as better alternatives arise. This problem is especially pertinent in natural language generation which requires ever-improving suites of datasets, metrics, and human evaluation to make definitive claims. To make following best model evaluation practices easier, we introduce GEMv2. The new version of the Generation, Evaluation, and Metrics Benchmark introduces a modular infrastructure for dataset, model, and metric developers to benefit from each others work. GEMv2 supports 40 documented datasets in 51 languages. Models for all datasets can be evaluated online and our interactive data card creation and rendering tools make it easier to add new datasets to the living benchmark.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.11249](https://arxiv.org/abs/2206.11249) [cs.CL]** |
|           | (or **[arXiv:2206.11249v1](https://arxiv.org/abs/2206.11249v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.11249Focus to learn more |





# 2022-06-22

[Return to Index](#Index)



<h2 id="2022-06-22-1">1. BN-HTRd: A Benchmark Dataset for Document Level Offline Bangla Handwritten Text Recognition (HTR) and Line Segmentation
</h2>

Title: [BN-HTRd: A Benchmark Dataset for Document Level Offline Bangla Handwritten Text Recognition (HTR) and Line Segmentation](https://arxiv.org/abs/2206.08977)

Authors: [Md. Ataur Rahman](https://arxiv.org/search/cs?searchtype=author&query=Rahman%2C+M+A), [Nazifa Tabassum](https://arxiv.org/search/cs?searchtype=author&query=Tabassum%2C+N), [Mitu Paul](https://arxiv.org/search/cs?searchtype=author&query=Paul%2C+M), [Riya Pal](https://arxiv.org/search/cs?searchtype=author&query=Pal%2C+R), [Mohammad Khairul Islam](https://arxiv.org/search/cs?searchtype=author&query=Islam%2C+M+K)

> We introduce a new dataset for offline Handwritten Text Recognition (HTR) from images of Bangla scripts comprising words, lines, and document-level annotations. The BN-HTRd dataset is based on the BBC Bangla News corpus, meant to act as ground truth texts. These texts were subsequently used to generate the annotations that were filled out by people with their handwriting. Our dataset includes 788 images of handwritten pages produced by approximately 150 different writers. It can be adopted as a basis for various handwriting classification tasks such as end-to-end document recognition, word-spotting, word or line segmentation, and so on. We also propose a scheme to segment Bangla handwritten document images into corresponding lines in an unsupervised manner. Our line segmentation approach takes care of the variability involved in different writing styles, accurately segmenting complex handwritten text lines of curvilinear nature. Along with a bunch of pre-processing and morphological operations, both Hough line and circle transforms were employed to distinguish different linear components. In order to arrange those components into their corresponding lines, we followed an unsupervised clustering approach. The average success rate of our segmentation technique is 81.57% in terms of FM metrics (similar to F-measure) with a mean Average Precision (mAP) of 0.547.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.08977](https://arxiv.org/abs/2206.08977) [cs.CV]** |
|           | (or **[arXiv:2206.08977v1](https://arxiv.org/abs/2206.08977v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.08977Focus to learn more |





<h2 id="2022-06-22-2">2. Towards Adversarial Attack on Vision-Language Pre-training Models
</h2>

Title: [Towards Adversarial Attack on Vision-Language Pre-training Models](https://arxiv.org/abs/2206.09391)

Authors: [Jiaming Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+J), [Qi Yi](https://arxiv.org/search/cs?searchtype=author&query=Yi%2C+Q), [Jitao Sang](https://arxiv.org/search/cs?searchtype=author&query=Sang%2C+J)

> While vision-language pre-training model (VLP) has shown revolutionary improvements on various vision-language (V+L) tasks, the studies regarding its adversarial robustness remain largely unexplored. This paper studied the adversarial attack on popular VLP models and V+L tasks. First, we analyzed the performance of adversarial attacks under different settings. By examining the influence of different perturbed objects and attack targets, we concluded some key observations as guidance on both designing strong multimodal adversarial attack and constructing robust VLP models. Second, we proposed a novel multimodal attack method on the VLP models called Collaborative Multimodal Adversarial Attack (Co-Attack), which collectively carries out the attacks on the image modality and the text modality. Experimental results demonstrated that the proposed method achieves improved attack performances on different V+L downstream tasks and VLP models. The analysis observations and novel attack method hopefully provide new understanding into the adversarial robustness of VLP models, so as to contribute their safe and reliable deployment in more real-world scenarios.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV); Multimedia (cs.MM) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.09391](https://arxiv.org/abs/2206.09391) [cs.LG]** |
|           | (or **[arXiv:2206.09391v1](https://arxiv.org/abs/2206.09391v1) [cs.LG]** for this version) |





<h2 id="2022-06-22-3">3. CLiMB: A Continual Learning Benchmark for Vision-and-Language Tasks
</h2>

Title: [CLiMB: A Continual Learning Benchmark for Vision-and-Language Tasks](https://arxiv.org/abs/2206.09059)

Authors: [Tejas Srinivasan](https://arxiv.org/search/cs?searchtype=author&query=Srinivasan%2C+T), [Ting-Yun Chang](https://arxiv.org/search/cs?searchtype=author&query=Chang%2C+T), [Leticia Leonor Pinto Alva](https://arxiv.org/search/cs?searchtype=author&query=Alva%2C+L+L+P), [Georgios Chochlakis](https://arxiv.org/search/cs?searchtype=author&query=Chochlakis%2C+G), [Mohammad Rostami](https://arxiv.org/search/cs?searchtype=author&query=Rostami%2C+M), [Jesse Thomason](https://arxiv.org/search/cs?searchtype=author&query=Thomason%2C+J)

> Current state-of-the-art vision-and-language models are evaluated on tasks either individually or in a multi-task setting, overlooking the challenges of continually learning (CL) tasks as they arrive. Existing CL benchmarks have facilitated research on task adaptation and mitigating "catastrophic forgetting", but are limited to vision-only and language-only tasks. We present CLiMB, a benchmark to study the challenge of learning multimodal tasks in a CL setting, and to systematically evaluate how upstream continual learning can rapidly generalize to new multimodal and unimodal tasks. CLiMB includes implementations of several CL algorithms and a modified Vision-Language Transformer (ViLT) model that can be deployed on both multimodal and unimodal tasks. We find that common CL methods can help mitigate forgetting during multimodal task learning, but do not enable cross-task knowledge transfer. We envision that CLiMB will facilitate research on a new class of CL algorithms for this challenging multimodal setting.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.09059](https://arxiv.org/abs/2206.09059) [cs.CL]** |
|           | (or **[arXiv:2206.09059v1](https://arxiv.org/abs/2206.09059v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.09059Focus to learn more |





<h2 id="2022-06-22-4">4. Learning Multiscale Transformer Models for Sequence Generation
</h2>

Title: [Learning Multiscale Transformer Models for Sequence Generation](https://arxiv.org/abs/2206.09337)

Authors: [Bei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+B), [Tong Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+T), [Yi Jing](https://arxiv.org/search/cs?searchtype=author&query=Jing%2C+Y), [Chengbo Jiao](https://arxiv.org/search/cs?searchtype=author&query=Jiao%2C+C), [Tong Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+T), [Jingbo Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+J)

> Multiscale feature hierarchies have been witnessed the success in the computer vision area. This further motivates researchers to design multiscale Transformer for natural language processing, mostly based on the self-attention mechanism. For example, restricting the receptive field across heads or extracting local fine-grained features via convolutions. However, most of existing works directly modeled local features but ignored the word-boundary information. This results in redundant and ambiguous attention distributions, which lacks of interpretability. In this work, we define those scales in different linguistic units, including sub-words, words and phrases. We built a multiscale Transformer model by establishing relationships among scales based on word-boundary information and phrase-level prior knowledge. The proposed \textbf{U}niversal \textbf{M}ulti\textbf{S}cale \textbf{T}ransformer, namely \textsc{Umst}, was evaluated on two sequence generation tasks. Notably, it yielded consistent performance gains over the strong baseline on several test sets without sacrificing the efficiency.

| Comments: | accepted by ICML2022                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2206.09337](https://arxiv.org/abs/2206.09337) [cs.CL]** |
|           | (or **[arXiv:2206.09337v1](https://arxiv.org/abs/2206.09337v1) [cs.CL]** for this version) |





<h2 id="2022-06-22-5">5. LayoutXLM vs. GNN: An Empirical Evaluation of Relation Extraction for Documents
</h2>

Title: [LayoutXLM vs. GNN: An Empirical Evaluation of Relation Extraction for Documents](https://arxiv.org/abs/2206.10304)

Authors: [Hervé Déjean](https://arxiv.org/search/cs?searchtype=author&query=Déjean%2C+H), [Stéphane Clinchant](https://arxiv.org/search/cs?searchtype=author&query=Clinchant%2C+S), [Jean-Luc Meunier](https://arxiv.org/search/cs?searchtype=author&query=Meunier%2C+J)

> This paper investigates the Relation Extraction task in documents by benchmarking two different neural network models: a multi-modal language model (LayoutXLM) and a Graph Neural Network: Edge Convolution Network (ECN). For this benchmark, we use the XFUND dataset, released along with LayoutXLM. While both models reach similar results, they both exhibit very different characteristics. This raises the question on how to integrate various modalities in a neural network: by merging all modalities thanks to additional pretraining (LayoutXLM), or in a cascaded way (ECN). We conclude by discussing some methodological issues that must be considered for new datasets and task definition in the domain of Information Extraction with complex documents.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.10304](https://arxiv.org/abs/2206.10304) [cs.CL]** |
|           | (or **[arXiv:2206.10304v1](https://arxiv.org/abs/2206.10304v1) [cs.CL]** for this version) |





<h2 id="2022-06-22-6">6. Plug and Play Counterfactual Text Generation for Model Robustness
</h2>

Title: [Plug and Play Counterfactual Text Generation for Model Robustness](https://arxiv.org/abs/2206.10429)

Authors: [Nishtha Madaan](https://arxiv.org/search/cs?searchtype=author&query=Madaan%2C+N), [Srikanta Bedathur](https://arxiv.org/search/cs?searchtype=author&query=Bedathur%2C+S), [Diptikalyan Saha](https://arxiv.org/search/cs?searchtype=author&query=Saha%2C+D)

> Generating counterfactual test-cases is an important backbone for testing NLP models and making them as robust and reliable as traditional software. In generating the test-cases, a desired property is the ability to control the test-case generation in a flexible manner to test for a large variety of failure cases and to explain and repair them in a targeted manner. In this direction, significant progress has been made in the prior works by manually writing rules for generating controlled counterfactuals. However, this approach requires heavy manual supervision and lacks the flexibility to easily introduce new controls. Motivated by the impressive flexibility of the plug-and-play approach of PPLM, we propose bringing the framework of plug-and-play to counterfactual test case generation task. We introduce CASPer, a plug-and-play counterfactual generation framework to generate test cases that satisfy goal attributes on demand. Our plug-and-play model can steer the test case generation process given any attribute model without requiring attribute-specific training of the model. In experiments, we show that CASPer effectively generates counterfactual text that follow the steering provided by an attribute model while also being fluent, diverse and preserving the original content. We also show that the generated counterfactuals from CASPer can be used for augmenting the training data and thereby fixing and making the test model more robust.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.10429](https://arxiv.org/abs/2206.10429) [cs.CL]** |
|           | (or **[arXiv:2206.10429v1](https://arxiv.org/abs/2206.10429v1) [cs.CL]** for this version) |





# 2022-06-20

[Return to Index](#Index)



<h2 id="2022-06-20-1">1. VLMbench: A Compositional Benchmark for Vision-and-Language Manipulation
</h2>

Title: [VLMbench: A Compositional Benchmark for Vision-and-Language Manipulation](https://arxiv.org/abs/2206.08522)

Authors: [Kaizhi Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+K), [Xiaotong Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+X), [Odest Chadwicke Jenkins](https://arxiv.org/search/cs?searchtype=author&query=Jenkins%2C+O+C), [Xin Eric Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X+E)

> Benefiting from language flexibility and compositionality, humans naturally intend to use language to command an embodied agent for complex tasks such as navigation and object manipulation. In this work, we aim to fill the blank of the last mile of embodied agents -- object manipulation by following human guidance, e.g., "move the red mug next to the box while keeping it upright." To this end, we introduce an Automatic Manipulation Solver (AMSolver) simulator and build a Vision-and-Language Manipulation benchmark (VLMbench) based on it, containing various language instructions on categorized robotic manipulation tasks. Specifically, modular rule-based task templates are created to automatically generate robot demonstrations with language instructions, consisting of diverse object shapes and appearances, action types, and motion constraints. We also develop a keypoint-based model 6D-CLIPort to deal with multi-view observations and language input and output a sequence of 6 degrees of freedom (DoF) actions. We hope the new simulator and benchmark will facilitate future research on language-guided robotic manipulation.

| Subjects: | **Robotics (cs.RO)**; Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.08522](https://arxiv.org/abs/2206.08522) [cs.RO]** |
|           | (or **[arXiv:2206.08522v1](https://arxiv.org/abs/2206.08522v1) [cs.RO]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.08522Focus to learn more |





<h2 id="2022-06-20-2">2. Bridge-Tower: Building Bridges Between Encoders in Vision-Language Representation Learning
</h2>

Title: [Bridge-Tower: Building Bridges Between Encoders in Vision-Language Representation Learning](https://arxiv.org/abs/2206.08657)

Authors: [Xiao Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+X), [Chenfei Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+C), [Shachar Rosenman](https://arxiv.org/search/cs?searchtype=author&query=Rosenman%2C+S), [Vasudev Lal](https://arxiv.org/search/cs?searchtype=author&query=Lal%2C+V), [Nan Duan](https://arxiv.org/search/cs?searchtype=author&query=Duan%2C+N)

> Vision-Language (VL) models with the Two-Tower architecture have dominated visual-language representation learning in recent years. Current VL models either use lightweight uni-modal encoders and learn to extract, align and fuse both modalities simultaneously in a cross-modal encoder, or feed the last-layer uni-modal features directly into the top cross-modal encoder, ignoring the semantic information at the different levels in the deep uni-modal encoders. Both approaches possibly restrict vision-language representation learning and limit model performance. In this paper, we introduce multiple bridge layers that build a connection between the top layers of uni-modal encoders and each layer of the cross-modal encoder. This enables comprehensive bottom-up interactions between visual and textual representations at different semantic levels, resulting in more effective cross-modal alignment and fusion. Our proposed Bridge-Tower, pre-trained with only 4M images, achieves state-of-the-art performance on various downstream vision-language tasks. On the VQAv2 test-std set, Bridge-Tower achieves an accuracy of 78.73%, outperforming the previous state-of-the-art METER model by 1.09% with the same pre-training data and almost no additional parameters and computational cost. Notably, when further scaling the model, Bridge-Tower achieves an accuracy of 81.15%, surpassing models that are pre-trained on orders-of-magnitude larger datasets. Code is available at [this https URL](https://github.com/microsoft/BridgeTower).

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.08657](https://arxiv.org/abs/2206.08657) [cs.CV]** |
|           | (or **[arXiv:2206.08657v1](https://arxiv.org/abs/2206.08657v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.08657Focus to learn more |





<h2 id="2022-06-20-3">3. Automatic Correction of Human Translations
</h2>

Title: [Automatic Correction of Human Translations](https://arxiv.org/abs/2206.08593)

Authors: [Jessy Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+J), [Geza Kovacs](https://arxiv.org/search/cs?searchtype=author&query=Kovacs%2C+G), [Aditya Shastry](https://arxiv.org/search/cs?searchtype=author&query=Shastry%2C+A), [Joern Wuebker](https://arxiv.org/search/cs?searchtype=author&query=Wuebker%2C+J), [John DeNero](https://arxiv.org/search/cs?searchtype=author&query=DeNero%2C+J)

> We introduce translation error correction (TEC), the task of automatically correcting human-generated translations. Imperfections in machine translations (MT) have long motivated systems for improving translations post-hoc with automatic post-editing. In contrast, little attention has been devoted to the problem of automatically correcting human translations, despite the intuition that humans make distinct errors that machines would be well-suited to assist with, from typos to inconsistencies in translation conventions. To investigate this, we build and release the Aced corpus with three TEC datasets. We show that human errors in TEC exhibit a more diverse range of errors and far fewer translation fluency errors than the MT errors in automatic post-editing datasets, suggesting the need for dedicated TEC models that are specialized to correct human errors. We show that pre-training instead on synthetic errors based on human errors improves TEC F-score by as much as 5.1 points. We conducted a human-in-the-loop user study with nine professional translation editors and found that the assistance of our TEC system led them to produce significantly higher quality revised translations.

| Comments: | NAACL 2022. Dataset available at: [this https URL](https://github.com/lilt/tec) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2206.08593](https://arxiv.org/abs/2206.08593) [cs.CL]** |
|           | (or **[arXiv:2206.08593v1](https://arxiv.org/abs/2206.08593v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.08593Focus to learn more |





<h2 id="2022-06-20-4">4. Language with Vision: a Study on Grounded Word and Sentence Embeddings
</h2>

Title: [Language with Vision: a Study on Grounded Word and Sentence Embeddings](https://arxiv.org/abs/2206.08823)

Authors: [Hassan Shahmohammadi](https://arxiv.org/search/cs?searchtype=author&query=Shahmohammadi%2C+H), [Maria Heitmeier](https://arxiv.org/search/cs?searchtype=author&query=Heitmeier%2C+M), [Elnaz Shafaei-Bajestan](https://arxiv.org/search/cs?searchtype=author&query=Shafaei-Bajestan%2C+E), [Hendrik P. A. Lensch](https://arxiv.org/search/cs?searchtype=author&query=Lensch%2C+H+P+A), [Harald Baayen](https://arxiv.org/search/cs?searchtype=author&query=Baayen%2C+H)

> Language grounding to vision is an active field of research aiming to enrich text-based representations of word meanings by leveraging perceptual knowledge from vision. Despite many attempts at language grounding, it is still unclear how to effectively inject visual knowledge into the word embeddings of a language in such a way that a proper balance of textual and visual knowledge is maintained. Some common concerns are the following. Is visual grounding beneficial for abstract words or is its contribution only limited to concrete words? What is the optimal way of bridging the gap between text and vision? How much do we gain by visually grounding textual embeddings? The present study addresses these questions by proposing a simple yet very effective grounding approach for pre-trained word embeddings. Our model aligns textual embeddings with vision while largely preserving the distributional statistics that characterize word use in text corpora. By applying a learned alignment, we are able to generate visually grounded embeddings for unseen words, including abstract words. A series of evaluations on word similarity benchmarks shows that visual grounding is beneficial not only for concrete words, but also for abstract words. We also show that our method for visual grounding offers advantages for contextualized embeddings, but only when these are trained on corpora of relatively modest size. Code and grounded embeddings for English are available at [this https URL](https://github.com/Hazel1994/Visually_Grounded_Word_Embeddings_2).

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.08823](https://arxiv.org/abs/2206.08823) [cs.CL]** |
|           | (or **[arXiv:2206.08823v1](https://arxiv.org/abs/2206.08823v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.08823Focus to learn more |






# 2022-06-17

[Return to Index](#Index)



<h2 id="2022-06-17-1">1. How Adults Understand What Young Children Say
</h2>

Title: [How Adults Understand What Young Children Say](https://arxiv.org/abs/2206.07807)

Authors: [Stephan C. Meylan](https://arxiv.org/search/cs?searchtype=author&query=Meylan%2C+S+C), [Ruthe Foushee](https://arxiv.org/search/cs?searchtype=author&query=Foushee%2C+R), [Nicole H. Wong](https://arxiv.org/search/cs?searchtype=author&query=Wong%2C+N+H), [Elika Bergelson](https://arxiv.org/search/cs?searchtype=author&query=Bergelson%2C+E), [Roger P. Levy](https://arxiv.org/search/cs?searchtype=author&query=Levy%2C+R+P)

> Children's early speech often bears little resemblance to adult speech in form or content, and yet caregivers often find meaning in young children's utterances. Precisely how caregivers are able to do this remains poorly understood. We propose that successful early communication (an essential building block of language development) relies not just on children's growing linguistic knowledge, but also on adults' sophisticated inferences. These inferences, we further propose, are optimized for fine-grained details of how children speak. We evaluate these ideas using a set of candidate computational models of spoken word recognition based on deep learning and Bayesian inference, which instantiate competing hypotheses regarding the information sources used by adults to understand children. We find that the best-performing models (evaluated on datasets of adult interpretations of child speech) are those that have strong prior expectations about what children are likely to want to communicate, rather than the actual phonetic contents of what children say. We further find that adults' behavior is best characterized as well-tuned to specific children: the more closely a word recognition model is tuned to the particulars of an individual child's actual linguistic behavior, the better it predicts adults' inferences about what the child has said. These results offer a comprehensive investigation into the role of caregivers as child-directed listeners, with broader consequences for theories of language acquisition.

| Comments: | 19 pages, 6 figures, 2 tables                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2206.07807](https://arxiv.org/abs/2206.07807) [cs.CL]** |
|           | (or **[arXiv:2206.07807v1](https://arxiv.org/abs/2206.07807v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.07807Focus to learn more |





<h2 id="2022-06-17-2">2. Alexa Teacher Model: Pretraining and Distilling Multi-Billion-Parameter Encoders for Natural Language Understanding Systems
</h2>

Title: [Alexa Teacher Model: Pretraining and Distilling Multi-Billion-Parameter Encoders for Natural Language Understanding Systems](https://arxiv.org/abs/2206.07808)

Authors: [Jack FitzGerald](https://arxiv.org/search/cs?searchtype=author&query=FitzGerald%2C+J), [Shankar Ananthakrishnan](https://arxiv.org/search/cs?searchtype=author&query=Ananthakrishnan%2C+S), [Konstantine Arkoudas](https://arxiv.org/search/cs?searchtype=author&query=Arkoudas%2C+K), [Davide Bernardi](https://arxiv.org/search/cs?searchtype=author&query=Bernardi%2C+D), [Abhishek Bhagia](https://arxiv.org/search/cs?searchtype=author&query=Bhagia%2C+A), [Claudio Delli Bovi](https://arxiv.org/search/cs?searchtype=author&query=Bovi%2C+C+D), [Jin Cao](https://arxiv.org/search/cs?searchtype=author&query=Cao%2C+J), [Rakesh Chada](https://arxiv.org/search/cs?searchtype=author&query=Chada%2C+R), [Amit Chauhan](https://arxiv.org/search/cs?searchtype=author&query=Chauhan%2C+A), [Luoxin Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+L), [Anurag Dwarakanath](https://arxiv.org/search/cs?searchtype=author&query=Dwarakanath%2C+A), [Satyam Dwivedi](https://arxiv.org/search/cs?searchtype=author&query=Dwivedi%2C+S), [Turan Gojayev](https://arxiv.org/search/cs?searchtype=author&query=Gojayev%2C+T), [Karthik Gopalakrishnan](https://arxiv.org/search/cs?searchtype=author&query=Gopalakrishnan%2C+K), [Thomas Gueudre](https://arxiv.org/search/cs?searchtype=author&query=Gueudre%2C+T), [Dilek Hakkani-Tur](https://arxiv.org/search/cs?searchtype=author&query=Hakkani-Tur%2C+D), [Wael Hamza](https://arxiv.org/search/cs?searchtype=author&query=Hamza%2C+W), [Jonathan Hueser](https://arxiv.org/search/cs?searchtype=author&query=Hueser%2C+J), [Kevin Martin Jose](https://arxiv.org/search/cs?searchtype=author&query=Jose%2C+K+M), [Haidar Khan](https://arxiv.org/search/cs?searchtype=author&query=Khan%2C+H), [Beiye Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+B), [Jianhua Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+J), [Alessandro Manzotti](https://arxiv.org/search/cs?searchtype=author&query=Manzotti%2C+A), [Pradeep Natarajan](https://arxiv.org/search/cs?searchtype=author&query=Natarajan%2C+P), [Karolina Owczarzak](https://arxiv.org/search/cs?searchtype=author&query=Owczarzak%2C+K), [Gokmen Oz](https://arxiv.org/search/cs?searchtype=author&query=Oz%2C+G), [Enrico Palumbo](https://arxiv.org/search/cs?searchtype=author&query=Palumbo%2C+E), [Charith Peris](https://arxiv.org/search/cs?searchtype=author&query=Peris%2C+C), [Chandana Satya Prakash](https://arxiv.org/search/cs?searchtype=author&query=Prakash%2C+C+S), [Stephen Rawls](https://arxiv.org/search/cs?searchtype=author&query=Rawls%2C+S), [Andy Rosenbaum](https://arxiv.org/search/cs?searchtype=author&query=Rosenbaum%2C+A), [Anjali Shenoy](https://arxiv.org/search/cs?searchtype=author&query=Shenoy%2C+A), [Saleh Soltan](https://arxiv.org/search/cs?searchtype=author&query=Soltan%2C+S), [Mukund Harakere Sridhar](https://arxiv.org/search/cs?searchtype=author&query=Sridhar%2C+M+H), [Liz Tan](https://arxiv.org/search/cs?searchtype=author&query=Tan%2C+L), [Fabian Triefenbach](https://arxiv.org/search/cs?searchtype=author&query=Triefenbach%2C+F), [Pan Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+P), [Haiyang Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+H), [Shuai Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+S), [Gokhan Tur](https://arxiv.org/search/cs?searchtype=author&query=Tur%2C+G), [Prem Natarajan](https://arxiv.org/search/cs?searchtype=author&query=Natarajan%2C+P)

> We present results from a large-scale experiment on pretraining encoders with non-embedding parameter counts ranging from 700M to 9.3B, their subsequent distillation into smaller models ranging from 17M-170M parameters, and their application to the Natural Language Understanding (NLU) component of a virtual assistant system. Though we train using 70% spoken-form data, our teacher models perform comparably to XLM-R and mT5 when evaluated on the written-form Cross-lingual Natural Language Inference (XNLI) corpus. We perform a second stage of pretraining on our teacher models using in-domain data from our system, improving error rates by 3.86% relative for intent classification and 7.01% relative for slot filling. We find that even a 170M-parameter model distilled from our Stage 2 teacher model has 2.88% better intent classification and 7.69% better slot filling error rates when compared to the 2.3B-parameter teacher trained only on public data (Stage 1), emphasizing the importance of in-domain data for pretraining. When evaluated offline using labeled NLU data, our 17M-parameter Stage 2 distilled model outperforms both XLM-R Base (85M params) and DistillBERT (42M params) by 4.23% to 6.14%, respectively. Finally, we present results from a full virtual assistant experimentation platform, where we find that models trained using our pretraining and distillation pipeline outperform models distilled from 85M-parameter teachers by 3.74%-4.91% on an automatic measurement of full-system user dissatisfaction.

| Comments:          | KDD 2022                                                     |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| ACM classes:       | I.2.7                                                        |
| Cite as:           | **[arXiv:2206.07808](https://arxiv.org/abs/2206.07808) [cs.CL]** |
|                    | (or **[arXiv:2206.07808v1](https://arxiv.org/abs/2206.07808v1) [cs.CL]** for this version) |
|                    | https://doi.org/10.48550/arXiv.2206.07808Focus to learn more |
| Journal reference: | Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '22), August 14-18, 2022, Washington, DC, USA |
| Related DOI:       | https://doi.org/10.1145/3534678.3539173Focus to learn more   |





<h2 id="2022-06-17-3">3. TransDrift: Modeling Word-Embedding Drift using Transformer
</h2>

Title: [TransDrift: Modeling Word-Embedding Drift using Transformer](https://arxiv.org/abs/2206.08081)

Authors: [Nishtha Madaan](https://arxiv.org/search/cs?searchtype=author&query=Madaan%2C+N), [Prateek Chaudhury](https://arxiv.org/search/cs?searchtype=author&query=Chaudhury%2C+P), [Nishant Kumar](https://arxiv.org/search/cs?searchtype=author&query=Kumar%2C+N), [Srikanta Bedathur](https://arxiv.org/search/cs?searchtype=author&query=Bedathur%2C+S)

> In modern NLP applications, word embeddings are a crucial backbone that can be readily shared across a number of tasks. However as the text distributions change and word semantics evolve over time, the downstream applications using the embeddings can suffer if the word representations do not conform to the data drift. Thus, maintaining word embeddings to be consistent with the underlying data distribution is a key problem. In this work, we tackle this problem and propose TransDrift, a transformer-based prediction model for word embeddings. Leveraging the flexibility of transformer, our model accurately learns the dynamics of the embedding drift and predicts the future embedding. In experiments, we compare with existing methods and show that our model makes significantly more accurate predictions of the word embedding than the baselines. Crucially, by applying the predicted embeddings as a backbone for downstream classification tasks, we show that our embeddings lead to superior performance compared to the previous methods.

| Comments: | 10 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2206.08081](https://arxiv.org/abs/2206.08081) [cs.CL]** |
|           | (or **[arXiv:2206.08081v1](https://arxiv.org/abs/2206.08081v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.08081Focus to learn more |





<h2 id="2022-06-17-4">4. Self-Generated In-Context Learning: Leveraging Auto-regressive Language Models as a Demonstration Generator
</h2>

Title: [Self-Generated In-Context Learning: Leveraging Auto-regressive Language Models as a Demonstration Generator](https://arxiv.org/abs/2206.08082)

Authors: [Hyuhng Joon Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+H+J), [Hyunsoo Cho](https://arxiv.org/search/cs?searchtype=author&query=Cho%2C+H), [Junyeob Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+J), [Taeuk Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+T), [Kang Min Yoo](https://arxiv.org/search/cs?searchtype=author&query=Yoo%2C+K+M), [Sang-goo Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+S)

> Large-scale pre-trained language models (PLMs) are well-known for being capable of solving a task simply by conditioning a few input-label pairs dubbed demonstrations on a prompt without being explicitly tuned for the desired downstream task. Such a process (i.e., in-context learning), however, naturally leads to high reliance on the demonstrations which are usually selected from external datasets. In this paper, we propose self-generated in-context learning (SG-ICL), which generates demonstrations for in-context learning from PLM itself to minimize the reliance on the external demonstration. We conduct experiments on four different text classification tasks and show SG-ICL significantly outperforms zero-shot learning and is generally worth approximately 0.6 gold training samples. Moreover, our generated demonstrations show more consistent performance with low variance compared to randomly selected demonstrations from the training dataset.

| Comments: | NAACL 2022 Workshop on Large-scale Pre-trained Language Models |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2206.08082](https://arxiv.org/abs/2206.08082) [cs.CL]** |
|           | (or **[arXiv:2206.08082v1](https://arxiv.org/abs/2206.08082v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.08082Focus to learn more |





<h2 id="2022-06-17-5">5. Deep Learning Architecture for Automatic Essay Scoring
</h2>

Title: [Deep Learning Architecture for Automatic Essay Scoring](https://arxiv.org/abs/2206.08232)

Authors: [Tsegaye Misikir Tashu](https://arxiv.org/search/cs?searchtype=author&query=Tashu%2C+T+M), [Chandresh Kumar Maurya](https://arxiv.org/search/cs?searchtype=author&query=Maurya%2C+C+K), [Tomas Horvath](https://arxiv.org/search/cs?searchtype=author&query=Horvath%2C+T)

> Automatic evaluation of essay (AES) and also called automatic essay scoring has become a severe problem due to the rise of online learning and evaluation platforms such as Coursera, Udemy, Khan academy, and so on. Researchers have recently proposed many techniques for automatic evaluation. However, many of these techniques use hand-crafted features and thus are limited from the feature representation point of view. Deep learning has emerged as a new paradigm in machine learning which can exploit the vast data and identify the features useful for essay evaluation. To this end, we propose a novel architecture based on recurrent networks (RNN) and convolution neural network (CNN). In the proposed architecture, the multichannel convolutional layer learns and captures the contextual features of the word n-gram from the word embedding vectors and the essential semantic concepts to form the feature vector at essay level using max-pooling operation. A variant of RNN called Bi-gated recurrent unit (BGRU) is used to access both previous and subsequent contextual representations. The experiment was carried out on eight data sets available on Kaggle for the task of AES. The experimental results show that our proposed system achieves significantly higher grading accuracy than other deep learning-based AES systems and also other state-of-the-art AES systems.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.08232](https://arxiv.org/abs/2206.08232) [cs.CL]** |
|           | (or **[arXiv:2206.08232v1](https://arxiv.org/abs/2206.08232v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.08232Focus to learn more |







# 2022-06-16

[Return to Index](#Index)



<h2 id="2022-06-16-1">1. Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone
</h2>

Title: [Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone](https://arxiv.org/abs/2206.07643)

Authors: [Zi-Yi Dou](https://arxiv.org/search/cs?searchtype=author&query=Dou%2C+Z), [Aishwarya Kamath](https://arxiv.org/search/cs?searchtype=author&query=Kamath%2C+A), [Zhe Gan](https://arxiv.org/search/cs?searchtype=author&query=Gan%2C+Z), [Pengchuan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+P), [Jianfeng Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+J), [Linjie Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L), [Zicheng Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Z), [Ce Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+C), [Yann LeCun](https://arxiv.org/search/cs?searchtype=author&query=LeCun%2C+Y), [Nanyun Peng](https://arxiv.org/search/cs?searchtype=author&query=Peng%2C+N), [Jianfeng Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+J), [Lijuan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L)

> Vision-language (VL) pre-training has recently received considerable attention. However, most existing end-to-end pre-training approaches either only aim to tackle VL tasks such as image-text retrieval, visual question answering (VQA) and image captioning that test high-level understanding of images, or only target region-level understanding for tasks such as phrase grounding and object detection. We present FIBER (Fusion-In-the-Backbone-based transformER), a new VL model architecture that can seamlessly handle both these types of tasks. Instead of having dedicated transformer layers for fusion after the uni-modal backbones, FIBER pushes multimodal fusion deep into the model by inserting cross-attention into the image and text backbones, bringing gains in terms of memory and performance. In addition, unlike previous work that is either only pre-trained on image-text data or on fine-grained data with box-level annotations, we present a two-stage pre-training strategy that uses both these kinds of data efficiently: (i) coarse-grained pre-training based on image-text data; followed by (ii) fine-grained pre-training based on image-text-box data. We conduct comprehensive experiments on a wide range of VL tasks, ranging from VQA, image captioning, and retrieval, to phrase grounding, referring expression comprehension, and object detection. Using deep multimodal fusion coupled with the two-stage pre-training, FIBER provides consistent performance improvements over strong baselines across all tasks, often outperforming methods using magnitudes more data. Code is available at [this https URL](https://github.com/microsoft/FIBER).

| Comments: | Project Website: [this https URL](https://ashkamath.github.io/FIBER_page) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2206.07643](https://arxiv.org/abs/2206.07643) [cs.CV]** |
|           | (or **[arXiv:2206.07643v1](https://arxiv.org/abs/2206.07643v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.07643Focus to learn more |





<h2 id="2022-06-16-2">2. A Unified Sequence Interface for Vision Tasks
</h2>

Title: [A Unified Sequence Interface for Vision Tasks](https://arxiv.org/abs/2206.07669)

Authors: [Ting Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+T), [Saurabh Saxena](https://arxiv.org/search/cs?searchtype=author&query=Saxena%2C+S), [Lala Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L), [Tsung-Yi Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+T), [David J. Fleet](https://arxiv.org/search/cs?searchtype=author&query=Fleet%2C+D+J), [Geoffrey Hinton](https://arxiv.org/search/cs?searchtype=author&query=Hinton%2C+G)

> While language tasks are naturally expressed in a single, unified, modeling framework, i.e., generating sequences of tokens, this has not been the case in computer vision. As a result, there is a proliferation of distinct architectures and loss functions for different vision tasks. In this work we show that a diverse set of "core" computer vision tasks can also be unified if formulated in terms of a shared pixel-to-sequence interface. We focus on four tasks, namely, object detection, instance segmentation, keypoint detection, and image captioning, all with diverse types of outputs, e.g., bounding boxes or dense masks. Despite that, by formulating the output of each task as a sequence of discrete tokens with a unified interface, we show that one can train a neural network with a single model architecture and loss function on all these tasks, with no task-specific customization. To solve a specific task, we use a short prompt as task description, and the sequence output adapts to the prompt so it can produce task-specific output. We show that such a model can achieve competitive performance compared to well-established task-specific models.

| Comments: | The first three authors contributed equally                  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2206.07669](https://arxiv.org/abs/2206.07669) [cs.CV]** |
|           | (or **[arXiv:2206.07669v1](https://arxiv.org/abs/2206.07669v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.07669Focus to learn more |





<h2 id="2022-06-16-3">3. Prefix Language Models are Unified Modal Learners
</h2>

Title: [Prefix Language Models are Unified Modal Learners](https://arxiv.org/abs/2206.07699)

Authors: [Shizhe Diao](https://arxiv.org/search/cs?searchtype=author&query=Diao%2C+S), [Wangchunshu Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+W), [Xinsong Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+X), [Jiawei Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+J)

> With the success of vision-language pre-training, we have witnessed the state-of-the-art has been pushed on multi-modal understanding and generation. However, the current pre-training paradigm is either incapable of targeting all modalities at once (e.g., text generation and image generation), or requires multi-fold well-designed tasks which significantly limits the scalability. We demonstrate that a unified modal model could be learned with a prefix language modeling objective upon text and image sequences. Thanks to the simple but powerful pre-training paradigm, our proposed model, DaVinci, is simple to train, scalable to huge data, and adaptable to a variety of downstream tasks across modalities (language / vision / vision+language), types (understanding / generation) and settings (e.g., zero-shot, fine-tuning, linear evaluation) with a single unified architecture. DaVinci achieves the competitive performance on a wide range of 26 understanding / generation tasks, and outperforms previous unified vision-language models on most tasks, including ImageNet classification (+1.6%), VQAv2 (+1.4%), COCO caption generation (BLEU@4 +1.1%, CIDEr +1.5%) and COCO image generation (IS +0.9%, FID -1.0%), at the comparable model and data scale. Furthermore, we offer a well-defined benchmark for future research by reporting the performance on different scales of the pre-training dataset on a heterogeneous and wide distribution coverage. Our results establish new, stronger baselines for future comparisons at different data scales and shed light on the difficulties of comparing VLP models more generally.

| Comments: | 22 pages, 3 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2206.07699](https://arxiv.org/abs/2206.07699) [cs.CV]** |
|           | (or **[arXiv:2206.07699v1](https://arxiv.org/abs/2206.07699v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.07699Focus to learn more |





<h2 id="2022-06-16-4">4. Human Heuristics for AI-Generated Language Are Flawed
</h2>

Title: [Human Heuristics for AI-Generated Language Are Flawed](https://arxiv.org/abs/2206.07271)

Authors: [Maurice Jakesch](https://arxiv.org/search/cs?searchtype=author&query=Jakesch%2C+M), [Jeffrey Hancock](https://arxiv.org/search/cs?searchtype=author&query=Hancock%2C+J), [Mor Naaman](https://arxiv.org/search/cs?searchtype=author&query=Naaman%2C+M)

> Human communication is increasingly intermixed with language generated by AI. Across chat, email, and social media, AI systems produce smart replies, autocompletes, and translations. AI-generated language is often not identified as such but poses as human language, raising concerns about novel forms of deception and manipulation. Here, we study how humans discern whether one of the most personal and consequential forms of language - a self-presentation - was generated by AI. Across six experiments, participants (N = 4,650) tried to identify self-presentations generated by state-of-the-art language models. Across professional, hospitality, and romantic settings, we find that humans are unable to identify AI-generated self-presentations. Combining qualitative analyses with language feature engineering, we find that human judgments of AI-generated language are handicapped by intuitive but flawed heuristics such as associating first-person pronouns, authentic words, or family topics with humanity. We show that these heuristics make human judgment of generated language predictable and manipulable, allowing AI systems to produce language perceived as more human than human. We conclude by discussing solutions - such as AI accents or fair use policies - to reduce the deceptive potential of generated language, limiting the subversion of human intuition.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Computers and Society (cs.CY); Human-Computer Interaction (cs.HC) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.07271](https://arxiv.org/abs/2206.07271) [cs.CL]** |
|           | (or **[arXiv:2206.07271v1](https://arxiv.org/abs/2206.07271v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.07271Focus to learn more |





<h2 id="2022-06-16-5">5. MPI: Evaluating and Inducing Personality in Pre-trained Language Models
</h2>

Title: [MPI: Evaluating and Inducing Personality in Pre-trained Language Models](https://arxiv.org/abs/2206.07550)

Authors: [Guangyuan Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+G), [Manjie Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+M), [Song-Chun Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+S), [Wenjuan Han](https://arxiv.org/search/cs?searchtype=author&query=Han%2C+W), [Chi Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+C), [Yixin Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+Y)

> Originated as a philosophical quest, personality discerns how individuals differ from each other in terms of thinking, feeling, and behaving. Towards building social machines that work with humans on a daily basis, we are motivated to ask: (1) Do existing pre-trained language models possess personality, akin to their human counterpart? If so, (2) how can we evaluate them? Further, given this evaluation framework, (3) how can we induce a certain personality in a fully controllable fashion? To tackle these three questions, we propose the Machine Personality Inventory (MPI) dataset for evaluating the machine personality; MPI follows standardized personality tests, built upon the Big Five Personality Factors (Big Five) theory and personality assessment inventories. By evaluating models with MPI, we provide the first piece of evidence showing the existence of personality in pre-trained language models. We further devise a Chain Prompting method to induce the language model with a specific personality in a controllable manner, capable of producing diversified behaviors. We hope to shed light on future studies by adopting personality as the essential psychological guidance for various downstream tasks, building more human-like and in situ dialogue agents.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.07550](https://arxiv.org/abs/2206.07550) [cs.CL]** |
|           | (or **[arXiv:2206.07550v1](https://arxiv.org/abs/2206.07550v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.07550Focus to learn more |





<h2 id="2022-06-16-6">6. Emergent Abilities of Large Language Models
</h2>

Title: [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682)

Authors: [Jason Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+J), [Yi Tay](https://arxiv.org/search/cs?searchtype=author&query=Tay%2C+Y), [Rishi Bommasani](https://arxiv.org/search/cs?searchtype=author&query=Bommasani%2C+R), [Colin Raffel](https://arxiv.org/search/cs?searchtype=author&query=Raffel%2C+C), [Barret Zoph](https://arxiv.org/search/cs?searchtype=author&query=Zoph%2C+B), [Sebastian Borgeaud](https://arxiv.org/search/cs?searchtype=author&query=Borgeaud%2C+S), [Dani Yogatama](https://arxiv.org/search/cs?searchtype=author&query=Yogatama%2C+D), [Maarten Bosma](https://arxiv.org/search/cs?searchtype=author&query=Bosma%2C+M), [Denny Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+D), [Donald Metzler](https://arxiv.org/search/cs?searchtype=author&query=Metzler%2C+D), [Ed H. Chi](https://arxiv.org/search/cs?searchtype=author&query=Chi%2C+E+H), [Tatsunori Hashimoto](https://arxiv.org/search/cs?searchtype=author&query=Hashimoto%2C+T), [Oriol Vinyals](https://arxiv.org/search/cs?searchtype=author&query=Vinyals%2C+O), [Percy Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+P), [Jeff Dean](https://arxiv.org/search/cs?searchtype=author&query=Dean%2C+J), [William Fedus](https://arxiv.org/search/cs?searchtype=author&query=Fedus%2C+W)

> Scaling up language models has been shown to predictably improve performance and sample efficiency on a wide range of downstream tasks. This paper instead discusses an unpredictable phenomenon that we refer to as emergent abilities of large language models. We consider an ability to be emergent if it is not present in smaller models but is present in larger models. Thus, emergent abilities cannot be predicted simply by extrapolating the performance of smaller models. The existence of such emergence implies that additional scaling could further expand the range of capabilities of language models.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.07682](https://arxiv.org/abs/2206.07682) [cs.CL]** |
|           | (or **[arXiv:2206.07682v1](https://arxiv.org/abs/2206.07682v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.07682Focus to learn more |






# 2022-06-15

[Return to Index](#Index)



<h2 id="2022-06-15-1">1. LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning
</h2>

Title: [LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning](https://arxiv.org/abs/2206.06522)

Authors: [Yi-Lin Sung](https://arxiv.org/search/cs?searchtype=author&query=Sung%2C+Y), [Jaemin Cho](https://arxiv.org/search/cs?searchtype=author&query=Cho%2C+J), [Mohit Bansal](https://arxiv.org/search/cs?searchtype=author&query=Bansal%2C+M)

> Fine-tuning large pre-trained models on downstream tasks has been adopted in a variety of domains recently. However, it is costly to update the entire parameter set of large pre-trained models. Although recently proposed parameter-efficient transfer learning (PETL) techniques allow updating a small subset of parameters (e.g. only using 2% of parameters) inside a pre-trained backbone network for a new task, they only reduce the training memory requirement by up to 30%. This is because the gradient computation for the trainable parameters still requires backpropagation through the large pre-trained backbone model. To address this, we propose Ladder Side-Tuning (LST), a new PETL technique that reduces training memory requirements by more substantial amounts. Unlike existing parameter-efficient methods that insert additional parameters inside backbone networks, we train a ladder side network, a small and separate network that takes intermediate activations as input via shortcut connections (ladders) from backbone networks and makes predictions. LST has significantly lower memory requirements than previous methods, because it does not require backpropagation through the backbone network, but instead only through the side network and ladder connections. We evaluate our method with various models (T5, CLIP-T5) on both NLP (GLUE) and vision-language (VQA, GQA, NLVR2, MSCOCO) tasks. LST saves 69% of the memory costs to fine-tune the whole network, while other methods only save 26% of that in similar parameter usages (hence, 2.7x more memory savings). Moreover, LST achieves higher accuracy than Adapter and LoRA in a low-memory regime. To further show the advantage of this better memory efficiency, we also apply LST to larger T5 models (T5-large, T5-3B), attaining better GLUE performance than full fine-tuning and other PETL methods. The exact same trend also holds in our experiments on VL tasks.

| Comments: | 13 pages; our code is available at: [this https URL](https://github.com/ylsung/Ladder-Side-Tuning) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV) |
| Cite as:  | **[arXiv:2206.06522](https://arxiv.org/abs/2206.06522) [cs.CL]** |
|           | (or **[arXiv:2206.06522v1](https://arxiv.org/abs/2206.06522v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.06522Focus to learn more |

