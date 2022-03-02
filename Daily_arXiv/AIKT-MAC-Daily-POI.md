# MA C.'s Daily Paper Of Interest - March, 2022

# Index

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



