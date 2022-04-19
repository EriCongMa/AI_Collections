# MA C.'s Daily Paper Of Interest - April b., 2022

# Index

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

