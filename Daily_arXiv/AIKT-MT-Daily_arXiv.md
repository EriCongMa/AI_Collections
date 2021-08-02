# Daily arXiv: Machine Translation - August, 2021

# Index


- [2021-08-02](#2021-08-02)

  - [1. Difficulty-Aware Machine Translation Evaluation](#2021-08-02-1)
  - [2. Residual Tree Aggregation of Layers for Neural Machine Translation](#2021-08-02-2)
  - [3. Neural Variational Learning for Grounded Language Acquisition](#2021-08-02-3)
  - [4. Multi-stage Pre-training over Simplified Multimodal Pre-training Models](#2021-08-02-4)
  - [5. MDQE: A More Accurate Direct Pretraining for Machine Translation Quality Estimation](#2021-08-02-5)
  - [6. Towards Universality in Multilingual Text Rewriting](#2021-08-02-6)
  - [7. ChrEnTranslate: Cherokee-English Machine Translation Demo with Quality Estimation and Corrective Feedback](#2021-08-02-7)
- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MT-Daily_arXiv-index.md)



# 2021-08-02

[Return to Index](#Index)



<h2 id="2021-08-02-1">1. Difficulty-Aware Machine Translation Evaluation
</h2>

Title: [Difficulty-Aware Machine Translation Evaluation](https://arxiv.org/abs/2107.14402)

Authors: [Runzhe Zhan](https://arxiv.org/search/cs?searchtype=author&query=Zhan%2C+R), [Xuebo Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+X), [Derek F. Wong](https://arxiv.org/search/cs?searchtype=author&query=Wong%2C+D+F), [Lidia S. Chao](https://arxiv.org/search/cs?searchtype=author&query=Chao%2C+L+S)

> The high-quality translation results produced by machine translation (MT) systems still pose a huge challenge for automatic evaluation. Current MT evaluation pays the same attention to each sentence component, while the questions of real-world examinations (e.g., university examinations) have different difficulties and weightings. In this paper, we propose a novel difficulty-aware MT evaluation metric, expanding the evaluation dimension by taking translation difficulty into consideration. A translation that fails to be predicted by most MT systems will be treated as a difficult one and assigned a large weight in the final score function, and conversely. Experimental results on the WMT19 English-German Metrics shared tasks show that our proposed method outperforms commonly used MT metrics in terms of human correlation. In particular, our proposed method performs well even when all the MT systems are very competitive, which is when most existing metrics fail to distinguish between them. The source code is freely available at [this https URL](https://github.com/NLP2CT/Difficulty-Aware-MT-Evaluation).

| Comments: | Accepted to ACL 2021                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2107.14402](https://arxiv.org/abs/2107.14402) [cs.CL]** |
|           | (or **[arXiv:2107.14402v1](https://arxiv.org/abs/2107.14402v1) [cs.CL]** for this version) |





<h2 id="2021-08-02-2">2. Residual Tree Aggregation of Layers for Neural Machine Translation
</h2>

Title: [Residual Tree Aggregation of Layers for Neural Machine Translation](https://arxiv.org/abs/2107.14590)

Authors: [GuoLiang Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+G), [Yiyang Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y)

> Although attention-based Neural Machine Translation has achieved remarkable progress in recent layers, it still suffers from issue of making insufficient use of the output of each layer. In transformer, it only uses the top layer of encoder and decoder in the subsequent process, which makes it impossible to take advantage of the useful information in other layers. To address this issue, we propose a residual tree aggregation of layers for Transformer(RTAL), which helps to fuse information across layers. Specifically, we try to fuse the information across layers by constructing a post-order binary tree. In additional to the last node, we add the residual connection to the process of generating child nodes. Our model is based on the Neural Machine Translation model Transformer and we conduct our experiments on WMT14 English-to-German and WMT17 English-to-France translation tasks. Experimental results across language pairs show that the proposed approach outperforms the strong baseline model significantly

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2107.14590](https://arxiv.org/abs/2107.14590) [cs.CL]** |
|           | (or **[arXiv:2107.14590v1](https://arxiv.org/abs/2107.14590v1) [cs.CL]** for this version) |





<h2 id="2021-08-02-3">3. Neural Variational Learning for Grounded Language Acquisition
</h2>

Title: [Neural Variational Learning for Grounded Language Acquisition](https://arxiv.org/abs/2107.14593)

Authors: [Nisha Pillai](https://arxiv.org/search/cs?searchtype=author&query=Pillai%2C+N), [Cynthia Matuszek](https://arxiv.org/search/cs?searchtype=author&query=Matuszek%2C+C), [Francis Ferraro](https://arxiv.org/search/cs?searchtype=author&query=Ferraro%2C+F)

> We propose a learning system in which language is grounded in visual percepts without specific pre-defined categories of terms. We present a unified generative method to acquire a shared semantic/visual embedding that enables the learning of language about a wide range of real-world objects. We evaluate the efficacy of this learning by predicting the semantics of objects and comparing the performance with neural and non-neural inputs. We show that this generative approach exhibits promising results in language grounding without pre-specifying visual categories under low resource settings. Our experiments demonstrate that this approach is generalizable to multilingual, highly varied datasets.

| Subjects:          | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Robotics (cs.RO) |
| ------------------ | ------------------------------------------------------------ |
| Journal reference: | 2021 30th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN) |
| Cite as:           | **[arXiv:2107.14593](https://arxiv.org/abs/2107.14593) [cs.CL]** |
|                    | (or **[arXiv:2107.14593v1](https://arxiv.org/abs/2107.14593v1) [cs.CL]** for this version) |





<h2 id="2021-08-02-4">4. Multi-stage Pre-training over Simplified Multimodal Pre-training Models
</h2>

Title: [Multi-stage Pre-training over Simplified Multimodal Pre-training Models](https://arxiv.org/abs/2107.14596)

Authors: [Tongtong Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+T), [Fangxiang Feng](https://arxiv.org/search/cs?searchtype=author&query=Feng%2C+F), [Xiaojie Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X)

> Multimodal pre-training models, such as LXMERT, have achieved excellent results in downstream tasks. However, current pre-trained models require large amounts of training data and have huge model sizes, which make them difficult to apply in low-resource situations. How to obtain similar or even better performance than a larger model under the premise of less pre-training data and smaller model size has become an important problem. In this paper, we propose a new Multi-stage Pre-training (MSP) method, which uses information at different granularities from word, phrase to sentence in both texts and images to pre-train the model in stages. We also design several different pre-training tasks suitable for the information granularity in different stage in order to efficiently capture the diverse knowledge from a limited corpus. We take a Simplified LXMERT (LXMERT- S), which has only 45.9% parameters of the original LXMERT model and 11.76% of the original pre-training data as the testbed of our MSP method. Experimental results show that our method achieves comparable performance to the original LXMERT model in all downstream tasks, and even outperforms the original model in Image-Text Retrieval task.

| Comments: | 10 pages, 4 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2107.14596](https://arxiv.org/abs/2107.14596) [cs.CL]** |
|           | (or **[arXiv:2107.14596v1](https://arxiv.org/abs/2107.14596v1) [cs.CL]** for this version) |





<h2 id="2021-08-02-5">5. MDQE: A More Accurate Direct Pretraining for Machine Translation Quality Estimation
</h2>

Title: [MDQE: A More Accurate Direct Pretraining for Machine Translation Quality Estimation](https://arxiv.org/abs/2107.14600)

Authors: [Lei Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+L)

> It is expensive to evaluate the results of Machine Translation(MT), which usually requires manual translation as a reference. Machine Translation Quality Estimation (QE) is a task of predicting the quality of machine translations without relying on any reference. Recently, the emergence of predictor-estimator framework which trains the predictor as a feature extractor and estimator as a QE predictor, and pre-trained language models(PLM) have achieved promising QE performance. However, we argue that there are still gaps between the predictor and the estimator in both data quality and training objectives, which preclude QE models from benefiting from a large number of parallel corpora more directly. Based on previous related work that have alleviated gaps to some extent, we propose a novel framework that provides a more accurate direct pretraining for QE tasks. In this framework, a generator is trained to produce pseudo data that is closer to the real QE data, and a estimator is pretrained on these data with novel objectives that are the same as the QE task. Experiments on widely used benchmarks show that our proposed framework outperforms existing methods, without using any pretraining models such as BERT.

| Comments: | arXiv admin note: substantial text overlap with [arXiv:2105.07149](https://arxiv.org/abs/2105.07149) by other authors |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2107.14600](https://arxiv.org/abs/2107.14600) [cs.CL]** |
|           | (or **[arXiv:2107.14600v1](https://arxiv.org/abs/2107.14600v1) [cs.CL]** for this version) |





<h2 id="2021-08-02-6">6. Towards Universality in Multilingual Text Rewriting
</h2>

Title: [Towards Universality in Multilingual Text Rewriting](https://arxiv.org/abs/2107.14749)

Authors: [Xavier Garcia](https://arxiv.org/search/cs?searchtype=author&query=Garcia%2C+X), [Noah Constant](https://arxiv.org/search/cs?searchtype=author&query=Constant%2C+N), [Mandy Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+M), [Orhan Firat](https://arxiv.org/search/cs?searchtype=author&query=Firat%2C+O)

> In this work, we take the first steps towards building a universal rewriter: a model capable of rewriting text in any language to exhibit a wide variety of attributes, including styles and languages, while preserving as much of the original semantics as possible. In addition to obtaining state-of-the-art results on unsupervised translation, we also demonstrate the ability to do zero-shot sentiment transfer in non-English languages using only English exemplars for sentiment. We then show that our model is able to modify multiple attributes at once, for example adjusting both language and sentiment jointly. Finally, we show that our model is capable of performing zero-shot formality-sensitive translation.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2107.14749](https://arxiv.org/abs/2107.14749) [cs.CL]** |
|           | (or **[arXiv:2107.14749v1](https://arxiv.org/abs/2107.14749v1) [cs.CL]** for this version) |





<h2 id="2021-08-02-7">7. ChrEnTranslate: Cherokee-English Machine Translation Demo with Quality Estimation and Corrective Feedback
</h2>

Title: [ChrEnTranslate: Cherokee-English Machine Translation Demo with Quality Estimation and Corrective Feedback](https://arxiv.org/abs/2107.14800)

Authors: [Shiyue Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+S), [Benjamin Frey](https://arxiv.org/search/cs?searchtype=author&query=Frey%2C+B), [Mohit Bansal](https://arxiv.org/search/cs?searchtype=author&query=Bansal%2C+M)

> We introduce ChrEnTranslate, an online machine translation demonstration system for translation between English and an endangered language Cherokee. It supports both statistical and neural translation models as well as provides quality estimation to inform users of reliability, two user feedback interfaces for experts and common users respectively, example inputs to collect human translations for monolingual data, word alignment visualization, and relevant terms from the Cherokee-English dictionary. The quantitative evaluation demonstrates that our backbone translation models achieve state-of-the-art translation performance and our quality estimation well correlates with both BLEU and human judgment. By analyzing 216 pieces of expert feedback, we find that NMT is preferable because it copies less than SMT, and, in general, current models can translate fragments of the source sentence but make major mistakes. When we add these 216 expert-corrected parallel texts into the training set and retrain models, equal or slightly better performance is observed, which demonstrates indicates the potential of human-in-the-loop learning. Our online demo is at [this https URL](https://chren.cs.unc.edu/;) our code is open-sourced at [this https URL](https://github.com/ZhangShiyue/ChrEnTranslate;) and our data is available at [this https URL](https://github.com/ZhangShiyue/ChrEn).

| Comments: | ACL 2021 Demo (8 pages)                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2107.14800](https://arxiv.org/abs/2107.14800) [cs.CL]** |
|           | (or **[arXiv:2107.14800v1](https://arxiv.org/abs/2107.14800v1) [cs.CL]** for this version) |




