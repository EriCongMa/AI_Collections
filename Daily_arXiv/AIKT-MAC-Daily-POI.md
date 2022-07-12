# MA C.'s Daily Paper Of Interest - July a., 2022

# Index

- [2022-07-12](#2022-07-12)
  - [1. A Study of Syntactic Multi-Modality in Non-Autoregressive Machine Translation](#2022-07-12-1)

  - [2. UM4: Unified Multilingual Multiple Teacher-Student Model for Zero-Resource Neural Machine Translation](#2022-07-12-2)
  
  - [3. Exploring Length Generalization in Large Language Models](#2022-07-12-3)
  
  - [4. High-resource Language-specific Training for Multilingual Neural Machine Translation](#2022-07-12-4)
  
  - [5. Embedding Recycling for Language Models](#2022-07-12-5)
  
- [2022-07-11](#2022-07-11)
  - [1. Meta-Learning the Difference: Preparing Large Language Models for Efficient Adaptation](#2022-07-11-1)

- [2022-07-06](#2022-07-06)
  - [1. Vision-and-Language Pretraining](#2022-07-06-1)

  - [2. ASR-Generated Text for Language Model Pre-training Applied to Speech Tasks](#2022-07-06-2)

- [2022-07-05](#2022-07-05)
  - [1. Dynamic Contrastive Distillation for Image-Text Retrieval](#2022-07-05-1)

  - [2. M-Adapter: Modality Adaptation for End-to-End Speech-to-Text Translation](#2022-07-05-2)

- [2022-07-04](#2022-07-04)
  - [1. MultiViz: An Analysis Benchmark for Visualizing and Understanding Multimodal Models](#2022-07-04-1)

  - [2. Towards Human-Agent Communication via the Information Bottleneck Principle](#2022-07-04-2)

  - [3. VL-CheckList: Evaluating Pre-trained Vision-Language Models with Objects, Attributes and Relations](#2022-07-04-3)

- [2022-07-01](#2022-07-01)
  - [1. Building Multilingual Machine Translation Systems That Serve Arbitrary X-Y Translations](#2022-07-01-1)

  - [2. GSCLIP : A Framework for Explaining Distribution Shifts in Natural Language](#2022-07-01-2)

  - [3. FL-Tuning: Layer Tuning for Feed-Forward Network in Transformer](#2022-07-01-3)

- [2022-06-29](#2022-06-29)
  - [1. Wav2Vec-Aug: Improved self-supervised training with limited data](#2022-06-29-1)

- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



# 2022-07-12

[Return to Index](#Index)



<h2 id="2022-07-12-1">1. A Study of Syntactic Multi-Modality in Non-Autoregressive Machine Translation
</h2>

Title: [A Study of Syntactic Multi-Modality in Non-Autoregressive Machine Translation](https://arxiv.org/abs/2207.04206)

Authors: [Kexun Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+K), [Rui Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+R), [Xu Tan](https://arxiv.org/search/cs?searchtype=author&query=Tan%2C+X), [Junliang Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+J), [Yi Ren](https://arxiv.org/search/cs?searchtype=author&query=Ren%2C+Y), [Tao Qin](https://arxiv.org/search/cs?searchtype=author&query=Qin%2C+T), [Tie-Yan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+T)

> It is difficult for non-autoregressive translation (NAT) models to capture the multi-modal distribution of target translations due to their conditional independence assumption, which is known as the "multi-modality problem", including the lexical multi-modality and the syntactic multi-modality. While the first one has been well studied, the syntactic multi-modality brings severe challenge to the standard cross entropy (XE) loss in NAT and is under studied. In this paper, we conduct a systematic study on the syntactic multi-modality problem. Specifically, we decompose it into short- and long-range syntactic multi-modalities and evaluate several recent NAT algorithms with advanced loss functions on both carefully designed synthesized datasets and real datasets. We find that the Connectionist Temporal Classification (CTC) loss and the Order-Agnostic Cross Entropy (OAXE) loss can better handle short- and long-range syntactic multi-modalities respectively. Furthermore, we take the best of both and design a new loss function to better handle the complicated syntactic multi-modality in real-world datasets. To facilitate practical usage, we provide a guide to use different loss functions for different kinds of syntactic multi-modality.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2207.04206](https://arxiv.org/abs/2207.04206) [cs.CL]** |
|           | (or **[arXiv:2207.04206v1](https://arxiv.org/abs/2207.04206v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2207.04206Focus to learn more |





<h2 id="2022-07-12-2">2. UM4: Unified Multilingual Multiple Teacher-Student Model for Zero-Resource Neural Machine Translation
</h2>

Title: [UM4: Unified Multilingual Multiple Teacher-Student Model for Zero-Resource Neural Machine Translation](https://arxiv.org/abs/2207.04900)

Authors: [Jian Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+J), [Yuwei Yin](https://arxiv.org/search/cs?searchtype=author&query=Yin%2C+Y), [Shuming Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+S), [Dongdong Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+D), [Shuangzhi Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+S), [Hongcheng Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+H), [Zhoujun Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Z), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F)

> Most translation tasks among languages belong to the zero-resource translation problem where parallel corpora are unavailable. Multilingual neural machine translation (MNMT) enables one-pass translation using shared semantic space for all languages compared to the two-pass pivot translation but often underperforms the pivot-based method. In this paper, we propose a novel method, named as Unified Multilingual Multiple teacher-student Model for NMT (UM4). Our method unifies source-teacher, target-teacher, and pivot-teacher models to guide the student model for the zero-resource translation. The source teacher and target teacher force the student to learn the direct source to target translation by the distilled knowledge on both source and target sides. The monolingual corpus is further leveraged by the pivot-teacher model to enhance the student model. Experimental results demonstrate that our model of 72 directions significantly outperforms previous methods on the WMT benchmark.

| Comments: | 7 pages, 5 figures, IJCAI-ECAI 2022                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2207.04900](https://arxiv.org/abs/2207.04900) [cs.CL]** |
|           | (or **[arXiv:2207.04900v1](https://arxiv.org/abs/2207.04900v1) [cs.CL]** for this version) |





<h2 id="2022-07-12-3">3. Exploring Length Generalization in Large Language Models
</h2>

Title: [Exploring Length Generalization in Large Language Models](https://arxiv.org/abs/2207.04901)

Authors: [Cem Anil](https://arxiv.org/search/cs?searchtype=author&query=Anil%2C+C), [Yuhuai Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+Y), [Anders Andreassen](https://arxiv.org/search/cs?searchtype=author&query=Andreassen%2C+A), [Aitor Lewkowycz](https://arxiv.org/search/cs?searchtype=author&query=Lewkowycz%2C+A), [Vedant Misra](https://arxiv.org/search/cs?searchtype=author&query=Misra%2C+V), [Vinay Ramasesh](https://arxiv.org/search/cs?searchtype=author&query=Ramasesh%2C+V), [Ambrose Slone](https://arxiv.org/search/cs?searchtype=author&query=Slone%2C+A), [Guy Gur-Ari](https://arxiv.org/search/cs?searchtype=author&query=Gur-Ari%2C+G), [Ethan Dyer](https://arxiv.org/search/cs?searchtype=author&query=Dyer%2C+E), [Behnam Neyshabur](https://arxiv.org/search/cs?searchtype=author&query=Neyshabur%2C+B)

> The ability to extrapolate from short problem instances to longer ones is an important form of out-of-distribution generalization in reasoning tasks, and is crucial when learning from datasets where longer problem instances are rare. These include theorem proving, solving quantitative mathematics problems, and reading/summarizing novels. In this paper, we run careful empirical studies exploring the length generalization capabilities of transformer-based language models. We first establish that naively finetuning transformers on length generalization tasks shows significant generalization deficiencies independent of model scale. We then show that combining pretrained large language models' in-context learning abilities with scratchpad prompting (asking the model to output solution steps before producing an answer) results in a dramatic improvement in length generalization. We run careful failure analyses on each of the learning modalities and identify common sources of mistakes that highlight opportunities in equipping language models with the ability to generalize to longer problems.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2207.04901](https://arxiv.org/abs/2207.04901) [cs.CL]** |
|           | (or **[arXiv:2207.04901v1](https://arxiv.org/abs/2207.04901v1) [cs.CL]** for this version) |





<h2 id="2022-07-12-4">4. High-resource Language-specific Training for Multilingual Neural Machine Translation
</h2>

Title: [High-resource Language-specific Training for Multilingual Neural Machine Translation](https://arxiv.org/abs/2207.04906)

Authors: [Jian Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+J), [Yuwei Yin](https://arxiv.org/search/cs?searchtype=author&query=Yin%2C+Y), [Shuming Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+S), [Dongdong Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+D), [Zhoujun Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Z), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F)

> Multilingual neural machine translation (MNMT) trained in multiple language pairs has attracted considerable attention due to fewer model parameters and lower training costs by sharing knowledge among multiple languages. Nonetheless, multilingual training is plagued by language interference degeneration in shared parameters because of the negative interference among different translation directions, especially on high-resource languages. In this paper, we propose the multilingual translation model with the high-resource language-specific training (HLT-MT) to alleviate the negative interference, which adopts the two-stage training with the language-specific selection mechanism. Specifically, we first train the multilingual model only with the high-resource pairs and select the language-specific modules at the top of the decoder to enhance the translation quality of high-resource directions. Next, the model is further trained on all available corpora to transfer knowledge from high-resource languages (HRLs) to low-resource languages (LRLs). Experimental results show that HLT-MT outperforms various strong baselines on WMT-10 and OPUS-100 benchmarks. Furthermore, the analytic experiments validate the effectiveness of our method in mitigating the negative interference in multilingual training.

| Comments: | 7 pages, 7 figures, IJCAI-ECAI 2022                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2207.04906](https://arxiv.org/abs/2207.04906) [cs.CL]** |
|           | (or **[arXiv:2207.04906v1](https://arxiv.org/abs/2207.04906v1) [cs.CL]** for this version) |





<h2 id="2022-07-12-5">5. Embedding Recycling for Language Models
</h2>

Title: [Embedding Recycling for Language Models](https://arxiv.org/abs/2207.04993)

Authors: [Jon Saad-Falcon](https://arxiv.org/search/cs?searchtype=author&query=Saad-Falcon%2C+J), [Amanpreet Singh](https://arxiv.org/search/cs?searchtype=author&query=Singh%2C+A), [Luca Soldaini](https://arxiv.org/search/cs?searchtype=author&query=Soldaini%2C+L), [Mike D'Arcy](https://arxiv.org/search/cs?searchtype=author&query=D'Arcy%2C+M), [Arman Cohan](https://arxiv.org/search/cs?searchtype=author&query=Cohan%2C+A), [Doug Downey](https://arxiv.org/search/cs?searchtype=author&query=Downey%2C+D)

> Training and inference with large neural models is expensive. However, for many application domains, while new tasks and models arise frequently, the underlying documents being modeled remain mostly unchanged. We study how to decrease computational cost in such settings through embedding recycling (ER): re-using activations from previous model runs when performing training or inference. In contrast to prior work focusing on freezing small classification heads for finetuning which often leads to notable drops in performance, we propose caching an intermediate layer's output from a pretrained model and finetuning the remaining layers for new tasks. We show that our method provides a 100% speedup during training and a 55-86% speedup for inference, and has negligible impacts on accuracy for text classification and entity recognition tasks in the scientific domain. For general-domain question answering tasks, ER offers a similar speedup and lowers accuracy by a small amount. Finally, we identify several open challenges and future directions for ER.

| Comments: | Preprint                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2207.04993](https://arxiv.org/abs/2207.04993) [cs.CL]** |
|           | (or **[arXiv:2207.04993v1](https://arxiv.org/abs/2207.04993v1) [cs.CL]** for this version) |





# 2022-07-11

[Return to Index](#Index)



<h2 id="2022-07-11-1">1. Meta-Learning the Difference: Preparing Large Language Models for Efficient Adaptation
</h2>

Title: [Meta-Learning the Difference: Preparing Large Language Models for Efficient Adaptation](https://arxiv.org/abs/2207.03509)

Authors: [Zejiang Hou](https://arxiv.org/search/cs?searchtype=author&query=Hou%2C+Z), [Julian Salazar](https://arxiv.org/search/cs?searchtype=author&query=Salazar%2C+J), [George Polovets](https://arxiv.org/search/cs?searchtype=author&query=Polovets%2C+G)

> Large pretrained language models (PLMs) are often domain- or task-adapted via fine-tuning or prompting. Finetuning requires modifying all of the parameters and having enough data to avoid overfitting while prompting requires no training and few examples but limits performance. Instead, we prepare PLMs for data- and parameter-efficient adaptation by learning to learn the difference between general and adapted PLMs. This difference is expressed in terms of model weights and sublayer structure through our proposed dynamic low-rank reparameterization and learned architecture controller. Experiments on few-shot dialogue completion, low-resource abstractive summarization, and multi-domain language modeling show improvements in adaptation time and performance over direct finetuning or preparation via domain-adaptive pretraining. Ablations show our task-adaptive reparameterization (TARP) and model search (TAMS) components individually improve on other parameter-efficient transfer like adapters and structure-learning methods like learned sparsification.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2207.03509](https://arxiv.org/abs/2207.03509) [cs.CL]** |
|           | (or **[arXiv:2207.03509v1](https://arxiv.org/abs/2207.03509v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2207.03509Focus to learn more |






# 2022-07-06

[Return to Index](#Index)



<h2 id="2022-07-06-1">1. Vision-and-Language Pretraining
</h2>

Title: [Vision-and-Language Pretraining](https://arxiv.org/abs/2207.01772)

Authors: [Thong Nguyen](https://arxiv.org/search/cs?searchtype=author&query=Nguyen%2C+T), [Cong-Duy Nguyen](https://arxiv.org/search/cs?searchtype=author&query=Nguyen%2C+C), [Xiaobao Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+X), [Anh Tuan Luu](https://arxiv.org/search/cs?searchtype=author&query=Luu%2C+A+T)

> With the burgeoning amount of data of image-text pairs and diversity of Vision-and-Language (V&L) tasks, scholars have introduced an abundance of deep learning models in this research domain. Furthermore, in recent years, transfer learning has also shown tremendous success in Computer Vision for tasks such as Image Classification, Object Detection, etc., and in Natural Language Processing for Question Answering, Machine Translation, etc. Inheriting the spirit of Transfer Learning, research works in V&L have devised multiple pretraining techniques on large-scale datasets in order to enhance the performance of downstream tasks. The aim of this article is to provide a comprehensive revision of contemporary V&L pretraining models. In particular, we categorize and delineate pretraining approaches, along with the summary of state-of-the-art vision-and-language pre-trained models. Moreover, a list of training datasets and downstream tasks is supplied to further polish the perspective on V&L pretraining. Lastly, we decided to take a further step to discuss numerous directions for future research.

| Comments: | 35 pages, 3 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2207.01772](https://arxiv.org/abs/2207.01772) [cs.CL]** |
|           | (or **[arXiv:2207.01772v1](https://arxiv.org/abs/2207.01772v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2207.01772Focus to learn more |





<h2 id="2022-07-06-2">2. ASR-Generated Text for Language Model Pre-training Applied to Speech Tasks
</h2>

Title: [ASR-Generated Text for Language Model Pre-training Applied to Speech Tasks](https://arxiv.org/abs/2207.01893)

Authors: [Valentin Pelloin](https://arxiv.org/search/cs?searchtype=author&query=Pelloin%2C+V), [Franck Dary](https://arxiv.org/search/cs?searchtype=author&query=Dary%2C+F), [Nicolas Herve](https://arxiv.org/search/cs?searchtype=author&query=Herve%2C+N), [Benoit Favre](https://arxiv.org/search/cs?searchtype=author&query=Favre%2C+B), [Nathalie Camelin](https://arxiv.org/search/cs?searchtype=author&query=Camelin%2C+N), [Antoine Laurent](https://arxiv.org/search/cs?searchtype=author&query=Laurent%2C+A), [Laurent Besacier](https://arxiv.org/search/cs?searchtype=author&query=Besacier%2C+L)

> We aim at improving spoken language modeling (LM) using very large amount of automatically transcribed speech. We leverage the INA (French National Audiovisual Institute) collection and obtain 19GB of text after applying ASR on 350,000 hours of diverse TV shows. From this, spoken language models are trained either by fine-tuning an existing LM (FlauBERT) or through training a LM from scratch. New models (FlauBERT-Oral) are shared with the community and evaluated for 3 downstream tasks: spoken language understanding, classification of TV shows and speech syntactic parsing. Results show that FlauBERT-Oral can be beneficial compared to its initial FlauBERT version demonstrating that, despite its inherent noisy nature, ASR-generated text can be used to build spoken language models.

| Comments: | Interspeech 2022 (Camera Ready)                              |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2207.01893](https://arxiv.org/abs/2207.01893) [cs.CL]** |
|           | (or **[arXiv:2207.01893v1](https://arxiv.org/abs/2207.01893v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2207.01893Focus to learn more |





# 2022-07-05

[Return to Index](#Index)



<h2 id="2022-07-05-1">1. Dynamic Contrastive Distillation for Image-Text Retrieval
</h2>

Title: [Dynamic Contrastive Distillation for Image-Text Retrieval](https://arxiv.org/abs/2207.01426)

Authors: [Jun Rao](https://arxiv.org/search/cs?searchtype=author&query=Rao%2C+J), [Liang Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+L), [Shuhan Qi](https://arxiv.org/search/cs?searchtype=author&query=Qi%2C+S), [Meng Fang](https://arxiv.org/search/cs?searchtype=author&query=Fang%2C+M), [Yang Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y), [Li Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+L), [Dacheng Tao](https://arxiv.org/search/cs?searchtype=author&query=Tao%2C+D)

> Although the vision-and-language pretraining (VLP) equipped cross-modal image-text retrieval (ITR) has achieved remarkable progress in the past two years, it suffers from a major drawback: the ever-increasing size of VLP models restricts its deployment to real-world search scenarios (where the high latency is unacceptable). To alleviate this problem, we present a novel plug-in dynamic contrastive distillation (DCD) framework to compress the large VLP models for the ITR task. Technically, we face the following two challenges: 1) the typical uni-modal metric learning approach is difficult to directly apply to the cross-modal tasks, due to the limited GPU memory to optimize too many negative samples during handling cross-modal fusion features. 2) it is inefficient to static optimize the student network from different hard samples, which have different effects on distillation learning and student network optimization. We try to overcome these challenges from two points. First, to achieve multi-modal contrastive learning, and balance the training costs and effects, we propose to use a teacher network to estimate the difficult samples for students, making the students absorb the powerful knowledge from pre-trained teachers, and master the knowledge from hard samples. Second, to dynamic learn from hard sample pairs, we propose dynamic distillation to dynamically learn samples of different difficulties, from the perspective of better balancing the difficulty of knowledge and students' self-learning ability. We successfully apply our proposed DCD strategy to two state-of-the-art vision-language pretrained models, i.e. ViLT and METER. Extensive experiments on MS-COCO and Flickr30K benchmarks show the effectiveness and efficiency of our DCD framework. Encouragingly, we can speed up the inference at least 129× compared to the existing ITR models.

| Subjects: | **Multimedia (cs.MM)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2207.01426](https://arxiv.org/abs/2207.01426) [cs.MM]** |
|           | (or **[arXiv:2207.01426v1](https://arxiv.org/abs/2207.01426v1) [cs.MM]** for this version) |
|           | https://doi.org/10.48550/arXiv.2207.01426Focus to learn more |





<h2 id="2022-07-05-2">2. M-Adapter: Modality Adaptation for End-to-End Speech-to-Text Translation
</h2>

Title: [M-Adapter: Modality Adaptation for End-to-End Speech-to-Text Translation](https://arxiv.org/abs/2207.00952)

Authors: [Jinming Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+J), [Hao Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+H), [Ehsan Shareghi](https://arxiv.org/search/cs?searchtype=author&query=Shareghi%2C+E), [Gholamreza Haffari](https://arxiv.org/search/cs?searchtype=author&query=Haffari%2C+G)

> End-to-end speech-to-text translation models are often initialized with pre-trained speech encoder and pre-trained text decoder. This leads to a significant training gap between pre-training and fine-tuning, largely due to the modality differences between speech outputs from the encoder and text inputs to the decoder. In this work, we aim to bridge the modality gap between speech and text to improve translation quality. We propose M-Adapter, a novel Transformer-based module, to adapt speech representations to text. While shrinking the speech sequence, M-Adapter produces features desired for speech-to-text translation via modelling global and local dependencies of a speech sequence. Our experimental results show that our model outperforms a strong baseline by up to 1 BLEU score on the Must-C En→DE dataset.\footnote{Our code is available at [this https URL](https://github.com/mingzi151/w2v2-st).}

| Comments: | Interspeech2022                                              |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2207.00952](https://arxiv.org/abs/2207.00952) [cs.CL]** |
|           | (or **[arXiv:2207.00952v1](https://arxiv.org/abs/2207.00952v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2207.00952Focus to learn more |



# 2022-07-04

[Return to Index](#Index)



<h2 id="2022-07-04-1">1. MultiViz: An Analysis Benchmark for Visualizing and Understanding Multimodal Models
</h2>

Title: [MultiViz: An Analysis Benchmark for Visualizing and Understanding Multimodal Models](https://arxiv.org/abs/2207.00056)
Authors: [Paul Pu Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+P+P), [Yiwei Lyu](https://arxiv.org/search/cs?searchtype=author&query=Lyu%2C+Y), [Gunjan Chhablani](https://arxiv.org/search/cs?searchtype=author&query=Chhablani%2C+G), [Nihal Jain](https://arxiv.org/search/cs?searchtype=author&query=Jain%2C+N), [Zihao Deng](https://arxiv.org/search/cs?searchtype=author&query=Deng%2C+Z), [Xingbo Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Louis-Philippe Morency](https://arxiv.org/search/cs?searchtype=author&query=Morency%2C+L), [Ruslan Salakhutdinov](https://arxiv.org/search/cs?searchtype=author&query=Salakhutdinov%2C+R)

> The promise of multimodal models for real-world applications has inspired research in visualizing and understanding their internal mechanics with the end goal of empowering stakeholders to visualize model behavior, perform model debugging, and promote trust in machine learning models. However, modern multimodal models are typically black-box neural networks, which makes it challenging to understand their internal mechanics. How can we visualize the internal modeling of multimodal interactions in these models? Our paper aims to fill this gap by proposing MultiViz, a method for analyzing the behavior of multimodal models by scaffolding the problem of interpretability into 4 stages: (1) unimodal importance: how each modality contributes towards downstream modeling and prediction, (2) cross-modal interactions: how different modalities relate with each other, (3) multimodal representations: how unimodal and cross-modal interactions are represented in decision-level features, and (4) multimodal prediction: how decision-level features are composed to make a prediction. MultiViz is designed to operate on diverse modalities, models, tasks, and research areas. Through experiments on 8 trained models across 6 real-world tasks, we show that the complementary stages in MultiViz together enable users to (1) simulate model predictions, (2) assign interpretable concepts to features, (3) perform error analysis on model misclassifications, and (4) use insights from error analysis to debug models. MultiViz is publicly available, will be regularly updated with new interpretation tools and metrics, and welcomes inputs from the community.

| Comments: | Code available at: [this https URL](https://github.com/pliang279/MultiViz). arXiv admin note: substantial text overlap with [arXiv:2107.07502](https://arxiv.org/abs/2107.07502) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV); Multimedia (cs.MM) |
| Cite as:  | **[arXiv:2207.00056](https://arxiv.org/abs/2207.00056) [cs.LG]** |
|           | (or **[arXiv:2207.00056v1](https://arxiv.org/abs/2207.00056v1) [cs.LG]** for this version) |
|           | https://doi.org/10.48550/arXiv.2207.00056Focus to learn more |





<h2 id="2022-07-04-2">2. Towards Human-Agent Communication via the Information Bottleneck Principle
</h2>

Title: [Towards Human-Agent Communication via the Information Bottleneck Principle](https://arxiv.org/abs/2207.00088)
Authors: [Mycal Tucker](https://arxiv.org/search/cs?searchtype=author&query=Tucker%2C+M), [Julie Shah](https://arxiv.org/search/cs?searchtype=author&query=Shah%2C+J), [Roger Levy](https://arxiv.org/search/cs?searchtype=author&query=Levy%2C+R), [Noga Zaslavsky](https://arxiv.org/search/cs?searchtype=author&query=Zaslavsky%2C+N)

> Emergent communication research often focuses on optimizing task-specific utility as a driver for communication. However, human languages appear to evolve under pressure to efficiently compress meanings into communication signals by optimizing the Information Bottleneck tradeoff between informativeness and complexity. In this work, we study how trading off these three factors -- utility, informativeness, and complexity -- shapes emergent communication, including compared to human communication. To this end, we propose Vector-Quantized Variational Information Bottleneck (VQ-VIB), a method for training neural agents to compress inputs into discrete signals embedded in a continuous space. We train agents via VQ-VIB and compare their performance to previously proposed neural architectures in grounded environments and in a Lewis reference game. Across all neural architectures and settings, taking into account communicative informativeness benefits communication convergence rates, and penalizing communicative complexity leads to human-like lexicon sizes while maintaining high utility. Additionally, we find that VQ-VIB outperforms other discrete communication methods. This work demonstrates how fundamental principles that are believed to characterize human language evolution may inform emergent communication in artificial agents.

| Subjects: | **Artificial Intelligence (cs.AI)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2207.00088](https://arxiv.org/abs/2207.00088) [cs.AI]** |
|           | (or **[arXiv:2207.00088v1](https://arxiv.org/abs/2207.00088v1) [cs.AI]** for this version) |
|           | https://doi.org/10.48550/arXiv.2207.00088Focus to learn more |





<h2 id="2022-07-04-3">3. VL-CheckList: Evaluating Pre-trained Vision-Language Models with Objects, Attributes and Relations
</h2>

Title: [VL-CheckList: Evaluating Pre-trained Vision-Language Models with Objects, Attributes and Relations](https://arxiv.org/abs/2207.00221)
Authors: [Tiancheng Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+T), [Tianqi Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+T), [Mingwei Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+M), [Haozhan Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+H), [Kyusong Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+K), [Xiaopeng Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+X), [Jianwei Yin](https://arxiv.org/search/cs?searchtype=author&query=Yin%2C+J)

> Vision-Language Pretraining (VLP) models have recently successfully facilitated many cross-modal downstream tasks. Most existing works evaluated their systems by comparing the fine-tuned downstream task performance. However, only average downstream task accuracy provides little information about the pros and cons of each VLP method, let alone provides insights on how the community can improve the systems in the future. Inspired by the CheckList for testing natural language processing, we introduce VL-CheckList, a novel framework to understand the capabilities of VLP models. The proposed method divides the image-texting ability of a VLP model into three categories: objects, attributes, and relations, and uses a novel taxonomy to further break down these three aspects. We conduct comprehensive studies to analyze seven recently popular VLP models via the proposed framework. Results confirm the effectiveness of the proposed method by revealing fine-grained differences among the compared models that were not visible from downstream task-only evaluation. Further results show promising research direction in building better VLP models. Data and Code: [this https URL](https://github.com/om-ai-lab/VL-CheckList)

| Comments: | 9 pages, preprint                                            |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2207.00221](https://arxiv.org/abs/2207.00221) [cs.CV]** |
|           | (or **[arXiv:2207.00221v1](https://arxiv.org/abs/2207.00221v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2207.00221Focus to learn more |






# 2022-07-01

[Return to Index](#Index)



<h2 id="2022-07-01-1">1. Building Multilingual Machine Translation Systems That Serve Arbitrary X-Y Translations
</h2>

Title: [Building Multilingual Machine Translation Systems That Serve Arbitrary X-Y Translations](https://arxiv.org/abs/2206.14982)

Authors: [Akiko Eriguchi](https://arxiv.org/search/cs?searchtype=author&query=Eriguchi%2C+A), [Shufang Xie](https://arxiv.org/search/cs?searchtype=author&query=Xie%2C+S), [Tao Qin](https://arxiv.org/search/cs?searchtype=author&query=Qin%2C+T), [Hany Hassan Awadalla](https://arxiv.org/search/cs?searchtype=author&query=Awadalla%2C+H+H)

> Multilingual Neural Machine Translation (MNMT) enables one system to translate sentences from multiple source languages to multiple target languages, greatly reducing deployment costs compared with conventional bilingual systems. The MNMT training benefit, however, is often limited to many-to-one directions. The model suffers from poor performance in one-to-many and many-to-many with zero-shot setup. To address this issue, this paper discusses how to practically build MNMT systems that serve arbitrary X-Y translation directions while leveraging multilinguality with a two-stage training strategy of pretraining and finetuning. Experimenting with the WMT'21 multilingual translation task, we demonstrate that our systems outperform the conventional baselines of direct bilingual models and pivot translation models for most directions, averagely giving +6.0 and +4.1 BLEU, without the need for architecture change or extra data collection. Moreover, we also examine our proposed approach in an extremely large-scale data setting to accommodate practical deployment scenarios.

| Comments: | NAACL 2022                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2206.14982](https://arxiv.org/abs/2206.14982) [cs.CL]** |
|           | (or **[arXiv:2206.14982v1](https://arxiv.org/abs/2206.14982v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.14982Focus to learn more |





<h2 id="2022-07-01-2">2. GSCLIP : A Framework for Explaining Distribution Shifts in Natural Language
</h2>

Title: [GSCLIP : A Framework for Explaining Distribution Shifts in Natural Language](https://arxiv.org/abs/2206.15007)

Authors: [Zhiying Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+Z), [Weixin Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+W), [James Zou](https://arxiv.org/search/cs?searchtype=author&query=Zou%2C+J)

> Helping end users comprehend the abstract distribution shifts can greatly facilitate AI deployment. Motivated by this, we propose a novel task, dataset explanation. Given two image data sets, dataset explanation aims to automatically point out their dataset-level distribution shifts with natural language. Current techniques for monitoring distribution shifts provide inadequate information to understand datasets with the goal of improving data quality. Therefore, we introduce GSCLIP, a training-free framework to solve the dataset explanation task. In GSCLIP, we propose the selector as the first quantitative evaluation method to identify explanations that are proper to summarize dataset shifts. Furthermore, we leverage this selector to demonstrate the superiority of a generator based on language model generation. Systematic evaluation on natural data shift verifies that GSCLIP, a combined system of a hybrid generator group and an efficient selector is not only easy-to-use but also powerful for dataset explanation at scale.

| Comments: | Accepted by ICML 2022 DataPerf                               |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2206.15007](https://arxiv.org/abs/2206.15007) [cs.CL]** |
|           | (or **[arXiv:2206.15007v1](https://arxiv.org/abs/2206.15007v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.15007Focus to learn more |





<h2 id="2022-07-01-3">3. FL-Tuning: Layer Tuning for Feed-Forward Network in Transformer
</h2>

Title: [FL-Tuning: Layer Tuning for Feed-Forward Network in Transformer](https://arxiv.org/abs/2206.15312)

Authors: [Jingping Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+J), [Yuqiu Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+Y), [Kui Xue](https://arxiv.org/search/cs?searchtype=author&query=Xue%2C+K), [Hongli Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+H), [Chao Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+C), [Lihan Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+L), [Haiyun Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+H), [Jiaqing Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+J), [Tong Ruan](https://arxiv.org/search/cs?searchtype=author&query=Ruan%2C+T)

> Prompt tuning is an emerging way of adapting pre-trained language models to downstream tasks. However, the existing studies are mainly to add prompts to the input sequence. This way would not work as expected due to the intermediate multi-head self-attention and feed-forward network computation, making model optimization not very smooth. Hence, we propose a novel tuning way called layer tuning, aiming to add learnable parameters in Transformer layers. Specifically, we focus on layer tuning for feed-forward network in the Transformer, namely FL-tuning. It introduces additional units into the hidden layer of each feed-forward network. We conduct extensive experiments on the public CLUE benchmark. The results show that: 1) Our FL-tuning outperforms prompt tuning methods under both full-data and few-shot settings in almost all cases. In particular, it improves accuracy by 17.93% (full-data setting) on WSC 1.0 and F1 by 16.142% (few-shot setting) on CLUENER over P-tuning v2. 2) Our FL-tuning is more stable and converges about 1.17 times faster than P-tuning v2. 3) With only about 3% of Transformer's parameters to be trained, FL-tuning is comparable with fine-tuning on most datasets, and significantly outperforms fine-tuning (e.g., accuracy improved by 12.9% on WSC 1.1) on several datasets. The source codes are available at [this https URL](https://github.com/genggui001/FL-Tuning).

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.15312](https://arxiv.org/abs/2206.15312) [cs.CL]** |
|           | (or **[arXiv:2206.15312v1](https://arxiv.org/abs/2206.15312v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.15312Focus to learn more |






# 2022-06-29

[Return to Index](#Index)



<h2 id="2022-06-29-1">1. Wav2Vec-Aug: Improved self-supervised training with limited data
</h2>

Title: [Wav2Vec-Aug: Improved self-supervised training with limited data](https://arxiv.org/abs/2206.13654)

Authors:  [Anuroop Sriram](https://arxiv.org/search/cs?searchtype=author&query=Sriram%2C+A), [Michael Auli](https://arxiv.org/search/cs?searchtype=author&query=Auli%2C+M), [Alexei Baevski](https://arxiv.org/search/cs?searchtype=author&query=Baevski%2C+A)

> Self-supervised learning (SSL) of speech representations has received much attention over the last few years but most work has focused on languages and domains with an abundance of unlabeled data. However, for many languages there is a shortage even in the unlabeled data which limits the effectiveness of SSL. In this work, we focus on the problem of applying SSL to domains with limited available data by leveraging data augmentation for Wav2Vec 2.0 pretraining. Further, we propose improvements to each component of the model which result in a combined relative word error rate (WER) improvement of up to 13% compared to Wav2Vec 2.0 on Librispeech test-clean / other.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.13654](https://arxiv.org/abs/2206.13654) [cs.CL]** |
|           | (or **[arXiv:2206.13654v1](https://arxiv.org/abs/2206.13654v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.13654Focus to learn more |



