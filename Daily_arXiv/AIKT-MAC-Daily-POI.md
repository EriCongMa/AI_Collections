# MA C.'s Daily Paper Of Interest - May b., 2022

# Index

- [2022-05-27](#2022-05-27)
  - [1. Open-Domain Sign Language Translation Learned from Online Video](#2022-05-27-1)
  
  - [2. AdaMix: Mixture-of-Adapter for Parameter-efficient Tuning of Large Language Models](#2022-05-27-2)
  
  - [3. Know Where You're Going: Meta-Learning for Parameter-Efficient Fine-tuning](#2022-05-27-3)

  - [4. Improving CTC-based ASR Models with Gated Interlayer Collaboration](#2022-05-27-4)
  
  - [5. Machine Translation Robustness to Natural Asemantic Variation](#2022-05-27-5)
  
  - [6. TranSpeech: Speech-to-Speech Translation With Bilateral Perturbation](#2022-05-27-6)
  
  - [7. Are Large Pre-Trained Language Models Leaking Your Personal Information?](#2022-05-27-7)
  
  - [8. Multimodal Knowledge Alignment with Reinforcement Learning](#2022-05-27-8)
  
  - [9. Discovering Language-neutral Sub-networks in Multilingual Language Models](#2022-05-27-9)
  
  - [10. Understanding Natural Language in Context](#2022-05-27-10)
  
  - [11. Eliciting Transferability in Multi-task Learning with Task-level Mixture-of-Experts](#2022-05-27-11)
  
- [2022-05-23](#2022-05-23)
  - [1. Translating Hanja historical documents to understandable Korean and English](#2022-05-23-1)

  - [2. Exploring Extreme Parameter Compression for Pre-trained Language Models](#2022-05-23-2)

  - [3. How to keep text private? A systematic review of deep learning methods for privacy-preserving natural language processing](#2022-05-23-3)

  - [4. Visually-Augmented Language Modeling](#2022-05-23-4)

  - [5. Visualizing and Explaining Language Models](#2022-05-23-5)

  - [6. Lossless Acceleration for Seq2seq Generation with Aggressive Decoding](#2022-05-23-6)

- [2022-05-20](#2022-05-20)
  - [1. PreQuEL: Quality Estimation of Machine Translation Outputs in Advance](#2022-05-20-1)
  - [2. Evaluating Subtitle Segmentation for End-to-end Generation Systems](#2022-05-20-2)
  - [3. Insights on Neural Representations for End-to-End Speech Recognition](#2022-05-20-3)
  - [4. Phylogeny-Inspired Adaptation of Multilingual Models to New Languages](#2022-05-20-4)
  - [5. Voxel-informed Language Grounding](#2022-05-20-5)

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



# 2022-05-27

[Return to Index](#Index)



<h2 id="2022-05-27-1">1. Open-Domain Sign Language Translation Learned from Online Video
</h2>

Title: [Open-Domain Sign Language Translation Learned from Online Video](https://arxiv.org/abs/2205.12870)

Authors: [Bowen Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+B), [Diane Brentari](https://arxiv.org/search/cs?searchtype=author&query=Brentari%2C+D), [Greg Shakhnarovich](https://arxiv.org/search/cs?searchtype=author&query=Shakhnarovich%2C+G), [Karen Livescu](https://arxiv.org/search/cs?searchtype=author&query=Livescu%2C+K)

> Existing work on sign language translation--that is, translation from sign language videos into sentences in a written language--has focused mainly on (1) data collected in a controlled environment or (2) data in a specific domain, which limits the applicability to real-world settings. In this paper, we introduce OpenASL, a large-scale ASL-English dataset collected from online video sites (e.g., YouTube). OpenASL contains 288 hours of ASL videos in various domains (news, VLOGs, etc.) from over 200 signers and is the largest publicly available ASL translation dataset to date. To tackle the challenges of sign language translation in realistic settings and without glosses, we propose a set of techniques including sign search as a pretext task for pre-training and fusion of mouthing and handshape features. The proposed techniques produce consistent and large improvements in translation quality, over baseline models based on prior work. Our data, code and model will be publicly available at [this https URL](https://github.com/chevalierNoir/OpenASL)

| Comments: | 17 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2205.12870](https://arxiv.org/abs/2205.12870) [cs.CV]** |
|           | (or **[arXiv:2205.12870v1](https://arxiv.org/abs/2205.12870v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.12870Focus to learn more |





<h2 id="2022-05-27-2">2. AdaMix: Mixture-of-Adapter for Parameter-efficient Tuning of Large Language Models
</h2>

Title: [AdaMix: Mixture-of-Adapter for Parameter-efficient Tuning of Large Language Models](https://arxiv.org/abs/2205.12410)

Authors: [Yaqing Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Y), [Subhabrata Mukherjee](https://arxiv.org/search/cs?searchtype=author&query=Mukherjee%2C+S), [Xiaodong Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+X), [Jing Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+J), [Ahmed Hassan Awadallah](https://arxiv.org/search/cs?searchtype=author&query=Awadallah%2C+A+H), [Jianfeng Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+J)

> Fine-tuning large-scale pre-trained language models to downstream tasks require updating hundreds of millions of parameters. This not only increases the serving cost to store a large copy of the model weights for every task, but also exhibits instability during few-shot task adaptation. Parameter-efficient techniques have been developed that tune small trainable components (e.g., adapters) injected in the large model while keeping most of the model weights frozen. The prevalent mechanism to increase adapter capacity is to increase the bottleneck dimension which increases the adapter parameters. In this work, we introduce a new mechanism to improve adapter capacity without increasing parameters or computational cost by two key techniques. (i) We introduce multiple shared adapter components in each layer of the Transformer architecture. We leverage sparse learning via random routing to update the adapter parameters (encoder is kept frozen) resulting in the same amount of computational cost (FLOPs) as that of training a single adapter. (ii) We propose a simple merging mechanism to average the weights of multiple adapter components to collapse to a single adapter in each Transformer layer, thereby, keeping the overall parameters also the same but with significant performance improvement. We demonstrate these techniques to work well across multiple task settings including fully supervised and few-shot Natural Language Understanding tasks. By only tuning 0.23% of a pre-trained language model's parameters, our model outperforms the full model fine-tuning performance and several competing methods.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.12410](https://arxiv.org/abs/2205.12410) [cs.CL]** |
|           | (or **[arXiv:2205.12410v1](https://arxiv.org/abs/2205.12410v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.12410Focus to learn more |





<h2 id="2022-05-27-3">3. Know Where You're Going: Meta-Learning for Parameter-Efficient Fine-tuning
</h2>

Title: [Know Where You're Going: Meta-Learning for Parameter-Efficient Fine-tuning](https://arxiv.org/abs/2205.12453)

Authors: [Mozhdeh Gheini](https://arxiv.org/search/cs?searchtype=author&query=Gheini%2C+M), [Xuezhe Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+X), [Jonathan May](https://arxiv.org/search/cs?searchtype=author&query=May%2C+J)

> A recent family of techniques, dubbed as lightweight fine-tuning methods, facilitates parameter-efficient transfer learning by updating only a small set of additional parameters while keeping the parameters of the pretrained language model frozen. While proven to be an effective method, there are no existing studies on if and how such knowledge of the downstream fine-tuning approach should affect the pretraining stage. In this work, we show that taking the ultimate choice of fine-tuning method into consideration boosts the performance of parameter-efficient fine-tuning. By relying on optimization-based meta-learning using MAML with certain modifications for our distinct purpose, we prime the pretrained model specifically for parameter-efficient fine-tuning, resulting in gains of up to 1.7 points on cross-lingual NER fine-tuning. Our ablation settings and analyses further reveal that the tweaks we introduce in MAML are crucial for the attained gains.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.12453](https://arxiv.org/abs/2205.12453) [cs.CL]** |
|           | (or **[arXiv:2205.12453v1](https://arxiv.org/abs/2205.12453v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.12453Focus to learn more |





<h2 id="2022-05-27-4">4. Improving CTC-based ASR Models with Gated Interlayer Collaboration
</h2>

Title: [Improving CTC-based ASR Models with Gated Interlayer Collaboration](https://arxiv.org/abs/2205.12462)

Authors: [Yuting Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Y), [Yuke Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Binbin Du](https://arxiv.org/search/cs?searchtype=author&query=Du%2C+B)

> For Automatic Speech Recognition (ASR), the CTC-based methods have become a dominant paradigm due to its simple architecture and efficient non-autoregressive inference manner. However, these methods without external language models usually lack the capacity of modeling the conditional dependencies and the textual interaction. In this work, we present a Gated Interlayer Collaboration (GIC) mechanism which introduces the contextual information into the models and relaxes the conditional independence assumption of the CTC-based models. Specifically, we train the model with intermediate CTC losses calculated by the interlayer outputs of the model, in which the probability distributions of the intermediate layers naturally serve as soft label sequences. The GIC block consists of an embedding layer to obtain the textual embedding of the soft label at each position, and a gate unit to fuse the textual embedding and the acoustic features. Experiments on AISHELL-1 and AIDATATANG benchmarks show that the proposed method outperforms the recently published CTC-based ASR models. Specifically, our method achieves CER of 4.0%/4.4% on AISHELL-1 dev/test sets and CER of 3.8%/4.4% on AIDATATANG dev/test sets using CTC greedy search decoding without external language models.

| Comments: | Submitted to INTERSPEECH2022                                 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2205.12462](https://arxiv.org/abs/2205.12462) [cs.CL]** |
|           | (or **[arXiv:2205.12462v1](https://arxiv.org/abs/2205.12462v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.12462Focus to learn more |





<h2 id="2022-05-27-5">5. Machine Translation Robustness to Natural Asemantic Variation
</h2>

Title: [Machine Translation Robustness to Natural Asemantic Variation](https://arxiv.org/abs/2205.12514)

Authors: [Jacob Bremerman](https://arxiv.org/search/cs?searchtype=author&query=Bremerman%2C+J), [Xiang Ren](https://arxiv.org/search/cs?searchtype=author&query=Ren%2C+X), [Jonathan May](https://arxiv.org/search/cs?searchtype=author&query=May%2C+J)

> We introduce and formalize an under-studied linguistic phenomenon we call Natural Asemantic Variation (NAV) and investigate it in the context of Machine Translation (MT) robustness. Standard MT models are shown to be less robust to rarer, nuanced language forms, and current robustness techniques do not account for this kind of perturbation despite their prevalence in "real world" data. Experiment results provide more insight into the nature of NAV and we demonstrate strategies to improve performance on NAV. We also show that NAV robustness can be transferred across languages and fine that synthetic perturbations can achieve some but not all of the benefits of human-generated NAV data.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.12514](https://arxiv.org/abs/2205.12514) [cs.CL]** |
|           | (or **[arXiv:2205.12514v1](https://arxiv.org/abs/2205.12514v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.12514Focus to learn more |





<h2 id="2022-05-27-6">6. TranSpeech: Speech-to-Speech Translation With Bilateral Perturbation
</h2>

Title: [TranSpeech: Speech-to-Speech Translation With Bilateral Perturbation](https://arxiv.org/abs/2205.12523)

Authors: [Rongjie Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+R), [Zhou Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+Z), [Jinglin Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+J), [Huadai Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+H), [Yi Ren](https://arxiv.org/search/cs?searchtype=author&query=Ren%2C+Y), [Lichao Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+L), [Jinzheng He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+J)

> Direct speech-to-speech translation (S2ST) systems leverage recent progress in speech representation learning, where a sequence of discrete representations (units) derived in a self-supervised manner, are predicted from the model and passed to a vocoder for speech synthesis, still facing the following challenges: 1) Acoustic multimodality: the discrete units derived from speech with same content could be indeterministic due to the acoustic property (e.g., rhythm, pitch, and energy), which causes deterioration of translation accuracy; 2) high latency: current S2ST systems utilize autoregressive models which predict each unit conditioned on the sequence previously generated, failing to take full advantage of parallelism. In this work, we propose TranSpeech, a speech-to-speech translation model with bilateral perturbation. To alleviate the acoustic multimodal problem, we propose bilateral perturbation, which consists of the style normalization and information enhancement stages, to learn only the linguistic information from speech samples and generate more deterministic representations. With reduced multimodality, we step forward and become the first to establish a non-autoregressive S2ST technique, which repeatedly masks and predicts unit choices and produces high-accuracy results in just a few cycles. Experimental results on three language pairs demonstrate the state-of-the-art results by up to 2.5 BLEU points over the best publicly-available textless S2ST baseline. Moreover, TranSpeech shows a significant improvement in inference latency, enabling speedup up to 21.4x than autoregressive technique. Audio samples are available at \url{[this https URL](https://transpeech.github.io/)}

| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.12523](https://arxiv.org/abs/2205.12523) [cs.CL]** |
|           | (or **[arXiv:2205.12523v1](https://arxiv.org/abs/2205.12523v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.12523Focus to learn more |





<h2 id="2022-05-27-7">7. Are Large Pre-Trained Language Models Leaking Your Personal Information?
</h2>

Title: [Are Large Pre-Trained Language Models Leaking Your Personal Information?](https://arxiv.org/abs/2205.12628)

Authors: [Jie Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+J), [Hanyin Shao](https://arxiv.org/search/cs?searchtype=author&query=Shao%2C+H), [Kevin Chen-Chuan Chang](https://arxiv.org/search/cs?searchtype=author&query=Chang%2C+K+C)

> Large Pre-Trained Language Models (PLMs) have facilitated and dominated many NLP tasks in recent years. However, despite the great success of PLMs, there are also privacy concerns brought with PLMs. For example, recent studies show that PLMs memorize a lot of training data, including sensitive information, while the information may be leaked unintentionally and be utilized by malicious attackers. 
> In this paper, we propose to measure whether PLMs are prone to leaking personal information. Specifically, we attempt to query PLMs for email addresses with contexts of the email address or prompts containing the owner's name. We find that PLMs do leak personal information due to memorization. However, the risk of specific personal information being extracted by attackers is low because the models are weak at associating the personal information with its owner. We hope this work could help the community to better understand the privacy risk of PLMs and bring new insights to make PLMs safe.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Cryptography and Security (cs.CR) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.12628](https://arxiv.org/abs/2205.12628) [cs.CL]** |
|           | (or **[arXiv:2205.12628v1](https://arxiv.org/abs/2205.12628v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.12628Focus to learn more |





<h2 id="2022-05-27-8">8. Multimodal Knowledge Alignment with Reinforcement Learning
</h2>

Title: [Multimodal Knowledge Alignment with Reinforcement Learning](https://arxiv.org/abs/2205.12630)

Authors: [Youngjae Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+Y), [Jiwan Chung](https://arxiv.org/search/cs?searchtype=author&query=Chung%2C+J), [Heeseung Yun](https://arxiv.org/search/cs?searchtype=author&query=Yun%2C+H), [Jack Hessel](https://arxiv.org/search/cs?searchtype=author&query=Hessel%2C+J), [JaeSung Park](https://arxiv.org/search/cs?searchtype=author&query=Park%2C+J), [Ximing Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+X), [Prithviraj Ammanabrolu](https://arxiv.org/search/cs?searchtype=author&query=Ammanabrolu%2C+P), [Rowan Zellers](https://arxiv.org/search/cs?searchtype=author&query=Zellers%2C+R), [Ronan Le Bras](https://arxiv.org/search/cs?searchtype=author&query=Bras%2C+R+L), [Gunhee Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+G), [Yejin Choi](https://arxiv.org/search/cs?searchtype=author&query=Choi%2C+Y)

> Large language models readily adapt to novel settings, even without task-specific training data. Can their zero-shot capacity be extended to multimodal inputs? In this work, we propose ESPER which extends language-only zero-shot models to unseen multimodal tasks, like image and audio captioning. Our key novelty is to use reinforcement learning to align multimodal inputs to language model generations without direct supervision: for example, in the image case our reward optimization relies only on cosine similarity derived from CLIP, and thus requires no additional explicitly paired (image, caption) data. Because the parameters of the language model are left unchanged, the model maintains its capacity for zero-shot generalization. Experiments demonstrate that ESPER outperforms baselines and prior work on a variety of zero-shot tasks; these include a new benchmark we collect+release, ESP dataset, which tasks models with generating several diversely-styled captions for each image.

| Subjects:    | **Computation and Language (cs.CL)**; Computer Vision and Pattern Recognition (cs.CV) |
| ------------ | ------------------------------------------------------------ |
| ACM classes: | I.2.7; I.4.9                                                 |
| Cite as:     | **[arXiv:2205.12630](https://arxiv.org/abs/2205.12630) [cs.CL]** |
|              | (or **[arXiv:2205.12630v1](https://arxiv.org/abs/2205.12630v1) [cs.CL]** for this version) |
|              | https://doi.org/10.48550/arXiv.2205.12630Focus to learn more |





<h2 id="2022-05-27-9">9. Discovering Language-neutral Sub-networks in Multilingual Language Models
</h2>

Title: [Discovering Language-neutral Sub-networks in Multilingual Language Models](https://arxiv.org/abs/2205.12672)

Authors: [Negar Foroutan](https://arxiv.org/search/cs?searchtype=author&query=Foroutan%2C+N), [Mohammadreza Banaei](https://arxiv.org/search/cs?searchtype=author&query=Banaei%2C+M), [Remi Lebret](https://arxiv.org/search/cs?searchtype=author&query=Lebret%2C+R), [Antoine Bosselut](https://arxiv.org/search/cs?searchtype=author&query=Bosselut%2C+A), [Karl Aberer](https://arxiv.org/search/cs?searchtype=author&query=Aberer%2C+K)

> Multilingual pre-trained language models perform remarkably well on cross-lingual transfer for downstream tasks. Despite their impressive performance, our understanding of their language neutrality (i.e., the extent to which they use shared representations to encode similar phenomena across languages) and its role in achieving such performance remain open questions. In this work, we conceptualize language neutrality of multilingual models as a function of the overlap between language-encoding sub-networks of these models. Using mBERT as a foundation, we employ the lottery ticket hypothesis to discover sub-networks that are individually optimized for various languages and tasks. Using three distinct tasks and eleven typologically-diverse languages in our evaluation, we show that the sub-networks found for different languages are in fact quite similar, supporting the idea that mBERT jointly encodes multiple languages in shared parameters. We conclude that mBERT is comprised of a language-neutral sub-network shared among many languages, along with multiple ancillary language-specific sub-networks, with the former playing a more prominent role in mBERT's impressive cross-lingual performance.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.12672](https://arxiv.org/abs/2205.12672) [cs.CL]** |
|           | (or **[arXiv:2205.12672v1](https://arxiv.org/abs/2205.12672v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.12672Focus to learn more |





<h2 id="2022-05-27-10">10. Understanding Natural Language in Context
</h2>

Title: [Understanding Natural Language in Context](https://arxiv.org/abs/2205.12691)

Authors: [Avichai Levy](https://arxiv.org/search/cs?searchtype=author&query=Levy%2C+A), [Erez Karpas](https://arxiv.org/search/cs?searchtype=author&query=Karpas%2C+E)

> Recent years have seen an increasing number of applications that have a natural language interface, either in the form of chatbots or via personal assistants such as Alexa (Amazon), Google Assistant, Siri (Apple), and Cortana (Microsoft). To use these applications, a basic dialog between the robot and the human is required. 
> While this kind of dialog exists today mainly within "static" robots that do not make any movement in the household space, the challenge of reasoning about the information conveyed by the environment increases significantly when dealing with robots that can move and manipulate objects in our home environment. 
> In this paper, we focus on cognitive robots, which have some knowledge-based models of the world and operate by reasoning and planning with this model. Thus, when the robot and the human communicate, there is already some formalism they can use - the robot's knowledge representation formalism. 
> Our goal in this research is to translate natural language utterances into this robot's formalism, allowing much more complicated household tasks to be completed. We do so by combining off-the-shelf SOTA language models, planning tools, and the robot's knowledge-base for better communication. In addition, we analyze different directive types and illustrate the contribution of the world's context to the translation process.

| Subjects: | **Computation and Language (cs.CL)**; Robotics (cs.RO)       |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.12691](https://arxiv.org/abs/2205.12691) [cs.CL]** |
|           | (or **[arXiv:2205.12691v1](https://arxiv.org/abs/2205.12691v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.12691Focus to learn more |





<h2 id="2022-05-27-11">11. Eliciting Transferability in Multi-task Learning with Task-level Mixture-of-Experts
</h2>

Title: [Eliciting Transferability in Multi-task Learning with Task-level Mixture-of-Experts](https://arxiv.org/abs/2205.12701)

Authors: [Qinyuan Ye](https://arxiv.org/search/cs?searchtype=author&query=Ye%2C+Q), [Juan Zha](https://arxiv.org/search/cs?searchtype=author&query=Zha%2C+J), [Xiang Ren](https://arxiv.org/search/cs?searchtype=author&query=Ren%2C+X)

> Recent work suggests that transformer models are capable of multi-task learning on diverse NLP tasks. However, the potential of these models may be limited as they use the same set of parameters for all tasks. In contrast, humans tackle tasks in a more flexible way, by making proper presumptions on what skills and knowledge are relevant and executing only the necessary computations. Inspired by this, we propose to use task-level mixture-of-expert models, which has a collection of transformer layers (i.e., experts) and a router component to choose among these experts dynamically and flexibly. We show that the learned routing decisions and experts partially rediscover human categorization of NLP tasks -- certain experts are strongly associated with extractive tasks, some with classification tasks, and some with tasks requiring world knowledge.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.12701](https://arxiv.org/abs/2205.12701) [cs.CL]** |
|           | (or **[arXiv:2205.12701v1](https://arxiv.org/abs/2205.12701v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.12701Focus to learn more |










# 2022-05-23

[Return to Index](#Index)



<h2 id="2022-05-23-1">1. Translating Hanja historical documents to understandable Korean and English
</h2>

Title: [Translating Hanja historical documents to understandable Korean and English](https://arxiv.org/abs/2205.10019)

Authors: [Juhee Son](https://arxiv.org/search/cs?searchtype=author&query=Son%2C+J), [Jiho Jin](https://arxiv.org/search/cs?searchtype=author&query=Jin%2C+J), [Haneul Yoo](https://arxiv.org/search/cs?searchtype=author&query=Yoo%2C+H), [JinYeong Bak](https://arxiv.org/search/cs?searchtype=author&query=Bak%2C+J), [Kyunghyun Cho](https://arxiv.org/search/cs?searchtype=author&query=Cho%2C+K), [Alice Oh](https://arxiv.org/search/cs?searchtype=author&query=Oh%2C+A)

> The Annals of Joseon Dynasty (AJD) contain the daily records of the Kings of Joseon, the 500-year kingdom preceding the modern nation of Korea. The Annals were originally written in an archaic Korean writing system, `Hanja', and translated into Korean from 1968 to 1993. However, this translation was literal and contained many archaic Korean words; thus, a new expert translation effort began in 2012, completing the records of only one king in a decade. Also, expert translators are working on an English translation, of which only one king's records are available because of the high cost and slow progress. Thus, we propose H2KE, the neural machine translation model that translates Hanja historical documents to understandable Korean and English. Based on the multilingual neural machine translation approach, it translates the historical document written in Hanja, using both the full dataset of outdated Korean translation and a small dataset of recently translated Korean and English. We compare our method with two baselines: one is a recent model that simultaneously learns to restore and translate Hanja historical document and the other is the transformer that trained on newly translated corpora only. The results show that our method significantly outperforms the baselines in terms of BLEU score in both modern Korean and English translations. We also conduct a human evaluation that shows that our translation is preferred over the original expert translation.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.10019](https://arxiv.org/abs/2205.10019) [cs.CL]** |
|           | (or **[arXiv:2205.10019v1](https://arxiv.org/abs/2205.10019v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.10019Focus to learn more |





<h2 id="2022-05-23-2">2. Exploring Extreme Parameter Compression for Pre-trained Language Models
</h2>

Title: [Exploring Extreme Parameter Compression for Pre-trained Language Models](https://arxiv.org/abs/2205.10036)

Authors: [Yuxin Ren](https://arxiv.org/search/cs?searchtype=author&query=Ren%2C+Y), [Benyou Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+B), [Lifeng Shang](https://arxiv.org/search/cs?searchtype=author&query=Shang%2C+L), [Xin Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+X), [Qun Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Q)

> Recent work explored the potential of large-scale Transformer-based pre-trained models, especially Pre-trained Language Models (PLMs) in natural language processing. This raises many concerns from various perspectives, e.g., financial costs and carbon emissions. Compressing PLMs like BERT with negligible performance loss for faster inference and cheaper deployment has attracted much attention. In this work, we aim to explore larger compression ratios for PLMs, among which tensor decomposition is a potential but under-investigated one. Two decomposition and reconstruction protocols are further proposed to improve the effectiveness and efficiency during compression. Our compressed BERT with 1/7 parameters in Transformer layers performs on-par with, sometimes slightly better than the original BERT in GLUE benchmark. A tiny version achieves 96.7% performance of BERT-base with 1/48 encoder parameters (i.e., less than 2M parameters excluding the embedding layer) and 2.7× faster on inference. To show that the proposed method is orthogonal to existing compression methods like knowledge distillation, we also explore the benefit of the proposed method on a distilled BERT.

| Comments: | Accepted at ICLR2022. Code available at [this https URL](https://github.com/twinkle0331/Xcompression) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2205.10036](https://arxiv.org/abs/2205.10036) [cs.CL]** |
|           | (or **[arXiv:2205.10036v1](https://arxiv.org/abs/2205.10036v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.10036Focus to learn more |





<h2 id="2022-05-23-3">3. How to keep text private? A systematic review of deep learning methods for privacy-preserving natural language processing
</h2>

Title: [How to keep text private? A systematic review of deep learning methods for privacy-preserving natural language processing](https://arxiv.org/abs/2205.10095)

Authors: [Samuel Sousa](https://arxiv.org/search/cs?searchtype=author&query=Sousa%2C+S), [Roman Kern](https://arxiv.org/search/cs?searchtype=author&query=Kern%2C+R)

> Deep learning (DL) models for natural language processing (NLP) tasks often handle private data, demanding protection against breaches and disclosures. Data protection laws, such as the European Union's General Data Protection Regulation (GDPR), thereby enforce the need for privacy. Although many privacy-preserving NLP methods have been proposed in recent years, no categories to organize them have been introduced yet, making it hard to follow the progress of the literature. To close this gap, this article systematically reviews over sixty DL methods for privacy-preserving NLP published between 2016 and 2020, covering theoretical foundations, privacy-enhancing technologies, and analysis of their suitability for real-world scenarios. First, we introduce a novel taxonomy for classifying the existing methods into three categories: data safeguarding methods, trusted methods, and verification methods. Second, we present an extensive summary of privacy threats, datasets for applications, and metrics for privacy evaluation. Third, throughout the review, we describe privacy issues in the NLP pipeline in a holistic view. Further, we discuss open challenges in privacy-preserving NLP regarding data traceability, computation overhead, dataset size, the prevalence of human biases in embeddings, and the privacy-utility tradeoff. Finally, this review presents future research directions to guide successive research and development of privacy-preserving NLP models.

| Comments: | 59 pages, 15 figures                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.10095](https://arxiv.org/abs/2205.10095) [cs.CL]** |
|           | (or **[arXiv:2205.10095v1](https://arxiv.org/abs/2205.10095v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.10095Focus to learn more |





<h2 id="2022-05-23-4">4. Visually-Augmented Language Modeling
</h2>

Title: [Visually-Augmented Language Modeling](https://arxiv.org/abs/2205.10178)

Authors: [Weizhi Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+W), [Li Dong](https://arxiv.org/search/cs?searchtype=author&query=Dong%2C+L), [Hao Cheng](https://arxiv.org/search/cs?searchtype=author&query=Cheng%2C+H), [Haoyu Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+H), [Xiaodong Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+X), [Xifeng Yan](https://arxiv.org/search/cs?searchtype=author&query=Yan%2C+X), [Jianfeng Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+J), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F)

> Human language is grounded on multimodal knowledge including visual knowledge like colors, sizes, and shapes. However, current large-scale pre-trained language models rely on the text-only self-supervised training with massive text data, which precludes them from utilizing relevant visual information when necessary. To address this, we propose a novel pre-training framework, named VaLM, to Visually-augment text tokens with retrieved relevant images for Language Modeling. Specifically, VaLM builds on a novel text-vision alignment method via an image retrieval module to fetch corresponding images given a textual context. With the visually-augmented context, VaLM uses a visual knowledge fusion layer to enable multimodal grounded language modeling by attending on both text context and visual knowledge in images. We evaluate the proposed model on various multimodal commonsense reasoning tasks, which require visual information to excel. VaLM outperforms the text-only baseline with substantial gains of +8.66% and +37.81% accuracy on object color and size reasoning, respectively.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.10178](https://arxiv.org/abs/2205.10178) [cs.CL]** |
|           | (or **[arXiv:2205.10178v1](https://arxiv.org/abs/2205.10178v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.10178Focus to learn more |





<h2 id="2022-05-23-5">5. Visualizing and Explaining Language Models
</h2>

Title: [Visualizing and Explaining Language Models](https://arxiv.org/abs/2205.10238)

Authors: [Adrian M.P. Braşoveanu](https://arxiv.org/search/cs?searchtype=author&query=Braşoveanu%2C+A+M), [Răzvan Andonie](https://arxiv.org/search/cs?searchtype=author&query=Andonie%2C+R)

> During the last decade, Natural Language Processing has become, after Computer Vision, the second field of Artificial Intelligence that was massively changed by the advent of Deep Learning. Regardless of the architecture, the language models of the day need to be able to process or generate text, as well as predict missing words, sentences or relations depending on the task. Due to their black-box nature, such models are difficult to interpret and explain to third parties. Visualization is often the bridge that language model designers use to explain their work, as the coloring of the salient words and phrases, clustering or neuron activations can be used to quickly understand the underlying models. This paper showcases the techniques used in some of the most popular Deep Learning for NLP visualizations, with a special focus on interpretability and explainability.

| Subjects: | **Computation and Language (cs.CL)**; Human-Computer Interaction (cs.HC); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.10238](https://arxiv.org/abs/2205.10238) [cs.CL]** |
|           | (or **[arXiv:2205.10238v1](https://arxiv.org/abs/2205.10238v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.10238Focus to learn more |





<h2 id="2022-05-23-6">6. Lossless Acceleration for Seq2seq Generation with Aggressive Decoding
</h2>

Title: [Lossless Acceleration for Seq2seq Generation with Aggressive Decoding](https://arxiv.org/abs/2205.10350)

Authors: [Tao Ge](https://arxiv.org/search/cs?searchtype=author&query=Ge%2C+T), [Heming Xia](https://arxiv.org/search/cs?searchtype=author&query=Xia%2C+H), [Xin Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+X), [Si-Qing Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+S), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F)

> We study lossless acceleration for seq2seq generation with a novel decoding algorithm -- Aggressive Decoding. Unlike the previous efforts (e.g., non-autoregressive decoding) speeding up seq2seq generation at the cost of quality loss, our approach aims to yield the identical (or better) generation compared with autoregressive decoding but in a significant speedup, achieved by innovative cooperation of aggressive decoding and verification that are both efficient due to parallel computing. 
> We propose two Aggressive Decoding paradigms for 2 kinds of seq2seq tasks: 1) For the seq2seq tasks whose inputs and outputs are highly similar (e.g., Grammatical Error Correction), we propose Input-guided Aggressive Decoding (IAD) that aggressively copies from the input sentence as drafted decoded tokens to verify in parallel; 2) For other general seq2seq tasks (e.g., Machine Translation), we propose Generalized Aggressive Decoding (GAD) that first employs an additional non-autoregressive decoding model for aggressive decoding and then verifies in parallel in the autoregressive manner. 
> We test Aggressive Decoding on the most popular 6-layer Transformer model on GPU in multiple seq2seq tasks: 1) For IAD, we show that it can introduce a 7x-9x speedup for the Transformer in Grammatical Error Correction and Text Simplification tasks with the identical results as greedy decoding; 2) For GAD, we observe a 3x-5x speedup with the identical or even better quality in two important seq2seq tasks: Machine Translation and Abstractive Summarization. Moreover, Aggressive Decoding can benefit even more from stronger computing devices that are better at parallel computing. Given the lossless quality as well as significant and promising speedup, we believe Aggressive Decoding may potentially evolve into a de facto standard for efficient and lossless seq2seq generation in the near future.

| Comments: | 24-page Microsoft Research Technical Report. Content overlap with [arXiv:2106.04970](https://arxiv.org/abs/2106.04970) and [arXiv:2203.16487](https://arxiv.org/abs/2203.16487) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2205.10350](https://arxiv.org/abs/2205.10350) [cs.CL]** |
|           | (or **[arXiv:2205.10350v1](https://arxiv.org/abs/2205.10350v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.10350Focus to learn more |








# 2022-05-20

[Return to Index](#Index)



<h2 id="2022-05-20-1">1. PreQuEL: Quality Estimation of Machine Translation Outputs in Advance
</h2>

Title: [PreQuEL: Quality Estimation of Machine Translation Outputs in Advance](https://arxiv.org/abs/2205.09178)

Authors: [Shachar Don-Yehiya](https://arxiv.org/search/cs?searchtype=author&query=Don-Yehiya%2C+S), [Leshem Choshen](https://arxiv.org/search/cs?searchtype=author&query=Choshen%2C+L), [Omri Abend](https://arxiv.org/search/cs?searchtype=author&query=Abend%2C+O)

> We present the task of PreQuEL, Pre-(Quality-Estimation) Learning. A PreQuEL system predicts how well a given sentence will be translated, without recourse to the actual translation, thus eschewing unnecessary resource allocation when translation quality is bound to be low. PreQuEL can be defined relative to a given MT system (e.g., some industry service) or generally relative to the state-of-the-art. From a theoretical perspective, PreQuEL places the focus on the source text, tracing properties, possibly linguistic features, that make a sentence harder to machine translate. 
> We develop a baseline model for the task and analyze its performance. We also develop a data augmentation method (from parallel corpora), that improves results substantially. We show that this augmentation method can improve the performance of the Quality-Estimation task as well. We investigate the properties of the input text that our model is sensitive to, by testing it on challenge sets and different languages. We conclude that it is aware of syntactic and semantic distinctions, and correlates and even over-emphasizes the importance of standard NLP features.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.09178](https://arxiv.org/abs/2205.09178) [cs.CL]** |
|           | (or **[arXiv:2205.09178v1](https://arxiv.org/abs/2205.09178v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.09178Focus to learn more |





<h2 id="2022-05-20-2">2. Evaluating Subtitle Segmentation for End-to-end Generation Systems
</h2>

Title: [Evaluating Subtitle Segmentation for End-to-end Generation Systems](https://arxiv.org/abs/2205.09360)

Authors: [Alina Karakanta](https://arxiv.org/search/cs?searchtype=author&query=Karakanta%2C+A), [François Buet](https://arxiv.org/search/cs?searchtype=author&query=Buet%2C+F), [Mauro Cettolo](https://arxiv.org/search/cs?searchtype=author&query=Cettolo%2C+M), [François Yvon](https://arxiv.org/search/cs?searchtype=author&query=Yvon%2C+F)

> Subtitles appear on screen as short pieces of text, segmented based on formal constraints (length) and syntactic/semantic criteria. Subtitle segmentation can be evaluated with sequence segmentation metrics against a human reference. However, standard segmentation metrics cannot be applied when systems generate outputs different than the reference, e.g. with end-to-end subtitling systems. In this paper, we study ways to conduct reference-based evaluations of segmentation accuracy irrespective of the textual content. We first conduct a systematic analysis of existing metrics for evaluating subtitle segmentation. We then introduce Sigma, a new Subtitle Segmentation Score derived from an approximate upper-bound of BLEU on segmentation boundaries, which allows us to disentangle the effect of good segmentation from text quality. To compare Sigma with existing metrics, we further propose a boundary projection method from imperfect hypotheses to the true reference. Results show that all metrics are able to reward high quality output but for similar outputs system ranking depends on each metric's sensitivity to error type. Our thorough analyses suggest Sigma is a promising segmentation candidate but its reliability over other segmentation metrics remains to be validated through correlations with human judgements.

| Comments: | Accepted at LREC 2022                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.09360](https://arxiv.org/abs/2205.09360) [cs.CL]** |
|           | (or **[arXiv:2205.09360v1](https://arxiv.org/abs/2205.09360v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.09360Focus to learn more |





<h2 id="2022-05-20-3">3. Insights on Neural Representations for End-to-End Speech Recognition
</h2>

Title: [Insights on Neural Representations for End-to-End Speech Recognition](https://arxiv.org/abs/2205.09456)

Authors: [Anna Ollerenshaw](https://arxiv.org/search/cs?searchtype=author&query=Ollerenshaw%2C+A), [Md Asif Jalal](https://arxiv.org/search/cs?searchtype=author&query=Jalal%2C+M+A), [Thomas Hain](https://arxiv.org/search/cs?searchtype=author&query=Hain%2C+T)

> End-to-end automatic speech recognition (ASR) models aim to learn a generalised speech representation. However, there are limited tools available to understand the internal functions and the effect of hierarchical dependencies within the model architecture. It is crucial to understand the correlations between the layer-wise representations, to derive insights on the relationship between neural representations and performance. 
> Previous investigations of network similarities using correlation analysis techniques have not been explored for End-to-End ASR models. This paper analyses and explores the internal dynamics between layers during training with CNN, LSTM and Transformer based approaches using Canonical correlation analysis (CCA) and centered kernel alignment (CKA) for the experiments. It was found that neural representations within CNN layers exhibit hierarchical correlation dependencies as layer depth increases but this is mostly limited to cases where neural representation correlates more closely. This behaviour is not observed in LSTM architecture, however there is a bottom-up pattern observed across the training process, while Transformer encoder layers exhibit irregular coefficiency correlation as neural depth increases. Altogether, these results provide new insights into the role that neural architectures have upon speech recognition performance. More specifically, these techniques can be used as indicators to build better performing speech recognition models.

| Comments:          | Submitted to Interspeech 2021                                |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:           | **[arXiv:2205.09456](https://arxiv.org/abs/2205.09456) [cs.CL]** |
|                    | (or **[arXiv:2205.09456v1](https://arxiv.org/abs/2205.09456v1) [cs.CL]** for this version) |
|                    | https://doi.org/10.48550/arXiv.2205.09456Focus to learn more |
| Journal reference: | Proc. Interspeech 2021, 4079-4083                            |
| Related DOI:       | https://doi.org/10.21437/Interspeech.2021-1516Focus to learn more |





<h2 id="2022-05-20-4">4. Phylogeny-Inspired Adaptation of Multilingual Models to New Languages
</h2>

Title: [Phylogeny-Inspired Adaptation of Multilingual Models to New Languages](https://arxiv.org/abs/2205.09634)

Authors: [Fahim Faisal](https://arxiv.org/search/cs?searchtype=author&query=Faisal%2C+F), [Antonios Anastasopoulos](https://arxiv.org/search/cs?searchtype=author&query=Anastasopoulos%2C+A)

> Large pretrained multilingual models, trained on dozens of languages, have delivered promising results due to cross-lingual learning capabilities on variety of language tasks. Further adapting these models to specific languages, especially ones unseen during pre-training, is an important goal towards expanding the coverage of language technologies. In this study, we show how we can use language phylogenetic information to improve cross-lingual transfer leveraging closely related languages in a structured, linguistically-informed manner. We perform adapter-based training on languages from diverse language families (Germanic, Uralic, Tupian, Uto-Aztecan) and evaluate on both syntactic and semantic tasks, obtaining more than 20% relative performance improvements over strong commonly used baselines, especially on languages unseen during pre-training.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.09634](https://arxiv.org/abs/2205.09634) [cs.CL]** |
|           | (or **[arXiv:2205.09634v1](https://arxiv.org/abs/2205.09634v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.09634Focus to learn more |





<h2 id="2022-05-20-5">5. Voxel-informed Language Grounding
</h2>

Title: [Voxel-informed Language Grounding](https://arxiv.org/abs/2205.09710)

Authors: [Rodolfo Corona](https://arxiv.org/search/cs?searchtype=author&query=Corona%2C+R), [Shizhan Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+S), [Dan Klein](https://arxiv.org/search/cs?searchtype=author&query=Klein%2C+D), [Trevor Darrell](https://arxiv.org/search/cs?searchtype=author&query=Darrell%2C+T)

> Natural language applied to natural 2D images describes a fundamentally 3D world. We present the Voxel-informed Language Grounder (VLG), a language grounding model that leverages 3D geometric information in the form of voxel maps derived from the visual input using a volumetric reconstruction model. We show that VLG significantly improves grounding accuracy on SNARE, an object reference game task. At the time of writing, VLG holds the top place on the SNARE leaderboard, achieving SOTA results with a 2.0% absolute improvement.

| Comments: | ACL 2022                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV) |
| Cite as:  | **[arXiv:2205.09710](https://arxiv.org/abs/2205.09710) [cs.CL]** |
|           | (or **[arXiv:2205.09710v1](https://arxiv.org/abs/2205.09710v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.09710Focus to learn more |





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

