# MA C.'s Daily Paper Of Interest - June a., 2022

# Index

- [2022-06-03](#2022-06-03)
  - [1. Squeezeformer: An Efficient Transformer for Automatic Speech Recognition](#2022-06-03-1)
  
  - [2. VL-BEiT: Generative Vision-Language Pretraining](#2022-06-03-2)
  - [3. BayesFormer: Transformer with Uncertainty Estimation](#2022-06-03-3)
  - [4. Finding the Right Recipe for Low Resource Domain Adaptation in Neural Machine Translation](#2022-06-03-4)
  
- [2022-06-02](#2022-06-02)
  - [1. VALHALLA: Visual Hallucination for Machine Translation](#2022-06-02-1)

  - [2. Discovering the Hidden Vocabulary of DALLE-2](#2022-06-02-2)

  - [3. On Layer Normalizations and Residual Connections in Transformers](#2022-06-02-3)

  - [4. Optical character recognition quality affects perceived usefulness of historical newspaper clippings](#2022-06-02-4)

  - [5. Exploring Diversity in Back Translation for Low-Resource Machine Translation](#2022-06-02-5)

  - [6. Cross-View Language Modeling: Towards Unified Cross-Lingual Cross-Modal Pre-training](#2022-06-02-6)

- [2022-06-01](#2022-06-01)
  - [1. Parameter-Efficient and Student-Friendly Knowledge Distillation](#2022-06-01-1)

  - [2. ADAPT: Vision-Language Navigation with Modality-Aligned Action Prompts](#2022-06-01-2)

  - [3. CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers](#2022-06-01-3)

  - [4. EMS: Efficient and Effective Massively Multilingual Sentence Representation Learning](#2022-06-01-4)

- [2022-05-31](#2022-05-31)
  - [1. VLUE: A Multi-Task Benchmark for Evaluating Vision-Language Models](#2022-05-31-1)

- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



# 2022-06-03

[Return to Index](#Index)



<h2 id="2022-06-03-1">1. Squeezeformer: An Efficient Transformer for Automatic Speech Recognition
</h2>

Title: [Squeezeformer: An Efficient Transformer for Automatic Speech Recognition](https://arxiv.org/abs/2206.00888)

Authors: [Sehoon Kim](https://arxiv.org/search/eess?searchtype=author&query=Kim%2C+S), [Amir Gholami](https://arxiv.org/search/eess?searchtype=author&query=Gholami%2C+A), [Albert Shaw](https://arxiv.org/search/eess?searchtype=author&query=Shaw%2C+A), [Nicholas Lee](https://arxiv.org/search/eess?searchtype=author&query=Lee%2C+N), [Karttikeya Mangalam](https://arxiv.org/search/eess?searchtype=author&query=Mangalam%2C+K), [Jitendra Malik](https://arxiv.org/search/eess?searchtype=author&query=Malik%2C+J), [Michael W. Mahoney](https://arxiv.org/search/eess?searchtype=author&query=Mahoney%2C+M+W), [Kurt Keutzer](https://arxiv.org/search/eess?searchtype=author&query=Keutzer%2C+K)

> The recently proposed Conformer model has become the de facto backbone model for various downstream speech tasks based on its hybrid attention-convolution architecture that captures both local and global features. However, through a series of systematic studies, we find that the Conformer architecture's design choices are not optimal. After reexamining the design choices for both the macro and micro-architecture of Conformer, we propose the Squeezeformer model, which consistently outperforms the state-of-the-art ASR models under the same training schemes. In particular, for the macro-architecture, Squeezeformer incorporates (i) the Temporal U-Net structure, which reduces the cost of the multi-head attention modules on long sequences, and (ii) a simpler block structure of feed-forward module, followed up by multi-head attention or convolution modules, instead of the Macaron structure proposed in Conformer. Furthermore, for the micro-architecture, Squeezeformer (i) simplifies the activations in the convolutional block, (ii) removes redundant Layer Normalization operations, and (iii) incorporates an efficient depth-wise downsampling layer to efficiently sub-sample the input signal. Squeezeformer achieves state-of-the-art results of 7.5%, 6.5%, and 6.0% word-error-rate on Librispeech test-other without external language models. This is 3.1%, 1.4%, and 0.6% better than Conformer-CTC with the same number of FLOPs. Our code is open-sourced and available online.

| Subjects: | **Audio and Speech Processing (eess.AS)**; Computation and Language (cs.CL); Sound (cs.SD) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.00888](https://arxiv.org/abs/2206.00888) [eess.AS]** |
|           | (or **[arXiv:2206.00888v1](https://arxiv.org/abs/2206.00888v1) [eess.AS]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.00888Focus to learn more |





<h2 id="2022-06-03-2">2. VL-BEiT: Generative Vision-Language Pretraining
</h2>

Title: [VL-BEiT: Generative Vision-Language Pretraining](https://arxiv.org/abs/2206.01127)

Authors: [Hangbo Bao](https://arxiv.org/search/cs?searchtype=author&query=Bao%2C+H), [Wenhui Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+W), [Li Dong](https://arxiv.org/search/cs?searchtype=author&query=Dong%2C+L), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F)

> We introduce a vision-language foundation model called VL-BEiT, which is a bidirectional multimodal Transformer learned by generative pretraining. Our minimalist solution conducts masked prediction on both monomodal and multimodal data with a shared Transformer. Specifically, we perform masked vision-language modeling on image-text pairs, masked language modeling on texts, and masked image modeling on images. VL-BEiT is learned from scratch with one unified pretraining task, one shared backbone, and one-stage training. Our method is conceptually simple and empirically effective. Experimental results show that VL-BEiT obtains strong results on various vision-language benchmarks, such as visual question answering, visual reasoning, and image-text retrieval. Moreover, our method learns transferable visual features, achieving competitive performance on image classification, and semantic segmentation.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.01127](https://arxiv.org/abs/2206.01127) [cs.CV]** |
|           | (or **[arXiv:2206.01127v1](https://arxiv.org/abs/2206.01127v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.01127Focus to learn more |





<h2 id="2022-06-03-3">3. BayesFormer: Transformer with Uncertainty Estimation
</h2>

Title: [BayesFormer: Transformer with Uncertainty Estimation](https://arxiv.org/abs/2206.00826)

Authors: [Karthik Abinav Sankararaman](https://arxiv.org/search/cs?searchtype=author&query=Sankararaman%2C+K+A), [Sinong Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+S), [Han Fang](https://arxiv.org/search/cs?searchtype=author&query=Fang%2C+H)

> Transformer has become ubiquitous due to its dominant performance in various NLP and image processing tasks. However, it lacks understanding of how to generate mathematically grounded uncertainty estimates for transformer architectures. Models equipped with such uncertainty estimates can typically improve predictive performance, make networks robust, avoid over-fitting and used as acquisition function in active learning. In this paper, we introduce BayesFormer, a Transformer model with dropouts designed by Bayesian theory. We proposed a new theoretical framework to extend the approximate variational inference-based dropout to Transformer-based architectures. Through extensive experiments, we validate the proposed architecture in four paradigms and show improvements across the board: language modeling and classification, long-sequence understanding, machine translation and acquisition function for active learning.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.00826](https://arxiv.org/abs/2206.00826) [cs.CL]** |
|           | (or **[arXiv:2206.00826v1](https://arxiv.org/abs/2206.00826v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.00826Focus to learn more |





<h2 id="2022-06-03-4">4. Finding the Right Recipe for Low Resource Domain Adaptation in Neural Machine Translation
</h2>

Title: [Finding the Right Recipe for Low Resource Domain Adaptation in Neural Machine Translation](https://arxiv.org/abs/2206.01137)

Authors: [Virginia Adams](https://arxiv.org/search/cs?searchtype=author&query=Adams%2C+V), [Sandeep Subramanian](https://arxiv.org/search/cs?searchtype=author&query=Subramanian%2C+S), [Mike Chrzanowski](https://arxiv.org/search/cs?searchtype=author&query=Chrzanowski%2C+M), [Oleksii Hrinchuk](https://arxiv.org/search/cs?searchtype=author&query=Hrinchuk%2C+O), [Oleksii Kuchaiev](https://arxiv.org/search/cs?searchtype=author&query=Kuchaiev%2C+O)

> General translation models often still struggle to generate accurate translations in specialized domains. To guide machine translation practitioners and characterize the effectiveness of domain adaptation methods under different data availability scenarios, we conduct an in-depth empirical exploration of monolingual and parallel data approaches to domain adaptation of pre-trained, third-party, NMT models in settings where architecture change is impractical. We compare data centric adaptation methods in isolation and combination. We study method effectiveness in very low resource (8k parallel examples) and moderately low resource (46k parallel examples) conditions and propose an ensemble approach to alleviate reductions in original domain translation quality. Our work includes three domains: consumer electronic, clinical, and biomedical and spans four language pairs - Zh-En, Ja-En, Es-En, and Ru-En. We also make concrete recommendations for achieving high in-domain performance and release our consumer electronic and medical domain datasets for all languages and make our code publicly available.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.01137](https://arxiv.org/abs/2206.01137) [cs.CL]** |
|           | (or **[arXiv:2206.01137v1](https://arxiv.org/abs/2206.01137v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.01137Focus to learn more |







# 2022-06-02

[Return to Index](#Index)



<h2 id="2022-06-02-1">1. VALHALLA: Visual Hallucination for Machine Translation
</h2>

Title: [VALHALLA: Visual Hallucination for Machine Translation](https://arxiv.org/abs/2206.00100)

Authors: [Yi Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Rameswar Panda](https://arxiv.org/search/cs?searchtype=author&query=Panda%2C+R), [Yoon Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+Y), [Chun-Fu](https://arxiv.org/search/cs?searchtype=author&query=Chun-Fu) (Richard)[Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen), [Rogerio Feris](https://arxiv.org/search/cs?searchtype=author&query=Feris%2C+R), [David Cox](https://arxiv.org/search/cs?searchtype=author&query=Cox%2C+D), [Nuno Vasconcelos](https://arxiv.org/search/cs?searchtype=author&query=Vasconcelos%2C+N)

> Designing better machine translation systems by considering auxiliary inputs such as images has attracted much attention in recent years. While existing methods show promising performance over the conventional text-only translation systems, they typically require paired text and image as input during inference, which limits their applicability to real-world scenarios. In this paper, we introduce a visual hallucination framework, called VALHALLA, which requires only source sentences at inference time and instead uses hallucinated visual representations for multimodal machine translation. In particular, given a source sentence an autoregressive hallucination transformer is used to predict a discrete visual representation from the input text, and the combined text and hallucinated representations are utilized to obtain the target translation. We train the hallucination transformer jointly with the translation transformer using standard backpropagation with cross-entropy losses while being guided by an additional loss that encourages consistency between predictions using either ground-truth or hallucinated visual representations. Extensive experiments on three standard translation datasets with a diverse set of language pairs demonstrate the effectiveness of our approach over both text-only baselines and state-of-the-art methods. Project page: [this http URL](http://www.svcl.ucsd.edu/projects/valhalla).

| Comments: | CVPR 2022                                                    |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2206.00100](https://arxiv.org/abs/2206.00100) [cs.CV]** |
|           | (or **[arXiv:2206.00100v1](https://arxiv.org/abs/2206.00100v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.00100Focus to learn more |





<h2 id="2022-06-02-2">2. Discovering the Hidden Vocabulary of DALLE-2
</h2>

Title: [Discovering the Hidden Vocabulary of DALLE-2](https://arxiv.org/abs/2206.00169)

Authors: [Giannis Daras](https://arxiv.org/search/cs?searchtype=author&query=Daras%2C+G), [Alexandros G. Dimakis](https://arxiv.org/search/cs?searchtype=author&query=Dimakis%2C+A+G)

> We discover that DALLE-2 seems to have a hidden vocabulary that can be used to generate images with absurd prompts. For example, it seems that \texttt{Apoploe vesrreaitais} means birds and \texttt{Contarra ccetnxniams luryca tanniounons} (sometimes) means bugs or pests. We find that these prompts are often consistent in isolation but also sometimes in combinations. We present our black-box method to discover words that seem random but have some correspondence to visual concepts. This creates important security and interpretability challenges.

| Comments: | 6 pages, 4 figures                                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Cryptography and Security (cs.CR); Computer Vision and Pattern Recognition (cs.CV) |
| Cite as:  | **[arXiv:2206.00169](https://arxiv.org/abs/2206.00169) [cs.LG]** |
|           | (or **[arXiv:2206.00169v1](https://arxiv.org/abs/2206.00169v1) [cs.LG]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.00169Focus to learn more |





<h2 id="2022-06-02-3">3. On Layer Normalizations and Residual Connections in Transformers
</h2>

Title: [On Layer Normalizations and Residual Connections in Transformers](https://arxiv.org/abs/2206.00330)

Authors: [Sho Takase](https://arxiv.org/search/cs?searchtype=author&query=Takase%2C+S), [Shun Kiyono](https://arxiv.org/search/cs?searchtype=author&query=Kiyono%2C+S), [Sosuke Kobayashi](https://arxiv.org/search/cs?searchtype=author&query=Kobayashi%2C+S), [Jun Suzuki](https://arxiv.org/search/cs?searchtype=author&query=Suzuki%2C+J)

> In the perspective of a layer normalization (LN) position, the architecture of Transformers can be categorized into two types: Post-LN and Pre-LN. Recent Transformers prefer to select Pre-LN because the training in Post-LN with deep Transformers, e.g., ten or more layers, often becomes unstable, resulting in useless models. However, in contrast, Post-LN has also consistently achieved better performance than Pre-LN in relatively shallow Transformers, e.g., six or fewer layers. This study first investigates the reason for these discrepant observations empirically and theoretically and discovers 1, the LN in Post-LN is the source of the vanishing gradient problem that mainly leads the unstable training whereas Pre-LN prevents it, and 2, Post-LN tends to preserve larger gradient norms in higher layers during the back-propagation that may lead an effective training. Exploiting the new findings, we propose a method that can equip both higher stability and effective training by a simple modification from Post-LN. We conduct experiments on a wide range of text generation tasks and demonstrate that our method outperforms Pre-LN, and stable training regardless of the shallow or deep layer settings.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.00330](https://arxiv.org/abs/2206.00330) [cs.LG]** |
|           | (or **[arXiv:2206.00330v1](https://arxiv.org/abs/2206.00330v1) [cs.LG]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.00330Focus to learn more |





<h2 id="2022-06-02-4">4. Optical character recognition quality affects perceived usefulness of historical newspaper clippings
</h2>

Title: [Optical character recognition quality affects perceived usefulness of historical newspaper clippings](https://arxiv.org/abs/2206.00369)

Authors: [Kimmo Kettunen](https://arxiv.org/search/cs?searchtype=author&query=Kettunen%2C+K), [Heikki Keskustalo](https://arxiv.org/search/cs?searchtype=author&query=Keskustalo%2C+H), [Sanna Kumpulainen](https://arxiv.org/search/cs?searchtype=author&query=Kumpulainen%2C+S), [Tuula Pääkkönen](https://arxiv.org/search/cs?searchtype=author&query=Pääkkönen%2C+T), [Juha Rautiainen](https://arxiv.org/search/cs?searchtype=author&query=Rautiainen%2C+J)

> Introduction. We study effect of different quality optical character recognition in interactive information retrieval with a collection of one digitized historical Finnish newspaper. Method. This study is based on the simulated interactive information retrieval work task model. Thirty-two users made searches to an article collection of Finnish newspaper Uusi Suometar 1869-1918 with ca. 1.45 million auto segmented articles. Our article search database had two versions of each article with different quality optical character recognition. Each user performed six pre-formulated and six self-formulated short queries and evaluated subjectively the top-10 results using graded relevance scale of 0-3 without knowing about the optical character recognition quality differences of the otherwise identical articles. Analysis. Analysis of the user evaluations was performed by comparing mean averages of evaluations scores in user sessions. Differences of query results were detected by analysing lengths of returned articles in pre-formulated and self-formulated queries and number of different documents retrieved overall in these two sessions. Results. The main result of the study is that improved optical character recognition quality affects perceived usefulness of historical newspaper articles positively. Conclusions. We were able to show that improvement in optical character recognition quality of documents leads to higher mean relevance evaluation scores of query results in our historical newspaper collection. To the best of our knowledge this simulated interactive user-task is the first one showing empirically that users' subjective relevance assessments are affected by a change in the quality of optically read text.

| Comments: | 21 pages, 6 figures, 2 tables, 1 appendix. arXiv admin note: substantial text overlap with [arXiv:2203.03557](https://arxiv.org/abs/2203.03557) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2206.00369](https://arxiv.org/abs/2206.00369) [cs.CL]** |
|           | (or **[arXiv:2206.00369v1](https://arxiv.org/abs/2206.00369v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.00369Focus to learn more |





<h2 id="2022-06-02-5">5. Exploring Diversity in Back Translation for Low-Resource Machine Translation
</h2>

Title: [Exploring Diversity in Back Translation for Low-Resource Machine Translation](https://arxiv.org/abs/2206.00564)

Authors: [Laurie Burchell](https://arxiv.org/search/cs?searchtype=author&query=Burchell%2C+L), [Alexandra Birch](https://arxiv.org/search/cs?searchtype=author&query=Birch%2C+A), [Kenneth Heafield](https://arxiv.org/search/cs?searchtype=author&query=Heafield%2C+K)

> Back translation is one of the most widely used methods for improving the performance of neural machine translation systems. Recent research has sought to enhance the effectiveness of this method by increasing the 'diversity' of the generated translations. We argue that the definitions and metrics used to quantify 'diversity' in previous work have been insufficient. This work puts forward a more nuanced framework for understanding diversity in training data, splitting it into lexical diversity and syntactic diversity. We present novel metrics for measuring these different aspects of diversity and carry out empirical analysis into the effect of these types of diversity on final neural machine translation model performance for low-resource English↔Turkish and mid-resource English↔Icelandic. Our findings show that generating back translation using nucleus sampling results in higher final model performance, and that this method of generation has high levels of both lexical and syntactic diversity. We also find evidence that lexical diversity is more important than syntactic for back translation performance.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.00564](https://arxiv.org/abs/2206.00564) [cs.CL]** |
|           | (or **[arXiv:2206.00564v1](https://arxiv.org/abs/2206.00564v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.00564Focus to learn more |





<h2 id="2022-06-02-6">6. Cross-View Language Modeling: Towards Unified Cross-Lingual Cross-Modal Pre-training
</h2>

Title: [Cross-View Language Modeling: Towards Unified Cross-Lingual Cross-Modal Pre-training](https://arxiv.org/abs/2206.00621)

Authors: [Yan Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+Y), [Wangchunshu Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+W), [Ao Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+A), [Xinsong Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+X)

> In this paper, we introduce Cross-View Language Modeling, a simple and effective language model pre-training framework that unifies cross-lingual cross-modal pre-training with shared architectures and objectives. Our approach is motivated by a key observation that cross-lingual and cross-modal pre-training share the same goal of aligning two different views of the same object into a common semantic space. To this end, the cross-view language modeling framework considers both multi-modal data (i.e., image-caption pairs) and multi-lingual data (i.e., parallel sentence pairs) as two different views of the same object, and trains the model to align the two views by maximizing the mutual information between them with conditional masked language modeling and contrastive learning. We pre-train CCLM, a Cross-lingual Cross-modal Language Model, with the cross-view language modeling framework. Empirical results on IGLUE, a multi-lingual multi-modal benchmark, and two multi-lingual image-text retrieval datasets show that while conceptually simpler, CCLM significantly outperforms the prior state-of-the-art with an average absolute improvement of over 10%. Notably, CCLM is the first multi-lingual multi-modal model that surpasses the translate-test performance of representative English vision-language models by zero-shot cross-lingual transfer.

| Comments: | 19 pages, 3 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2206.00621](https://arxiv.org/abs/2206.00621) [cs.CL]** |
|           | (or **[arXiv:2206.00621v1](https://arxiv.org/abs/2206.00621v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.00621Focus to learn more |







# 2022-06-01

[Return to Index](#Index)



<h2 id="2022-06-01-1">1. Parameter-Efficient and Student-Friendly Knowledge Distillation
</h2>

Title: [Parameter-Efficient and Student-Friendly Knowledge Distillation](https://arxiv.org/abs/2205.15308)

Authors: [Jun Rao](https://arxiv.org/search/cs?searchtype=author&query=Rao%2C+J), [Xv Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+X), [Liang Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+L), [Shuhan Qi](https://arxiv.org/search/cs?searchtype=author&query=Qi%2C+S), [Dacheng Tao](https://arxiv.org/search/cs?searchtype=author&query=Tao%2C+D)

> Knowledge distillation (KD) has been extensively employed to transfer the knowledge from a large teacher model to the smaller students, where the parameters of the teacher are fixed (or partially) during training. Recent studies show that this mode may cause difficulties in knowledge transfer due to the mismatched model capacities. To alleviate the mismatch problem, teacher-student joint training methods, e.g., online distillation, have been proposed, but it always requires expensive computational cost. In this paper, we present a parameter-efficient and student-friendly knowledge distillation method, namely PESF-KD, to achieve efficient and sufficient knowledge transfer by updating relatively few partial parameters. Technically, we first mathematically formulate the mismatch as the sharpness gap between their predictive distributions, where we show such a gap can be narrowed with the appropriate smoothness of the soft label. Then, we introduce an adapter module for the teacher and only update the adapter to obtain soft labels with appropriate smoothness. Experiments on a variety of benchmarks show that PESF-KD can significantly reduce the training cost while obtaining competitive results compared to advanced online distillation methods. Code will be released upon acceptance.

| Subjects: | **Machine Learning (cs.LG)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.15308](https://arxiv.org/abs/2205.15308) [cs.LG]** |
|           | (or **[arXiv:2205.15308v1](https://arxiv.org/abs/2205.15308v1) [cs.LG]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.15308Focus to learn more |





<h2 id="2022-06-01-2">2. ADAPT: Vision-Language Navigation with Modality-Aligned Action Prompts
</h2>

Title: [ADAPT: Vision-Language Navigation with Modality-Aligned Action Prompts](https://arxiv.org/abs/2205.15509)

Authors: [Bingqian Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+B), [Yi Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+Y), [Zicong Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Z), [Xiwen Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+X), [Jianzhuang Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+J), [Xiaodan Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+X)

> Vision-Language Navigation (VLN) is a challenging task that requires an embodied agent to perform action-level modality alignment, i.e., make instruction-asked actions sequentially in complex visual environments. Most existing VLN agents learn the instruction-path data directly and cannot sufficiently explore action-level alignment knowledge inside the multi-modal inputs. In this paper, we propose modAlity-aligneD Action PrompTs (ADAPT), which provides the VLN agent with action prompts to enable the explicit learning of action-level modality alignment to pursue successful navigation. Specifically, an action prompt is defined as a modality-aligned pair of an image sub-prompt and a text sub-prompt, where the former is a single-view observation and the latter is a phrase like ''walk past the chair''. When starting navigation, the instruction-related action prompt set is retrieved from a pre-built action prompt base and passed through a prompt encoder to obtain the prompt feature. Then the prompt feature is concatenated with the original instruction feature and fed to a multi-layer transformer for action prediction. To collect high-quality action prompts into the prompt base, we use the Contrastive Language-Image Pretraining (CLIP) model which has powerful cross-modality alignment ability. A modality alignment loss and a sequential consistency loss are further introduced to enhance the alignment of the action prompt and enforce the agent to focus on the related prompt sequentially. Experimental results on both R2R and RxR show the superiority of ADAPT over state-of-the-art methods.

| Comments: | Accepted to CVPR 2022                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2205.15509](https://arxiv.org/abs/2205.15509) [cs.CV]** |
|           | (or **[arXiv:2205.15509v1](https://arxiv.org/abs/2205.15509v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.15509Focus to learn more |





<h2 id="2022-06-01-3">3. CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers
</h2>

Title: [CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers](https://arxiv.org/abs/2205.15868)

Authors: [Wenyi Hong](https://arxiv.org/search/cs?searchtype=author&query=Hong%2C+W), [Ming Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+M), [Wendi Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+W), [Xinghan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+X), [Jie Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+J)

> Large-scale pretrained transformers have created milestones in text (GPT-3) and text-to-image (DALL-E and CogView) generation. Its application to video generation is still facing many challenges: The potential huge computation cost makes the training from scratch unaffordable; The scarcity and weak relevance of text-video datasets hinder the model understanding complex movement semantics. In this work, we present 9B-parameter transformer CogVideo, trained by inheriting a pretrained text-to-image model, CogView2. We also propose multi-frame-rate hierarchical training strategy to better align text and video clips. As (probably) the first open-source large-scale pretrained text-to-video model, CogVideo outperforms all publicly available models at a large margin in machine and human evaluations.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.15868](https://arxiv.org/abs/2205.15868) [cs.CV]** |
|           | (or **[arXiv:2205.15868v1](https://arxiv.org/abs/2205.15868v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.15868Focus to learn more |





<h2 id="2022-06-01-4">4. EMS: Efficient and Effective Massively Multilingual Sentence Representation Learning
</h2>

Title: [EMS: Efficient and Effective Massively Multilingual Sentence Representation Learning](https://arxiv.org/abs/2205.15744)

Authors: [Zhuoyuan Mao](https://arxiv.org/search/cs?searchtype=author&query=Mao%2C+Z), [Chenhui Chu](https://arxiv.org/search/cs?searchtype=author&query=Chu%2C+C), [Sadao Kurohashi](https://arxiv.org/search/cs?searchtype=author&query=Kurohashi%2C+S)

> Massively multilingual sentence representation models, e.g., LASER, SBERT-distill, and LaBSE, help significantly improve cross-lingual downstream tasks. However, multiple training procedures, the use of a large amount of data, or inefficient model architectures result in heavy computation to train a new model according to our preferred languages and domains. To resolve this issue, we introduce efficient and effective massively multilingual sentence representation learning (EMS), using cross-lingual sentence reconstruction (XTR) and sentence-level contrastive learning as training objectives. Compared with related studies, the proposed model can be efficiently trained using significantly fewer parallel sentences and GPU computation resources without depending on large-scale pre-trained models. Empirical results show that the proposed model significantly yields better or comparable results with regard to bi-text mining, zero-shot cross-lingual genre classification, and sentiment classification. Ablative analyses demonstrate the effectiveness of each component of the proposed model. We release the codes for model training and the EMS pre-trained model, which supports 62 languages ([this https URL](https://github.com/Mao-KU/EMS)).

| Comments: | This work is an extension of [arXiv:2105.13856](https://arxiv.org/abs/2105.13856). This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.15744](https://arxiv.org/abs/2205.15744) [cs.CL]** |
|           | (or **[arXiv:2205.15744v1](https://arxiv.org/abs/2205.15744v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.15744Focus to learn more |






# 2022-05-31

[Return to Index](#Index)



<h2 id="2022-05-31-1">1. VLUE: A Multi-Task Benchmark for Evaluating Vision-Language Models
</h2>

Title: [VLUE: A Multi-Task Benchmark for Evaluating Vision-Language Models](https://arxiv.org/abs/2205.15237)

Authors: [Wangchunshu Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+W), [Yan Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+Y), [Shizhe Diao](https://arxiv.org/search/cs?searchtype=author&query=Diao%2C+S), [Xinsong Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+X)

> Recent advances in vision-language pre-training (VLP) have demonstrated impressive performance in a range of vision-language (VL) tasks. However, there exist several challenges for measuring the community's progress in building general multi-modal intelligence. First, most of the downstream VL datasets are annotated using raw images that are already seen during pre-training, which may result in an overestimation of current VLP models' generalization ability. Second, recent VLP work mainly focuses on absolute performance but overlooks the efficiency-performance trade-off, which is also an important indicator for measuring progress. 
> To this end, we introduce the Vision-Language Understanding Evaluation (VLUE) benchmark, a multi-task multi-dimension benchmark for evaluating the generalization capabilities and the efficiency-performance trade-off (``Pareto SOTA'') of VLP models. We demonstrate that there is a sizable generalization gap for all VLP models when testing on out-of-distribution test sets annotated on images from a more diverse distribution that spreads across cultures. Moreover, we find that measuring the efficiency-performance trade-off of VLP models leads to complementary insights for several design choices of VLP. We release the VLUE benchmark to promote research on building vision-language models that generalize well to more diverse images and concepts unseen during pre-training, and are practical in terms of efficiency-performance trade-off.

| Comments: | ICML 2022, Benchmark website at [this https URL](https://vlue-benchmark.github.io/) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2205.15237](https://arxiv.org/abs/2205.15237) [cs.CV]** |
|           | (or **[arXiv:2205.15237v1](https://arxiv.org/abs/2205.15237v1) [cs.CV]** for this version) |



