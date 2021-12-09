# MA C.'s Daily Paper Of Interest - December, 2021

# Index


- [2021-12-9](#2021-12-9)

  - [1. Transformer-Based Approach for Joint Handwriting and Named Entity Recognition in Historical documents](#2021-12-9-1)
  - [2. MLP Architectures for Vision-and-Language Modeling: An Empirical Study](#2021-12-9-2)
  - [3. Bidimensional Leaderboards: Generate and Evaluate Language Hand in Hand](#2021-12-9-3)
  - [4. Improving language models by retrieving from trillions of tokens](#2021-12-9-4)
  
- [2021-12-8](#2021-12-8)

  - [1. CMA-CLIP: Cross-Modality Attention CLIP for Image-Text Classification](#2021-12-8-1)
  - [2. Grounded Language-Image Pre-training](#2021-12-8-2)
  - [3. Parsing with Pretrained Language Models, Multiple Datasets, and Dataset Embeddings](#2021-12-8-3)
  - [4. Natural Answer Generation: From Factoid Answer to Full-length Answer using Grammar Correction](#2021-12-8-4)

- [2021-12-7](#2021-12-7)

  - [1. Legal Document Retrieval using Document Vector Embeddings and Deep Learning](#2021-12-7-1)
  - [2. VT-CLIP: Enhancing Vision-Language Models with Visual-guided Texts](#2021-12-7-2)
  - [3. VarCLR: Variable Semantic Representation Pre-training via Contrastive Learning](#2021-12-7-3)
  - [4. Embedding Arithmetic for Text-driven Image Transformation](#2021-12-7-4)
  - [5. Text2Mesh: Text-Driven Neural Stylization for Meshes](#2021-12-7-5)
  - [6. CLASSIC: Continual and Contrastive Learning of Aspect Sentiment Classification Tasks](#2021-12-7-6)
  - [7. Towards More Robust Natural Language Understanding](#2021-12-7-7)
  - [8. Quantifying Adaptability in Pre-trained Language Models with 500 Tasks](#2021-12-7-8)

- [2021-12-6](#2021-12-6)

  - [1. Linear algebra with transformers](#2021-12-6-1)
  - [2. Multitask Finetuning for Improving Neural Machine Translation in Indian Languages](#2021-12-6-2)
  - [3. Translating Politeness Across Cultures: Case of Hindi and English](#2021-12-6-3)
  - [4. Semantic Segmentation of Legal Documents via Rhetorical Roles](#2021-12-6-4)
  - [5. A Proposal of Automatic Error Correction in Text](#2021-12-6-5)

- [2021-12-3](#2021-12-3)
  - [1. Consensus Graph Representation Learning for Better Grounded Image Captioning](#2021-12-3-1)
  - [2. A Mixture of Expert Based Deep Neural Network for Improved ASR](#2021-12-3-2)
  - [3. DenseCLIP: Extract Free Dense Labels from CLIP](#2021-12-3-3)
- [2021-12-2](#2021-12-2)
  - [1. CLIPstyler: Image Style Transfer with a Single Text Condition](#2021-12-2-1)
  - [2. Translation-equivariant Image Quantizer for Bi-directional Image-Text Generation](#2021-12-2-2)
  - [3. DPRK-BERT: The Supreme Language Model](#2021-12-2-3)
- [2021-12-1](#2021-12-1)

  - [1. Do We Still Need Automatic Speech Recognition for Spoken Language Understanding?](#2021-12-1-1)
  - [2. Improvement in Machine Translation with Generative Adversarial Networks](#2021-12-1-2)
  - [3. Pureformer: Do We Even Need Attention?](#2021-12-1-3)
- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



# 2021-12-9

[Return to Index](#Index)



<h2 id="2021-12-9-1">1. Transformer-Based Approach for Joint Handwriting and Named Entity Recognition in Historical documents
</h2>

Title: [Transformer-Based Approach for Joint Handwriting and Named Entity Recognition in Historical documents](https://arxiv.org/abs/2112.04189)

Authors: [Ahmed Cheikh Rouhoua](https://arxiv.org/search/cs?searchtype=author&query=Rouhoua%2C+A+C), [Marwa Dhiaf](https://arxiv.org/search/cs?searchtype=author&query=Dhiaf%2C+M), [Yousri Kessentini](https://arxiv.org/search/cs?searchtype=author&query=Kessentini%2C+Y), [Sinda Ben Salem](https://arxiv.org/search/cs?searchtype=author&query=Salem%2C+S+B)

> The extraction of relevant information carried out by named entities in handwriting documents is still a challenging task. Unlike traditional information extraction approaches that usually face text transcription and named entity recognition as separate subsequent tasks, we propose in this paper an end-to-end transformer-based approach to jointly perform these two tasks. The proposed approach operates at the paragraph level, which brings two main benefits. First, it allows the model to avoid unrecoverable early errors due to line segmentation. Second, it allows the model to exploit larger bi-dimensional context information to identify the semantic categories, reaching a higher final prediction accuracy. We also explore different training scenarios to show their effect on the performance and we demonstrate that a two-stage learning strategy can make the model reach a higher final prediction accuracy. As far as we know, this work presents the first approach that adopts the transformer networks for named entity recognition in handwritten documents. We achieve the new state-of-the-art performance in the ICDAR 2017 Information Extraction competition using the Esposalles database, for the complete task, even though the proposed technique does not use any dictionaries, language modeling, or post-processing.

| Subjects:          | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| ------------------ | ------------------------------------------------------------ |
| Journal reference: | Pattern Recognition Letters, 2022                            |
| Cite as:           | **[arXiv:2112.04189](https://arxiv.org/abs/2112.04189) [cs.CV]** |
|                    | (or **[arXiv:2112.04189v1](https://arxiv.org/abs/2112.04189v1) [cs.CV]** for this version) |





<h2 id="2021-12-9-2">2. MLP Architectures for Vision-and-Language Modeling: An Empirical Study
</h2>

Title: [MLP Architectures for Vision-and-Language Modeling: An Empirical Study](https://arxiv.org/abs/2112.04453)

Authors: [Yixin Nie](https://arxiv.org/search/cs?searchtype=author&query=Nie%2C+Y), [Linjie Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L), [Zhe Gan](https://arxiv.org/search/cs?searchtype=author&query=Gan%2C+Z), [Shuohang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+S), [Chenguang Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+C), [Michael Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+M), [Zicheng Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Z), [Mohit Bansal](https://arxiv.org/search/cs?searchtype=author&query=Bansal%2C+M), [Lijuan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L)

> We initiate the first empirical study on the use of MLP architectures for vision-and-language (VL) fusion. Through extensive experiments on 5 VL tasks and 5 robust VQA benchmarks, we find that: (i) Without pre-training, using MLPs for multimodal fusion has a noticeable performance gap compared to transformers; (ii) However, VL pre-training can help close the performance gap; (iii) Instead of heavy multi-head attention, adding tiny one-head attention to MLPs is sufficient to achieve comparable performance to transformers. Moreover, we also find that the performance gap between MLPs and transformers is not widened when being evaluated on the harder robust VQA benchmarks, suggesting using MLPs for VL fusion can generalize roughly to a similar degree as using transformers. These results hint that MLPs can effectively learn to align vision and text features extracted from lower-level encoders without heavy reliance on self-attention. Based on this, we ask an even bolder question: can we have an all-MLP architecture for VL modeling, where both VL fusion and the vision encoder are replaced with MLPs? Our result shows that an all-MLP VL model is sub-optimal compared to state-of-the-art full-featured VL models when both of them get pre-trained. However, pre-training an all-MLP can surprisingly achieve a better average score than full-featured transformer models without pre-training. This indicates the potential of large-scale pre-training of MLP-like architectures for VL modeling and inspires the future research direction on simplifying well-established VL modeling with less inductive design bias. Our code is publicly available at: [this https URL](https://github.com/easonnie/mlp-vil)

| Comments: | 15 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2112.04453](https://arxiv.org/abs/2112.04453) [cs.CV]** |
|           | (or **[arXiv:2112.04453v1](https://arxiv.org/abs/2112.04453v1) [cs.CV]** for this version) |





<h2 id="2021-12-9-3">3. Bidimensional Leaderboards: Generate and Evaluate Language Hand in Hand
</h2>

Title: [Bidimensional Leaderboards: Generate and Evaluate Language Hand in Hand](https://arxiv.org/abs/2112.04139)

Authors: [Jungo Kasai](https://arxiv.org/search/cs?searchtype=author&query=Kasai%2C+J), [Keisuke Sakaguchi](https://arxiv.org/search/cs?searchtype=author&query=Sakaguchi%2C+K), [Ronan Le Bras](https://arxiv.org/search/cs?searchtype=author&query=Bras%2C+R+L), [Lavinia Dunagan](https://arxiv.org/search/cs?searchtype=author&query=Dunagan%2C+L), [Jacob Morrison](https://arxiv.org/search/cs?searchtype=author&query=Morrison%2C+J), [Alexander R. Fabbri](https://arxiv.org/search/cs?searchtype=author&query=Fabbri%2C+A+R), [Yejin Choi](https://arxiv.org/search/cs?searchtype=author&query=Choi%2C+Y), [Noah A. Smith](https://arxiv.org/search/cs?searchtype=author&query=Smith%2C+N+A)

> Natural language processing researchers have identified limitations of evaluation methodology for generation tasks, with new questions raised about the validity of automatic metrics and of crowdworker judgments. Meanwhile, efforts to improve generation models tend to focus on simple n-gram overlap metrics (e.g., BLEU, ROUGE). We argue that new advances on models and metrics should each more directly benefit and inform the other. We therefore propose a generalization of leaderboards, bidimensional leaderboards (Billboards), that simultaneously tracks progress in language generation tasks and metrics for their evaluation. Unlike conventional unidimensional leaderboards that sort submitted systems by predetermined metrics, a Billboard accepts both generators and evaluation metrics as competing entries. A Billboard automatically creates an ensemble metric that selects and linearly combines a few metrics based on a global analysis across generators. Further, metrics are ranked based on their correlations with human judgments. We release four Billboards for machine translation, summarization, and image captioning. We demonstrate that a linear ensemble of a few diverse metrics sometimes substantially outperforms existing metrics in isolation. Our mixed-effects model analysis shows that most automatic metrics, especially the reference-based ones, overrate machine over human generation, demonstrating the importance of updating metrics as generation models become stronger (and perhaps more similar to humans) in the future.

| Comments: | Project website: [this https URL](https://nlp.cs.washington.edu/billboard/) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2112.04139](https://arxiv.org/abs/2112.04139) [cs.CL]** |
|           | (or **[arXiv:2112.04139v1](https://arxiv.org/abs/2112.04139v1) [cs.CL]** for this version) |





<h2 id="2021-12-9-4">4. Improving language models by retrieving from trillions of tokens
</h2>

Title: [Improving language models by retrieving from trillions of tokens](https://arxiv.org/abs/2112.04426)

Authors: [Sebastian Borgeaud](https://arxiv.org/search/cs?searchtype=author&query=Borgeaud%2C+S), [Arthur Mensch](https://arxiv.org/search/cs?searchtype=author&query=Mensch%2C+A), [Jordan Hoffmann](https://arxiv.org/search/cs?searchtype=author&query=Hoffmann%2C+J), [Trevor Cai](https://arxiv.org/search/cs?searchtype=author&query=Cai%2C+T), [Eliza Rutherford](https://arxiv.org/search/cs?searchtype=author&query=Rutherford%2C+E), [Katie Millican](https://arxiv.org/search/cs?searchtype=author&query=Millican%2C+K), [George van den Driessche](https://arxiv.org/search/cs?searchtype=author&query=van+den+Driessche%2C+G), [Jean-Baptiste Lespiau](https://arxiv.org/search/cs?searchtype=author&query=Lespiau%2C+J), [Bogdan Damoc](https://arxiv.org/search/cs?searchtype=author&query=Damoc%2C+B), [Aidan Clark](https://arxiv.org/search/cs?searchtype=author&query=Clark%2C+A), [Diego de Las Casas](https://arxiv.org/search/cs?searchtype=author&query=de+Las+Casas%2C+D), [Aurelia Guy](https://arxiv.org/search/cs?searchtype=author&query=Guy%2C+A), [Jacob Menick](https://arxiv.org/search/cs?searchtype=author&query=Menick%2C+J), [Roman Ring](https://arxiv.org/search/cs?searchtype=author&query=Ring%2C+R), [Tom Hennigan](https://arxiv.org/search/cs?searchtype=author&query=Hennigan%2C+T), [Saffron Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+S), [Loren Maggiore](https://arxiv.org/search/cs?searchtype=author&query=Maggiore%2C+L), [Chris Jones](https://arxiv.org/search/cs?searchtype=author&query=Jones%2C+C), [Albin Cassirer](https://arxiv.org/search/cs?searchtype=author&query=Cassirer%2C+A), [Andy Brock](https://arxiv.org/search/cs?searchtype=author&query=Brock%2C+A), [Michela Paganini](https://arxiv.org/search/cs?searchtype=author&query=Paganini%2C+M), [Geoffrey Irving](https://arxiv.org/search/cs?searchtype=author&query=Irving%2C+G), [Oriol Vinyals](https://arxiv.org/search/cs?searchtype=author&query=Vinyals%2C+O), [Simon Osindero](https://arxiv.org/search/cs?searchtype=author&query=Osindero%2C+S), [Karen Simonyan](https://arxiv.org/search/cs?searchtype=author&query=Simonyan%2C+K), [Jack W. Rae](https://arxiv.org/search/cs?searchtype=author&query=Rae%2C+J+W), [Erich Elsen](https://arxiv.org/search/cs?searchtype=author&query=Elsen%2C+E), [Laurent Sifre](https://arxiv.org/search/cs?searchtype=author&query=Sifre%2C+L)

> We enhance auto-regressive language models by conditioning on document chunks retrieved from a large corpus, based on local similarity with preceding tokens. With a 2 trillion token database, our Retrieval-Enhanced Transformer (RETRO) obtains comparable performance to GPT-3 and Jurassic-1 on the Pile, despite using 25× fewer parameters. After fine-tuning, RETRO performance translates to downstream knowledge-intensive tasks such as question answering. RETRO combines a frozen Bert retriever, a differentiable encoder and a chunked cross-attention mechanism to predict tokens based on an order of magnitude more data than what is typically consumed during training. We typically train RETRO from scratch, yet can also rapidly RETROfit pre-trained transformers with retrieval and still achieve good performance. Our work opens up new avenues for improving language models through explicit memory at unprecedented scale.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2112.04426](https://arxiv.org/abs/2112.04426) [cs.CL]** |
|           | (or **[arXiv:2112.04426v1](https://arxiv.org/abs/2112.04426v1) [cs.CL]** for this version) |





# 2021-12-8

[Return to Index](#Index)



<h2 id="2021-12-8-1">1. CMA-CLIP: Cross-Modality Attention CLIP for Image-Text Classification
</h2>

Title: [CMA-CLIP: Cross-Modality Attention CLIP for Image-Text Classification](https://arxiv.org/abs/2112.03562)

Authors: [Huidong Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+H) (1), [Shaoyuan Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+S) (2), [Jinmiao Fu](https://arxiv.org/search/cs?searchtype=author&query=Fu%2C+J) (2), [Yang Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y) (2), [Ning Xie](https://arxiv.org/search/cs?searchtype=author&query=Xie%2C+N) (2), [Chien-chih Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+C) (2), [Bryan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+B) (2), [Yi Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+Y) (2) ((1) Stony Brook University, (2) Amazon Inc.)

> Modern Web systems such as social media and e-commerce contain rich contents expressed in images and text. Leveraging information from multi-modalities can improve the performance of machine learning tasks such as classification and recommendation. In this paper, we propose the Cross-Modality Attention Contrastive Language-Image Pre-training (CMA-CLIP), a new framework which unifies two types of cross-modality attentions, sequence-wise attention and modality-wise attention, to effectively fuse information from image and text pairs. The sequence-wise attention enables the framework to capture the fine-grained relationship between image patches and text tokens, while the modality-wise attention weighs each modality by its relevance to the downstream tasks. In addition, by adding task specific modality-wise attentions and multilayer perceptrons, our proposed framework is capable of performing multi-task classification with multi-modalities. 
> We conduct experiments on a Major Retail Website Product Attribute (MRWPA) dataset and two public datasets, Food101 and Fashion-Gen. The results show that CMA-CLIP outperforms the pre-trained and fine-tuned CLIP by an average of 11.9% in recall at the same level of precision on the MRWPA dataset for multi-task classification. It also surpasses the state-of-the-art method on Fashion-Gen Dataset by 5.5% in accuracy and achieves competitive performance on Food101 Dataset. Through detailed ablation studies, we further demonstrate the effectiveness of both cross-modality attention modules and our method's robustness against noise in image and text inputs, which is a common challenge in practice.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2112.03562](https://arxiv.org/abs/2112.03562) [cs.CV]** |
|           | (or **[arXiv:2112.03562v1](https://arxiv.org/abs/2112.03562v1) [cs.CV]** for this version) |





<h2 id="2021-12-8-2">2. Grounded Language-Image Pre-training
</h2>

Title: [Grounded Language-Image Pre-training](https://arxiv.org/abs/2112.03857)

Authors: [Liunian Harold Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L+H), [Pengchuan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+P), [Haotian Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+H), [Jianwei Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+J), [Chunyuan Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+C), [Yiwu Zhong](https://arxiv.org/search/cs?searchtype=author&query=Zhong%2C+Y), [Lijuan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L), [Lu Yuan](https://arxiv.org/search/cs?searchtype=author&query=Yuan%2C+L), [Lei Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+L), [Jenq-Neng Hwang](https://arxiv.org/search/cs?searchtype=author&query=Hwang%2C+J), [Kai-Wei Chang](https://arxiv.org/search/cs?searchtype=author&query=Chang%2C+K), [Jianfeng Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+J)

> This paper presents a grounded language-image pre-training (GLIP) model for learning object-level, language-aware, and semantic-rich visual representations. GLIP unifies object detection and phrase grounding for pre-training. The unification brings two benefits: 1) it allows GLIP to learn from both detection and grounding data to improve both tasks and bootstrap a good grounding model; 2) GLIP can leverage massive image-text pairs by generating grounding boxes in a self-training fashion, making the learned representation semantic-rich. In our experiments, we pre-train GLIP on 27M grounding data, including 3M human-annotated and 24M web-crawled image-text pairs. The learned representations demonstrate strong zero-shot and few-shot transferability to various object-level recognition tasks. 1) When directly evaluated on COCO and LVIS (without seeing any images in COCO during pre-training), GLIP achieves 49.8 AP and 26.9 AP, respectively, surpassing many supervised baselines. 2) After fine-tuned on COCO, GLIP achieves 60.8 AP on val and 61.5 AP on test-dev, surpassing prior SoTA. 3) When transferred to 13 downstream object detection tasks, a 1-shot GLIP rivals with a fully-supervised Dynamic Head. Code will be released at [this https URL](https://github.com/microsoft/GLIP).

| Comments: | Code will be released at [this https URL](https://github.com/microsoft/GLIP) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG); Multimedia (cs.MM) |
| Cite as:  | **[arXiv:2112.03857](https://arxiv.org/abs/2112.03857) [cs.CV]** |
|           | (or **[arXiv:2112.03857v1](https://arxiv.org/abs/2112.03857v1) [cs.CV]** for this version) |





<h2 id="2021-12-8-3">3. Parsing with Pretrained Language Models, Multiple Datasets, and Dataset Embeddings
</h2>

Title: [Parsing with Pretrained Language Models, Multiple Datasets, and Dataset Embeddings](https://arxiv.org/abs/2112.03625)

Authors: [Rob van der Goot](https://arxiv.org/search/cs?searchtype=author&query=van+der+Goot%2C+R), [Miryam de Lhoneux](https://arxiv.org/search/cs?searchtype=author&query=de+Lhoneux%2C+M)

> With an increase of dataset availability, the potential for learning from a variety of data sources has increased. One particular method to improve learning from multiple data sources is to embed the data source during training. This allows the model to learn generalizable features as well as distinguishing features between datasets. However, these dataset embeddings have mostly been used before contextualized transformer-based embeddings were introduced in the field of Natural Language Processing. In this work, we compare two methods to embed datasets in a transformer-based multilingual dependency parser, and perform an extensive evaluation. We show that: 1) embedding the dataset is still beneficial with these models 2) performance increases are highest when embedding the dataset at the encoder level 3) unsurprisingly, we confirm that performance increases are highest for small datasets and datasets with a low baseline score. 4) we show that training on the combination of all datasets performs similarly to designing smaller clusters based on language-relatedness.

| Comments: | Accepted to TLT at SyntaxFest 2021                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2112.03625](https://arxiv.org/abs/2112.03625) [cs.CL]** |
|           | (or **[arXiv:2112.03625v1](https://arxiv.org/abs/2112.03625v1) [cs.CL]** for this version) |





<h2 id="2021-12-8-4">4. Natural Answer Generation: From Factoid Answer to Full-length Answer using Grammar Correction
</h2>

Title: [Natural Answer Generation: From Factoid Answer to Full-length Answer using Grammar Correction](https://arxiv.org/abs/2112.03849)

Authors: [Manas Jain](https://arxiv.org/search/cs?searchtype=author&query=Jain%2C+M), [Sriparna Saha](https://arxiv.org/search/cs?searchtype=author&query=Saha%2C+S), [Pushpak Bhattacharyya](https://arxiv.org/search/cs?searchtype=author&query=Bhattacharyya%2C+P), [Gladvin Chinnadurai](https://arxiv.org/search/cs?searchtype=author&query=Chinnadurai%2C+G), [Manish Kumar Vatsa](https://arxiv.org/search/cs?searchtype=author&query=Vatsa%2C+M+K)

> Question Answering systems these days typically use template-based language generation. Though adequate for a domain-specific task, these systems are too restrictive and predefined for domain-independent systems. This paper proposes a system that outputs a full-length answer given a question and the extracted factoid answer (short spans such as named entities) as the input. Our system uses constituency and dependency parse trees of questions. A transformer-based Grammar Error Correction model GECToR (2020), is used as a post-processing step for better fluency. We compare our system with (i) Modified Pointer Generator (SOTA) and (ii) Fine-tuned DialoGPT for factoid questions. We also test our approach on existential (yes-no) questions with better results. Our model generates accurate and fluent answers than the state-of-the-art (SOTA) approaches. The evaluation is done on NewsQA and SqUAD datasets with an increment of 0.4 and 0.9 percentage points in ROUGE-1 score respectively. Also the inference time is reduced by 85\% as compared to the SOTA. The improved datasets used for our evaluation will be released as part of the research contribution.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2112.03849](https://arxiv.org/abs/2112.03849) [cs.CL]** |
|           | (or **[arXiv:2112.03849v1](https://arxiv.org/abs/2112.03849v1) [cs.CL]** for this version) |





# 2021-12-7

[Return to Index](#Index)



<h2 id="2021-12-7-1">1. Legal Document Retrieval using Document Vector Embeddings and Deep Learning
</h2>

Title: [Legal Document Retrieval using Document Vector Embeddings and Deep Learning](https://arxiv.org/abs/1805.10685)

Authors: [Keet Sugathadasa](https://arxiv.org/search/cs?searchtype=author&query=Sugathadasa%2C+K), [Buddhi Ayesha](https://arxiv.org/search/cs?searchtype=author&query=Ayesha%2C+B), [Nisansa de Silva](https://arxiv.org/search/cs?searchtype=author&query=de+Silva%2C+N), [Amal Shehan Perera](https://arxiv.org/search/cs?searchtype=author&query=Perera%2C+A+S), [Vindula Jayawardana](https://arxiv.org/search/cs?searchtype=author&query=Jayawardana%2C+V), [Dimuthu Lakmal](https://arxiv.org/search/cs?searchtype=author&query=Lakmal%2C+D), [Madhavi Perera](https://arxiv.org/search/cs?searchtype=author&query=Perera%2C+M)

> Domain specific information retrieval process has been a prominent and ongoing research in the field of natural language processing. Many researchers have incorporated different techniques to overcome the technical and domain specificity and provide a mature model for various domains of interest. The main bottleneck in these studies is the heavy coupling of domain experts, that makes the entire process to be time consuming and cumbersome. In this study, we have developed three novel models which are compared against a golden standard generated via the on line repositories provided, specifically for the legal domain. The three different models incorporated vector space representations of the legal domain, where document vector generation was done in two different mechanisms and as an ensemble of the above two. This study contains the research being carried out in the process of representing legal case documents into different vector spaces, whilst incorporating semantic word measures and natural language processing techniques. The ensemble model built in this study, shows a significantly higher accuracy level, which indeed proves the need for incorporation of domain specific semantic similarity measures into the information retrieval process. This study also shows, the impact of varying distribution of the word similarity measures, against varying document vector dimensions, which can lead to improvements in the process of legal information retrieval.

| Subjects: | **Information Retrieval (cs.IR)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:1805.10685](https://arxiv.org/abs/1805.10685) [cs.IR]** |
|           | (or **[arXiv:1805.10685v1](https://arxiv.org/abs/1805.10685v1) [cs.IR]** for this version) |





<h2 id="2021-12-7-2">2. VT-CLIP: Enhancing Vision-Language Models with Visual-guided Texts
</h2>

Title: [VT-CLIP: Enhancing Vision-Language Models with Visual-guided Texts](https://arxiv.org/abs/2112.02399)

Authors: [Renrui Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+R), [Longtian Qiu](https://arxiv.org/search/cs?searchtype=author&query=Qiu%2C+L), [Wei Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+W), [Ziyao Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+Z)

> Contrastive Vision-Language Pre-training (CLIP) has drown increasing attention recently for its transferable visual representation learning. Supervised by large-scale image-text pairs, CLIP is able to align paired images and texts and thus conduct zero-shot recognition in open-vocabulary scenarios. However, there exists semantic gap between the specific application and generally pre-trained knowledge, which makes the matching sub-optimal on downstream tasks. In this paper, we propose VT-CLIP to enhance vision-language modeling via visual-guided texts. Specifically, we guide the text feature to adaptively explore informative regions on the image and aggregate the visual feature by cross-attention machanism. In this way, the visual-guided text become more semantically correlated with the image, which greatly benefits the matching process. In few-shot settings, we evaluate our VT-CLIP on 11 well-known classification datasets and experiment extensive ablation studies to demonstrate the effectiveness of VT-CLIP. The code will be released soon.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2112.02399](https://arxiv.org/abs/2112.02399) [cs.CV]** |
|           | (or **[arXiv:2112.02399v1](https://arxiv.org/abs/2112.02399v1) [cs.CV]** for this version) |





<h2 id="2021-12-7-3">3. VarCLR: Variable Semantic Representation Pre-training via Contrastive Learning
</h2>

Title: [VarCLR: Variable Semantic Representation Pre-training via Contrastive Learning](https://arxiv.org/abs/2112.02650)

Authors: [Qibin Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Q), [Jeremy Lacomis](https://arxiv.org/search/cs?searchtype=author&query=Lacomis%2C+J), [Edward J. Schwartz](https://arxiv.org/search/cs?searchtype=author&query=Schwartz%2C+E+J), [Graham Neubig](https://arxiv.org/search/cs?searchtype=author&query=Neubig%2C+G), [Bogdan Vasilescu](https://arxiv.org/search/cs?searchtype=author&query=Vasilescu%2C+B), [Claire Le Goues](https://arxiv.org/search/cs?searchtype=author&query=Goues%2C+C+L)

> Variable names are critical for conveying intended program behavior. Machine learning-based program analysis methods use variable name representations for a wide range of tasks, such as suggesting new variable names and bug detection. Ideally, such methods could capture semantic relationships between names beyond syntactic similarity, e.g., the fact that the names average and mean are similar. Unfortunately, previous work has found that even the best of previous representation approaches primarily capture relatedness (whether two variables are linked at all), rather than similarity (whether they actually have the same meaning). 
> We propose VarCLR, a new approach for learning semantic representations of variable names that effectively captures variable similarity in this stricter sense. We observe that this problem is an excellent fit for contrastive learning, which aims to minimize the distance between explicitly similar inputs, while maximizing the distance between dissimilar inputs. This requires labeled training data, and thus we construct a novel, weakly-supervised variable renaming dataset mined from GitHub edits. We show that VarCLR enables the effective application of sophisticated, general-purpose language models like BERT, to variable name representation and thus also to related downstream tasks like variable name similarity search or spelling correction. VarCLR produces models that significantly outperform the state-of-the-art on IdBench, an existing benchmark that explicitly captures variable similarity (as distinct from relatedness). Finally, we contribute a release of all data, code, and pre-trained models, aiming to provide a drop-in replacement for variable representations used in either existing or future program analyses that rely on variable names.

| Comments: | Accepted by ICSE 2022                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Software Engineering (cs.SE)**; Computation and Language (cs.CL); Machine Learning (cs.LG); Programming Languages (cs.PL) |
| Cite as:  | **[arXiv:2112.02650](https://arxiv.org/abs/2112.02650) [cs.SE]** |
|           | (or **[arXiv:2112.02650v1](https://arxiv.org/abs/2112.02650v1) [cs.SE]** for this version) |





<h2 id="2021-12-7-4">4. Embedding Arithmetic for Text-driven Image Transformation
</h2>

Title: [Embedding Arithmetic for Text-driven Image Transformation](https://arxiv.org/abs/2112.03162)

Authors: [Guillaume Couairon](https://arxiv.org/search/cs?searchtype=author&query=Couairon%2C+G), [Matthieu Cord](https://arxiv.org/search/cs?searchtype=author&query=Cord%2C+M), [Matthijs Douze](https://arxiv.org/search/cs?searchtype=author&query=Douze%2C+M), [Holger Schwenk](https://arxiv.org/search/cs?searchtype=author&query=Schwenk%2C+H)

> Latent text representations exhibit geometric regularities, such as the famous analogy: queen is to king what woman is to man. Such structured semantic relations were not demonstrated on image representations. Recent works aiming at bridging this semantic gap embed images and text into a multimodal space, enabling the transfer of text-defined transformations to the image modality. 
> We introduce the SIMAT dataset to evaluate the task of text-driven image transformation. SIMAT contains 6k images and 18k "transformation queries" that aim at either replacing scene elements or changing their pairwise relationships. The goal is to retrieve an image consistent with the (source image, transformation) query. We use an image/text matching oracle (OSCAR) to assess whether the image transformation is successful. The SIMAT dataset will be publicly available. 
> We use SIMAT to show that vanilla CLIP multimodal embeddings are not very well suited for text-driven image transformation, but that a simple finetuning on the COCO dataset can bring dramatic improvements. We also study whether it is beneficial to leverage the geometric properties of pretrained universal sentence encoders (FastText, LASER and LaBSE).

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2112.03162](https://arxiv.org/abs/2112.03162) [cs.CV]** |
|           | (or **[arXiv:2112.03162v1](https://arxiv.org/abs/2112.03162v1) [cs.CV]** for this version) |





<h2 id="2021-12-7-5">5. Text2Mesh: Text-Driven Neural Stylization for Meshes
</h2>

Title: [Text2Mesh: Text-Driven Neural Stylization for Meshes](https://arxiv.org/abs/2112.03221)

Authors: [Oscar Michel](https://arxiv.org/search/cs?searchtype=author&query=Michel%2C+O), [Roi Bar-On](https://arxiv.org/search/cs?searchtype=author&query=Bar-On%2C+R), [Richard Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+R), [Sagie Benaim](https://arxiv.org/search/cs?searchtype=author&query=Benaim%2C+S), [Rana Hanocka](https://arxiv.org/search/cs?searchtype=author&query=Hanocka%2C+R)

> In this work, we develop intuitive controls for editing the style of 3D objects. Our framework, Text2Mesh, stylizes a 3D mesh by predicting color and local geometric details which conform to a target text prompt. We consider a disentangled representation of a 3D object using a fixed mesh input (content) coupled with a learned neural network, which we term neural style field network. In order to modify style, we obtain a similarity score between a text prompt (describing style) and a stylized mesh by harnessing the representational power of CLIP. Text2Mesh requires neither a pre-trained generative model nor a specialized 3D mesh dataset. It can handle low-quality meshes (non-manifold, boundaries, etc.) with arbitrary genus, and does not require UV parameterization. We demonstrate the ability of our technique to synthesize a myriad of styles over a wide variety of 3D meshes.

| Comments: | project page: [this https URL](https://threedle.github.io/text2mesh/) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Graphics (cs.GR) |
| Cite as:  | **[arXiv:2112.03221](https://arxiv.org/abs/2112.03221) [cs.CV]** |
|           | (or **[arXiv:2112.03221v1](https://arxiv.org/abs/2112.03221v1) [cs.CV]** for this version) |





<h2 id="2021-12-7-6">6. CLASSIC: Continual and Contrastive Learning of Aspect Sentiment Classification Tasks
</h2>

Title: [CLASSIC: Continual and Contrastive Learning of Aspect Sentiment Classification Tasks](https://arxiv.org/abs/2112.02714)

Authors: [Zixuan Ke](https://arxiv.org/search/cs?searchtype=author&query=Ke%2C+Z), [Bing Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+B), [Hu Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+H), [Lei Shu](https://arxiv.org/search/cs?searchtype=author&query=Shu%2C+L)

> This paper studies continual learning (CL) of a sequence of aspect sentiment classification(ASC) tasks in a particular CL setting called domain incremental learning (DIL). Each task is from a different domain or product. The DIL setting is particularly suited to ASC because in testing the system needs not know the task/domain to which the test data belongs. To our knowledge, this setting has not been studied before for ASC. This paper proposes a novel model called CLASSIC. The key novelty is a contrastive continual learning method that enables both knowledge transfer across tasks and knowledge distillation from old tasks to the new task, which eliminates the need for task ids in testing. Experimental results show the high effectiveness of CLASSIC.

| Subjects:          | **Computation and Language (cs.CL)**                         |
| ------------------ | ------------------------------------------------------------ |
| Journal reference: | EMNLP 2021                                                   |
| Cite as:           | **[arXiv:2112.02714](https://arxiv.org/abs/2112.02714) [cs.CL]** |
|                    | (or **[arXiv:2112.02714v1](https://arxiv.org/abs/2112.02714v1) [cs.CL]** for this version) |





<h2 id="2021-12-7-7">7. Towards More Robust Natural Language Understanding
</h2>

Title: [Towards More Robust Natural Language Understanding](https://arxiv.org/abs/2112.02992)

Authors: [Xinliang Frederick Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+X+F)

> Natural Language Understanding (NLU) is a branch of Natural Language Processing (NLP) that uses intelligent computer software to understand texts that encode human knowledge. Recent years have witnessed notable progress across various NLU tasks with deep learning techniques, especially with pretrained language models. Besides proposing more advanced model architectures, constructing more reliable and trustworthy datasets also plays a huge role in improving NLU systems, without which it would be impossible to train a decent NLU model. It's worth noting that the human ability of understanding natural language is flexible and robust. On the contrary, most of existing NLU systems fail to achieve desirable performance on out-of-domain data or struggle on handling challenging items (e.g., inherently ambiguous items, adversarial items) in the real world. Therefore, in order to have NLU models understand human language more effectively, it is expected to prioritize the study on robust natural language understanding. In this thesis, we deem that NLU systems are consisting of two components: NLU models and NLU datasets. As such, we argue that, to achieve robust NLU, the model architecture/training and the dataset are equally important. Specifically, we will focus on three NLU tasks to illustrate the robustness problem in different NLU tasks and our contributions (i.e., novel models and new datasets) to help achieve more robust natural language understanding. Moving forward, the ultimate goal for robust natural language understanding is to build NLU models which can behave humanly. That is, it's expected that robust NLU systems are capable to transfer the knowledge from training corpus to unseen documents more reliably and survive when encountering challenging items even if the system doesn't know a priori of users' inputs.

| Comments: | Undergraduate Research Thesis, The Ohio State University     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2112.02992](https://arxiv.org/abs/2112.02992) [cs.CL]** |
|           | (or **[arXiv:2112.02992v1](https://arxiv.org/abs/2112.02992v1) [cs.CL]** for this version) |





<h2 id="2021-12-7-8">8. Quantifying Adaptability in Pre-trained Language Models with 500 Tasks
</h2>

Title: [Quantifying Adaptability in Pre-trained Language Models with 500 Tasks](https://arxiv.org/abs/2112.03204)

Authors: [Belinda Z. Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+B+Z), [Jane Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+J), [Madian Khabsa](https://arxiv.org/search/cs?searchtype=author&query=Khabsa%2C+M), [Luke Zettlemoyer](https://arxiv.org/search/cs?searchtype=author&query=Zettlemoyer%2C+L), [Alon Halevy](https://arxiv.org/search/cs?searchtype=author&query=Halevy%2C+A), [Jacob Andreas](https://arxiv.org/search/cs?searchtype=author&query=Andreas%2C+J)

> When a neural language model (LM) is adapted to perform a new task, what aspects of the task predict the eventual performance of the model? In NLP, systematic features of LM generalization to individual examples are well characterized, but systematic aspects of LM adaptability to new tasks are not nearly as well understood. We present a large-scale empirical study of the features and limits of LM adaptability using a new benchmark, TaskBench500, built from 500 procedurally generated sequence modeling tasks. These tasks combine core aspects of language processing, including lexical semantics, sequence processing, memorization, logical reasoning, and world knowledge. Using TaskBench500, we evaluate three facets of adaptability, finding that: (1) adaptation procedures differ dramatically in their ability to memorize small datasets; (2) within a subset of task types, adaptation procedures exhibit compositional adaptability to complex tasks; and (3) failure to match training label distributions is explained by mismatches in the intrinsic difficulty of predicting individual labels. Our experiments show that adaptability to new tasks, like generalization to new examples, can be systematically described and understood, and we conclude with a discussion of additional aspects of adaptability that could be studied using the new benchmark.

| Comments: | 18 pages, 5 figures, 8 tables                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2112.03204](https://arxiv.org/abs/2112.03204) [cs.CL]** |
|           | (or **[arXiv:2112.03204v1](https://arxiv.org/abs/2112.03204v1) [cs.CL]** for this version) |







# 2021-12-6

[Return to Index](#Index)



<h2 id="2021-12-6-1">1. Linear algebra with transformers
</h2>

Title: [Linear algebra with transformers](https://arxiv.org/abs/2112.01898)

Authors: [François Charton](https://arxiv.org/search/cs?searchtype=author&query=Charton%2C+F)

> Most applications of transformers to mathematics, from integration to theorem proving, focus on symbolic computation. In this paper, we show that transformers can be trained to perform numerical calculations with high accuracy. We consider problems of linear algebra: matrix transposition, addition, multiplication, eigenvalues and vectors, singular value decomposition, and inversion. Training small transformers (up to six layers) over datasets of random matrices, we achieve high accuracies (over 90%) on all problems. We also show that trained models can generalize out of their training distribution, and that out-of-domain accuracy can be greatly improved by working from more diverse datasets (in particular, by training from matrices with non-independent and identically distributed coefficients). Finally, we show that few-shot learning can be leveraged to re-train models to solve larger problems.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2112.01898](https://arxiv.org/abs/2112.01898) [cs.LG]** |
|           | (or **[arXiv:2112.01898v1](https://arxiv.org/abs/2112.01898v1) [cs.LG]** for this version) |





<h2 id="2021-12-6-2">2. Multitask Finetuning for Improving Neural Machine Translation in Indian Languages
</h2>

Title: [Multitask Finetuning for Improving Neural Machine Translation in Indian Languages](https://arxiv.org/abs/2112.01742)

Authors: [Shaily Desai](https://arxiv.org/search/cs?searchtype=author&query=Desai%2C+S), [Atharva Kshirsagar](https://arxiv.org/search/cs?searchtype=author&query=Kshirsagar%2C+A), [Manisha Marathe](https://arxiv.org/search/cs?searchtype=author&query=Marathe%2C+M)

> Transformer based language models have led to impressive results across all domains in Natural Language Processing. Pretraining these models on language modeling tasks and finetuning them on downstream tasks such as Text Classification, Question Answering and Neural Machine Translation has consistently shown exemplary results. In this work, we propose a Multitask Finetuning methodology which combines the Bilingual Machine Translation task with an auxiliary Causal Language Modeling task to improve performance on the former task on Indian Languages. We conduct an empirical study on three language pairs, Marathi-Hindi, Marathi-English and Hindi-English, where we compare the multitask finetuning approach to the standard finetuning approach, for which we use the mBART50 model. Our study indicates that the multitask finetuning method could be a better technique than standard finetuning, and could improve Bilingual Machine Translation across language pairs.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2112.01742](https://arxiv.org/abs/2112.01742) [cs.CL]** |
|           | (or **[arXiv:2112.01742v1](https://arxiv.org/abs/2112.01742v1) [cs.CL]** for this version) |





<h2 id="2021-12-6-3">3. Translating Politeness Across Cultures: Case of Hindi and English
</h2>

Title: [Translating Politeness Across Cultures: Case of Hindi and English](https://arxiv.org/abs/2112.01822)

Authors: [Ritesh Kumar](https://arxiv.org/search/cs?searchtype=author&query=Kumar%2C+R), [Girish Nath Jha](https://arxiv.org/search/cs?searchtype=author&query=Jha%2C+G+N)

> In this paper, we present a corpus based study of politeness across two languages-English and Hindi. It studies the politeness in a translated parallel corpus of Hindi and English and sees how politeness in a Hindi text is translated into English. We provide a detailed theoretical background in which the comparison is carried out, followed by a brief description of the translated data within this theoretical model. Since politeness may become one of the major reasons of conflict and misunderstanding, it is a very important phenomenon to be studied and understood cross-culturally, particularly for such purposes as machine translation.

| Subjects:          | **Computation and Language (cs.CL)**                         |
| ------------------ | ------------------------------------------------------------ |
| Journal reference: | Proceedings of the 3rd ACM International Conference on Inter-Cultural Collaboration (ICIC-2010), Copenhagen Business School, Denmark, pp. 175-178, 2010 |
| Cite as:           | **[arXiv:2112.01822](https://arxiv.org/abs/2112.01822) [cs.CL]** |
|                    | (or **[arXiv:2112.01822v1](https://arxiv.org/abs/2112.01822v1) [cs.CL]** for this version) |





<h2 id="2021-12-6-4">4. Semantic Segmentation of Legal Documents via Rhetorical Roles
</h2>

Title: [Semantic Segmentation of Legal Documents via Rhetorical Roles](https://arxiv.org/abs/2112.01836)

Authors: [Vijit Malik](https://arxiv.org/search/cs?searchtype=author&query=Malik%2C+V), [Rishabh Sanjay](https://arxiv.org/search/cs?searchtype=author&query=Sanjay%2C+R), [Shouvik Kumar Guha](https://arxiv.org/search/cs?searchtype=author&query=Guha%2C+S+K), [Shubham Kumar Nigam](https://arxiv.org/search/cs?searchtype=author&query=Nigam%2C+S+K), [Angshuman Hazarika](https://arxiv.org/search/cs?searchtype=author&query=Hazarika%2C+A), [Arnab Bhattacharya](https://arxiv.org/search/cs?searchtype=author&query=Bhattacharya%2C+A), [Ashutosh Modi](https://arxiv.org/search/cs?searchtype=author&query=Modi%2C+A)

> Legal documents are unstructured, use legal jargon, and have considerable length, making it difficult to process automatically via conventional text processing techniques. A legal document processing system would benefit substantially if the documents could be semantically segmented into coherent units of information. This paper proposes a Rhetorical Roles (RR) system for segmenting a legal document into semantically coherent units: facts, arguments, statute, issue, precedent, ruling, and ratio. With the help of legal experts, we propose a set of 13 fine-grained rhetorical role labels and create a new corpus of legal documents annotated with the proposed RR. We develop a system for segmenting a document into rhetorical role units. In particular, we develop a multitask learning-based deep learning model with document rhetorical role label shift as an auxiliary task for segmenting a legal document. We experiment extensively with various deep learning models for predicting rhetorical roles in a document, and the proposed model shows superior performance over the existing models. Further, we apply RR for predicting the judgment of legal cases and show that the use of RR enhances the prediction compared to the transformer-based models.

| Comments: | 16 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2112.01836](https://arxiv.org/abs/2112.01836) [cs.CL]** |
|           | (or **[arXiv:2112.01836v1](https://arxiv.org/abs/2112.01836v1) [cs.CL]** for this version) |





<h2 id="2021-12-6-5">5. A Proposal of Automatic Error Correction in Text
</h2>

Title: [A Proposal of Automatic Error Correction in Text](https://arxiv.org/abs/2112.01846)

Authors: [Wulfrano A. Luna-Ramírez](https://arxiv.org/search/cs?searchtype=author&query=Luna-Ramírez%2C+W+A), [Carlos R. Jaimez-González](https://arxiv.org/search/cs?searchtype=author&query=Jaimez-González%2C+C+R)

> The great amount of information that can be stored in electronic media is growing up daily. Many of them is got mainly by typing, such as the huge of information obtained from web 2.0 sites; or scaned and processing by an Optical Character Recognition software, like the texts of libraries and goverment offices. Both processes introduce error in texts, so it is difficult to use the data for other purposes than just to read it, i.e. the processing of those texts by other applications like e-learning, learning of languages, electronic tutorials, data minning, information retrieval and even more specialized systems such as tiflologic software, specifically blinded people-oriented applications like automatic reading, where the text would be error free as possible in order to make easier the text to speech task, and so on. In this paper it is showed an application of automatic recognition and correction of ortographic errors in electronic texts. This task is composed of three stages: a) error detection; b) candidate corrections generation; and c) correction -selection of the best candidate. The proposal is based in part of speech text categorization, word similarity, word diccionaries, statistical measures, morphologic analisys and n-grams based language model of Spanish.

| Comments:          | 15 pages, 3 figures, 11 tables, 1 algorithm. Formerly published on Journal of Research in Computer Science - Intl Conference on Computer CORE2012 |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Journal reference: | Luna-Ramírez, A., Jaimez-González, C. R. (2012). A Proposal of Automatic Error Correction in Text. Revista Research in Computing Science: Advances in Computing Science, Vol. 58, pp. 323-337. ISSN: 1870-4069 |
| Cite as:           | **[arXiv:2112.01846](https://arxiv.org/abs/2112.01846) [cs.CL]** |
|                    | (or **[arXiv:2112.01846v1](https://arxiv.org/abs/2112.01846v1) [cs.CL]** for this version) |






# 2021-12-3

[Return to Index](#Index)



<h2 id="2021-12-3-1">1. Consensus Graph Representation Learning for Better Grounded Image Captioning
</h2>

Title: [Consensus Graph Representation Learning for Better Grounded Image Captioning](https://arxiv.org/abs/2112.00974)

Authors: [Wenqiao Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+W), [Haochen Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+H), [Siliang Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+S), [Jun Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+J), [Qiang Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+Q), [Yueting Zhuang](https://arxiv.org/search/cs?searchtype=author&query=Zhuang%2C+Y)

> The contemporary visual captioning models frequently hallucinate objects that are not actually in a scene, due to the visual misclassification or over-reliance on priors that resulting in the semantic inconsistency between the visual information and the target lexical words. The most common way is to encourage the captioning model to dynamically link generated object words or phrases to appropriate regions of the image, i.e., the grounded image captioning (GIC). However, GIC utilizes an auxiliary task (grounding objects) that has not solved the key issue of object hallucination, i.e., the semantic inconsistency. In this paper, we take a novel perspective on the issue above - exploiting the semantic coherency between the visual and language modalities. Specifically, we propose the Consensus Rraph Representation Learning framework (CGRL) for GIC that incorporates a consensus representation into the grounded captioning pipeline. The consensus is learned by aligning the visual graph (e.g., scene graph) to the language graph that consider both the nodes and edges in a graph. With the aligned consensus, the captioning model can capture both the correct linguistic characteristics and visual relevance, and then grounding appropriate image regions further. We validate the effectiveness of our model, with a significant decline in object hallucination (-9% CHAIRi) on the Flickr30k Entities dataset. Besides, our CGRL also evaluated by several automatic metrics and human evaluation, the results indicate that the proposed approach can simultaneously improve the performance of image captioning (+2.9 Cider) and grounding (+2.3 F1LOC).

| Comments: | 9 pages, 5 figures, AAAI 2021                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2112.00974](https://arxiv.org/abs/2112.00974) [cs.CV]** |
|           | (or **[arXiv:2112.00974v1](https://arxiv.org/abs/2112.00974v1) [cs.CV]** for this version) |





<h2 id="2021-12-3-2">2. A Mixture of Expert Based Deep Neural Network for Improved ASR
</h2>

Title: [A Mixture of Expert Based Deep Neural Network for Improved ASR](https://arxiv.org/abs/2112.01025)

Authors: [Vishwanath Pratap Singh](https://arxiv.org/search/eess?searchtype=author&query=Singh%2C+V+P), [Shakti P. Rath](https://arxiv.org/search/eess?searchtype=author&query=Rath%2C+S+P), [Abhishek Pandey](https://arxiv.org/search/eess?searchtype=author&query=Pandey%2C+A)

> This paper presents a novel deep learning architecture for acoustic model in the context of Automatic Speech Recognition (ASR), termed as MixNet. Besides the conventional layers, such as fully connected layers in DNN-HMM and memory cells in LSTM-HMM, the model uses two additional layers based on Mixture of Experts (MoE). The first MoE layer operating at the input is based on pre-defined broad phonetic classes and the second layer operating at the penultimate layer is based on automatically learned acoustic classes. In natural speech, overlap in distribution across different acoustic classes is inevitable, which leads to inter-class mis-classification. The ASR accuracy is expected to improve if the conventional architecture of acoustic model is modified to make them more suitable to account for such overlaps. MixNet is developed keeping this in mind. Analysis conducted by means of scatter diagram verifies that MoE indeed improves the separation between classes that translates to better ASR accuracy. Experiments are conducted on a large vocabulary ASR task which show that the proposed architecture provides 13.6% and 10.0% relative reduction in word error rates compared to the conventional models, namely, DNN and LSTM respectively, trained using sMBR criteria. In comparison to an existing method developed for phone-classification (by Eigen et al), our proposed method yields a significant improvement.

| Subjects: | **Audio and Speech Processing (eess.AS)**; Computation and Language (cs.CL); Sound (cs.SD) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2112.01025](https://arxiv.org/abs/2112.01025) [eess.AS]** |
|           | (or **[arXiv:2112.01025v1](https://arxiv.org/abs/2112.01025v1) [eess.AS]** for this version) |





<h2 id="2021-12-3-3">3. DenseCLIP: Extract Free Dense Labels from CLIP
</h2>

Title: [DenseCLIP: Extract Free Dense Labels from CLIP](https://arxiv.org/abs/2112.01071)

Authors: [Chong Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+C), [Chen Change Loy](https://arxiv.org/search/cs?searchtype=author&query=Loy%2C+C+C), [Bo Dai](https://arxiv.org/search/cs?searchtype=author&query=Dai%2C+B)

> Contrastive Language-Image Pre-training (CLIP) has made a remarkable breakthrough in open-vocabulary zero-shot image recognition. Many recent studies leverage the pre-trained CLIP models for image-level classification and manipulation. In this paper, we further explore the potentials of CLIP for pixel-level dense prediction, specifically in semantic segmentation. Our method, DenseCLIP, in the absence of annotations and fine-tuning, yields reasonable segmentation results on open concepts across various datasets. By adding pseudo labeling and self-training, DenseCLIP+ surpasses SOTA transductive zero-shot semantic segmentation methods by large margins, e.g., mIoUs of unseen classes on PASCAL VOC/PASCAL Context/COCO Stuff are improved from 35.6/20.7/30.3 to 86.1/66.7/54.7. We also test the robustness of DenseCLIP under input corruption and evaluate its capability in discriminating fine-grained objects and novel concepts. Our finding suggests that DenseCLIP can serve as a new reliable source of supervision for dense prediction tasks to achieve annotation-free segmentation.

| Comments: | Tech report, 12 pages, 6 figures                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2112.01071](https://arxiv.org/abs/2112.01071) [cs.CV]** |
|           | (or **[arXiv:2112.01071v1](https://arxiv.org/abs/2112.01071v1) [cs.CV]** for this version) |






# 2021-12-2

[Return to Index](#Index)



<h2 id="2021-12-2-1">1. CLIPstyler: Image Style Transfer with a Single Text Condition
</h2>

Title: [CLIPstyler: Image Style Transfer with a Single Text Condition](https://arxiv.org/abs/2112.00374)

Authors: [Gihyun Kwon](https://arxiv.org/search/cs?searchtype=author&query=Kwon%2C+G), [Jong Chul Ye](https://arxiv.org/search/cs?searchtype=author&query=Ye%2C+J+C)

> Existing neural style transfer methods require reference style images to transfer texture information of style images to content images. However, in many practical situations, users may not have reference style images but still be interested in transferring styles by just imagining them. In order to deal with such applications, we propose a new framework that enables a style transfer `without' a style image, but only with a text description of the desired style. Using the pre-trained text-image embedding model of CLIP, we demonstrate the modulation of the style of content images only with a single text condition. Specifically, we propose a patch-wise text-image matching loss with multiview augmentations for realistic texture transfer. Extensive experimental results confirmed the successful image style transfer with realistic textures that reflect semantic query texts.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Image and Video Processing (eess.IV) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2112.00374](https://arxiv.org/abs/2112.00374) [cs.CV]** |
|           | (or **[arXiv:2112.00374v1](https://arxiv.org/abs/2112.00374v1) [cs.CV]** for this version) |





<h2 id="2021-12-2-2">2. Translation-equivariant Image Quantizer for Bi-directional Image-Text Generation
</h2>

Title: [Translation-equivariant Image Quantizer for Bi-directional Image-Text Generation](https://arxiv.org/abs/2112.00384)

Authors: [Woncheol Shin](https://arxiv.org/search/cs?searchtype=author&query=Shin%2C+W), [Gyubok Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+G), [Jiyoung Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+J), [Joonseok Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+J), [Edward Choi](https://arxiv.org/search/cs?searchtype=author&query=Choi%2C+E)

> Recently, vector-quantized image modeling has demonstrated impressive performance on generation tasks such as text-to-image generation. However, we discover that the current image quantizers do not satisfy translation equivariance in the quantized space due to aliasing, degrading performance in the downstream text-to-image generation and image-to-text generation, even in simple experimental setups. Instead of focusing on anti-aliasing, we take a direct approach to encourage translation equivariance in the quantized space. In particular, we explore a desirable property of image quantizers, called 'Translation Equivariance in the Quantized Space' and propose a simple but effective way to achieve translation equivariance by regularizing orthogonality in the codebook embedding vectors. Using this method, we improve accuracy by +22% in text-to-image generation and +26% in image-to-text generation, outperforming the VQGAN.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2112.00384](https://arxiv.org/abs/2112.00384) [cs.CV]** |
|           | (or **[arXiv:2112.00384v1](https://arxiv.org/abs/2112.00384v1) [cs.CV]** for this version) |





<h2 id="2021-12-2-3">3. DPRK-BERT: The Supreme Language Model
</h2>

Title: [DPRK-BERT: The Supreme Language Model](https://arxiv.org/abs/2112.00567)

Authors: [Arda Akdemir](https://arxiv.org/search/cs?searchtype=author&query=Akdemir%2C+A), [Yeojoo Jeon](https://arxiv.org/search/cs?searchtype=author&query=Jeon%2C+Y)

> Deep language models have achieved remarkable success in the NLP domain. The standard way to train a deep language model is to employ unsupervised learning from scratch on a large unlabeled corpus. However, such large corpora are only available for widely-adopted and high-resource languages and domains. This study presents the first deep language model, DPRK-BERT, for the DPRK language. We achieve this by compiling the first unlabeled corpus for the DPRK language and fine-tuning a preexisting the ROK language model. We compare the proposed model with existing approaches and show significant improvements on two DPRK datasets. We also present a cross-lingual version of this model which yields better generalization across the two Korean languages. Finally, we provide various NLP tools related to the DPRK language that would foster future research.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2112.00567](https://arxiv.org/abs/2112.00567) [cs.CL]** |
|           | (or **[arXiv:2112.00567v1](https://arxiv.org/abs/2112.00567v1) [cs.CL]** for this version) |





# 2021-12-1

[Return to Index](#Index)



<h2 id="2021-12-1-1">1. Do We Still Need Automatic Speech Recognition for Spoken Language Understanding?
</h2>

Title: [Do We Still Need Automatic Speech Recognition for Spoken Language Understanding?](https://arxiv.org/abs/2111.14842)

Authors: [Lasse Borgholt](https://arxiv.org/search/eess?searchtype=author&query=Borgholt%2C+L), [Jakob Drachmann Havtorn](https://arxiv.org/search/eess?searchtype=author&query=Havtorn%2C+J+D), [Mostafa Abdou](https://arxiv.org/search/eess?searchtype=author&query=Abdou%2C+M), [Joakim Edin](https://arxiv.org/search/eess?searchtype=author&query=Edin%2C+J), [Lars Maaløe](https://arxiv.org/search/eess?searchtype=author&query=Maaløe%2C+L), [Anders Søgaard](https://arxiv.org/search/eess?searchtype=author&query=Søgaard%2C+A), [Christian Igel](https://arxiv.org/search/eess?searchtype=author&query=Igel%2C+C)

> Spoken language understanding (SLU) tasks are usually solved by first transcribing an utterance with automatic speech recognition (ASR) and then feeding the output to a text-based model. Recent advances in self-supervised representation learning for speech data have focused on improving the ASR component. We investigate whether representation learning for speech has matured enough to replace ASR in SLU. We compare learned speech features from wav2vec 2.0, state-of-the-art ASR transcripts, and the ground truth text as input for a novel speech-based named entity recognition task, a cardiac arrest detection task on real-world emergency calls and two existing SLU benchmarks. We show that learned speech features are superior to ASR transcripts on three classification tasks. For machine translation, ASR transcripts are still the better choice. We highlight the intrinsic robustness of wav2vec 2.0 representations to out-of-vocabulary words as key to better performance.

| Comments: | Under review as a conference paper at ICASSP 2022            |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Audio and Speech Processing (eess.AS)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2111.14842](https://arxiv.org/abs/2111.14842) [eess.AS]** |
|           | (or **[arXiv:2111.14842v1](https://arxiv.org/abs/2111.14842v1) [eess.AS]** for this version) |





<h2 id="2021-12-1-2">2. Improvement in Machine Translation with Generative Adversarial Networks
</h2>

Title: [Improvement in Machine Translation with Generative Adversarial Networks](https://arxiv.org/abs/2111.15166)

Authors: [Jay Ahn](https://arxiv.org/search/cs?searchtype=author&query=Ahn%2C+J), [Hari Madhu](https://arxiv.org/search/cs?searchtype=author&query=Madhu%2C+H), [Viet Nguyen](https://arxiv.org/search/cs?searchtype=author&query=Nguyen%2C+V)

> In this paper, we explore machine translation improvement via Generative Adversarial Network (GAN) architecture. We take inspiration from RelGAN, a model for text generation, and NMT-GAN, an adversarial machine translation model, to implement a model that learns to transform awkward, non-fluent English sentences to fluent ones, while only being trained on monolingual corpora. We utilize a parameter λ to control the amount of deviation from the input sentence, i.e. a trade-off between keeping the original tokens and modifying it to be more fluent. Our results improved upon phrase-based machine translation in some cases. Especially, GAN with a transformer generator shows some promising results. We suggests some directions for future works to build upon this proof-of-concept.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2111.15166](https://arxiv.org/abs/2111.15166) [cs.CL]** |
|           | (or **[arXiv:2111.15166v1](https://arxiv.org/abs/2111.15166v1) [cs.CL]** for this version) |





<h2 id="2021-12-1-3">3. Pureformer: Do We Even Need Attention?
</h2>

Title: [Pureformer: Do We Even Need Attention?](https://arxiv.org/abs/2111.15588)

Authors: [Uladzislau Yorsh](https://arxiv.org/search/cs?searchtype=author&query=Yorsh%2C+U), [Alexander Kovalenko](https://arxiv.org/search/cs?searchtype=author&query=Kovalenko%2C+A)

> In this paper we propose that the dot product pairwise matching attention layer, which is widely used in transformer-based models, is redundant for the model performance. Attention in its original formulation has to be seen rather as a human-level tool to explore and/or visualize relevancy scores in the sequences. Instead, we present a simple and fast alternative without any approximation that, to the best of our knowledge, outperforms existing attention approximations on the text classification task from the Long-Range Arena benchmark.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2111.15588](https://arxiv.org/abs/2111.15588) [cs.CL]** |
|           | (or **[arXiv:2111.15588v1](https://arxiv.org/abs/2111.15588v1) [cs.CL]** for this version) |

