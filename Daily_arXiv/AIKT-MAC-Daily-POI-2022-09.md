# MA C.'s Daily Paper Of Interest - September a., 2022

# Index

- [2022-09-14](#2022-09-14)
  - [1. PreSTU: Pre-Training for Scene-Text Understanding](#2022-09-14-1)
  
  - [2. Learning ASR pathways: A sparse multilingual ASR model](#2022-09-14-2)
  
  - [3. StoryDALL-E: Adapting Pretrained Text-to-Image Transformers for Story Continuation](#2022-09-14-3)
  
  - [4. Rethink about the Word-level Quality Estimation for Machine Translation from Human Judgement](#2022-09-14-4)
  
  - [5. Multi-stage Distillation Framework for Cross-Lingual Semantic Similarity Matching](#2022-09-14-5)
  
  - [6. Don't Judge a Language Model by Its Last Layer: Contrastive Learning with Layer-Wise Attention Pooling](#2022-09-14-6)
  
- [2022-09-13](#2022-09-13)
  - [1. Simple and Effective Gradient-Based Tuning of Sequence-to-Sequence Models](#2022-09-13-1)

  - [2. CSL: A Large-scale Chinese Scientific Literature Dataset](#2022-09-13-2)

- [2022-09-07](#2022-09-07)
  - [1. Informative Language Representation Learning for Massively Multilingual Neural Machine Translation](#2022-09-07-1)
  - [2. Distilling the Knowledge of BERT for CTC-based ASR](#2022-09-07-2)
  - [3. Rare but Severe Neural Machine Translation Errors Induced by Minimal Deletion: An Empirical Study on Chinese and English](#2022-09-07-3)
  - [4. Analyzing Transformers in Embedding Space](#2022-09-07-4)

- [2022-09-05](#2022-09-05)
  - [1. Extend and Explain: Interpreting Very Long Language Models](#2022-09-05-1)
- [2022-08-30](#2022-08-30)
  - [1. Contrastive Audio-Language Learning for Music](#2022-08-30-1)


- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



# 2022-09-14

[Return to Index](#Index)



<h2 id="2022-09-14-1">1. PreSTU: Pre-Training for Scene-Text Understanding
</h2>

Title: [PreSTU: Pre-Training for Scene-Text Understanding](https://arxiv.org/abs/2209.05534)

Authors: [Jihyung Kil](https://arxiv.org/search/cs?searchtype=author&query=Kil%2C+J), [Soravit Changpinyo](https://arxiv.org/search/cs?searchtype=author&query=Changpinyo%2C+S), [Xi Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+X), [Hexiang Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+H), [Sebastian Goodman](https://arxiv.org/search/cs?searchtype=author&query=Goodman%2C+S), [Wei-Lun Chao](https://arxiv.org/search/cs?searchtype=author&query=Chao%2C+W), [Radu Soricut](https://arxiv.org/search/cs?searchtype=author&query=Soricut%2C+R)

> The ability to read and reason about texts in an image is often lacking in vision-and-language (V&L) models. How can we learn V&L models that exhibit strong scene-text understanding (STU)? In this paper, we propose PreSTU, a simple pre-training recipe specifically designed for scene-text understanding. PreSTU combines a simple OCR-aware pre-training objective with a large-scale image-text dataset with off-the-shelf OCR signals. We empirically demonstrate the superiority of this pre-training objective on TextVQA, TextCaps, ST-VQA, and VizWiz-VQA. We also study which factors affect STU performance, where we highlight the importance of image resolution and dataset scale during pre-training.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2209.05534](https://arxiv.org/abs/2209.05534) [cs.CV]** |
|           | (or **[arXiv:2209.05534v1](https://arxiv.org/abs/2209.05534v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2209.05534Focus to learn more |





<h2 id="2022-09-14-2">2. Learning ASR pathways: A sparse multilingual ASR model
</h2>

Title: [Learning ASR pathways: A sparse multilingual ASR model](https://arxiv.org/abs/2209.05735)

Authors: [Mu Yang](https://arxiv.org/search/eess?searchtype=author&query=Yang%2C+M), [Andros Tjandra](https://arxiv.org/search/eess?searchtype=author&query=Tjandra%2C+A), [Chunxi Liu](https://arxiv.org/search/eess?searchtype=author&query=Liu%2C+C), [David Zhang](https://arxiv.org/search/eess?searchtype=author&query=Zhang%2C+D), [Duc Le](https://arxiv.org/search/eess?searchtype=author&query=Le%2C+D), [John H. L. Hansen](https://arxiv.org/search/eess?searchtype=author&query=Hansen%2C+J+H+L), [Ozlem Kalinli](https://arxiv.org/search/eess?searchtype=author&query=Kalinli%2C+O)

> Neural network pruning can be effectively applied to compress automatic speech recognition (ASR) models. However, in multilingual ASR, performing language-agnostic pruning may lead to severe performance degradation on some languages because language-agnostic pruning masks may not fit all languages and discard important language-specific parameters. In this work, we present ASR pathways, a sparse multilingual ASR model that activates language-specific sub-networks ("pathways"), such that the parameters for each language are learned explicitly. With the overlapping sub-networks, the shared parameters can also enable knowledge transfer for lower resource languages via joint multilingual training. We propose a novel algorithm to learn ASR pathways, and evaluate the proposed method on 4 languages with a streaming RNN-T model. Our proposed ASR pathways outperform both dense models (-5.0% average WER) and a language-agnostically pruned model (-21.4% average WER), and provide better performance on low-resource languages compared to the monolingual sparse models.

| Comments: | 5 pages, 3 figures                                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Audio and Speech Processing (eess.AS)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2209.05735](https://arxiv.org/abs/2209.05735) [eess.AS]** |
|           | (or **[arXiv:2209.05735v1](https://arxiv.org/abs/2209.05735v1) [eess.AS]** for this version) |
|           | https://doi.org/10.48550/arXiv.2209.05735Focus to learn more |





<h2 id="2022-09-14-3">3. StoryDALL-E: Adapting Pretrained Text-to-Image Transformers for Story Continuation
</h2>

Title: [StoryDALL-E: Adapting Pretrained Text-to-Image Transformers for Story Continuation](https://arxiv.org/abs/2209.06192)

Authors: [Adyasha Maharana](https://arxiv.org/search/cs?searchtype=author&query=Maharana%2C+A), [Darryl Hannan](https://arxiv.org/search/cs?searchtype=author&query=Hannan%2C+D), [Mohit Bansal](https://arxiv.org/search/cs?searchtype=author&query=Bansal%2C+M)

> Recent advances in text-to-image synthesis have led to large pretrained transformers with excellent capabilities to generate visualizations from a given text. However, these models are ill-suited for specialized tasks like story visualization, which requires an agent to produce a sequence of images given a corresponding sequence of captions, forming a narrative. Moreover, we find that the story visualization task fails to accommodate generalization to unseen plots and characters in new narratives. Hence, we first propose the task of story continuation, where the generated visual story is conditioned on a source image, allowing for better generalization to narratives with new characters. Then, we enhance or 'retro-fit' the pretrained text-to-image synthesis models with task-specific modules for (a) sequential image generation and (b) copying relevant elements from an initial frame. Then, we explore full-model finetuning, as well as prompt-based tuning for parameter-efficient adaptation, of the pre-trained model. We evaluate our approach StoryDALL-E on two existing datasets, PororoSV and FlintstonesSV, and introduce a new dataset DiDeMoSV collected from a video-captioning dataset. We also develop a model StoryGANc based on Generative Adversarial Networks (GAN) for story continuation, and compare it with the StoryDALL-E model to demonstrate the advantages of our approach. We show that our retro-fitting approach outperforms GAN-based models for story continuation and facilitates copying of visual elements from the source image, thereby improving continuity in the generated visual story. Finally, our analysis suggests that pretrained transformers struggle to comprehend narratives containing several characters. Overall, our work demonstrates that pretrained text-to-image synthesis models can be adapted for complex and low-resource tasks like story continuation.

| Comments: | ECCV 2022 (33 pages; code, data, demo, model card available at [this https URL](https://github.com/adymaharana/storydalle)) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2209.06192](https://arxiv.org/abs/2209.06192) [cs.CV]** |
|           | (or **[arXiv:2209.06192v1](https://arxiv.org/abs/2209.06192v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2209.06192Focus to learn more |





<h2 id="2022-09-14-4">4. Rethink about the Word-level Quality Estimation for Machine Translation from Human Judgement
</h2>

Title: [Rethink about the Word-level Quality Estimation for Machine Translation from Human Judgement](https://arxiv.org/abs/2209.05695)

Authors: [Zhen Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Z), [Fandong Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+F), [Yuanmeng Yan](https://arxiv.org/search/cs?searchtype=author&query=Yan%2C+Y), [Jie Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+J)

> Word-level Quality Estimation (QE) of Machine Translation (MT) aims to find out potential translation errors in the translated sentence without reference. Typically, conventional works on word-level QE are designed to predict the translation quality in terms of the post-editing effort, where the word labels ("OK" and "BAD") are automatically generated by comparing words between MT sentences and the post-edited sentences through a Translation Error Rate (TER) toolkit. While the post-editing effort can be used to measure the translation quality to some extent, we find it usually conflicts with the human judgement on whether the word is well or poorly translated. To overcome the limitation, we first create a golden benchmark dataset, namely \emph{HJQE} (Human Judgement on Quality Estimation), where the expert translators directly annotate the poorly translated words on their judgements. Additionally, to further make use of the parallel corpus, we propose the self-supervised pre-training with two tag correcting strategies, namely tag refinement strategy and tree-based annotation strategy, to make the TER-based artificial QE corpus closer to \emph{HJQE}. We conduct substantial experiments based on the publicly available WMT En-De and En-Zh corpora. The results not only show our proposed dataset is more consistent with human judgment but also confirm the effectiveness of the proposed tag correcting strategies.\footnote{The data can be found at \url{[this https URL](https://github.com/ZhenYangIACAS/HJQE)}.}

| Comments: | 8 pages, 6 figures                                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2209.05695](https://arxiv.org/abs/2209.05695) [cs.CL]** |
|           | (or **[arXiv:2209.05695v1](https://arxiv.org/abs/2209.05695v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2209.05695Focus to learn more |





<h2 id="2022-09-14-5">5. Multi-stage Distillation Framework for Cross-Lingual Semantic Similarity Matching
</h2>

Title: [Multi-stage Distillation Framework for Cross-Lingual Semantic Similarity Matching](https://arxiv.org/abs/2209.05869)

Authors: [Kunbo Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+K), [Weijie Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+W), [Yuejian Fang](https://arxiv.org/search/cs?searchtype=author&query=Fang%2C+Y), [Zhe Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+Z), [Qi Ju](https://arxiv.org/search/cs?searchtype=author&query=Ju%2C+Q), [Xuefeng Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+X)

> Previous studies have proved that cross-lingual knowledge distillation can significantly improve the performance of pre-trained models for cross-lingual similarity matching tasks. However, the student model needs to be large in this operation. Otherwise, its performance will drop sharply, thus making it impractical to be deployed to memory-limited devices. To address this issue, we delve into cross-lingual knowledge distillation and propose a multi-stage distillation framework for constructing a small-size but high-performance cross-lingual model. In our framework, contrastive learning, bottleneck, and parameter recurrent strategies are combined to prevent performance from being compromised during the compression process. The experimental results demonstrate that our method can compress the size of XLM-R and MiniLM by more than 50\%, while the performance is only reduced by about 1%.

| Comments: | Published at Findings of NAACL, 2022                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2209.05869](https://arxiv.org/abs/2209.05869) [cs.CL]** |
|           | (or **[arXiv:2209.05869v1](https://arxiv.org/abs/2209.05869v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2209.05869Focus to learn more |





<h2 id="2022-09-14-6">6. Don't Judge a Language Model by Its Last Layer: Contrastive Learning with Layer-Wise Attention Pooling
</h2>

Title: [Don't Judge a Language Model by Its Last Layer: Contrastive Learning with Layer-Wise Attention Pooling](https://arxiv.org/abs/2209.05972)

Authors: [Dongsuk Oh](https://arxiv.org/search/cs?searchtype=author&query=Oh%2C+D), [Yejin Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+Y), [Hodong Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+H), [H. Howie Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+H+H), [Heuiseok Lim](https://arxiv.org/search/cs?searchtype=author&query=Lim%2C+H)

> Recent pre-trained language models (PLMs) achieved great success on many natural language processing tasks through learning linguistic features and contextualized sentence representation. Since attributes captured in stacked layers of PLMs are not clearly identified, straightforward approaches such as embedding the last layer are commonly preferred to derive sentence representations from PLMs. This paper introduces the attention-based pooling strategy, which enables the model to preserve layer-wise signals captured in each layer and learn digested linguistic features for downstream tasks. The contrastive learning objective can adapt the layer-wise attention pooling to both unsupervised and supervised manners. It results in regularizing the anisotropic space of pre-trained embeddings and being more uniform. We evaluate our model on standard semantic textual similarity (STS) and semantic search tasks. As a result, our method improved the performance of the base contrastive learned BERT_base and variants.

| Comments: | Accepted to COLING 2022                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2209.05972](https://arxiv.org/abs/2209.05972) [cs.CL]** |
|           | (or **[arXiv:2209.05972v1](https://arxiv.org/abs/2209.05972v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2209.05972Focus to learn more |






# 2022-09-13

[Return to Index](#Index)



<h2 id="2022-09-13-1">1. Simple and Effective Gradient-Based Tuning of Sequence-to-Sequence Models
</h2>

Title: [Simple and Effective Gradient-Based Tuning of Sequence-to-Sequence Models](https://arxiv.org/abs/2209.04683)

Authors: [Jared Lichtarge](https://arxiv.org/search/cs?searchtype=author&query=Lichtarge%2C+J), [Chris Alberti](https://arxiv.org/search/cs?searchtype=author&query=Alberti%2C+C), [Shankar Kumar](https://arxiv.org/search/cs?searchtype=author&query=Kumar%2C+S)

> Recent trends towards training ever-larger language models have substantially improved machine learning performance across linguistic tasks. However, the huge cost of training larger models can make tuning them prohibitively expensive, motivating the study of more efficient methods. Gradient-based hyper-parameter optimization offers the capacity to tune hyper-parameters during training, yet has not previously been studied in a sequence-to-sequence setting. We apply a simple and general gradient-based hyperparameter optimization method to sequence-to-sequence tasks for the first time, demonstrating both efficiency and performance gains over strong baselines for both Neural Machine Translation and Natural Language Understanding (NLU) tasks (via T5 pretraining). For translation, we show the method generalizes across language pairs, is more efficient than Bayesian hyper-parameter optimization, and that learned schedules for some hyper-parameters can out-perform even optimal constant-valued tuning. For T5, we show that learning hyper-parameters during pretraining can improve performance across downstream NLU tasks. When learning multiple hyper-parameters concurrently, we show that the global learning rate can follow a schedule over training that improves performance and is not explainable by the `short-horizon bias' of greedy methods \citep{wu2018}. We release the code used to facilitate further research.

| Comments:          | 18 pages, 6 figures, Accepted to AutoML 2022                 |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:           | **[arXiv:2209.04683](https://arxiv.org/abs/2209.04683) [cs.CL]** |
|                    | (or **[arXiv:2209.04683v1](https://arxiv.org/abs/2209.04683v1) [cs.CL]** for this version) |
|                    | https://doi.org/10.48550/arXiv.2209.04683Focus to learn more |
| Journal reference: | Lichtarge J., Alberti C., and Kumar S. (2022). Simple and Effective Gradient-Based Tuning of Sequence-to-Sequence Models. In Proceedings of AutoML 2022 (Workshop track), Baltimore, MD, USA |





<h2 id="2022-09-13-2">2. CSL: A Large-scale Chinese Scientific Literature Dataset
</h2>

Title: [CSL: A Large-scale Chinese Scientific Literature Dataset](https://arxiv.org/abs/2209.05034)

Authors: [Yudong Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Yuqing Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y), [Zhe Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+Z), [Linlin Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+L), [Weijie Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+W), [Weiquan Mao](https://arxiv.org/search/cs?searchtype=author&query=Mao%2C+W), [Hui Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+H)

> Scientific literature serves as a high-quality corpus, supporting a lot of Natural Language Processing (NLP) research. However, existing datasets are centered around the English language, which restricts the development of Chinese scientific NLP. In this work, we present CSL, a large-scale Chinese Scientific Literature dataset, which contains the titles, abstracts, keywords and academic fields of 396k papers. To our knowledge, CSL is the first scientific document dataset in Chinese. The CSL can serve as a Chinese corpus. Also, this semi-structured data is a natural annotation that can constitute many supervised NLP tasks. Based on CSL, we present a benchmark to evaluate the performance of models across scientific domain tasks, i.e., summarization, keyword generation and text classification. We analyze the behavior of existing text-to-text models on the evaluation tasks and reveal the challenges for Chinese scientific NLP tasks, which provides a valuable reference for future research. Data and code are available at [this https URL](https://github.com/ydli-ai/CSL)

| Comments: | to be published in COLING 2022                               |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2209.05034](https://arxiv.org/abs/2209.05034) [cs.CL]** |
|           | (or **[arXiv:2209.05034v1](https://arxiv.org/abs/2209.05034v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2209.05034Focus to learn more |



# 2022-09-07

[Return to Index](#Index)



<h2 id="2022-09-07-1">1. Informative Language Representation Learning for Massively Multilingual Neural Machine Translation
</h2>

Title: [Informative Language Representation Learning for Massively Multilingual Neural Machine Translation](https://arxiv.org/abs/2209.01530)

Authors: [Renren Jin](https://arxiv.org/search/cs?searchtype=author&query=Jin%2C+R), [Deyi Xiong](https://arxiv.org/search/cs?searchtype=author&query=Xiong%2C+D)

> In a multilingual neural machine translation model that fully shares parameters across all languages, an artificial language token is usually used to guide translation into the desired target language. However, recent studies show that prepending language tokens sometimes fails to navigate the multilingual neural machine translation models into right translation directions, especially on zero-shot translation. To mitigate this issue, we propose two methods, language embedding embodiment and language-aware multi-head attention, to learn informative language representations to channel translation into right directions. The former embodies language embeddings into different critical switching points along the information flow from the source to the target, aiming at amplifying translation direction guiding signals. The latter exploits a matrix, instead of a vector, to represent a language in the continuous space. The matrix is chunked into multiple heads so as to learn language representations in multiple subspaces. Experiment results on two datasets for massively multilingual neural machine translation demonstrate that language-aware multi-head attention benefits both supervised and zero-shot translation and significantly alleviates the off-target translation issue. Further linguistic typology prediction experiments show that matrix-based language representations learned by our methods are capable of capturing rich linguistic typology features.

| Comments: | Accepted by COLING 2022                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2209.01530](https://arxiv.org/abs/2209.01530) [cs.CL]** |
|           | (or **[arXiv:2209.01530v1](https://arxiv.org/abs/2209.01530v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2209.01530Focus to learn more |





<h2 id="2022-09-07-2">2. Distilling the Knowledge of BERT for CTC-based ASR
</h2>

Title: [Distilling the Knowledge of BERT for CTC-based ASR](https://arxiv.org/abs/2209.02030)

Authors: [Hayato Futami](https://arxiv.org/search/cs?searchtype=author&query=Futami%2C+H), [Hirofumi Inaguma](https://arxiv.org/search/cs?searchtype=author&query=Inaguma%2C+H), [Masato Mimura](https://arxiv.org/search/cs?searchtype=author&query=Mimura%2C+M), [Shinsuke Sakai](https://arxiv.org/search/cs?searchtype=author&query=Sakai%2C+S), [Tatsuya Kawahara](https://arxiv.org/search/cs?searchtype=author&query=Kawahara%2C+T)

> Connectionist temporal classification (CTC) -based models are attractive because of their fast inference in automatic speech recognition (ASR). Language model (LM) integration approaches such as shallow fusion and rescoring can improve the recognition accuracy of CTC-based ASR by taking advantage of the knowledge in text corpora. However, they significantly slow down the inference of CTC. In this study, we propose to distill the knowledge of BERT for CTC-based ASR, extending our previous study for attention-based ASR. CTC-based ASR learns the knowledge of BERT during training and does not use BERT during testing, which maintains the fast inference of CTC. Different from attention-based models, CTC-based models make frame-level predictions, so they need to be aligned with token-level predictions of BERT for distillation. We propose to obtain alignments by calculating the most plausible CTC paths. Experimental evaluations on the Corpus of Spontaneous Japanese (CSJ) and TED-LIUM2 show that our method improves the performance of CTC-based ASR without the cost of inference speed.

| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2209.02030](https://arxiv.org/abs/2209.02030) [cs.CL]** |
|           | (or **[arXiv:2209.02030v1](https://arxiv.org/abs/2209.02030v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2209.02030Focus to learn more |





<h2 id="2022-09-07-3">3. Rare but Severe Neural Machine Translation Errors Induced by Minimal Deletion: An Empirical Study on Chinese and English
</h2>

Title: [Rare but Severe Neural Machine Translation Errors Induced by Minimal Deletion: An Empirical Study on Chinese and English](https://arxiv.org/abs/2209.02145)

Authors: [Ruikang Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+R), [Alvin Grissom II](https://arxiv.org/search/cs?searchtype=author&query=II%2C+A+G), [Duc Minh Trinh](https://arxiv.org/search/cs?searchtype=author&query=Trinh%2C+D+M)

> We examine the inducement of rare but severe errors in English-Chinese and Chinese-English in-domain neural machine translation by minimal deletion of the source text with character-based models. By deleting a single character, we find that we can induce severe errors in the translation. We categorize these errors and compare the results of deleting single characters and single words. We also examine the effect of training data size on the number and types of pathological cases induced by these minimal perturbations, finding significant variation.

| Comments: | Accepted to COLING 2022                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2209.02145](https://arxiv.org/abs/2209.02145) [cs.CL]** |
|           | (or **[arXiv:2209.02145v1](https://arxiv.org/abs/2209.02145v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2209.02145Focus to learn more |





<h2 id="2022-09-07-4">4. Analyzing Transformers in Embedding Space
</h2>

Title: [Analyzing Transformers in Embedding Space](https://arxiv.org/abs/2209.02535)

Authors: [Guy Dar](https://arxiv.org/search/cs?searchtype=author&query=Dar%2C+G), [Mor Geva](https://arxiv.org/search/cs?searchtype=author&query=Geva%2C+M), [Ankit Gupta](https://arxiv.org/search/cs?searchtype=author&query=Gupta%2C+A), [Jonathan Berant](https://arxiv.org/search/cs?searchtype=author&query=Berant%2C+J)

> Understanding Transformer-based models has attracted significant attention, as they lie at the heart of recent technological advances across machine learning. While most interpretability methods rely on running models over inputs, recent work has shown that a zero-pass approach, where parameters are interpreted directly without a forward/backward pass is feasible for some Transformer parameters, and for two-layer attention networks. In this work, we present a theoretical analysis where all parameters of a trained Transformer are interpreted by projecting them into the embedding space, that is, the space of vocabulary items they operate on. We derive a simple theoretical framework to support our arguments and provide ample evidence for its validity. First, an empirical analysis showing that parameters of both pretrained and fine-tuned models can be interpreted in embedding space. Second, we present two applications of our framework: (a) aligning the parameters of different models that share a vocabulary, and (b) constructing a classifier without training by ``translating'' the parameters of a fine-tuned classifier to parameters of a different model that was only pretrained. Overall, our findings open the door to interpretation methods that, at least in part, abstract away from model specifics and operate in the embedding space only.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2209.02535](https://arxiv.org/abs/2209.02535) [cs.CL]** |
|           | (or **[arXiv:2209.02535v1](https://arxiv.org/abs/2209.02535v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2209.02535Focus to learn more |






# 2022-09-05

[Return to Index](#Index)



<h2 id="2022-09-05-1">1. Extend and Explain: Interpreting Very Long Language Models
</h2>

Title: [Extend and Explain: Interpreting Very Long Language Models](https://arxiv.org/abs/2209.01174)

Authors: [Joel Stremmel](https://arxiv.org/search/cs?searchtype=author&query=Stremmel%2C+J), [Brian L. Hill](https://arxiv.org/search/cs?searchtype=author&query=Hill%2C+B+L), [Jeffrey Hertzberg](https://arxiv.org/search/cs?searchtype=author&query=Hertzberg%2C+J), [Jaime Murillo](https://arxiv.org/search/cs?searchtype=author&query=Murillo%2C+J), [Llewelyn Allotey](https://arxiv.org/search/cs?searchtype=author&query=Allotey%2C+L), [Eran Halperin](https://arxiv.org/search/cs?searchtype=author&query=Halperin%2C+E)

> While Transformer language models (LMs) are state-of-the-art for information extraction, long text introduces computational challenges requiring suboptimal preprocessing steps or alternative model architectures. Sparse-attention LMs can represent longer sequences, overcoming performance hurdles. However, it remains unclear how to explain predictions from these models, as not all tokens attend to each other in the self-attention layers, and long sequences pose computational challenges for explainability algorithms when runtime depends on document length. These challenges are severe in the medical context where documents can be very long, and machine learning (ML) models must be auditable and trustworthy. We introduce a novel Masked Sampling Procedure (MSP) to identify the text blocks that contribute to a prediction, apply MSP in the context of predicting diagnoses from medical text, and validate our approach with a blind review by two clinicians. Our method identifies about 1.7x more clinically informative text blocks than the previous state-of-the-art, runs up to 100x faster, and is tractable for generating important phrase pairs. MSP is particularly well-suited to long LMs but can be applied to any text classifier. We provide a general implementation of MSP.

| Comments:    | 10 pages                                                     |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**                         |
| MSC classes: | I.2.7                                                        |
| Cite as:     | **[arXiv:2209.01174](https://arxiv.org/abs/2209.01174) [cs.CL]** |
|              | (or **[arXiv:2209.01174v1](https://arxiv.org/abs/2209.01174v1) [cs.CL]** for this version) |
|              | https://doi.org/10.48550/arXiv.2209.01174Focus to learn more |






# 2022-08-30

[Return to Index](#Index)



<h2 id="2022-08-30-1">1. Contrastive Audio-Language Learning for Music
</h2>

Title: [Contrastive Audio-Language Learning for Music](https://arxiv.org/abs/2208.12208)

Authors: [Ilaria Manco](https://arxiv.org/search/cs?searchtype=author&query=Manco%2C+I), [Emmanouil Benetos](https://arxiv.org/search/cs?searchtype=author&query=Benetos%2C+E), [Elio Quinton](https://arxiv.org/search/cs?searchtype=author&query=Quinton%2C+E), [György Fazekas](https://arxiv.org/search/cs?searchtype=author&query=Fazekas%2C+G)

> As one of the most intuitive interfaces known to humans, natural language has the potential to mediate many tasks that involve human-computer interaction, especially in application-focused fields like Music Information Retrieval. In this work, we explore cross-modal learning in an attempt to bridge audio and language in the music domain. To this end, we propose MusCALL, a framework for Music Contrastive Audio-Language Learning. Our approach consists of a dual-encoder architecture that learns the alignment between pairs of music audio and descriptive sentences, producing multimodal embeddings that can be used for text-to-audio and audio-to-text retrieval out-of-the-box. Thanks to this property, MusCALL can be transferred to virtually any task that can be cast as text-based retrieval. Our experiments show that our method performs significantly better than the baselines at retrieving audio that matches a textual description and, conversely, text that matches an audio query. We also demonstrate that the multimodal alignment capability of our model can be successfully extended to the zero-shot transfer scenario for genre classification and auto-tagging on two public datasets.

| Comments: | Accepted to ISMIR 2022                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Sound (cs.SD)**; Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2208.12208](https://arxiv.org/abs/2208.12208) [cs.SD]** |
|           | (or **[arXiv:2208.12208v1](https://arxiv.org/abs/2208.12208v1) [cs.SD]** for this version) |
|           | https://doi.org/10.48550/arXiv.2208.12208Focus to learn more |



