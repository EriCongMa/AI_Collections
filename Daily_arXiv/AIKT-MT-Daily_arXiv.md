# Daily arXiv: Machine Translation - September, 2021

# Index


- [2021-09-02](#2021-09-02)

  - [1. Sentence Bottleneck Autoencoders from Transformer Language Models](#2021-09-02-1)
  - [2. It's not Rocket Science : Interpreting Figurative Language in Narratives](#2021-09-02-2)
  - [3. Aligning Cross-lingual Sentence Representations with Dual Momentum Contrast](#2021-09-02-3)
  - [4. Discovering Representation Sprachbund For Multilingual Pre-Training](#2021-09-02-4)
  - [5. ∞-former: Infinite Memory Transformer](#2021-09-02-5)
  - [6. Masked Adversarial Generation for Neural Machine Translation](#2021-09-02-6)
  - [7. Position Masking for Improved Layout-Aware Document Understanding](#2021-09-02-7)
  - [8. Survey of Low-Resource Machine Translation](#2021-09-02-8)
- [2021-09-01](#2021-09-01)
  - [1. SimulLR: Simultaneous Lip Reading Transducer with Attention-Guided Adaptive Memory](#2021-09-01-1)
  - [2. Want To Reduce Labeling Cost? GPT-3 Can Help](#2021-09-01-2)
  - [3. T3-Vis: a visual analytic framework for Training and fine-Tuning Transformers in NLP](#2021-09-01-3)
  - [4. Enjoy the Salience: Towards Better Transformer-based Faithful Explanations with Word Salience](#2021-09-01-4)
  - [5. Thermostat: A Large Collection of NLP Model Explanations and Analysis Tools](#2021-09-01-5)
- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MT-Daily_arXiv-index.md)



# 2021-09-02

[Return to Index](#Index)



<h2 id="2021-09-02-1">1. Sentence Bottleneck Autoencoders from Transformer Language Models
</h2>

Title: [Sentence Bottleneck Autoencoders from Transformer Language Models](https://arxiv.org/abs/2109.00055)

Authors: [Ivan Montero](https://arxiv.org/search/cs?searchtype=author&query=Montero%2C+I), [Nikolaos Pappas](https://arxiv.org/search/cs?searchtype=author&query=Pappas%2C+N), [Noah A. Smith](https://arxiv.org/search/cs?searchtype=author&query=Smith%2C+N+A)

> Representation learning for text via pretraining a language model on a large corpus has become a standard starting point for building NLP systems. This approach stands in contrast to autoencoders, also trained on raw text, but with the objective of learning to encode each input as a vector that allows full reconstruction. Autoencoders are attractive because of their latent space structure and generative properties. We therefore explore the construction of a sentence-level autoencoder from a pretrained, frozen transformer language model. We adapt the masked language modeling objective as a generative, denoising one, while only training a sentence bottleneck and a single-layer modified transformer decoder. We demonstrate that the sentence representations discovered by our model achieve better quality than previous methods that extract representations from pretrained transformers on text similarity tasks, style transfer (an example of controlled generation), and single-sentence classification tasks in the GLUE benchmark, while using fewer parameters than large pretrained models.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2109.00055](https://arxiv.org/abs/2109.00055) [cs.CL]** |
|           | (or **[arXiv:2109.00055v1](https://arxiv.org/abs/2109.00055v1) [cs.CL]** for this version) |





<h2 id="2021-09-02-2">2. It's not Rocket Science : Interpreting Figurative Language in Narratives
</h2>

Title: [It's not Rocket Science : Interpreting Figurative Language in Narratives](https://arxiv.org/abs/2109.00087)

Authors: [Tuhin Chakrabarty](https://arxiv.org/search/cs?searchtype=author&query=Chakrabarty%2C+T), [Yejin Choi](https://arxiv.org/search/cs?searchtype=author&query=Choi%2C+Y), [Vered Shwartz](https://arxiv.org/search/cs?searchtype=author&query=Shwartz%2C+V)

> Figurative language is ubiquitous in English. Yet, the vast majority of NLP research focuses on literal language. Existing text representations by design rely on compositionality, while figurative language is often non-compositional. In this paper, we study the interpretation of two non-compositional figurative languages (idioms and similes). We collected datasets of fictional narratives containing a figurative expression along with crowd-sourced plausible and implausible continuations relying on the correct interpretation of the expression. We then trained models to choose or generate the plausible continuation. Our experiments show that models based solely on pre-trained language models perform substantially worse than humans on these tasks. We additionally propose knowledge-enhanced models, adopting human strategies for interpreting figurative language: inferring meaning from the context and relying on the constituent word's literal meanings. The knowledge-enhanced models improve the performance on both the discriminative and generative tasks, further bridging the gap from human performance.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2109.00087](https://arxiv.org/abs/2109.00087) [cs.CL]** |
|           | (or **[arXiv:2109.00087v1](https://arxiv.org/abs/2109.00087v1) [cs.CL]** for this version) |





<h2 id="2021-09-02-3">3. Aligning Cross-lingual Sentence Representations with Dual Momentum Contrast
</h2>

Title: [Aligning Cross-lingual Sentence Representations with Dual Momentum Contrast](https://arxiv.org/abs/2109.00253)

Authors: [Liang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L), [Wei Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+W), [Jingming Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+J)

> In this paper, we propose to align sentence representations from different languages into a unified embedding space, where semantic similarities (both cross-lingual and monolingual) can be computed with a simple dot product. Pre-trained language models are fine-tuned with the translation ranking task. Existing work (Feng et al., 2020) uses sentences within the same batch as negatives, which can suffer from the issue of easy negatives. We adapt MoCo (He et al., 2020) to further improve the quality of alignment. As the experimental results show, the sentence representations produced by our model achieve the new state-of-the-art on several tasks, including Tatoeba en-zh similarity search (Artetxe and Schwenk, 2019b), BUCC en-zh bitext mining, and semantic textual similarity on 7 datasets.

| Comments: | Accepted to EMNLP 2021 main conference                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Information Retrieval (cs.IR) |
| Cite as:  | **[arXiv:2109.00253](https://arxiv.org/abs/2109.00253) [cs.CL]** |
|           | (or **[arXiv:2109.00253v1](https://arxiv.org/abs/2109.00253v1) [cs.CL]** for this version) |





<h2 id="2021-09-02-4">4. Discovering Representation Sprachbund For Multilingual Pre-Training
</h2>

Title: [Discovering Representation Sprachbund For Multilingual Pre-Training](https://arxiv.org/abs/2109.00271)

Authors: [Yimin Fan](https://arxiv.org/search/cs?searchtype=author&query=Fan%2C+Y), [Yaobo Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+Y), [Alexandre Muzio](https://arxiv.org/search/cs?searchtype=author&query=Muzio%2C+A), [Hany Hassan](https://arxiv.org/search/cs?searchtype=author&query=Hassan%2C+H), [Houqiang Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+H), [Ming Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+M), [Nan Duan](https://arxiv.org/search/cs?searchtype=author&query=Duan%2C+N)

> Multilingual pre-trained models have demonstrated their effectiveness in many multilingual NLP tasks and enabled zero-shot or few-shot transfer from high-resource languages to low resource ones. However, due to significant typological differences and contradictions between some languages, such models usually perform poorly on many languages and cross-lingual settings, which shows the difficulty of learning a single model to handle massive diverse languages well at the same time. To alleviate this issue, we present a new multilingual pre-training pipeline. We propose to generate language representation from multilingual pre-trained models and conduct linguistic analysis to show that language representation similarity reflect linguistic similarity from multiple perspectives, including language family, geographical sprachbund, lexicostatistics and syntax. Then we cluster all the target languages into multiple groups and name each group as a representation sprachbund. Thus, languages in the same representation sprachbund are supposed to boost each other in both pre-training and fine-tuning as they share rich linguistic similarity. We pre-train one multilingual model for each representation sprachbund. Experiments are conducted on cross-lingual benchmarks and significant improvements are achieved compared to strong baselines.

| Comments: | To Appear at the Findings of EMNLP2021                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2109.00271](https://arxiv.org/abs/2109.00271) [cs.CL]** |
|           | (or **[arXiv:2109.00271v1](https://arxiv.org/abs/2109.00271v1) [cs.CL]** for this version) |





<h2 id="2021-09-02-5">5. ∞-former: Infinite Memory Transformer
</h2>

Title: [∞-former: Infinite Memory Transformer](https://arxiv.org/abs/2109.00301)

Authors: [Pedro Henrique Martins](https://arxiv.org/search/cs?searchtype=author&query=Martins%2C+P+H), [Zita Marinho](https://arxiv.org/search/cs?searchtype=author&query=Marinho%2C+Z), [André F. T. Martins](https://arxiv.org/search/cs?searchtype=author&query=Martins%2C+A+F+T)

> Transformers struggle when attending to long contexts, since the amount of computation grows with the context length, and therefore they cannot model long-term memories effectively. Several variations have been proposed to alleviate this problem, but they all have a finite memory capacity, being forced to drop old information. In this paper, we propose the ∞-former, which extends the vanilla transformer with an unbounded long-term memory. By making use of a continuous-space attention mechanism to attend over the long-term memory, the ∞-former's attention complexity becomes independent of the context length. Thus, it is able to model arbitrarily long contexts and maintain "sticky memories" while keeping a fixed computation budget. Experiments on a synthetic sorting task demonstrate the ability of the ∞-former to retain information from long sequences. We also perform experiments on language modeling, by training a model from scratch and by fine-tuning a pre-trained language model, which show benefits of unbounded long-term memories.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2109.00301](https://arxiv.org/abs/2109.00301) [cs.CL]** |
|           | (or **[arXiv:2109.00301v1](https://arxiv.org/abs/2109.00301v1) [cs.CL]** for this version) |





<h2 id="2021-09-02-6">6. Masked Adversarial Generation for Neural Machine Translation
</h2>

Title: [Masked Adversarial Generation for Neural Machine Translation](https://arxiv.org/abs/2109.00417)

Authors: [Badr Youbi Idrissi](https://arxiv.org/search/cs?searchtype=author&query=Idrissi%2C+B+Y), [Stéphane Clinchant](https://arxiv.org/search/cs?searchtype=author&query=Clinchant%2C+S)

> Attacking Neural Machine Translation models is an inherently combinatorial task on discrete sequences, solved with approximate heuristics. Most methods use the gradient to attack the model on each sample independently. Instead of mechanically applying the gradient, could we learn to produce meaningful adversarial attacks ? In contrast to existing approaches, we learn to attack a model by training an adversarial generator based on a language model. We propose the Masked Adversarial Generation (MAG) model, that learns to perturb the translation model throughout the training process. The experiments show that it improves the robustness of machine translation models, while being faster than competing methods.

| Comments: | 5 pages                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2109.00417](https://arxiv.org/abs/2109.00417) [cs.CL]** |
|           | (or **[arXiv:2109.00417v1](https://arxiv.org/abs/2109.00417v1) [cs.CL]** for this version) |





<h2 id="2021-09-02-7">7. Position Masking for Improved Layout-Aware Document Understanding
</h2>

Title: [Position Masking for Improved Layout-Aware Document Understanding](https://arxiv.org/abs/2109.00442)

Authors: [Anik Saha](https://arxiv.org/search/cs?searchtype=author&query=Saha%2C+A), [Catherine Finegan-Dollak](https://arxiv.org/search/cs?searchtype=author&query=Finegan-Dollak%2C+C), [Ashish Verma](https://arxiv.org/search/cs?searchtype=author&query=Verma%2C+A)

> Natural language processing for document scans and PDFs has the potential to enormously improve the efficiency of business processes. Layout-aware word embeddings such as LayoutLM have shown promise for classification of and information extraction from such documents. This paper proposes a new pre-training task called that can improve performance of layout-aware word embeddings that incorporate 2-D position embeddings. We compare models pre-trained with only language masking against models pre-trained with both language masking and position masking, and we find that position masking improves performance by over 5% on a form understanding task.

| Comments: | Document Intelligence Workshop at KDD, 2021                  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2109.00442](https://arxiv.org/abs/2109.00442) [cs.CL]** |
|           | (or **[arXiv:2109.00442v1](https://arxiv.org/abs/2109.00442v1) [cs.CL]** for this version) |





<h2 id="2021-09-02-8">8. Survey of Low-Resource Machine Translation
</h2>

Title: [Survey of Low-Resource Machine Translation](https://arxiv.org/abs/2109.00486)

Authors: [Barry Haddow](https://arxiv.org/search/cs?searchtype=author&query=Haddow%2C+B), [Rachel Bawden](https://arxiv.org/search/cs?searchtype=author&query=Bawden%2C+R), [Antonio Valerio Miceli Barone](https://arxiv.org/search/cs?searchtype=author&query=Barone%2C+A+V+M), [Jindřich Helcl](https://arxiv.org/search/cs?searchtype=author&query=Helcl%2C+J), [Alexandra Birch](https://arxiv.org/search/cs?searchtype=author&query=Birch%2C+A)

> We present a survey covering the state of the art in low-resource machine translation. There are currently around 7000 languages spoken in the world and almost all language pairs lack significant resources for training machine translation models. There has been increasing interest in research addressing the challenge of producing useful translation models when very little translated training data is available. We present a high level summary of this topical field and provide an overview of best practices.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2109.00486](https://arxiv.org/abs/2109.00486) [cs.CL]** |
|           | (or **[arXiv:2109.00486v1](https://arxiv.org/abs/2109.00486v1) [cs.CL]** for this version) |








# 2021-09-01

[Return to Index](#Index)



<h2 id="2021-09-01-1">1. SimulLR: Simultaneous Lip Reading Transducer with Attention-Guided Adaptive Memory
</h2>


Title: [SimulLR: Simultaneous Lip Reading Transducer with Attention-Guided Adaptive Memory](https://arxiv.org/abs/2108.13630)

Authors: [Zhijie Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+Z), [Zhou Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+Z), [Haoyuan Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+H), [Jinglin Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+J), [Meng Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+M), [Xingshan Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+X), [Xiaofei He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+X)

> Lip reading, aiming to recognize spoken sentences according to the given video of lip movements without relying on the audio stream, has attracted great interest due to its application in many scenarios. Although prior works that explore lip reading have obtained salient achievements, they are all trained in a non-simultaneous manner where the predictions are generated requiring access to the full video. To breakthrough this constraint, we study the task of simultaneous lip reading and devise SimulLR, a simultaneous lip Reading transducer with attention-guided adaptive memory from three aspects: (1) To address the challenge of monotonic alignments while considering the syntactic structure of the generated sentences under simultaneous setting, we build a transducer-based model and design several effective training strategies including CTC pre-training, model warm-up and curriculum learning to promote the training of the lip reading transducer. (2) To learn better spatio-temporal representations for simultaneous encoder, we construct a truncated 3D convolution and time-restricted self-attention layer to perform the frame-to-frame interaction within a video segment containing fixed number of frames. (3) The history information is always limited due to the storage in real-time scenarios, especially for massive video data. Therefore, we devise a novel attention-guided adaptive memory to organize semantic information of history segments and enhance the visual representations with acceptable computation-aware latency. The experiments show that the SimulLR achieves the translation speedup 9.10× compared with the state-of-the-art non-simultaneous methods, and also obtains competitive results, which indicates the effectiveness of our proposed methods.

| Comments: | ACMMM 2021                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2108.13630](https://arxiv.org/abs/2108.13630) [cs.CV]** |
|           | (or **[arXiv:2108.13630v1](https://arxiv.org/abs/2108.13630v1) [cs.CV]** for this version) |





<h2 id="2021-09-01-2">2. Want To Reduce Labeling Cost? GPT-3 Can Help
</h2>


Title: [Want To Reduce Labeling Cost? GPT-3 Can Help](https://arxiv.org/abs/2108.13487)

Authors: [Shuohang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+S), [Yang Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y), [Yichong Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+Y), [Chenguang Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+C), [Michael Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+M)

> Data annotation is a time-consuming and labor-intensive process for many NLP tasks. Although there exist various methods to produce pseudo data labels, they are often task-specific and require a decent amount of labeled data to start with. Recently, the immense language model GPT-3 with 175 billion parameters has achieved tremendous improvement across many few-shot learning tasks. In this paper, we explore ways to leverage GPT-3 as a low-cost data labeler to train other models. We find that, to make the downstream model achieve the same performance on a variety of NLU and NLG tasks, it costs 50% to 96% less to use labels from GPT-3 than using labels from humans. Furthermore, we propose a novel framework of combining pseudo labels from GPT-3 with human labels, which leads to even better performance with limited labeling budget. These results present a cost-effective data labeling methodology that is generalizable to many practical applications.

| Comments: | Findings of EMNLP 2021, 11 pages                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2108.13487](https://arxiv.org/abs/2108.13487) [cs.CL]** |
|           | (or **[arXiv:2108.13487v1](https://arxiv.org/abs/2108.13487v1) [cs.CL]** for this version) |





<h2 id="2021-09-01-3">3. T3-Vis: a visual analytic framework for Training and fine-Tuning Transformers in NLP
</h2>


Title: [T3-Vis: a visual analytic framework for Training and fine-Tuning Transformers in NLP](https://arxiv.org/abs/2108.13587)

Authors: [Raymond Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+R) (1), [Wen Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+W) (1), [Lanjun Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L) (2), [Hyeju Jang](https://arxiv.org/search/cs?searchtype=author&query=Jang%2C+H) (1), [Giuseppe Carenini](https://arxiv.org/search/cs?searchtype=author&query=Carenini%2C+G) (1) ((1) University of British Columbia, (2) Huawei Cananda Technologies Co. Ltd.)

> Transformers are the dominant architecture in NLP, but their training and fine-tuning is still very challenging. In this paper, we present the design and implementation of a visual analytic framework for assisting researchers in such process, by providing them with valuable insights about the model's intrinsic properties and behaviours. Our framework offers an intuitive overview that allows the user to explore different facets of the model (e.g., hidden states, attention) through interactive visualization, and allows a suite of built-in algorithms that compute the importance of model components and different parts of the input sequence. Case studies and feedback from a user focus group indicate that the framework is useful, and suggest several improvements.

| Comments: | 10 pages, 4 figures, accepted to EMNLP 2021 System Demonstration |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Human-Computer Interaction (cs.HC) |
| Cite as:  | **[arXiv:2108.13587](https://arxiv.org/abs/2108.13587) [cs.CL]** |
|           | (or **[arXiv:2108.13587v1](https://arxiv.org/abs/2108.13587v1) [cs.CL]** for this version) |





<h2 id="2021-09-01-4">4. Enjoy the Salience: Towards Better Transformer-based Faithful Explanations with Word Salience
</h2>


Title: [Enjoy the Salience: Towards Better Transformer-based Faithful Explanations with Word Salience](https://arxiv.org/abs/2108.13759)

Authors: [George Chrysostomou](https://arxiv.org/search/cs?searchtype=author&query=Chrysostomou%2C+G), [Nikolaos Aletras](https://arxiv.org/search/cs?searchtype=author&query=Aletras%2C+N)

> Pretrained transformer-based models such as BERT have demonstrated state-of-the-art predictive performance when adapted into a range of natural language processing tasks. An open problem is how to improve the faithfulness of explanations (rationales) for the predictions of these models. In this paper, we hypothesize that salient information extracted a priori from the training data can complement the task-specific information learned by the model during fine-tuning on a downstream task. In this way, we aim to help BERT not to forget assigning importance to informative input tokens when making predictions by proposing SaLoss; an auxiliary loss function for guiding the multi-head attention mechanism during training to be close to salient information extracted a priori using TextRank. Experiments for explanation faithfulness across five datasets, show that models trained with SaLoss consistently provide more faithful explanations across four different feature attribution methods compared to vanilla BERT. Using the rationales extracted from vanilla BERT and SaLoss models to train inherently faithful classifiers, we further show that the latter result in higher predictive performance in downstream tasks.

| Comments: | EMNLP 2021 Pre-print                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2108.13759](https://arxiv.org/abs/2108.13759) [cs.CL]** |
|           | (or **[arXiv:2108.13759v1](https://arxiv.org/abs/2108.13759v1) [cs.CL]** for this version) |





<h2 id="2021-09-01-5">5. Thermostat: A Large Collection of NLP Model Explanations and Analysis Tools
</h2>


Title: [Thermostat: A Large Collection of NLP Model Explanations and Analysis Tools](https://arxiv.org/abs/2108.13961)

Authors: [Nils Feldhus](https://arxiv.org/search/cs?searchtype=author&query=Feldhus%2C+N), [Robert Schwarzenberg](https://arxiv.org/search/cs?searchtype=author&query=Schwarzenberg%2C+R), [Sebastian Möller](https://arxiv.org/search/cs?searchtype=author&query=Möller%2C+S)

> In the language domain, as in other domains, neural explainability takes an ever more important role, with feature attribution methods on the forefront. Many such methods require considerable computational resources and expert knowledge about implementation details and parameter choices. To facilitate research, we present Thermostat which consists of a large collection of model explanations and accompanying analysis tools. Thermostat allows easy access to over 200k explanations for the decisions of prominent state-of-the-art models spanning across different NLP tasks, generated with multiple explainers. The dataset took over 10k GPU hours (> one year) to compile; compute time that the community now saves. The accompanying software tools allow to analyse explanations instance-wise but also accumulatively on corpus level. Users can investigate and compare models, datasets and explainers without the need to orchestrate implementation details. Thermostat is fully open source, democratizes explainability research in the language domain, circumvents redundant computations and increases comparability and replicability.

| Comments: | Accepted to EMNLP 2021 System Demonstrations                 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2108.13961](https://arxiv.org/abs/2108.13961) [cs.CL]** |
|           | (or **[arXiv:2108.13961v1](https://arxiv.org/abs/2108.13961v1) [cs.CL]** for this version) |



