# Daily arXiv: Machine Translation - November, 2021

# Index


- [2021-11-01](#2021-11-01)
  - [1. Decision Attentive Regularization to Improve Simultaneous Speech Translation Systems](#2021-11-01-1)
  - [2. Analysing the Effect of Masking Length Distribution of MLM: An Evaluation Framework and Case Study on Chinese MRC Datasets](#2021-11-01-2)
  - [3. Analysing the Effect of Masking Length Distribution of MLM: An Evaluation Framework and Case Study on Chinese MRC Datasets](#2021-11-01-3)
  - [4. Batch-Softmax Contrastive Loss for Pairwise Sentence Scoring Tasks](#2021-11-01-4)
  - [5. BERMo: What can BERT learn from ELMo?](#2021-11-01-5)
  - [6. MetaICL: Learning to Learn In Context](#2021-11-01-6)

- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MT-Daily_arXiv-index.md)



# 2021-11-01

[Return to Index](#Index)



<h2 id="2021-11-01-1">1. Decision Attentive Regularization to Improve Simultaneous Speech Translation Systems
</h2>

Title: [Decision Attentive Regularization to Improve Simultaneous Speech Translation Systems](https://arxiv.org/abs/2110.15729)

Authors: [Mohd Abbas Zaidi](https://arxiv.org/search/cs?searchtype=author&query=Zaidi%2C+M+A), [Beomseok Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+B), [Nikhil Kumar Lakumarapu](https://arxiv.org/search/cs?searchtype=author&query=Lakumarapu%2C+N+K), [Sangha Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+S), [Chanwoo Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+C)

> Simultaneous Speech-to-text Translation (SimulST) systems translate source speech in tandem with the speaker using partial input. Recent works have tried to leverage the text translation task to improve the performance of Speech Translation (ST) in the offline domain. Motivated by these improvements, we propose to add Decision Attentive Regularization (DAR) to Monotonic Multihead Attention (MMA) based SimulST systems. DAR improves the read/write decisions for speech using the Simultaneous text Translation (SimulMT) task. We also extend several techniques from the offline domain to the SimulST task. Our proposed system achieves significant performance improvements for the MuST-C English-German (EnDe) SimulST task, where we provide an average BLUE score improvement of around 4.57 points or 34.17% across different latencies. Further, the latency-quality tradeoffs establish that the proposed model achieves better results compared to the baseline.

| Comments: | 5 pages, 3 figures, 1 table                                  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Sound (cs.SD)**; Computation and Language (cs.CL); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2110.15729](https://arxiv.org/abs/2110.15729) [cs.SD]** |
|           | (or **[arXiv:2110.15729v1](https://arxiv.org/abs/2110.15729v1) [cs.SD]** for this version) |





<h2 id="2021-11-01-2">2. Analysing the Effect of Masking Length Distribution of MLM: An Evaluation Framework and Case Study on Chinese MRC Datasets
</h2>

Title: [Analysing the Effect of Masking Length Distribution of MLM: An Evaluation Framework and Case Study on Chinese MRC Datasets](https://arxiv.org/abs/2110.15712)

Authors: [Changchang. Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+C), [Shaobo. Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+S)

> Machine reading comprehension (MRC) is a challenging natural language processing (NLP) task. Recently, the emergence of pre-trained models (PTM) has brought this research field into a new era, in which the training objective plays a key role. The masked language model (MLM) is a self-supervised training objective that widely used in various PTMs. With the development of training objectives, many variants of MLM have been proposed, such as whole word masking, entity masking, phrase masking, span masking, and so on. In different MLM, the length of the masked tokens is different. Similarly, in different machine reading comprehension tasks, the length of the answer is also different, and the answer is often a word, phrase, or sentence. Thus, in MRC tasks with different answer lengths, whether the length of MLM is related to performance is a question worth studying. If this hypothesis is true, it can guide us how to pre-train the MLM model with a relatively suitable mask length distribution for MRC task. In this paper, we try to uncover how much of MLM's success in the machine reading comprehension tasks comes from the correlation between masking length distribution and answer length in MRC dataset. In order to address this issue, herein, (1) we propose four MRC tasks with different answer length distributions, namely short span extraction task, long span extraction task, short multiple-choice cloze task, long multiple-choice cloze task; (2) four Chinese MRC datasets are created for these tasks; (3) we also have pre-trained four masked language models according to the answer length distributions of these datasets; (4) ablation experiments are conducted on the datasets to verify our hypothesis. The experimental results demonstrate that our hypothesis is true.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2110.15712](https://arxiv.org/abs/2110.15712) [cs.CL]** |
|           | (or **[arXiv:2110.15712v1](https://arxiv.org/abs/2110.15712v1) [cs.CL]** for this version) |





<h2 id="2021-11-01-3">3. Building the Language Resource for a Cebuano-Filipino Neural Machine Translation System
</h2>

Title: [Building the Language Resource for a Cebuano-Filipino Neural Machine Translation System](https://arxiv.org/abs/2110.15716)

Authors: [Kristine Mae Adlaon](https://arxiv.org/search/cs?searchtype=author&query=Adlaon%2C+K+M), [Nelson Marcos](https://arxiv.org/search/cs?searchtype=author&query=Marcos%2C+N)

> Parallel corpus is a critical resource in machine learning-based translation. The task of collecting, extracting, and aligning texts in order to build an acceptable corpus for doing the translation is very tedious most especially for low-resource languages. In this paper, we present the efforts made to build a parallel corpus for Cebuano and Filipino from two different domains: biblical texts and the web. For the biblical resource, subword unit translation for verbs and copy-able approach for nouns were applied to correct inconsistencies in the translation. This correction mechanism was applied as a preprocessing technique. On the other hand, for Wikipedia being the main web resource, commonly occurring topic segments were extracted from both the source and the target languages. These observed topic segments are unique in 4 different categories. The identification of these topic segments may be used for the automatic extraction of sentences. A Recurrent Neural Network was used to implement the translation using OpenNMT sequence modeling tool in TensorFlow. The two different corpora were then evaluated by using them as two separate inputs in the neural network. Results have shown a difference in BLEU scores in both corpora.

| Comments:    | Published in the Proceedings of the 2019 3rd International Conference on Natural Language Processing and Information Retrieval. arXiv admin note: substantial text overlap with [arXiv:1902.07250](https://arxiv.org/abs/1902.07250) |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**                         |
| ACM classes: | A.2                                                          |
| DOI:         | [10.1145/3342827.3342833](https://arxiv.org/ct?url=https%3A%2F%2Fdx.doi.org%2F10.1145%2F3342827.3342833&v=b6a91262) |
| Cite as:     | **[arXiv:2110.15716](https://arxiv.org/abs/2110.15716) [cs.CL]** |
|              | (or **[arXiv:2110.15716v1](https://arxiv.org/abs/2110.15716v1) [cs.CL]** for this version) |





<h2 id="2021-11-01-4">4. Batch-Softmax Contrastive Loss for Pairwise Sentence Scoring Tasks
</h2>

Title: [Batch-Softmax Contrastive Loss for Pairwise Sentence Scoring Tasks](https://arxiv.org/abs/2110.15725)

Authors: [Anton Chernyavskiy](https://arxiv.org/search/cs?searchtype=author&query=Chernyavskiy%2C+A), [Dmitry Ilvovsky](https://arxiv.org/search/cs?searchtype=author&query=Ilvovsky%2C+D), [Pavel Kalinin](https://arxiv.org/search/cs?searchtype=author&query=Kalinin%2C+P), [Preslav Nakov](https://arxiv.org/search/cs?searchtype=author&query=Nakov%2C+P)

> The use of contrastive loss for representation learning has become prominent in computer vision, and it is now getting attention in Natural Language Processing (NLP). Here, we explore the idea of using a batch-softmax contrastive loss when fine-tuning large-scale pre-trained transformer models to learn better task-specific sentence embeddings for pairwise sentence scoring tasks. We introduce and study a number of variations in the calculation of the loss as well as in the overall training procedure; in particular, we find that data shuffling can be quite important. Our experimental results show sizable improvements on a number of datasets and pairwise sentence scoring tasks including classification, ranking, and regression. Finally, we offer detailed analysis and discussion, which should be useful for researchers aiming to explore the utility of contrastive loss in NLP.

| Comments:    | batch-softmax contrastive loss, pairwise sentence scoring, classification, ranking, and regression |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Information Retrieval (cs.IR); Machine Learning (cs.LG); Neural and Evolutionary Computing (cs.NE) |
| MSC classes: | 68T50                                                        |
| ACM classes: | F.2.2; I.2.7                                                 |
| Cite as:     | **[arXiv:2110.15725](https://arxiv.org/abs/2110.15725) [cs.CL]** |
|              | (or **[arXiv:2110.15725v1](https://arxiv.org/abs/2110.15725v1) [cs.CL]** for this version) |





<h2 id="2021-11-01-5">5. BERMo: What can BERT learn from ELMo?
</h2>

Title: [BERMo: What can BERT learn from ELMo?](https://arxiv.org/abs/2110.15802)

Authors: [Sangamesh Kodge](https://arxiv.org/search/cs?searchtype=author&query=Kodge%2C+S), [Kaushik Roy](https://arxiv.org/search/cs?searchtype=author&query=Roy%2C+K)

> We propose BERMo, an architectural modification to BERT, which makes predictions based on a hierarchy of surface, syntactic and semantic language features. We use linear combination scheme proposed in Embeddings from Language Models (ELMo) to combine the scaled internal representations from different network depths. Our approach has two-fold benefits: (1) improved gradient flow for the downstream task as every layer has a direct connection to the gradients of the loss function and (2) increased representative power as the model no longer needs to copy the features learned in the shallower layer which are necessary for the downstream task. Further, our model has a negligible parameter overhead as there is a single scalar parameter associated with each layer in the network. Experiments on the probing task from SentEval dataset show that our model performs up to 4.65% better in accuracy than the baseline with an average improvement of 2.67% on the semantic tasks. When subject to compression techniques, we find that our model enables stable pruning for compressing small datasets like SST-2, where the BERT model commonly diverges. We observe that our approach converges 1.67× and 1.15× faster than the baseline on MNLI and QQP tasks from GLUE dataset. Moreover, our results show that our approach can obtain better parameter efficiency for penalty based pruning approaches on QQP task.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2110.15802](https://arxiv.org/abs/2110.15802) [cs.CL]** |
|           | (or **[arXiv:2110.15802v1](https://arxiv.org/abs/2110.15802v1) [cs.CL]** for this version) |





<h2 id="2021-11-01-6">6. MetaICL: Learning to Learn In Context
</h2>

Title: [MetaICL: Learning to Learn In Context](https://arxiv.org/abs/2110.15943)

Authors: [Sewon Min](https://arxiv.org/search/cs?searchtype=author&query=Min%2C+S), [Mike Lewis](https://arxiv.org/search/cs?searchtype=author&query=Lewis%2C+M), [Luke Zettlemoyer](https://arxiv.org/search/cs?searchtype=author&query=Zettlemoyer%2C+L), [Hannaneh Hajishirzi](https://arxiv.org/search/cs?searchtype=author&query=Hajishirzi%2C+H)

> We introduce MetaICL (Meta-training for In-Context Learning), a new meta-training framework for few-shot learning where a pretrained language model is tuned to do in-context learn-ing on a large set of training tasks. This meta-training enables the model to more effectively learn a new task in context at test time, by simply conditioning on a few training examples with no parameter updates or task-specific templates. We experiment on a large, diverse collection of tasks consisting of 142 NLP datasets including classification, question answering, natural language inference, paraphrase detection and more, across seven different meta-training/target splits. MetaICL outperforms a range of baselines including in-context learning without meta-training and multi-task learning followed by zero-shot transfer. We find that the gains are particularly significant for target tasks that have domain shifts from the meta-training tasks, and that using a diverse set of the meta-training tasks is key to improvements. We also show that MetaICL approaches (and sometimes beats) the performance of models fully finetuned on the target task training data, and outperforms much bigger models with nearly 8x parameters.

| Comments: | 18 pages (9 pages for the main paper, 9 pages for references and appendices). 1 figure. Code available at [this https URL](https://github.com/facebookresearch/MetaICL) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2110.15943](https://arxiv.org/abs/2110.15943) [cs.CL]** |
|           | (or **[arXiv:2110.15943v1](https://arxiv.org/abs/2110.15943v1) [cs.CL]** for this version) |



