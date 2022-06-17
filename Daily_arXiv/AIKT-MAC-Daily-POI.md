# MA C.'s Daily Paper Of Interest - June b., 2022

# Index

- [2022-06-17](#2022-06-17)
  - [1. How Adults Understand What Young Children Say](#2022-06-17-1)

  - [2. Alexa Teacher Model: Pretraining and Distilling Multi-Billion-Parameter Encoders for Natural Language Understanding Systems](#2022-06-17-2)
  
  - [3. TransDrift: Modeling Word-Embedding Drift using Transformer](#2022-06-17-3)
  
  - [4. Self-Generated In-Context Learning: Leveraging Auto-regressive Language Models as a Demonstration Generator](#2022-06-17-4)
  
  - [5. Deep Learning Architecture for Automatic Essay Scoring](#2022-06-17-5)
  
- [2022-06-16](#2022-06-16)
  - [1. Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone](#2022-06-16-1)

  - [2. A Unified Sequence Interface for Vision Tasks](#2022-06-16-2)

  - [3. Prefix Language Models are Unified Modal Learners](#2022-06-16-3)

  - [4. Human Heuristics for AI-Generated Language Are Flawed](#2022-06-16-4)

  - [5. MPI: Evaluating and Inducing Personality in Pre-trained Language Models](#2022-06-16-5)

  - [6. Emergent Abilities of Large Language Models](#2022-06-16-6)

- [2022-06-15](#2022-06-15)
  - [1. LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning](#2022-06-15-1)

- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



# 2022-06-17

[Return to Index](#Index)



<h2 id="2022-06-17-1">1. How Adults Understand What Young Children Say
</h2>

Title: [How Adults Understand What Young Children Say](https://arxiv.org/abs/2206.07807)

Authors: [Stephan C. Meylan](https://arxiv.org/search/cs?searchtype=author&query=Meylan%2C+S+C), [Ruthe Foushee](https://arxiv.org/search/cs?searchtype=author&query=Foushee%2C+R), [Nicole H. Wong](https://arxiv.org/search/cs?searchtype=author&query=Wong%2C+N+H), [Elika Bergelson](https://arxiv.org/search/cs?searchtype=author&query=Bergelson%2C+E), [Roger P. Levy](https://arxiv.org/search/cs?searchtype=author&query=Levy%2C+R+P)

> Children's early speech often bears little resemblance to adult speech in form or content, and yet caregivers often find meaning in young children's utterances. Precisely how caregivers are able to do this remains poorly understood. We propose that successful early communication (an essential building block of language development) relies not just on children's growing linguistic knowledge, but also on adults' sophisticated inferences. These inferences, we further propose, are optimized for fine-grained details of how children speak. We evaluate these ideas using a set of candidate computational models of spoken word recognition based on deep learning and Bayesian inference, which instantiate competing hypotheses regarding the information sources used by adults to understand children. We find that the best-performing models (evaluated on datasets of adult interpretations of child speech) are those that have strong prior expectations about what children are likely to want to communicate, rather than the actual phonetic contents of what children say. We further find that adults' behavior is best characterized as well-tuned to specific children: the more closely a word recognition model is tuned to the particulars of an individual child's actual linguistic behavior, the better it predicts adults' inferences about what the child has said. These results offer a comprehensive investigation into the role of caregivers as child-directed listeners, with broader consequences for theories of language acquisition.

| Comments: | 19 pages, 6 figures, 2 tables                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2206.07807](https://arxiv.org/abs/2206.07807) [cs.CL]** |
|           | (or **[arXiv:2206.07807v1](https://arxiv.org/abs/2206.07807v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.07807Focus to learn more |





<h2 id="2022-06-17-2">2. Alexa Teacher Model: Pretraining and Distilling Multi-Billion-Parameter Encoders for Natural Language Understanding Systems
</h2>

Title: [Alexa Teacher Model: Pretraining and Distilling Multi-Billion-Parameter Encoders for Natural Language Understanding Systems](https://arxiv.org/abs/2206.07808)

Authors: [Jack FitzGerald](https://arxiv.org/search/cs?searchtype=author&query=FitzGerald%2C+J), [Shankar Ananthakrishnan](https://arxiv.org/search/cs?searchtype=author&query=Ananthakrishnan%2C+S), [Konstantine Arkoudas](https://arxiv.org/search/cs?searchtype=author&query=Arkoudas%2C+K), [Davide Bernardi](https://arxiv.org/search/cs?searchtype=author&query=Bernardi%2C+D), [Abhishek Bhagia](https://arxiv.org/search/cs?searchtype=author&query=Bhagia%2C+A), [Claudio Delli Bovi](https://arxiv.org/search/cs?searchtype=author&query=Bovi%2C+C+D), [Jin Cao](https://arxiv.org/search/cs?searchtype=author&query=Cao%2C+J), [Rakesh Chada](https://arxiv.org/search/cs?searchtype=author&query=Chada%2C+R), [Amit Chauhan](https://arxiv.org/search/cs?searchtype=author&query=Chauhan%2C+A), [Luoxin Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+L), [Anurag Dwarakanath](https://arxiv.org/search/cs?searchtype=author&query=Dwarakanath%2C+A), [Satyam Dwivedi](https://arxiv.org/search/cs?searchtype=author&query=Dwivedi%2C+S), [Turan Gojayev](https://arxiv.org/search/cs?searchtype=author&query=Gojayev%2C+T), [Karthik Gopalakrishnan](https://arxiv.org/search/cs?searchtype=author&query=Gopalakrishnan%2C+K), [Thomas Gueudre](https://arxiv.org/search/cs?searchtype=author&query=Gueudre%2C+T), [Dilek Hakkani-Tur](https://arxiv.org/search/cs?searchtype=author&query=Hakkani-Tur%2C+D), [Wael Hamza](https://arxiv.org/search/cs?searchtype=author&query=Hamza%2C+W), [Jonathan Hueser](https://arxiv.org/search/cs?searchtype=author&query=Hueser%2C+J), [Kevin Martin Jose](https://arxiv.org/search/cs?searchtype=author&query=Jose%2C+K+M), [Haidar Khan](https://arxiv.org/search/cs?searchtype=author&query=Khan%2C+H), [Beiye Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+B), [Jianhua Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+J), [Alessandro Manzotti](https://arxiv.org/search/cs?searchtype=author&query=Manzotti%2C+A), [Pradeep Natarajan](https://arxiv.org/search/cs?searchtype=author&query=Natarajan%2C+P), [Karolina Owczarzak](https://arxiv.org/search/cs?searchtype=author&query=Owczarzak%2C+K), [Gokmen Oz](https://arxiv.org/search/cs?searchtype=author&query=Oz%2C+G), [Enrico Palumbo](https://arxiv.org/search/cs?searchtype=author&query=Palumbo%2C+E), [Charith Peris](https://arxiv.org/search/cs?searchtype=author&query=Peris%2C+C), [Chandana Satya Prakash](https://arxiv.org/search/cs?searchtype=author&query=Prakash%2C+C+S), [Stephen Rawls](https://arxiv.org/search/cs?searchtype=author&query=Rawls%2C+S), [Andy Rosenbaum](https://arxiv.org/search/cs?searchtype=author&query=Rosenbaum%2C+A), [Anjali Shenoy](https://arxiv.org/search/cs?searchtype=author&query=Shenoy%2C+A), [Saleh Soltan](https://arxiv.org/search/cs?searchtype=author&query=Soltan%2C+S), [Mukund Harakere Sridhar](https://arxiv.org/search/cs?searchtype=author&query=Sridhar%2C+M+H), [Liz Tan](https://arxiv.org/search/cs?searchtype=author&query=Tan%2C+L), [Fabian Triefenbach](https://arxiv.org/search/cs?searchtype=author&query=Triefenbach%2C+F), [Pan Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+P), [Haiyang Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+H), [Shuai Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+S), [Gokhan Tur](https://arxiv.org/search/cs?searchtype=author&query=Tur%2C+G), [Prem Natarajan](https://arxiv.org/search/cs?searchtype=author&query=Natarajan%2C+P)

> We present results from a large-scale experiment on pretraining encoders with non-embedding parameter counts ranging from 700M to 9.3B, their subsequent distillation into smaller models ranging from 17M-170M parameters, and their application to the Natural Language Understanding (NLU) component of a virtual assistant system. Though we train using 70% spoken-form data, our teacher models perform comparably to XLM-R and mT5 when evaluated on the written-form Cross-lingual Natural Language Inference (XNLI) corpus. We perform a second stage of pretraining on our teacher models using in-domain data from our system, improving error rates by 3.86% relative for intent classification and 7.01% relative for slot filling. We find that even a 170M-parameter model distilled from our Stage 2 teacher model has 2.88% better intent classification and 7.69% better slot filling error rates when compared to the 2.3B-parameter teacher trained only on public data (Stage 1), emphasizing the importance of in-domain data for pretraining. When evaluated offline using labeled NLU data, our 17M-parameter Stage 2 distilled model outperforms both XLM-R Base (85M params) and DistillBERT (42M params) by 4.23% to 6.14%, respectively. Finally, we present results from a full virtual assistant experimentation platform, where we find that models trained using our pretraining and distillation pipeline outperform models distilled from 85M-parameter teachers by 3.74%-4.91% on an automatic measurement of full-system user dissatisfaction.

| Comments:          | KDD 2022                                                     |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| ACM classes:       | I.2.7                                                        |
| Cite as:           | **[arXiv:2206.07808](https://arxiv.org/abs/2206.07808) [cs.CL]** |
|                    | (or **[arXiv:2206.07808v1](https://arxiv.org/abs/2206.07808v1) [cs.CL]** for this version) |
|                    | https://doi.org/10.48550/arXiv.2206.07808Focus to learn more |
| Journal reference: | Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '22), August 14-18, 2022, Washington, DC, USA |
| Related DOI:       | https://doi.org/10.1145/3534678.3539173Focus to learn more   |





<h2 id="2022-06-17-3">3. TransDrift: Modeling Word-Embedding Drift using Transformer
</h2>

Title: [TransDrift: Modeling Word-Embedding Drift using Transformer](https://arxiv.org/abs/2206.08081)

Authors: [Nishtha Madaan](https://arxiv.org/search/cs?searchtype=author&query=Madaan%2C+N), [Prateek Chaudhury](https://arxiv.org/search/cs?searchtype=author&query=Chaudhury%2C+P), [Nishant Kumar](https://arxiv.org/search/cs?searchtype=author&query=Kumar%2C+N), [Srikanta Bedathur](https://arxiv.org/search/cs?searchtype=author&query=Bedathur%2C+S)

> In modern NLP applications, word embeddings are a crucial backbone that can be readily shared across a number of tasks. However as the text distributions change and word semantics evolve over time, the downstream applications using the embeddings can suffer if the word representations do not conform to the data drift. Thus, maintaining word embeddings to be consistent with the underlying data distribution is a key problem. In this work, we tackle this problem and propose TransDrift, a transformer-based prediction model for word embeddings. Leveraging the flexibility of transformer, our model accurately learns the dynamics of the embedding drift and predicts the future embedding. In experiments, we compare with existing methods and show that our model makes significantly more accurate predictions of the word embedding than the baselines. Crucially, by applying the predicted embeddings as a backbone for downstream classification tasks, we show that our embeddings lead to superior performance compared to the previous methods.

| Comments: | 10 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2206.08081](https://arxiv.org/abs/2206.08081) [cs.CL]** |
|           | (or **[arXiv:2206.08081v1](https://arxiv.org/abs/2206.08081v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.08081Focus to learn more |





<h2 id="2022-06-17-4">4. Self-Generated In-Context Learning: Leveraging Auto-regressive Language Models as a Demonstration Generator
</h2>

Title: [Self-Generated In-Context Learning: Leveraging Auto-regressive Language Models as a Demonstration Generator](https://arxiv.org/abs/2206.08082)

Authors: [Hyuhng Joon Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+H+J), [Hyunsoo Cho](https://arxiv.org/search/cs?searchtype=author&query=Cho%2C+H), [Junyeob Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+J), [Taeuk Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+T), [Kang Min Yoo](https://arxiv.org/search/cs?searchtype=author&query=Yoo%2C+K+M), [Sang-goo Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+S)

> Large-scale pre-trained language models (PLMs) are well-known for being capable of solving a task simply by conditioning a few input-label pairs dubbed demonstrations on a prompt without being explicitly tuned for the desired downstream task. Such a process (i.e., in-context learning), however, naturally leads to high reliance on the demonstrations which are usually selected from external datasets. In this paper, we propose self-generated in-context learning (SG-ICL), which generates demonstrations for in-context learning from PLM itself to minimize the reliance on the external demonstration. We conduct experiments on four different text classification tasks and show SG-ICL significantly outperforms zero-shot learning and is generally worth approximately 0.6 gold training samples. Moreover, our generated demonstrations show more consistent performance with low variance compared to randomly selected demonstrations from the training dataset.

| Comments: | NAACL 2022 Workshop on Large-scale Pre-trained Language Models |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2206.08082](https://arxiv.org/abs/2206.08082) [cs.CL]** |
|           | (or **[arXiv:2206.08082v1](https://arxiv.org/abs/2206.08082v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.08082Focus to learn more |





<h2 id="2022-06-17-5">5. Deep Learning Architecture for Automatic Essay Scoring
</h2>

Title: [Deep Learning Architecture for Automatic Essay Scoring](https://arxiv.org/abs/2206.08232)

Authors: [Tsegaye Misikir Tashu](https://arxiv.org/search/cs?searchtype=author&query=Tashu%2C+T+M), [Chandresh Kumar Maurya](https://arxiv.org/search/cs?searchtype=author&query=Maurya%2C+C+K), [Tomas Horvath](https://arxiv.org/search/cs?searchtype=author&query=Horvath%2C+T)

> Automatic evaluation of essay (AES) and also called automatic essay scoring has become a severe problem due to the rise of online learning and evaluation platforms such as Coursera, Udemy, Khan academy, and so on. Researchers have recently proposed many techniques for automatic evaluation. However, many of these techniques use hand-crafted features and thus are limited from the feature representation point of view. Deep learning has emerged as a new paradigm in machine learning which can exploit the vast data and identify the features useful for essay evaluation. To this end, we propose a novel architecture based on recurrent networks (RNN) and convolution neural network (CNN). In the proposed architecture, the multichannel convolutional layer learns and captures the contextual features of the word n-gram from the word embedding vectors and the essential semantic concepts to form the feature vector at essay level using max-pooling operation. A variant of RNN called Bi-gated recurrent unit (BGRU) is used to access both previous and subsequent contextual representations. The experiment was carried out on eight data sets available on Kaggle for the task of AES. The experimental results show that our proposed system achieves significantly higher grading accuracy than other deep learning-based AES systems and also other state-of-the-art AES systems.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.08232](https://arxiv.org/abs/2206.08232) [cs.CL]** |
|           | (or **[arXiv:2206.08232v1](https://arxiv.org/abs/2206.08232v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.08232Focus to learn more |







# 2022-06-16

[Return to Index](#Index)



<h2 id="2022-06-16-1">1. Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone
</h2>

Title: [Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone](https://arxiv.org/abs/2206.07643)

Authors: [Zi-Yi Dou](https://arxiv.org/search/cs?searchtype=author&query=Dou%2C+Z), [Aishwarya Kamath](https://arxiv.org/search/cs?searchtype=author&query=Kamath%2C+A), [Zhe Gan](https://arxiv.org/search/cs?searchtype=author&query=Gan%2C+Z), [Pengchuan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+P), [Jianfeng Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+J), [Linjie Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L), [Zicheng Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Z), [Ce Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+C), [Yann LeCun](https://arxiv.org/search/cs?searchtype=author&query=LeCun%2C+Y), [Nanyun Peng](https://arxiv.org/search/cs?searchtype=author&query=Peng%2C+N), [Jianfeng Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+J), [Lijuan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L)

> Vision-language (VL) pre-training has recently received considerable attention. However, most existing end-to-end pre-training approaches either only aim to tackle VL tasks such as image-text retrieval, visual question answering (VQA) and image captioning that test high-level understanding of images, or only target region-level understanding for tasks such as phrase grounding and object detection. We present FIBER (Fusion-In-the-Backbone-based transformER), a new VL model architecture that can seamlessly handle both these types of tasks. Instead of having dedicated transformer layers for fusion after the uni-modal backbones, FIBER pushes multimodal fusion deep into the model by inserting cross-attention into the image and text backbones, bringing gains in terms of memory and performance. In addition, unlike previous work that is either only pre-trained on image-text data or on fine-grained data with box-level annotations, we present a two-stage pre-training strategy that uses both these kinds of data efficiently: (i) coarse-grained pre-training based on image-text data; followed by (ii) fine-grained pre-training based on image-text-box data. We conduct comprehensive experiments on a wide range of VL tasks, ranging from VQA, image captioning, and retrieval, to phrase grounding, referring expression comprehension, and object detection. Using deep multimodal fusion coupled with the two-stage pre-training, FIBER provides consistent performance improvements over strong baselines across all tasks, often outperforming methods using magnitudes more data. Code is available at [this https URL](https://github.com/microsoft/FIBER).

| Comments: | Project Website: [this https URL](https://ashkamath.github.io/FIBER_page) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2206.07643](https://arxiv.org/abs/2206.07643) [cs.CV]** |
|           | (or **[arXiv:2206.07643v1](https://arxiv.org/abs/2206.07643v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.07643Focus to learn more |





<h2 id="2022-06-16-2">2. A Unified Sequence Interface for Vision Tasks
</h2>

Title: [A Unified Sequence Interface for Vision Tasks](https://arxiv.org/abs/2206.07669)

Authors: [Ting Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+T), [Saurabh Saxena](https://arxiv.org/search/cs?searchtype=author&query=Saxena%2C+S), [Lala Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L), [Tsung-Yi Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+T), [David J. Fleet](https://arxiv.org/search/cs?searchtype=author&query=Fleet%2C+D+J), [Geoffrey Hinton](https://arxiv.org/search/cs?searchtype=author&query=Hinton%2C+G)

> While language tasks are naturally expressed in a single, unified, modeling framework, i.e., generating sequences of tokens, this has not been the case in computer vision. As a result, there is a proliferation of distinct architectures and loss functions for different vision tasks. In this work we show that a diverse set of "core" computer vision tasks can also be unified if formulated in terms of a shared pixel-to-sequence interface. We focus on four tasks, namely, object detection, instance segmentation, keypoint detection, and image captioning, all with diverse types of outputs, e.g., bounding boxes or dense masks. Despite that, by formulating the output of each task as a sequence of discrete tokens with a unified interface, we show that one can train a neural network with a single model architecture and loss function on all these tasks, with no task-specific customization. To solve a specific task, we use a short prompt as task description, and the sequence output adapts to the prompt so it can produce task-specific output. We show that such a model can achieve competitive performance compared to well-established task-specific models.

| Comments: | The first three authors contributed equally                  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2206.07669](https://arxiv.org/abs/2206.07669) [cs.CV]** |
|           | (or **[arXiv:2206.07669v1](https://arxiv.org/abs/2206.07669v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.07669Focus to learn more |





<h2 id="2022-06-16-3">3. Prefix Language Models are Unified Modal Learners
</h2>

Title: [Prefix Language Models are Unified Modal Learners](https://arxiv.org/abs/2206.07699)

Authors: [Shizhe Diao](https://arxiv.org/search/cs?searchtype=author&query=Diao%2C+S), [Wangchunshu Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+W), [Xinsong Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+X), [Jiawei Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+J)

> With the success of vision-language pre-training, we have witnessed the state-of-the-art has been pushed on multi-modal understanding and generation. However, the current pre-training paradigm is either incapable of targeting all modalities at once (e.g., text generation and image generation), or requires multi-fold well-designed tasks which significantly limits the scalability. We demonstrate that a unified modal model could be learned with a prefix language modeling objective upon text and image sequences. Thanks to the simple but powerful pre-training paradigm, our proposed model, DaVinci, is simple to train, scalable to huge data, and adaptable to a variety of downstream tasks across modalities (language / vision / vision+language), types (understanding / generation) and settings (e.g., zero-shot, fine-tuning, linear evaluation) with a single unified architecture. DaVinci achieves the competitive performance on a wide range of 26 understanding / generation tasks, and outperforms previous unified vision-language models on most tasks, including ImageNet classification (+1.6%), VQAv2 (+1.4%), COCO caption generation (BLEU@4 +1.1%, CIDEr +1.5%) and COCO image generation (IS +0.9%, FID -1.0%), at the comparable model and data scale. Furthermore, we offer a well-defined benchmark for future research by reporting the performance on different scales of the pre-training dataset on a heterogeneous and wide distribution coverage. Our results establish new, stronger baselines for future comparisons at different data scales and shed light on the difficulties of comparing VLP models more generally.

| Comments: | 22 pages, 3 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2206.07699](https://arxiv.org/abs/2206.07699) [cs.CV]** |
|           | (or **[arXiv:2206.07699v1](https://arxiv.org/abs/2206.07699v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.07699Focus to learn more |





<h2 id="2022-06-16-4">4. Human Heuristics for AI-Generated Language Are Flawed
</h2>

Title: [Human Heuristics for AI-Generated Language Are Flawed](https://arxiv.org/abs/2206.07271)

Authors: [Maurice Jakesch](https://arxiv.org/search/cs?searchtype=author&query=Jakesch%2C+M), [Jeffrey Hancock](https://arxiv.org/search/cs?searchtype=author&query=Hancock%2C+J), [Mor Naaman](https://arxiv.org/search/cs?searchtype=author&query=Naaman%2C+M)

> Human communication is increasingly intermixed with language generated by AI. Across chat, email, and social media, AI systems produce smart replies, autocompletes, and translations. AI-generated language is often not identified as such but poses as human language, raising concerns about novel forms of deception and manipulation. Here, we study how humans discern whether one of the most personal and consequential forms of language - a self-presentation - was generated by AI. Across six experiments, participants (N = 4,650) tried to identify self-presentations generated by state-of-the-art language models. Across professional, hospitality, and romantic settings, we find that humans are unable to identify AI-generated self-presentations. Combining qualitative analyses with language feature engineering, we find that human judgments of AI-generated language are handicapped by intuitive but flawed heuristics such as associating first-person pronouns, authentic words, or family topics with humanity. We show that these heuristics make human judgment of generated language predictable and manipulable, allowing AI systems to produce language perceived as more human than human. We conclude by discussing solutions - such as AI accents or fair use policies - to reduce the deceptive potential of generated language, limiting the subversion of human intuition.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Computers and Society (cs.CY); Human-Computer Interaction (cs.HC) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.07271](https://arxiv.org/abs/2206.07271) [cs.CL]** |
|           | (or **[arXiv:2206.07271v1](https://arxiv.org/abs/2206.07271v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.07271Focus to learn more |





<h2 id="2022-06-16-5">5. MPI: Evaluating and Inducing Personality in Pre-trained Language Models
</h2>

Title: [MPI: Evaluating and Inducing Personality in Pre-trained Language Models](https://arxiv.org/abs/2206.07550)

Authors: [Guangyuan Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+G), [Manjie Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+M), [Song-Chun Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+S), [Wenjuan Han](https://arxiv.org/search/cs?searchtype=author&query=Han%2C+W), [Chi Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+C), [Yixin Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+Y)

> Originated as a philosophical quest, personality discerns how individuals differ from each other in terms of thinking, feeling, and behaving. Towards building social machines that work with humans on a daily basis, we are motivated to ask: (1) Do existing pre-trained language models possess personality, akin to their human counterpart? If so, (2) how can we evaluate them? Further, given this evaluation framework, (3) how can we induce a certain personality in a fully controllable fashion? To tackle these three questions, we propose the Machine Personality Inventory (MPI) dataset for evaluating the machine personality; MPI follows standardized personality tests, built upon the Big Five Personality Factors (Big Five) theory and personality assessment inventories. By evaluating models with MPI, we provide the first piece of evidence showing the existence of personality in pre-trained language models. We further devise a Chain Prompting method to induce the language model with a specific personality in a controllable manner, capable of producing diversified behaviors. We hope to shed light on future studies by adopting personality as the essential psychological guidance for various downstream tasks, building more human-like and in situ dialogue agents.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.07550](https://arxiv.org/abs/2206.07550) [cs.CL]** |
|           | (or **[arXiv:2206.07550v1](https://arxiv.org/abs/2206.07550v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.07550Focus to learn more |





<h2 id="2022-06-16-6">6. Emergent Abilities of Large Language Models
</h2>

Title: [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682)

Authors: [Jason Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+J), [Yi Tay](https://arxiv.org/search/cs?searchtype=author&query=Tay%2C+Y), [Rishi Bommasani](https://arxiv.org/search/cs?searchtype=author&query=Bommasani%2C+R), [Colin Raffel](https://arxiv.org/search/cs?searchtype=author&query=Raffel%2C+C), [Barret Zoph](https://arxiv.org/search/cs?searchtype=author&query=Zoph%2C+B), [Sebastian Borgeaud](https://arxiv.org/search/cs?searchtype=author&query=Borgeaud%2C+S), [Dani Yogatama](https://arxiv.org/search/cs?searchtype=author&query=Yogatama%2C+D), [Maarten Bosma](https://arxiv.org/search/cs?searchtype=author&query=Bosma%2C+M), [Denny Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+D), [Donald Metzler](https://arxiv.org/search/cs?searchtype=author&query=Metzler%2C+D), [Ed H. Chi](https://arxiv.org/search/cs?searchtype=author&query=Chi%2C+E+H), [Tatsunori Hashimoto](https://arxiv.org/search/cs?searchtype=author&query=Hashimoto%2C+T), [Oriol Vinyals](https://arxiv.org/search/cs?searchtype=author&query=Vinyals%2C+O), [Percy Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+P), [Jeff Dean](https://arxiv.org/search/cs?searchtype=author&query=Dean%2C+J), [William Fedus](https://arxiv.org/search/cs?searchtype=author&query=Fedus%2C+W)

> Scaling up language models has been shown to predictably improve performance and sample efficiency on a wide range of downstream tasks. This paper instead discusses an unpredictable phenomenon that we refer to as emergent abilities of large language models. We consider an ability to be emergent if it is not present in smaller models but is present in larger models. Thus, emergent abilities cannot be predicted simply by extrapolating the performance of smaller models. The existence of such emergence implies that additional scaling could further expand the range of capabilities of language models.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.07682](https://arxiv.org/abs/2206.07682) [cs.CL]** |
|           | (or **[arXiv:2206.07682v1](https://arxiv.org/abs/2206.07682v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.07682Focus to learn more |






# 2022-06-15

[Return to Index](#Index)



<h2 id="2022-06-15-1">1. LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning
</h2>

Title: [LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning](https://arxiv.org/abs/2206.06522)

Authors: [Yi-Lin Sung](https://arxiv.org/search/cs?searchtype=author&query=Sung%2C+Y), [Jaemin Cho](https://arxiv.org/search/cs?searchtype=author&query=Cho%2C+J), [Mohit Bansal](https://arxiv.org/search/cs?searchtype=author&query=Bansal%2C+M)

> Fine-tuning large pre-trained models on downstream tasks has been adopted in a variety of domains recently. However, it is costly to update the entire parameter set of large pre-trained models. Although recently proposed parameter-efficient transfer learning (PETL) techniques allow updating a small subset of parameters (e.g. only using 2% of parameters) inside a pre-trained backbone network for a new task, they only reduce the training memory requirement by up to 30%. This is because the gradient computation for the trainable parameters still requires backpropagation through the large pre-trained backbone model. To address this, we propose Ladder Side-Tuning (LST), a new PETL technique that reduces training memory requirements by more substantial amounts. Unlike existing parameter-efficient methods that insert additional parameters inside backbone networks, we train a ladder side network, a small and separate network that takes intermediate activations as input via shortcut connections (ladders) from backbone networks and makes predictions. LST has significantly lower memory requirements than previous methods, because it does not require backpropagation through the backbone network, but instead only through the side network and ladder connections. We evaluate our method with various models (T5, CLIP-T5) on both NLP (GLUE) and vision-language (VQA, GQA, NLVR2, MSCOCO) tasks. LST saves 69% of the memory costs to fine-tune the whole network, while other methods only save 26% of that in similar parameter usages (hence, 2.7x more memory savings). Moreover, LST achieves higher accuracy than Adapter and LoRA in a low-memory regime. To further show the advantage of this better memory efficiency, we also apply LST to larger T5 models (T5-large, T5-3B), attaining better GLUE performance than full fine-tuning and other PETL methods. The exact same trend also holds in our experiments on VL tasks.

| Comments: | 13 pages; our code is available at: [this https URL](https://github.com/ylsung/Ladder-Side-Tuning) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV) |
| Cite as:  | **[arXiv:2206.06522](https://arxiv.org/abs/2206.06522) [cs.CL]** |
|           | (or **[arXiv:2206.06522v1](https://arxiv.org/abs/2206.06522v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.06522Focus to learn more |


