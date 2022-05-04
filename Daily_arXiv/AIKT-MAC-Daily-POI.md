# MA C.'s Daily Paper Of Interest - May a., 2022

# Index

- [2022-05-04](#2022-05-04)
  - [1. Hausa Visual Genome: A Dataset for Multi-Modal English to Hausa Machine Translation](#2022-05-04-1)
  - [2. Contrastive Learning for Prompt-Based Few-Shot Language Learners](#2022-05-04-2)
  - [3. Meta Learning for Natural Language Processing: A Survey](#2022-05-04-3)
  - [4. Learning to Transfer Prompts for Text Generation](#2022-05-04-4)
  - [5. Adaptable Adapters](#2022-05-04-5)
  - [6. Training Mixed-Domain Translation Models via Federated Learning](#2022-05-04-6)
  - [7. OmniKnight: Multilingual Neural Machine Translation with Language-Specific Self-Distillation](#2022-05-04-7)
- [2022-05-03](#2022-05-03)
  - [1. Multimodal Representation Learning With Text and Images](#2022-05-03-1)
  - [2. EasyNLP: A Comprehensive and Easy-to-use Toolkit for Natural Language Processing](#2022-05-03-2)
  - [3. AdapterBias: Parameter-efficient Token-dependent Representation Shift for Adapters in NLP Tasks](#2022-05-03-3)
  - [4. Nearest Neighbor Knowledge Distillation for Neural Machine Translation](#2022-05-03-4)
  - [5. Bilingual End-to-End ASR with Byte-Level Subwords](#2022-05-03-5)
  - [6. Debiased Contrastive Learning of Unsupervised Sentence Representations](#2022-05-03-6)
  - [7. The Implicit Length Bias of Label Smoothing on Beam Search Decoding](#2022-05-03-7)
  - [8. Quality-Aware Decoding for Neural Machine Translation](#2022-05-03-8)
  - [9. OPT: Open Pre-trained Transformer Language Models](#2022-05-03-9)
  - [10. Wav2Seq: Pre-training Speech-to-Text Encoder-Decoder Models Using Pseudo Languages](#2022-05-03-10)
- [2022-05-02](#2022-05-02)
  - [1. Vision-Language Pre-Training for Boosting Scene Text Detectors](#2022-05-03-1)
  - [2. Polyglot Prompt: Multilingual Multitask PrompTraining](#2022-05-03-2)
  - [3. How Robust is Neural Machine Translation to Language Imbalance in Multilingual Tokenizer Training?](#2022-05-03-3)
- [2022-04-29](#2022-04-29)	
  - [1. UniTE: Unified Translation Evaluation](#2022-04-29-1)


- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



# 2022-05-04

[Return to Index](#Index)



<h2 id="2022-05-04-1">1. Hausa Visual Genome: A Dataset for Multi-Modal English to Hausa Machine Translation
</h2>

Title: [Hausa Visual Genome: A Dataset for Multi-Modal English to Hausa Machine Translation](https://arxiv.org/abs/2205.01133)

Authors: [Idris Abdulmumin](https://arxiv.org/search/cs?searchtype=author&query=Abdulmumin%2C+I), [Satya Ranjan Dash](https://arxiv.org/search/cs?searchtype=author&query=Dash%2C+S+R), [Musa Abdullahi Dawud](https://arxiv.org/search/cs?searchtype=author&query=Dawud%2C+M+A), [Shantipriya Parida](https://arxiv.org/search/cs?searchtype=author&query=Parida%2C+S), [Shamsuddeen Hassan Muhammad](https://arxiv.org/search/cs?searchtype=author&query=Muhammad%2C+S+H), [Ibrahim Sa'id Ahmad](https://arxiv.org/search/cs?searchtype=author&query=Ahmad%2C+I+S), [Subhadarshi Panda](https://arxiv.org/search/cs?searchtype=author&query=Panda%2C+S), [Ondřej Bojar](https://arxiv.org/search/cs?searchtype=author&query=Bojar%2C+O), [Bashir Shehu Galadanci](https://arxiv.org/search/cs?searchtype=author&query=Galadanci%2C+B+S), [Bello Shehu Bello](https://arxiv.org/search/cs?searchtype=author&query=Bello%2C+B+S)

> Multi-modal Machine Translation (MMT) enables the use of visual information to enhance the quality of translations. The visual information can serve as a valuable piece of context information to decrease the ambiguity of input sentences. Despite the increasing popularity of such a technique, good and sizeable datasets are scarce, limiting the full extent of their potential. Hausa, a Chadic language, is a member of the Afro-Asiatic language family. It is estimated that about 100 to 150 million people speak the language, with more than 80 million indigenous speakers. This is more than any of the other Chadic languages. Despite a large number of speakers, the Hausa language is considered low-resource in natural language processing (NLP). This is due to the absence of sufficient resources to implement most NLP tasks. While some datasets exist, they are either scarce, machine-generated, or in the religious domain. Therefore, there is a need to create training and evaluation data for implementing machine learning tasks and bridging the research gap in the language. This work presents the Hausa Visual Genome (HaVG), a dataset that contains the description of an image or a section within the image in Hausa and its equivalent in English. To prepare the dataset, we started by translating the English description of the images in the Hindi Visual Genome (HVG) into Hausa automatically. Afterward, the synthetic Hausa data was carefully post-edited considering the respective images. The dataset comprises 32,923 images and their descriptions that are divided into training, development, test, and challenge test set. The Hausa Visual Genome is the first dataset of its kind and can be used for Hausa-English machine translation, multi-modal research, and image description, among various other natural language processing and generation tasks.

| Comments: | Accepted at Language Resources and Evaluation Conference 2022 (LREC2022) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2205.01133](https://arxiv.org/abs/2205.01133) [cs.CL]** |
|           | (or **[arXiv:2205.01133v1](https://arxiv.org/abs/2205.01133v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.01133Focus to learn more |





<h2 id="2022-05-04-2">2. Contrastive Learning for Prompt-Based Few-Shot Language Learners
</h2>

Title: [Contrastive Learning for Prompt-Based Few-Shot Language Learners](https://arxiv.org/abs/2205.01308)

Authors: [Yiren Jian](https://arxiv.org/search/cs?searchtype=author&query=Jian%2C+Y), [Chongyang Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+C), [Soroush Vosoughi](https://arxiv.org/search/cs?searchtype=author&query=Vosoughi%2C+S)

> The impressive performance of GPT-3 using natural language prompts and in-context learning has inspired work on better fine-tuning of moderately-sized models under this paradigm. Following this line of work, we present a contrastive learning framework that clusters inputs from the same class for better generality of models trained with only limited examples. Specifically, we propose a supervised contrastive framework that clusters inputs from the same class under different augmented "views" and repel the ones from different classes. We create different "views" of an example by appending it with different language prompts and contextual demonstrations. Combining a contrastive loss with the standard masked language modeling (MLM) loss in prompt-based few-shot learners, the experimental results show that our method can improve over the state-of-the-art methods in a diverse set of 15 language tasks. Our framework makes minimal assumptions on the task or the base model, and can be applied to many recent methods with little modification. The code will be made available at: [this https URL](https://github.com/yiren-jian/LM-SupCon).

| Comments: | accepted to NAACL 2022                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2205.01308](https://arxiv.org/abs/2205.01308) [cs.CL]** |
|           | (or **[arXiv:2205.01308v1](https://arxiv.org/abs/2205.01308v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.01308Focus to learn more |





<h2 id="2022-05-04-3">3. Meta Learning for Natural Language Processing: A Survey
</h2>

Title: [Meta Learning for Natural Language Processing: A Survey](https://arxiv.org/abs/2205.01500)

Authors: [Hung-yi Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+H), [Shang-Wen Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+S), [Ngoc Thang Vu](https://arxiv.org/search/cs?searchtype=author&query=Vu%2C+N+T)

> Deep learning has been the mainstream technique in natural language processing (NLP) area. However, the techniques require many labeled data and are less generalizable across domains. Meta-learning is an arising field in machine learning studying approaches to learn better learning algorithms. Approaches aim at improving algorithms in various aspects, including data efficiency and generalizability. Efficacy of approaches has been shown in many NLP tasks, but there is no systematic survey of these approaches in NLP, which hinders more researchers from joining the field. Our goal with this survey paper is to offer researchers pointers to relevant meta-learning works in NLP and attract more attention from the NLP community to drive future innovation. This paper first introduces the general concepts of meta-learning and the common approaches. Then we summarize task construction settings and application of meta-learning for various NLP problems and review the development of meta-learning in NLP community.

| Comments: | Accepted by NAACL 2022                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2205.01500](https://arxiv.org/abs/2205.01500) [cs.CL]** |
|           | (or **[arXiv:2205.01500v1](https://arxiv.org/abs/2205.01500v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.01500Focus to learn more |





<h2 id="2022-05-04-4">4. Learning to Transfer Prompts for Text Generation
</h2>

Title: [Learning to Transfer Prompts for Text Generation](https://arxiv.org/abs/2205.01543)

Authors: [Junyi Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J), [Tianyi Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+T), [Jian-Yun Nie](https://arxiv.org/search/cs?searchtype=author&query=Nie%2C+J), [Ji-Rong Wen](https://arxiv.org/search/cs?searchtype=author&query=Wen%2C+J), [Wayne Xin Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+W+X)

> Pretrained language models (PLMs) have made remarkable progress in text generation tasks via fine-tuning. While, it is challenging to fine-tune PLMs in a data-scarce situation. Therefore, it is non-trivial to develop a general and lightweight model that can adapt to various text generation tasks based on PLMs. To fulfill this purpose, the recent prompt-based learning offers a potential solution. In this paper, we improve this technique and propose a novel prompt-based method (PTG) for text generation in a transferable setting. First, PTG learns a set of source prompts for various source generation tasks and then transfers these prompts as target prompts to perform target generation tasks. To consider both task- and instance-level information, we design an adaptive attention mechanism to derive the target prompts. For each data instance, PTG learns a specific target prompt by attending to highly relevant source prompts. In extensive experiments, PTG yields competitive or better results than fine-tuning methods. We release our source prompts as an open resource, where users can add or reuse them to improve new text generation tasks for future research. Code and data can be available at [this https URL](https://github.com/RUCAIBox/Transfer-Prompts-for-Text-Generation).

| Comments: | Accepted by NAACL 2022 main conference (Long Paper)          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.01543](https://arxiv.org/abs/2205.01543) [cs.CL]** |
|           | (or **[arXiv:2205.01543v1](https://arxiv.org/abs/2205.01543v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.01543Focus to learn more |





<h2 id="2022-05-04-5">5. Adaptable Adapters
</h2>

Title: [Adaptable Adapters](https://arxiv.org/abs/2205.01549)

Authors: [Nafise Sadat Moosavi](https://arxiv.org/search/cs?searchtype=author&query=Moosavi%2C+N+S), [Quentin Delfosse](https://arxiv.org/search/cs?searchtype=author&query=Delfosse%2C+Q), [Kristian Kersting](https://arxiv.org/search/cs?searchtype=author&query=Kersting%2C+K), [Iryna Gurevych](https://arxiv.org/search/cs?searchtype=author&query=Gurevych%2C+I)

> State-of-the-art pretrained NLP models contain a hundred million to trillion parameters. Adapters provide a parameter-efficient alternative for the full finetuning in which we can only finetune lightweight neural network layers on top of pretrained weights. Adapter layers are initialized randomly. However, existing work uses the same adapter architecture -- i.e., the same adapter layer on top of each layer of the pretrained model -- for every dataset, regardless of the properties of the dataset or the amount of available training data. In this work, we introduce adaptable adapters that contain (1) learning different activation functions for different layers and different input data, and (2) a learnable switch to select and only use the beneficial adapter layers. We show that adaptable adapters achieve on-par performances with the standard adapter architecture while using a considerably smaller number of adapter layers. In addition, we show that the selected adapter architecture by adaptable adapters transfers well across different data settings and similar tasks. We propose to use adaptable adapters for designing efficient and effective adapter architectures. The resulting adapters (a) contain about 50% of the learning parameters of the standard adapter and are therefore more efficient at training and inference, and require less storage space, and (b) achieve considerably higher performances in low-data settings.

| Comments: | Accepted at NAACL-2022 main conference                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2205.01549](https://arxiv.org/abs/2205.01549) [cs.CL]** |
|           | (or **[arXiv:2205.01549v1](https://arxiv.org/abs/2205.01549v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.01549Focus to learn more |





<h2 id="2022-05-04-6">6. Training Mixed-Domain Translation Models via Federated Learning
</h2>

Title: [Training Mixed-Domain Translation Models via Federated Learning](https://arxiv.org/abs/2205.01557)

Authors: [Peyman Passban](https://arxiv.org/search/cs?searchtype=author&query=Passban%2C+P), [Tanya Roosta](https://arxiv.org/search/cs?searchtype=author&query=Roosta%2C+T), [Rahul Gupta](https://arxiv.org/search/cs?searchtype=author&query=Gupta%2C+R), [Ankit Chadha](https://arxiv.org/search/cs?searchtype=author&query=Chadha%2C+A), [Clement Chung](https://arxiv.org/search/cs?searchtype=author&query=Chung%2C+C)

> Training mixed-domain translation models is a complex task that demands tailored architectures and costly data preparation techniques. In this work, we leverage federated learning (FL) in order to tackle the problem. Our investigation demonstrates that with slight modifications in the training process, neural machine translation (NMT) engines can be easily adapted when an FL-based aggregation is applied to fuse different domains. Experimental results also show that engines built via FL are able to perform on par with state-of-the-art baselines that rely on centralized training techniques. We evaluate our hypothesis in the presence of five datasets with different sizes, from different domains, to translate from German into English and discuss how FL and NMT can mutually benefit from each other. In addition to providing benchmarking results on the union of FL and NMT, we also propose a novel technique to dynamically control the communication bandwidth by selecting impactful parameters during FL updates. This is a significant achievement considering the large size of NMT engines that need to be exchanged between FL parties.

| Comments: | accepted at NAACL 2022 (main conference)                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.01557](https://arxiv.org/abs/2205.01557) [cs.CL]** |
|           | (or **[arXiv:2205.01557v1](https://arxiv.org/abs/2205.01557v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.01557Focus to learn more |





<h2 id="2022-05-04-7">7. OmniKnight: Multilingual Neural Machine Translation with Language-Specific Self-Distillation
</h2>

Title: [OmniKnight: Multilingual Neural Machine Translation with Language-Specific Self-Distillation](https://arxiv.org/abs/2205.01620)

Authors: [Yichong Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+Y), [Xiaocheng Feng](https://arxiv.org/search/cs?searchtype=author&query=Feng%2C+X), [Xinwei Geng](https://arxiv.org/search/cs?searchtype=author&query=Geng%2C+X), [Bing Qin](https://arxiv.org/search/cs?searchtype=author&query=Qin%2C+B)

> Although all-in-one-model multilingual neural machine translation (MNMT) has achieved remarkable progress in recent years, its selected best overall checkpoint fails to achieve the best performance simultaneously in all language pairs. It is because that the best checkpoints for each individual language pair (i.e., language-specific best checkpoints) scatter in different epochs. In this paper, we present a novel training strategy dubbed Language-Specific Self-Distillation (LSSD) for bridging the gap between language-specific best checkpoints and the overall best checkpoint. In detail, we regard each language-specific best checkpoint as a teacher to distill the overall best checkpoint. Moreover, we systematically explore three variants of our LSSD, which perform distillation statically, selectively, and adaptively. Experimental results on two widely-used benchmarks show that LSSD obtains consistent improvements towards all language pairs and achieves the state-of-the-art

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.01620](https://arxiv.org/abs/2205.01620) [cs.CL]** |
|           | (or **[arXiv:2205.01620v1](https://arxiv.org/abs/2205.01620v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.01620Focus to learn more |





# 2022-05-03

[Return to Index](#Index)



<h2 id="2022-05-03-1">1. Multimodal Representation Learning With Text and Images
</h2>

Title: [Multimodal Representation Learning With Text and Images](https://arxiv.org/abs/2205.00142)

Authors:[Aishwarya Jayagopal](https://arxiv.org/search/cs?searchtype=author&query=Jayagopal%2C+A), [Ankireddy Monica Aiswarya](https://arxiv.org/search/cs?searchtype=author&query=Aiswarya%2C+A+M), [Ankita Garg](https://arxiv.org/search/cs?searchtype=author&query=Garg%2C+A), [Srinivasan Kolumam Nandakumar](https://arxiv.org/search/cs?searchtype=author&query=Nandakumar%2C+S+K)

> In recent years, multimodal AI has seen an upward trend as researchers are integrating data of different types such as text, images, speech into modelling to get the best results. This project leverages multimodal AI and matrix factorization techniques for representation learning, on text and image data simultaneously, thereby employing the widely used techniques of Natural Language Processing (NLP) and Computer Vision. The learnt representations are evaluated using downstream classification and regression tasks. The methodology adopted can be extended beyond the scope of this project as it uses Auto-Encoders for unsupervised representation learning.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.00142](https://arxiv.org/abs/2205.00142) [cs.LG]** |
|           | (or **[arXiv:2205.00142v1](https://arxiv.org/abs/2205.00142v1) [cs.LG]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.00142Focus to learn more |





<h2 id="2022-05-03-2">2. EasyNLP: A Comprehensive and Easy-to-use Toolkit for Natural Language Processing
</h2>

Title: [EasyNLP: A Comprehensive and Easy-to-use Toolkit for Natural Language Processing](https://arxiv.org/abs/2205.00258)

Authors:[Chengyu Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+C), [Minghui Qiu](https://arxiv.org/search/cs?searchtype=author&query=Qiu%2C+M), [Taolin Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+T), [Tingting Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+T), [Lei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L), [Jianing Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+J), [Ming Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+M), [Jun Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+J), [Wei Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+W)

> The success of Pre-Trained Models (PTMs) has reshaped the development of Natural Language Processing (NLP). Yet, it is not easy to obtain high-performing models and deploy them online for industrial practitioners. To bridge this gap, EasyNLP is designed to make it easy to build NLP applications, which supports a comprehensive suite of NLP algorithms. It further features knowledge-enhanced pre-training, knowledge distillation and few-shot learning functionalities for large-scale PTMs, and provides a unified framework of model training, inference and deployment for real-world applications. Currently, EasyNLP has powered over ten business units within Alibaba Group and is seamlessly integrated to the Platform of AI (PAI) products on Alibaba Cloud. The source code of our EasyNLP toolkit is released at GitHub ([this https URL](https://github.com/alibaba/EasyNLP)).

| Comments: | 8 pages                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.00258](https://arxiv.org/abs/2205.00258) [cs.CL]** |
|           | (or **[arXiv:2205.00258v1](https://arxiv.org/abs/2205.00258v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.00258Focus to learn more |





<h2 id="2022-05-03-3">3. AdapterBias: Parameter-efficient Token-dependent Representation Shift for Adapters in NLP Tasks
</h2>

Title: [AdapterBias: Parameter-efficient Token-dependent Representation Shift for Adapters in NLP Tasks](https://arxiv.org/abs/2205.00305)

Authors:[Chin-Lun Fu](https://arxiv.org/search/cs?searchtype=author&query=Fu%2C+C), [Zih-Ching Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Z), [Yun-Ru Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+Y), [Hung-yi Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+H)

> Transformer-based pre-trained models with millions of parameters require large storage. Recent approaches tackle this shortcoming by training adapters, but these approaches still require a relatively large number of parameters. In this study, AdapterBias, a surprisingly simple yet effective adapter architecture, is proposed. AdapterBias adds a token-dependent shift to the hidden output of transformer layers to adapt to downstream tasks with only a vector and a linear layer. Extensive experiments are conducted to demonstrate the effectiveness of AdapterBias. The experiments show that our proposed method can dramatically reduce the trainable parameters compared to the previous works with a minimal decrease in task performances compared with fine-tuned pre-trained models. We further find that AdapterBias automatically learns to assign more significant representation shifts to the tokens related to the task in consideration.

| Comments: | The first two authors contributed equally. This paper will be published in Findings of NAACL 2022 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.00305](https://arxiv.org/abs/2205.00305) [cs.CL]** |
|           | (or **[arXiv:2205.00305v1](https://arxiv.org/abs/2205.00305v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.00305Focus to learn more |





<h2 id="2022-05-03-4">4. Nearest Neighbor Knowledge Distillation for Neural Machine Translation
</h2>

Title: [Nearest Neighbor Knowledge Distillation for Neural Machine Translation](https://arxiv.org/abs/2205.00479)

Authors:[Zhixian Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Z), [Renliang Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+R), [Xiaojun Wan](https://arxiv.org/search/cs?searchtype=author&query=Wan%2C+X)

> k-nearest-neighbor machine translation (NN-MT), proposed by Khandelwal et al. (2021), has achieved many state-of-the-art results in machine translation tasks. Although effective, NN-MT requires conducting NN searches through the large datastore for each decoding step during inference, prohibitively increasing the decoding cost and thus leading to the difficulty for the deployment in real-world applications. In this paper, we propose to move the time-consuming NN search forward to the preprocessing phase, and then introduce Nearest Neighbor Knowledge Distillation (NN-KD) that trains the base NMT model to directly learn the knowledge of NN. Distilling knowledge retrieved by NN can encourage the NMT model to take more reasonable target tokens into consideration, thus addressing the overcorrection problem. Extensive experimental results show that, the proposed method achieves consistent improvement over the state-of-the-art baselines including NN-MT, while maintaining the same training and decoding speed as the standard NMT model.

| Comments: | Accepted to NAACL 2022 Main Conference                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.00479](https://arxiv.org/abs/2205.00479) [cs.CL]** |
|           | (or **[arXiv:2205.00479v1](https://arxiv.org/abs/2205.00479v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.00479Focus to learn more |





<h2 id="2022-05-03-5">5. Bilingual End-to-End ASR with Byte-Level Subwords
</h2>

Title: [Bilingual End-to-End ASR with Byte-Level Subwords](https://arxiv.org/abs/2205.00485)

Authors:[Liuhui Deng](https://arxiv.org/search/cs?searchtype=author&query=Deng%2C+L), [Roger Hsiao](https://arxiv.org/search/cs?searchtype=author&query=Hsiao%2C+R), [Arnab Ghoshal](https://arxiv.org/search/cs?searchtype=author&query=Ghoshal%2C+A)

> In this paper, we investigate how the output representation of an end-to-end neural network affects multilingual automatic speech recognition (ASR). We study different representations including character-level, byte-level, byte pair encoding (BPE), and byte-level byte pair encoding (BBPE) representations, and analyze their strengths and weaknesses. We focus on developing a single end-to-end model to support utterance-based bilingual ASR, where speakers do not alternate between two languages in a single utterance but may change languages across utterances. We conduct our experiments on English and Mandarin dictation tasks, and we find that BBPE with penalty schemes can improve utterance-based bilingual ASR performance by 2% to 5% relative even with smaller number of outputs and fewer parameters. We conclude with analysis that indicates directions for further improving multilingual ASR.

| Comments: | 5 pages, to be published in IEEE ICASSP 2022                 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2205.00485](https://arxiv.org/abs/2205.00485) [cs.CL]** |
|           | (or **[arXiv:2205.00485v1](https://arxiv.org/abs/2205.00485v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.00485Focus to learn more |





<h2 id="2022-05-03-6">6. Debiased Contrastive Learning of Unsupervised Sentence Representations
</h2>

Title: [Debiased Contrastive Learning of Unsupervised Sentence Representations](https://arxiv.org/abs/2205.00656)

Authors:[Kun Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+K), [Beichen Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+B), [Wayne Xin Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+W+X), [Ji-Rong Wen](https://arxiv.org/search/cs?searchtype=author&query=Wen%2C+J)

> Recently, contrastive learning has been shown to be effective in improving pre-trained language models (PLM) to derive high-quality sentence representations. It aims to pull close positive examples to enhance the alignment while push apart irrelevant negatives for the uniformity of the whole representation space. However, previous works mostly adopt in-batch negatives or sample from training data at random. Such a way may cause the sampling bias that improper negatives (e.g. false negatives and anisotropy representations) are used to learn sentence representations, which will hurt the uniformity of the representation space. To address it, we present a new framework \textbf{DCLR} (\underline{D}ebiased \underline{C}ontrastive \underline{L}earning of unsupervised sentence \underline{R}epresentations) to alleviate the influence of these improper negatives. In DCLR, we design an instance weighting method to punish false negatives and generate noise-based negatives to guarantee the uniformity of the representation space. Experiments on seven semantic textual similarity tasks show that our approach is more effective than competitive baselines. Our code and data are publicly available at the link: \textcolor{blue}{\url{[this https URL](https://github.com/RUCAIBox/DCLR)}}.

| Comments: | 11 pages, accepted by ACL 2022 main conference               |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.00656](https://arxiv.org/abs/2205.00656) [cs.CL]** |
|           | (or **[arXiv:2205.00656v1](https://arxiv.org/abs/2205.00656v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.00656Focus to learn more |





<h2 id="2022-05-03-7">7. The Implicit Length Bias of Label Smoothing on Beam Search Decoding
</h2>

Title: [The Implicit Length Bias of Label Smoothing on Beam Search Decoding](https://arxiv.org/abs/2205.00659)

Authors:[Bowen Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+B), [Pidong Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+P), [Yuan Cao](https://arxiv.org/search/cs?searchtype=author&query=Cao%2C+Y)

> Label smoothing is ubiquitously applied in Neural Machine Translation (NMT) training. While label smoothing offers a desired regularization effect during model training, in this paper we demonstrate that it nevertheless introduces length biases in the beam search decoding procedure. Our analysis shows that label smoothing implicitly applies a length penalty term to output sequence, causing a bias towards shorter translations. We also show that for a model fully optimized with label smoothing, translation length is implicitly upper bounded by a fixed constant independent of input. We verify our theory by applying a simple rectification function at inference time to restore the unbiased distributions from the label-smoothed model predictions. This rectification method led to consistent quality improvements on WMT English-German, English-French, English-Czech and English-Chinese tasks, up to +0.3 BLEU at beam size 4 and +2.8 BLEU at beam size 200.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.00659](https://arxiv.org/abs/2205.00659) [cs.CL]** |
|           | (or **[arXiv:2205.00659v1](https://arxiv.org/abs/2205.00659v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.00659Focus to learn more |





<h2 id="2022-05-03-8">8. Quality-Aware Decoding for Neural Machine Translation
</h2>

Title: [Quality-Aware Decoding for Neural Machine Translation](https://arxiv.org/abs/2205.00978)

Authors:[Patrick Fernandes](https://arxiv.org/search/cs?searchtype=author&query=Fernandes%2C+P), [António Farinhas](https://arxiv.org/search/cs?searchtype=author&query=Farinhas%2C+A), [Ricardo Rei](https://arxiv.org/search/cs?searchtype=author&query=Rei%2C+R), [José G. C. de Souza](https://arxiv.org/search/cs?searchtype=author&query=de+Souza%2C+J+G+C), [Perez Ogayo](https://arxiv.org/search/cs?searchtype=author&query=Ogayo%2C+P), [Graham Neubig](https://arxiv.org/search/cs?searchtype=author&query=Neubig%2C+G), [André F. T. Martins](https://arxiv.org/search/cs?searchtype=author&query=Martins%2C+A+F+T)

> Despite the progress in machine translation quality estimation and evaluation in the last years, decoding in neural machine translation (NMT) is mostly oblivious to this and centers around finding the most probable translation according to the model (MAP decoding), approximated with beam search. In this paper, we bring together these two lines of research and propose quality-aware decoding for NMT, by leveraging recent breakthroughs in reference-free and reference-based MT evaluation through various inference methods like N-best reranking and minimum Bayes risk decoding. We perform an extensive comparison of various possible candidate generation and ranking methods across four datasets and two model classes and find that quality-aware decoding consistently outperforms MAP-based decoding according both to state-of-the-art automatic metrics (COMET and BLEURT) and to human assessments. Our code is available at [this https URL](https://github.com/deep-spin/qaware-decode).

| Comments: | NAACL2022                                                    |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.00978](https://arxiv.org/abs/2205.00978) [cs.CL]** |
|           | (or **[arXiv:2205.00978v1](https://arxiv.org/abs/2205.00978v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.00978Focus to learn more |





<h2 id="2022-05-03-9">9. OPT: Open Pre-trained Transformer Language Models
</h2>

Title: [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068)

Authors:[Susan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+S), [Stephen Roller](https://arxiv.org/search/cs?searchtype=author&query=Roller%2C+S), [Naman Goyal](https://arxiv.org/search/cs?searchtype=author&query=Goyal%2C+N), [Mikel Artetxe](https://arxiv.org/search/cs?searchtype=author&query=Artetxe%2C+M), [Moya Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+M), [Shuohui Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+S), [Christopher Dewan](https://arxiv.org/search/cs?searchtype=author&query=Dewan%2C+C), [Mona Diab](https://arxiv.org/search/cs?searchtype=author&query=Diab%2C+M), [Xian Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+X), [Xi Victoria Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+X+V), [Todor Mihaylov](https://arxiv.org/search/cs?searchtype=author&query=Mihaylov%2C+T), [Myle Ott](https://arxiv.org/search/cs?searchtype=author&query=Ott%2C+M), [Sam Shleifer](https://arxiv.org/search/cs?searchtype=author&query=Shleifer%2C+S), [Kurt Shuster](https://arxiv.org/search/cs?searchtype=author&query=Shuster%2C+K), [Daniel Simig](https://arxiv.org/search/cs?searchtype=author&query=Simig%2C+D), [Punit Singh Koura](https://arxiv.org/search/cs?searchtype=author&query=Koura%2C+P+S), [Anjali Sridhar](https://arxiv.org/search/cs?searchtype=author&query=Sridhar%2C+A), [Tianlu Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+T), [Luke Zettlemoyer](https://arxiv.org/search/cs?searchtype=author&query=Zettlemoyer%2C+L)

> Large language models, which are often trained for hundreds of thousands of compute days, have shown remarkable capabilities for zero- and few-shot learning. Given their computational cost, these models are difficult to replicate without significant capital. For the few that are available through APIs, no access is granted to the full model weights, making them difficult to study. We present Open Pre-trained Transformers (OPT), a suite of decoder-only pre-trained transformers ranging from 125M to 175B parameters, which we aim to fully and responsibly share with interested researchers. We show that OPT-175B is comparable to GPT-3, while requiring only 1/7th the carbon footprint to develop. We are also releasing our logbook detailing the infrastructure challenges we faced, along with code for experimenting with all of the released models.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.01068](https://arxiv.org/abs/2205.01068) [cs.CL]** |
|           | (or **[arXiv:2205.01068v1](https://arxiv.org/abs/2205.01068v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.01068Focus to learn more |





<h2 id="2022-05-03-10">10. Wav2Seq: Pre-training Speech-to-Text Encoder-Decoder Models Using Pseudo Languages
</h2>

Title: [Wav2Seq: Pre-training Speech-to-Text Encoder-Decoder Models Using Pseudo Languages](https://arxiv.org/abs/2205.01086)

Authors:[Felix Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+F), [Kwangyoun Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+K), [Shinji Watanabe](https://arxiv.org/search/cs?searchtype=author&query=Watanabe%2C+S), [Kyu Han](https://arxiv.org/search/cs?searchtype=author&query=Han%2C+K), [Ryan McDonald](https://arxiv.org/search/cs?searchtype=author&query=McDonald%2C+R), [Kilian Q. Weinberger](https://arxiv.org/search/cs?searchtype=author&query=Weinberger%2C+K+Q), [Yoav Artzi](https://arxiv.org/search/cs?searchtype=author&query=Artzi%2C+Y)

> We introduce Wav2Seq, the first self-supervised approach to pre-train both parts of encoder-decoder models for speech data. We induce a pseudo language as a compact discrete representation, and formulate a self-supervised pseudo speech recognition task -- transcribing audio inputs into pseudo subword sequences. This process stands on its own, or can be applied as low-cost second-stage pre-training. We experiment with automatic speech recognition (ASR), spoken named entity recognition, and speech-to-text translation. We set new state-of-the-art results for end-to-end spoken named entity recognition, and show consistent improvements on 20 language pairs for speech-to-text translation, even when competing methods use additional text data for training. Finally, on ASR, our approach enables encoder-decoder methods to benefit from pre-training for all parts of the network, and shows comparable performance to highly optimized recent methods.

| Comments: | Code available at [this https URL](https://github.com/asappresearch/wav2seq) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2205.01086](https://arxiv.org/abs/2205.01086) [cs.CL]** |
|           | (or **[arXiv:2205.01086v1](https://arxiv.org/abs/2205.01086v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.01086Focus to learn more |





# 2022-05-02

[Return to Index](#Index)



<h2 id="2022-05-02-1">1. Vision-Language Pre-Training for Boosting Scene Text Detectors
</h2>

Title: [Vision-Language Pre-Training for Boosting Scene Text Detectors](https://arxiv.org/abs/2204.13867)

Authors: [Sibo Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+S), [Jianqiang Wan](https://arxiv.org/search/cs?searchtype=author&query=Wan%2C+J), [Zhibo Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Z), [Jun Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+J), [Wenqing Cheng](https://arxiv.org/search/cs?searchtype=author&query=Cheng%2C+W), [Xiang Bai](https://arxiv.org/search/cs?searchtype=author&query=Bai%2C+X), [Cong Yao](https://arxiv.org/search/cs?searchtype=author&query=Yao%2C+C)

> Recently, vision-language joint representation learning has proven to be highly effective in various scenarios. In this paper, we specifically adapt vision-language joint learning for scene text detection, a task that intrinsically involves cross-modal interaction between the two modalities: vision and language, since text is the written form of language. Concretely, we propose to learn contextualized, joint representations through vision-language pre-training, for the sake of enhancing the performance of scene text detectors. Towards this end, we devise a pre-training architecture with an image encoder, a text encoder and a cross-modal encoder, as well as three pretext tasks: image-text contrastive learning (ITC), masked language modeling (MLM) and word-in-image prediction (WIP). The pre-trained model is able to produce more informative representations with richer semantics, which could readily benefit existing scene text detectors (such as EAST and PSENet) in the down-stream text detection task. Extensive experiments on standard benchmarks demonstrate that the proposed paradigm can significantly improve the performance of various representative text detectors, outperforming previous pre-training approaches. The code and pre-trained models will be publicly released.

| Comments: | Accepted by CVPR 2022                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2204.13867](https://arxiv.org/abs/2204.13867) [cs.CV]** |
|           | (or **[arXiv:2204.13867v1](https://arxiv.org/abs/2204.13867v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.13867Focus to learn more |





<h2 id="2022-05-02-2">2. Polyglot Prompt: Multilingual Multitask PrompTraining
</h2>

Title: [Polyglot Prompt: Multilingual Multitask PrompTraining](https://arxiv.org/abs/2204.14264)

Authors: [Jinlan Fu](https://arxiv.org/search/cs?searchtype=author&query=Fu%2C+J), [See-Kiong Ng](https://arxiv.org/search/cs?searchtype=author&query=Ng%2C+S), [Pengfei Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+P)

> This paper aims for a potential architectural breakthrough for multilingual learning and asks: could different tasks from different languages be modeled in a monolithic framework (without any task/language-specific module)? The benefit of achieving this is not only that systems trained on low resources scenario can be assisted by more other languages and tasks, but opening new doors for future multilingual research. We approach this goal by developing a learning framework Polyglot Prompt, where prompting methods are introduced to learn a unified semantic space for different languages and tasks after proper multilingual prompt engineering. Experimentally, we perform a comprehensive evaluation on 6 tasks (topic classification, sentiment classification, named entity recognition, question answering, natural language inference, summarization), 24 datasets, and 49 languages, which shows the efficacy of multilingual multitask prompting training and suggests several interesting observations. e.g., English prompts are polyglots since directly applying them to task samples in other languages could result in a better improvement. We also present an interpretable multilingual evaluation methodology and show how the proposed framework, multilingual multitask prompt training, works. We release all datasets prompted in the best setting and will release our code soon.

| Comments: | 19 pages, 64 figures                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2204.14264](https://arxiv.org/abs/2204.14264) [cs.CL]** |
|           | (or **[arXiv:2204.14264v1](https://arxiv.org/abs/2204.14264v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.14264Focus to learn more |





<h2 id="2022-05-02-3">3. How Robust is Neural Machine Translation to Language Imbalance in Multilingual Tokenizer Training?
</h2>

Title: [How Robust is Neural Machine Translation to Language Imbalance in Multilingual Tokenizer Training?](https://arxiv.org/abs/2204.14268)

Authors: [Shiyue Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+S), [Vishrav Chaudhary](https://arxiv.org/search/cs?searchtype=author&query=Chaudhary%2C+V), [Naman Goyal](https://arxiv.org/search/cs?searchtype=author&query=Goyal%2C+N), [James Cross](https://arxiv.org/search/cs?searchtype=author&query=Cross%2C+J), [Guillaume Wenzek](https://arxiv.org/search/cs?searchtype=author&query=Wenzek%2C+G), [Mohit Bansal](https://arxiv.org/search/cs?searchtype=author&query=Bansal%2C+M), [Francisco Guzman](https://arxiv.org/search/cs?searchtype=author&query=Guzman%2C+F)

> A multilingual tokenizer is a fundamental component of multilingual neural machine translation. It is trained from a multilingual corpus. Since a skewed data distribution is considered to be harmful, a sampling strategy is usually used to balance languages in the corpus. However, few works have systematically answered how language imbalance in tokenizer training affects downstream performance. In this work, we analyze how translation performance changes as the data ratios among languages vary in the tokenizer training corpus. We find that while relatively better performance is often observed when languages are more equally sampled, the downstream performance is more robust to language imbalance than we usually expected. Two features, UNK rate and closeness to the character level, can warn of poor downstream performance before performing the task. We also distinguish language sampling for tokenizer training from sampling for model training and show that the model is more sensitive to the latter.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2204.14268](https://arxiv.org/abs/2204.14268) [cs.CL]** |
|           | (or **[arXiv:2204.14268v1](https://arxiv.org/abs/2204.14268v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.14268Focus to learn more |





# 2022-04-29

[Return to Index](#Index)



<h2 id="2022-04-29-1">1. UniTE: Unified Translation Evaluation
</h2>


Title: [UniTE: Unified Translation Evaluation](https://arxiv.org/abs/2204.13346)

Authors: [Yu Wan](https://arxiv.org/search/cs?searchtype=author&query=Wan%2C+Y), [Dayiheng Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+D), [Baosong Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+B), [Haibo Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+H), [Boxing Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+B), [Derek F. Wong](https://arxiv.org/search/cs?searchtype=author&query=Wong%2C+D+F), [Lidia S. Chao](https://arxiv.org/search/cs?searchtype=author&query=Chao%2C+L+S)

> Translation quality evaluation plays a crucial role in machine translation. According to the input format, it is mainly separated into three tasks, i.e., reference-only, source-only and source-reference-combined. Recent methods, despite their promising results, are specifically designed and optimized on one of them. This limits the convenience of these methods, and overlooks the commonalities among tasks. In this paper, we propose UniTE, which is the first unified framework engaged with abilities to handle all three evaluation tasks. Concretely, we propose monotonic regional attention to control the interaction among input segments, and unified pretraining to better adapt multi-task learning. We testify our framework on WMT 2019 Metrics and WMT 2020 Quality Estimation benchmarks. Extensive analyses show that our \textit{single model} can universally surpass various state-of-the-art or winner methods across tasks. Both source code and associated models are available at [this https URL](https://github.com/NLP2CT/UniTE).

| Comments: | ACL2022                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2204.13346](https://arxiv.org/abs/2204.13346) [cs.CL]** |
|           | (or **[arXiv:2204.13346v1](https://arxiv.org/abs/2204.13346v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.13346Focus to learn more |

