# MA C.'s Daily Paper Of Interest - December, 2021

# Index


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

