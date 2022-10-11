# MA C.'s Daily Paper Of Interest - October a., 2022

# Index

- [2022-10-11](#2022-10-11)
  - [1. AlphaTuning: Quantization-Aware Parameter-Efficient Adaptation of Large-Scale Pre-Trained Language Models](#2022-10-11-1)
  
  - [2. MAMO: Masked Multimodal Modeling for Fine-Grained Vision-Language Representation Learning](#2022-10-11-2)
  
  - [3. What the DAAM: Interpreting Stable Diffusion Using Cross Attention](#2022-10-11-3)
  
  - [4. Visualize Before You Write: Imagination-Guided Open-Ended Text Generation](#2022-10-11-4)
  
  - [5. Breaking BERT: Evaluating and Optimizing Sparsified Attention](#2022-10-11-5)
  
  - [6. Improving End-to-End Text Image Translation From the Auxiliary Text Translation Task](#2022-10-11-6)
  
  - [7. Detecting Label Errors in Token Classification Data](#2022-10-11-7)
  
  - [8. Sparse Teachers Can Be Dense with Knowledge](#2022-10-11-8)
  
  - [9. Non-Monotonic Latent Alignments for CTC-Based Non-Autoregressive Machine Translation](#2022-10-11-9)
  
  - [10. SDA: Simple Discrete Augmentation for Contrastive Sentence Representation Learning](#2022-10-11-10)
  
  - [11. KG-MTT-BERT: Knowledge Graph Enhanced BERT for Multi-Type Medical Text Classification](#2022-10-11-11)
  
  - [12. Cross-Align: Modeling Deep Cross-lingual Interactions for Word Alignment](#2022-10-11-12)
  
  - [13. SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters](#2022-10-11-13)
  
  - [14. Parameter-Efficient Tuning with Special Token Adaptation](#2022-10-11-14)
  
  - [15. Distill the Image to Nowhere: Inversion Knowledge Distillation for Multimodal Machine Translation](#2022-10-11-15)
  
  - [16. Automatic Evaluation and Analysis of Idioms in Neural Machine Translation](#2022-10-11-16)
  
  - [17. A Survey of Methods for Addressing Class Imbalance in Deep-Learning Based Natural Language Processing](#2022-10-11-17)
  
- [2022-10-10](#2022-10-10)
  - [1. Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding](#2022-10-10-1)

  - [2. NMTSloth: Understanding and Testing Efficiency Degradation of Neural Machine Translation Systems](#2022-10-10-2)

  - [3. SpeechUT: Bridging Speech and Text with Hidden-Unit for Encoder-Decoder Based Speech-Text Pre-training](#2022-10-10-3)

- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



# 2022-10-11

[Return to Index](#Index)



<h2 id="2022-10-11-1">1. AlphaTuning: Quantization-Aware Parameter-Efficient Adaptation of Large-Scale Pre-Trained Language Models
</h2>

Title: [AlphaTuning: Quantization-Aware Parameter-Efficient Adaptation of Large-Scale Pre-Trained Language Models](https://arxiv.org/abs/2210.03858)

Authors: [Se Jung Kwon](https://arxiv.org/search/cs?searchtype=author&query=Kwon%2C+S+J), [Jeonghoon Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+J), [Jeongin Bae](https://arxiv.org/search/cs?searchtype=author&query=Bae%2C+J), [Kang Min Yoo](https://arxiv.org/search/cs?searchtype=author&query=Yoo%2C+K+M), [Jin-Hwa Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+J), [Baeseong Park](https://arxiv.org/search/cs?searchtype=author&query=Park%2C+B), [Byeongwook Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+B), [Jung-Woo Ha](https://arxiv.org/search/cs?searchtype=author&query=Ha%2C+J), [Nako Sung](https://arxiv.org/search/cs?searchtype=author&query=Sung%2C+N), [Dongsoo Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+D)

> There are growing interests in adapting large-scale language models using parameter-efficient fine-tuning methods. However, accelerating the model itself and achieving better inference efficiency through model compression has not been thoroughly explored yet. Model compression could provide the benefits of reducing memory footprints, enabling low-precision computations, and ultimately achieving cost-effective inference. To combine parameter-efficient adaptation and model compression, we propose AlphaTuning consisting of post-training quantization of the pre-trained language model and fine-tuning only some parts of quantized parameters for a target task. Specifically, AlphaTuning works by employing binary-coding quantization, which factorizes the full-precision parameters into binary parameters and a separate set of scaling factors. During the adaptation phase, the binary values are frozen for all tasks, while the scaling factors are fine-tuned for the downstream task. We demonstrate that AlphaTuning, when applied to GPT-2 and OPT, performs competitively with full fine-tuning on a variety of downstream tasks while achieving >10x compression ratio under 4-bit quantization and >1,000x reduction in the number of trainable parameters.

| Comments: | Findings of EMNLP 2022                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2210.03858](https://arxiv.org/abs/2210.03858) [cs.LG]** |
|           | (or **[arXiv:2210.03858v1](https://arxiv.org/abs/2210.03858v1) [cs.LG]** for this version) |
|           | https://doi.org/10.48550/arXiv.2210.03858Focus to learn more |





<h2 id="2022-10-11-2">2. MAMO: Masked Multimodal Modeling for Fine-Grained Vision-Language Representation Learning
</h2>

Title: [MAMO: Masked Multimodal Modeling for Fine-Grained Vision-Language Representation Learning](https://arxiv.org/abs/2210.04183)

Authors: [Zijia Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+Z), [Longteng Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+L), [Xingjian He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+X), [Shuai Shao](https://arxiv.org/search/cs?searchtype=author&query=Shao%2C+S), [Zehuan Yuan](https://arxiv.org/search/cs?searchtype=author&query=Yuan%2C+Z), [Jing Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+J)

> Multimodal representation learning has shown promising improvements on various vision-language tasks. Most existing methods excel at building global-level alignment between vision and language while lacking effective fine-grained image-text interaction. In this paper, we propose a jointly masked multimodal modeling method to learn fine-grained multimodal representations. Our method performs joint masking on image-text input and integrates both implicit and explicit targets for the masked signals to recover. The implicit target provides a unified and debiased objective for vision and language, where the model predicts latent multimodal representations of the unmasked input. The explicit target further enriches the multimodal representations by recovering high-level and semantically meaningful information: momentum visual features of image patches and concepts of word tokens. Through such a masked modeling process, our model not only learns fine-grained multimodal interaction, but also avoids the semantic gap between high-level representations and low- or mid-level prediction targets (e.g. image pixels), thus producing semantically rich multimodal representations that perform well on both zero-shot and fine-tuned settings. Our pre-trained model (named MAMO) achieves state-of-the-art performance on various downstream vision-language tasks, including image-text retrieval, visual question answering, visual reasoning, and weakly-supervised visual grounding.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG); Multimedia (cs.MM) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2210.04183](https://arxiv.org/abs/2210.04183) [cs.CV]** |
|           | (or **[arXiv:2210.04183v1](https://arxiv.org/abs/2210.04183v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2210.04183Focus to learn more |





<h2 id="2022-10-11-3">3. What the DAAM: Interpreting Stable Diffusion Using Cross Attention
</h2>

Title: [What the DAAM: Interpreting Stable Diffusion Using Cross Attention](https://arxiv.org/abs/2210.04885)

Authors: [Raphael Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+R), [Akshat Pandey](https://arxiv.org/search/cs?searchtype=author&query=Pandey%2C+A), [Zhiying Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+Z), [Gefei Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+G), [Karun Kumar](https://arxiv.org/search/cs?searchtype=author&query=Kumar%2C+K), [Jimmy Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+J), [Ferhan Ture](https://arxiv.org/search/cs?searchtype=author&query=Ture%2C+F)

> Large-scale diffusion neural networks represent a substantial milestone in text-to-image generation, with some performing similar to real photographs in human evaluation. However, they remain poorly understood, lacking explainability and interpretability analyses, largely due to their proprietary, closed-source nature. In this paper, to shine some much-needed light on text-to-image diffusion models, we perform a text-image attribution analysis on Stable Diffusion, a recently open-sourced large diffusion model. To produce pixel-level attribution maps, we propose DAAM, a novel method based on upscaling and aggregating cross-attention activations in the latent denoising subnetwork. We support its correctness by evaluating its unsupervised instance segmentation quality on its own generated imagery, compared to supervised segmentation models. We show that DAAM performs strongly on COCO caption-generated images, achieving an average precision (AP) of 61.0, and it outperforms supervised models on full-vocabulary segmentation, for an AP of 51.5. We further find that certain parts of speech, like punctuation and conjunctions, influence the generated imagery most, which agrees with the prior literature, while determiners and numerals the least, suggesting poor numeracy. To our knowledge, we are the first to propose and study word--pixel attribution for large-scale text-to-image diffusion models. Our code and data are at [this https URL](https://github.com/castorini/daam)

| Comments: | 5 pages, 5 figures                                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2210.04885](https://arxiv.org/abs/2210.04885) [cs.CV]** |
|           | (or **[arXiv:2210.04885v1](https://arxiv.org/abs/2210.04885v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2210.04885Focus to learn more |





<h2 id="2022-10-11-4">4. Visualize Before You Write: Imagination-Guided Open-Ended Text Generation
</h2>

Title: [Visualize Before You Write: Imagination-Guided Open-Ended Text Generation](https://arxiv.org/abs/2210.03765)

Authors: [Wanrong Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+W), [An Yan](https://arxiv.org/search/cs?searchtype=author&query=Yan%2C+A), [Yujie Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+Y), [Wenda Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+W), [Xin Eric Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X+E), [Miguel Eckstein](https://arxiv.org/search/cs?searchtype=author&query=Eckstein%2C+M), [William Yang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+W+Y)

> Recent advances in text-to-image synthesis make it possible to visualize machine imaginations for a given context. On the other hand, when generating text, human writers are gifted at creative visualization, which enhances their writings by forming imaginations as blueprints before putting down the stories in words. Inspired by such a cognitive process, we ask the natural question of whether we can endow machines with the same ability to utilize visual information and construct a general picture of the context to guide text generation. In this work, we propose iNLG that uses machine-generated images to guide language models (LM) in open-ended text generation. The experiments and analyses demonstrate the effectiveness of iNLG on open-ended text generation tasks, including text completion, story generation, and concept-to-text generation in few-shot scenarios. Both automatic metrics and human evaluations verify that the text snippets generated by our iNLG are coherent and informative while displaying minor degeneration.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2210.03765](https://arxiv.org/abs/2210.03765) [cs.CL]** |
|           | (or **[arXiv:2210.03765v1](https://arxiv.org/abs/2210.03765v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2210.03765Focus to learn more |





<h2 id="2022-10-11-5">5. Breaking BERT: Evaluating and Optimizing Sparsified Attention
</h2>

Title: [Breaking BERT: Evaluating and Optimizing Sparsified Attention](https://arxiv.org/abs/2210.03841)

Authors: [Siddhartha Brahma](https://arxiv.org/search/cs?searchtype=author&query=Brahma%2C+S), [Polina Zablotskaia](https://arxiv.org/search/cs?searchtype=author&query=Zablotskaia%2C+P), [David Mimno](https://arxiv.org/search/cs?searchtype=author&query=Mimno%2C+D)

> Transformers allow attention between all pairs of tokens, but there is reason to believe that most of these connections - and their quadratic time and memory - may not be necessary. But which ones? We evaluate the impact of sparsification patterns with a series of ablation experiments. First, we compare masks based on syntax, lexical similarity, and token position to random connections, and measure which patterns reduce performance the least. We find that on three common finetuning tasks even using attention that is at least 78% sparse can have little effect on performance if applied at later transformer layers, but that applying sparsity throughout the network reduces performance significantly. Second, we vary the degree of sparsity for three patterns supported by previous work, and find that connections to neighbouring tokens are the most significant. Finally, we treat sparsity as an optimizable parameter, and present an algorithm to learn degrees of neighboring connections that gives a fine-grained control over the accuracy-sparsity trade-off while approaching the performance of existing methods.

| Comments: | Shorter version accepted to SNN2021 workshop                 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2210.03841](https://arxiv.org/abs/2210.03841) [cs.CL]** |
|           | (or **[arXiv:2210.03841v1](https://arxiv.org/abs/2210.03841v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2210.03841Focus to learn more |





<h2 id="2022-10-11-6">6. Improving End-to-End Text Image Translation From the Auxiliary Text Translation Task
</h2>

Title: [Improving End-to-End Text Image Translation From the Auxiliary Text Translation Task](https://arxiv.org/abs/2210.03887)

Authors: [Cong Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+C), [Yaping Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y), [Mei Tu](https://arxiv.org/search/cs?searchtype=author&query=Tu%2C+M), [Xu Han](https://arxiv.org/search/cs?searchtype=author&query=Han%2C+X), [Linghui Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+L), [Yang Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+Y), [Yu Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+Y)

> End-to-end text image translation (TIT), which aims at translating the source language embedded in images to the target language, has attracted intensive attention in recent research. However, data sparsity limits the performance of end-to-end text image translation. Multi-task learning is a non-trivial way to alleviate this problem via exploring knowledge from complementary related tasks. In this paper, we propose a novel text translation enhanced text image translation, which trains the end-to-end model with text translation as an auxiliary task. By sharing model parameters and multi-task training, our model is able to take full advantage of easily-available large-scale text parallel corpus. Extensive experimental results show our proposed method outperforms existing end-to-end methods, and the joint multi-task learning with both text translation and recognition tasks achieves better results, proving translation and recognition auxiliary tasks are complementary.

| Comments: | Accepted at the 26TH International Conference on Pattern Recognition (ICPR 2022) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2210.03887](https://arxiv.org/abs/2210.03887) [cs.CL]** |
|           | (or **[arXiv:2210.03887v1](https://arxiv.org/abs/2210.03887v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2210.03887Focus to learn more |





<h2 id="2022-10-11-7">7. Detecting Label Errors in Token Classification Data
</h2>

Title: [Detecting Label Errors in Token Classification Data](https://arxiv.org/abs/2210.03920)

Authors: [Wei-Chen Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+W), [Jonas Mueller](https://arxiv.org/search/cs?searchtype=author&query=Mueller%2C+J)

> Mislabeled examples are a common issue in real-world data, particularly for tasks like token classification where many labels must be chosen on a fine-grained basis. Here we consider the task of finding sentences that contain label errors in token classification datasets. We study 11 different straightforward methods that score tokens/sentences based on the predicted class probabilities output by a (any) token classification model (trained via any procedure). In precision-recall evaluations based on real-world label errors in entity recognition data from CoNLL-2003, we identify a simple and effective method that consistently detects those sentences containing label errors when applied with different token classification models.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2210.03920](https://arxiv.org/abs/2210.03920) [cs.CL]** |
|           | (or **[arXiv:2210.03920v1](https://arxiv.org/abs/2210.03920v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2210.03920Focus to learn more |





<h2 id="2022-10-11-8">8. Sparse Teachers Can Be Dense with Knowledge
</h2>

Title: [Sparse Teachers Can Be Dense with Knowledge](https://arxiv.org/abs/2210.03923)

Authors: [Yi Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Y), [Chen Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+C), [Dawei Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+D)

> Recent advances in distilling pretrained language models have discovered that, besides the expressiveness of knowledge, the student-friendliness should be taken into consideration to realize a truly knowledgable teacher. Based on a pilot study, we find that over-parameterized teachers can produce expressive yet student-unfriendly knowledge, and are thus limited in overall knowledgableness. To remove the parameters that result in student-unfriendliness, we propose a sparse teacher trick under the guidance of an overall knowledgable score for each teacher parameter. The knowledgable score is essentially an interpolation of the expressiveness and student-friendliness scores. The aim is to ensure that the expressive parameters are retained while the student-unfriendly ones are removed. Extensive experiments on the GLUE benchmark show that the proposed sparse teachers can be dense with knowledge and lead to students with compelling performance in comparison with a series of competitive baselines.

| Comments: | 12 pages, 8 figures, 6 tables, accepted to EMNLP 2022. Code is available at [this https URL](https://github.com/GeneZC/StarK) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2210.03923](https://arxiv.org/abs/2210.03923) [cs.CL]** |
|           | (or **[arXiv:2210.03923v1](https://arxiv.org/abs/2210.03923v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2210.03923Focus to learn more |





<h2 id="2022-10-11-9">9. Non-Monotonic Latent Alignments for CTC-Based Non-Autoregressive Machine Translation
</h2>

Title: [Non-Monotonic Latent Alignments for CTC-Based Non-Autoregressive Machine Translation](https://arxiv.org/abs/2210.03953)

Authors: [Chenze Shao](https://arxiv.org/search/cs?searchtype=author&query=Shao%2C+C), [Yang Feng](https://arxiv.org/search/cs?searchtype=author&query=Feng%2C+Y)

> Non-autoregressive translation (NAT) models are typically trained with the cross-entropy loss, which forces the model outputs to be aligned verbatim with the target sentence and will highly penalize small shifts in word positions. Latent alignment models relax the explicit alignment by marginalizing out all monotonic latent alignments with the CTC loss. However, they cannot handle non-monotonic alignments, which is non-negligible as there is typically global word reordering in machine translation. In this work, we explore non-monotonic latent alignments for NAT. We extend the alignment space to non-monotonic alignments to allow for the global word reordering and further consider all alignments that overlap with the target sentence. We non-monotonically match the alignments to the target sentence and train the latent alignment model to maximize the F1 score of non-monotonic matching. Extensive experiments on major WMT benchmarks show that our method substantially improves the translation performance of CTC-based models. Our best model achieves 30.06 BLEU on WMT14 En-De with only one-iteration decoding, closing the gap between non-autoregressive and autoregressive models.

| Comments: | NeurIPS 2022                                                 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2210.03953](https://arxiv.org/abs/2210.03953) [cs.CL]** |
|           | (or **[arXiv:2210.03953v1](https://arxiv.org/abs/2210.03953v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2210.03953Focus to learn more |





<h2 id="2022-10-11-10">10. SDA: Simple Discrete Augmentation for Contrastive Sentence Representation Learning
</h2>

Title: [SDA: Simple Discrete Augmentation for Contrastive Sentence Representation Learning](https://arxiv.org/abs/2210.03963)

Authors: [Zhenyu Mao](https://arxiv.org/search/cs?searchtype=author&query=Mao%2C+Z), [Dongsheng Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+D), [Jinghui Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+J), [Rui Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+R), [Fei Tan](https://arxiv.org/search/cs?searchtype=author&query=Tan%2C+F)

> Contrastive learning methods achieve state-of-the-art results in unsupervised sentence representation learning. Although playing essential roles in contrastive learning, data augmentation methods applied on sentences have not been fully explored. Current SOTA method SimCSE utilizes a simple dropout mechanism as continuous augmentation which outperforms discrete augmentations such as cropping, word deletion and synonym replacement. To understand the underlying rationales, we revisit existing approaches and attempt to hypothesize the desiderata of reasonable data augmentation methods: balance of semantic consistency and expression diversity. Based on the hypothesis, we propose three simple yet effective discrete sentence augmentation methods, i.e., punctuation insertion, affirmative auxiliary and double negation. The punctuation marks, auxiliaries and negative words act as minimal noises in lexical level to produce diverse sentence expressions. Unlike traditional augmentation methods which randomly modify the sentence, our augmentation rules are well designed for generating semantically consistent and grammatically correct sentences. We conduct extensive experiments on both English and Chinese semantic textual similarity datasets. The results show the robustness and effectiveness of the proposed methods.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2210.03963](https://arxiv.org/abs/2210.03963) [cs.CL]** |
|           | (or **[arXiv:2210.03963v1](https://arxiv.org/abs/2210.03963v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2210.03963Focus to learn more |





<h2 id="2022-10-11-11">11. KG-MTT-BERT: Knowledge Graph Enhanced BERT for Multi-Type Medical Text Classification
</h2>

Title: [KG-MTT-BERT: Knowledge Graph Enhanced BERT for Multi-Type Medical Text Classification](https://arxiv.org/abs/2210.03970)

Authors: [Yong He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+Y), [Cheng Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+C), [Shun Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+S), [Nan Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+N), [Zhaorong Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Z), [Zhenyu Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+Z)

> Medical text learning has recently emerged as a promising area to improve healthcare due to the wide adoption of electronic health record (EHR) systems. The complexity of the medical text such as diverse length, mixed text types, and full of medical jargon, poses a great challenge for developing effective deep learning models. BERT has presented state-of-the-art results in many NLP tasks, such as text classification and question answering. However, the standalone BERT model cannot deal with the complexity of the medical text, especially the lengthy clinical notes. Herein, we develop a new model called KG-MTT-BERT (Knowledge Graph Enhanced Multi-Type Text BERT) by extending the BERT model for long and multi-type text with the integration of the medical knowledge graph. Our model can outperform all baselines and other state-of-the-art models in diagnosis-related group (DRG) classification, which requires comprehensive medical text for accurate classification. We also demonstrated that our model can effectively handle multi-type text and the integration of medical knowledge graph can significantly improve the performance.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2210.03970](https://arxiv.org/abs/2210.03970) [cs.CL]** |
|           | (or **[arXiv:2210.03970v1](https://arxiv.org/abs/2210.03970v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2210.03970Focus to learn more |





<h2 id="2022-10-11-12">12. Cross-Align: Modeling Deep Cross-lingual Interactions for Word Alignment
</h2>

Title: [Cross-Align: Modeling Deep Cross-lingual Interactions for Word Alignment](https://arxiv.org/abs/2210.04141)

Authors: [Siyu Lai](https://arxiv.org/search/cs?searchtype=author&query=Lai%2C+S), [Zhen Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Z), [Fandong Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+F), [Yufeng Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Y), [Jinan Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+J), [Jie Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+J)

> Word alignment which aims to extract lexicon translation equivalents between source and target sentences, serves as a fundamental tool for natural language processing. Recent studies in this area have yielded substantial improvements by generating alignments from contextualized embeddings of the pre-trained multilingual language models. However, we find that the existing approaches capture few interactions between the input sentence pairs, which degrades the word alignment quality severely, especially for the ambiguous words in the monolingual context. To remedy this problem, we propose Cross-Align to model deep interactions between the input sentence pairs, in which the source and target sentences are encoded separately with the shared self-attention modules in the shallow layers, while cross-lingual interactions are explicitly constructed by the cross-attention modules in the upper layers. Besides, to train our model effectively, we propose a two-stage training framework, where the model is trained with a simple Translation Language Modeling (TLM) objective in the first stage and then finetuned with a self-supervised alignment objective in the second stage. Experiments show that the proposed Cross-Align achieves the state-of-the-art (SOTA) performance on four out of five language pairs.

| Comments: | Accepted by EMNLP 2022                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2210.04141](https://arxiv.org/abs/2210.04141) [cs.CL]** |
|           | (or **[arXiv:2210.04141v1](https://arxiv.org/abs/2210.04141v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2210.04141Focus to learn more |





<h2 id="2022-10-11-13">13. SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters
</h2>

Title: [SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters](https://arxiv.org/abs/2210.04284)

Authors: [Shwai He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+S), [Liang Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+L), [Daize Dong](https://arxiv.org/search/cs?searchtype=author&query=Dong%2C+D), [Miao Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+M), [Dacheng Tao](https://arxiv.org/search/cs?searchtype=author&query=Tao%2C+D)

> Adapter Tuning, which freezes the pretrained language models (PLMs) and only fine-tunes a few extra modules, becomes an appealing efficient alternative to the full model fine-tuning. Although computationally efficient, the recent Adapters often increase parameters (e.g. bottleneck dimension) for matching the performance of full model fine-tuning, which we argue goes against their original intention. In this work, we re-examine the parameter-efficiency of Adapters through the lens of network pruning (we name such plug-in concept as \texttt{SparseAdapter}) and find that SparseAdapter can achieve comparable or better performance than standard Adapters when the sparse ratio reaches up to 80\%. Based on our findings, we introduce an easy but effective setting ``\textit{Large-Sparse}'' to improve the model capacity of Adapters under the same parameter budget. Experiments on five competitive Adapters upon three advanced PLMs show that with proper sparse method (e.g. SNIP) and ratio (e.g. 40\%) SparseAdapter can consistently outperform their corresponding counterpart. Encouragingly, with the \textit{Large-Sparse} setting, we can obtain further appealing gains, even outperforming the full fine-tuning by a large margin. Our code will be released at: \url{[this https URL](https://github.com/Shwai-He/SparseAdapter)}.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2210.04284](https://arxiv.org/abs/2210.04284) [cs.CL]** |
|           | (or **[arXiv:2210.04284v1](https://arxiv.org/abs/2210.04284v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2210.04284Focus to learn more |





<h2 id="2022-10-11-14">14. Parameter-Efficient Tuning with Special Token Adaptation
</h2>

Title: [Parameter-Efficient Tuning with Special Token Adaptation](https://arxiv.org/abs/2210.04382)

Authors: [Xiaoocong Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+X), [James Y. Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+J+Y), [Wenxuan Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+W), [Muhao Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+M)

> Parameter-efficient tuning aims at updating only a small subset of parameters when adapting a pretrained model to downstream tasks. In this work, we introduce PASTA, in which we only modify the special token representations (e.g., [SEP] and [CLS] in BERT) before the self-attention module at each layer in Transformer-based models. PASTA achieves comparable performance to fine-tuning in natural language understanding tasks including text classification and NER with up to only 0.029% of total parameters trained. Our work not only provides a simple yet effective way of parameter-efficient tuning, which has a wide range of practical applications when deploying finetuned models for multiple tasks, but also demonstrates the pivotal role of special tokens in pretrained language models.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2210.04382](https://arxiv.org/abs/2210.04382) [cs.CL]** |
|           | (or **[arXiv:2210.04382v1](https://arxiv.org/abs/2210.04382v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2210.04382Focus to learn more |





<h2 id="2022-10-11-15">15. Distill the Image to Nowhere: Inversion Knowledge Distillation for Multimodal Machine Translation
</h2>

Title: [Distill the Image to Nowhere: Inversion Knowledge Distillation for Multimodal Machine Translation](https://arxiv.org/abs/2210.04468)

Authors: [Ru Peng](https://arxiv.org/search/cs?searchtype=author&query=Peng%2C+R), [Yawen Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+Y), [Junbo Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+J)

> Past works on multimodal machine translation (MMT) elevate bilingual setup by incorporating additional aligned vision information. However, an image-must requirement of the multimodal dataset largely hinders MMT's development -- namely that it demands an aligned form of [image, source text, target text]. This limitation is generally troublesome during the inference phase especially when the aligned image is not provided as in the normal NMT setup. Thus, in this work, we introduce IKD-MMT, a novel MMT framework to support the image-free inference phase via an inversion knowledge distillation scheme. In particular, a multimodal feature generator is executed with a knowledge distillation module, which directly generates the multimodal feature from (only) source texts as the input. While there have been a few prior works entertaining the possibility to support image-free inference for machine translation, their performances have yet to rival the image-must translation. In our experiments, we identify our method as the first image-free approach to comprehensively rival or even surpass (almost) all image-must frameworks, and achieved the state-of-the-art result on the often-used Multi30k benchmark. Our code and data are available at: [this https URL](https://github.com/pengr/IKD-mmt/tree/master)..

| Comments: | Long paper accepted by EMNLP2022 main conference             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2210.04468](https://arxiv.org/abs/2210.04468) [cs.CL]** |
|           | (or **[arXiv:2210.04468v1](https://arxiv.org/abs/2210.04468v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2210.04468Focus to learn more |





<h2 id="2022-10-11-16">16. Automatic Evaluation and Analysis of Idioms in Neural Machine Translation
</h2>

Title: [Automatic Evaluation and Analysis of Idioms in Neural Machine Translation](https://arxiv.org/abs/2210.04545)

Authors: [Christos Baziotis](https://arxiv.org/search/cs?searchtype=author&query=Baziotis%2C+C), [Prashant Mathur](https://arxiv.org/search/cs?searchtype=author&query=Mathur%2C+P), [Eva Hasler](https://arxiv.org/search/cs?searchtype=author&query=Hasler%2C+E)

> A major open problem in neural machine translation (NMT) is the translation of idiomatic expressions, such as "under the weather". The meaning of these expressions is not composed by the meaning of their constituent words, and NMT models tend to translate them literally (i.e., word-by-word), which leads to confusing and nonsensical translations. Research on idioms in NMT is limited and obstructed by the absence of automatic methods for quantifying these errors. In this work, first, we propose a novel metric for automatically measuring the frequency of literal translation errors without human involvement. Equipped with this metric, we present controlled translation experiments with models trained in different conditions (with/without the test-set idioms) and across a wide range of (global and targeted) metrics and test sets. We explore the role of monolingual pretraining and find that it yields substantial targeted improvements, even without observing any translation examples of the test-set idioms. In our analysis, we probe the role of idiom context. We find that the randomly initialized models are more local or "myopic" as they are relatively unaffected by variations of the idiom context, unlike the pretrained ones.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2210.04545](https://arxiv.org/abs/2210.04545) [cs.CL]** |
|           | (or **[arXiv:2210.04545v1](https://arxiv.org/abs/2210.04545v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2210.04545Focus to learn more |





<h2 id="2022-10-11-17">17. A Survey of Methods for Addressing Class Imbalance in Deep-Learning Based Natural Language Processing
</h2>

Title: [A Survey of Methods for Addressing Class Imbalance in Deep-Learning Based Natural Language Processing](https://arxiv.org/abs/2210.04675)

Authors: [Sophie Henning](https://arxiv.org/search/cs?searchtype=author&query=Henning%2C+S), [William H. Beluch](https://arxiv.org/search/cs?searchtype=author&query=Beluch%2C+W+H), [Alexander Fraser](https://arxiv.org/search/cs?searchtype=author&query=Fraser%2C+A), [Annemarie Friedrich](https://arxiv.org/search/cs?searchtype=author&query=Friedrich%2C+A)

> Many natural language processing (NLP) tasks are naturally imbalanced, as some target categories occur much more frequently than others in the real world. In such scenarios, current NLP models still tend to perform poorly on less frequent classes. Addressing class imbalance in NLP is an active research topic, yet, finding a good approach for a particular task and imbalance scenario is difficult. 
> With this survey, the first overview on class imbalance in deep-learning based NLP, we provide guidance for NLP researchers and practitioners dealing with imbalanced data. We first discuss various types of controlled and real-world class imbalance. Our survey then covers approaches that have been explicitly proposed for class-imbalanced NLP tasks or, originating in the computer vision community, have been evaluated on them. We organize the methods by whether they are based on sampling, data augmentation, choice of loss function, staged learning, or model design. Finally, we discuss open problems such as dealing with multi-label scenarios, and propose systematic benchmarking and reporting in order to move forward on this problem as a community.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2210.04675](https://arxiv.org/abs/2210.04675) [cs.CL]** |
|           | (or **[arXiv:2210.04675v1](https://arxiv.org/abs/2210.04675v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2210.04675Focus to learn more |



# 2022-10-10

[Return to Index](#Index)



<h2 id="2022-10-10-1">1. Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding
</h2>


Title: [Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding](https://arxiv.org/abs/2210.03347)

Authors: [Kenton Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+K), [Mandar Joshi](https://arxiv.org/search/cs?searchtype=author&query=Joshi%2C+M), [Iulia Turc](https://arxiv.org/search/cs?searchtype=author&query=Turc%2C+I), [Hexiang Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+H), [Fangyu Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+F), [Julian Eisenschlos](https://arxiv.org/search/cs?searchtype=author&query=Eisenschlos%2C+J), [Urvashi Khandelwal](https://arxiv.org/search/cs?searchtype=author&query=Khandelwal%2C+U), [Peter Shaw](https://arxiv.org/search/cs?searchtype=author&query=Shaw%2C+P), [Ming-Wei Chang](https://arxiv.org/search/cs?searchtype=author&query=Chang%2C+M), [Kristina Toutanova](https://arxiv.org/search/cs?searchtype=author&query=Toutanova%2C+K)

> Visually-situated language is ubiquitous -- sources range from textbooks with diagrams to web pages with images and tables, to mobile apps with buttons and forms. Perhaps due to this diversity, previous work has typically relied on domain-specific recipes with limited sharing of the underlying data, model architectures, and objectives. We present Pix2Struct, a pretrained image-to-text model for purely visual language understanding, which can be finetuned on tasks containing visually-situated language. Pix2Struct is pretrained by learning to parse masked screenshots of web pages into simplified HTML. The web, with its richness of visual elements cleanly reflected in the HTML structure, provides a large source of pretraining data well suited to the diversity of downstream tasks. Intuitively, this objective subsumes common pretraining signals such as OCR, language modeling, image captioning. In addition to the novel pretraining strategy, we introduce a variable-resolution input representation and a more flexible integration of language and vision inputs, where language prompts such as questions are rendered directly on top of the input image. For the first time, we show that a single pretrained model can achieve state-of-the-art results in six out of nine tasks across four domains: documents, illustrations, user interfaces, and natural images.

| Subjects: | **Computation and Language (cs.CL)**; Computer Vision and Pattern Recognition (cs.CV) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2210.03347](https://arxiv.org/abs/2210.03347) [cs.CL]** |
|           | (or **[arXiv:2210.03347v1](https://arxiv.org/abs/2210.03347v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2210.03347Focus to learn more |





<h2 id="2022-10-10-2">2. NMTSloth: Understanding and Testing Efficiency Degradation of Neural Machine Translation Systems
</h2>


Title: [NMTSloth: Understanding and Testing Efficiency Degradation of Neural Machine Translation Systems](https://arxiv.org/abs/2210.03696)

Authors: [Simin Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+S), [Cong Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+C), [Mirazul Haque](https://arxiv.org/search/cs?searchtype=author&query=Haque%2C+M), [Zihe Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+Z), [Wei Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+W)

> Neural Machine Translation (NMT) systems have received much recent attention due to their human-level accuracy. While existing works mostly focus on either improving accuracy or testing accuracy robustness, the computation efficiency of NMT systems, which is of paramount importance due to often vast translation demands and real-time requirements, has surprisingly received little attention. In this paper, we make the first attempt to understand and test potential computation efficiency robustness in state-of-the-art NMT systems. By analyzing the working mechanism and implementation of 1455 public-accessible NMT systems, we observe a fundamental property in NMT systems that could be manipulated in an adversarial manner to reduce computation efficiency significantly. Our key motivation is to generate test inputs that could sufficiently delay the generation of EOS such that NMT systems would have to go through enough iterations to satisfy the pre-configured threshold. We present NMTSloth, which develops a gradient-guided technique that searches for a minimal and unnoticeable perturbation at character-level, token-level, and structure-level, which sufficiently delays the appearance of EOS and forces these inputs to reach the naturally-unreachable threshold. To demonstrate the effectiveness of NMTSloth, we conduct a systematic evaluation on three public-available NMT systems: Google T5, AllenAI WMT14, and Helsinki-NLP translators. Experimental results show that NMTSloth can increase NMT systems' response latency and energy consumption by 85% to 3153% and 86% to 3052%, respectively, by perturbing just one character or token in the input sentence. Our case study shows that inputs generated by NMTSloth significantly affect the battery power in real-world mobile devices (i.e., drain more than 30 times battery power than normal inputs).

| Comments: | This paper has been accepted to ESEC/FSE 2022                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Software Engineering (cs.SE) |
| Cite as:  | **[arXiv:2210.03696](https://arxiv.org/abs/2210.03696) [cs.CL]** |
|           | (or **[arXiv:2210.03696v1](https://arxiv.org/abs/2210.03696v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2210.03696Focus to learn more |





<h2 id="2022-10-10-3">3. SpeechUT: Bridging Speech and Text with Hidden-Unit for Encoder-Decoder Based Speech-Text Pre-training
</h2>


Title: [SpeechUT: Bridging Speech and Text with Hidden-Unit for Encoder-Decoder Based Speech-Text Pre-training](https://arxiv.org/abs/2210.03730)

Authors: [Ziqiang Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Z), [Long Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+L), [Junyi Ao](https://arxiv.org/search/cs?searchtype=author&query=Ao%2C+J), [Shujie Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+S), [Lirong Dai](https://arxiv.org/search/cs?searchtype=author&query=Dai%2C+L), [Jinyu Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F)

> The rapid development of single-modal pre-training has prompted researchers to pay more attention to cross-modal pre-training methods. In this paper, we propose a unified-modal speech-unit-text pre-training model, SpeechUT, to connect the representations of a speech encoder and a text decoder with a shared unit encoder. Leveraging hidden-unit as an interface to align speech and text, we can decompose the speech-to-text model into a speech-to-unit model and a unit-to-text model, which can be jointly pre-trained with unpaired speech and text data respectively. Our proposed SpeechUT is fine-tuned and evaluated on automatic speech recognition (ASR) and speech translation (ST) tasks. Experimental results show that SpeechUT gets substantial improvements over strong baselines, and achieves state-of-the-art performance on both the LibriSpeech ASR and MuST-C ST tasks. To better understand the proposed SpeechUT, detailed analyses are conducted. The code and pre-trained models are available at [this https URL](https://aka.ms/SpeechUT).

| Comments: | 14 pages, accepted by EMNLP 2022                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2210.03730](https://arxiv.org/abs/2210.03730) [cs.CL]** |
|           | (or **[arXiv:2210.03730v1](https://arxiv.org/abs/2210.03730v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2210.03730Focus to learn more |

