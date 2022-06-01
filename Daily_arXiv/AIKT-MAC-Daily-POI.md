# MA C.'s Daily Paper Of Interest - June a., 2022

# Index

- [2022-06-01](#2022-06-01)
  - [1. Parameter-Efficient and Student-Friendly Knowledge Distillation](#2022-06-01-1)
  
  - [2. ADAPT: Vision-Language Navigation with Modality-Aligned Action Prompts](#2022-06-01-2)
  
  - [3. CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers](#2022-06-01-3)
  
  - [4. EMS: Efficient and Effective Massively Multilingual Sentence Representation Learning](#2022-06-01-4)
  
- [2022-05-31](#2022-05-31)
  - [1. VLUE: A Multi-Task Benchmark for Evaluating Vision-Language Models](#2022-05-31-1)

- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



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



