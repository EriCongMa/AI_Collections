# MA C.'s Daily Paper Of Interest - November, 2022

# Index

- [2023-11-29](#2023-11-29)
  - [1. A Benchmark for Evaluating Machine Translation Metrics on Dialects Without Standard Orthography](#2023-11-29-1)
  
  - [2. CharacterGLM: Customizing Chinese Conversational AI Characters with Large Language Models](#2023-11-29-2)
  
  - [3. Evaluating Optimal Reference Translations](#2023-11-29-3)
  
  - [4. Reducing Gender Bias in Machine Translation through Counterfactual Data Generation](#2023-11-29-4)
  
  - [5. MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training](#2023-11-29-5)
  
  - [6. LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models](#2023-11-29-6)
  
  - [7. Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding](#2023-11-29-7)
  
  - [8. RELIC: Investigating Large Language Model Responses using Self-Consistency](#2023-11-29-8)
  
  - [9. ChartLlama: A Multimodal LLM for Chart Understanding and Generation](#2023-11-29-9)
  
  - [10. ChatTraffc: Text-to-Traffic Generation via Diffusion Model](#2023-11-29-10)
  
  - [11. Pre-trained Language Models Do Not Help Auto-regressive Text-to-Image Generation](#2023-11-29-11)
  
- [2023-11-28](#2023-11-28)
  - [1. DUnE: Dataset for Unified Editing](#2023-11-28-1)

  - [2. MEDITRON-70B: Scaling Medical Pretraining for Large Language Models](#2023-11-28-2)

  - [3. A Quantitative Approach to Understand Self-Supervised Models as Cross-lingual Feature Extractors](#2023-11-28-3)

  - [4. WorldSense: A Synthetic Benchmark for Grounded Reasoning in Large Language Models](#2023-11-28-4)

  - [5. YUAN 2.0: A Large Language Model with Localized Filtering-based Attention](#2023-11-28-5)

  - [6. Knowledge Unlearning for LLMs: Tasks, Methods, and Challenges](#2023-11-28-6)

  - [7. Towards Vision Enhancing LLMs: Empowering Multimodal Knowledge Storage and Sharing in LLMs](#2023-11-28-7)

  - [8. MoDS: Model-oriented Data Selection for Instruction Tuning](#2023-11-28-8)

  - [9. LongStory: Coherent, Complete and Length Controlled Long story Generation](#2023-11-28-9)

  - [10. Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion](#2023-11-28-10)

  - [11. Offensive Language Identification in Transliterated and Code-Mixed Bangla](#2023-11-28-11)

  - [12. Vector-Quantized Prompt Learning for Paraphrase Generation](#2023-11-28-12)

  - [13. How Many Unicorns Are in This Image? A Safety Evaluation Benchmark for Vision LLMs](#2023-11-28-13)

  - [14. Data Generation for Post-OCR correction of Cyrillic handwriting](#2023-11-28-14)

  - [15. Can Vision-Language Models Think from a First-Person Perspective?](#2023-11-28-15)

  - [16. ChatGPT and Beyond: The Generative AI Revolution in Education](#2023-11-28-16)

- [2023-11-27](#2023-11-27)
  - [1. Calibrated Language Models Must Hallucinate](#2023-11-27-1)

  - [2. GPT Struct Me: Probing GPT Models on Narrative Entity Extraction](#2023-11-27-2)

  - [3. Data-Efficient Alignment of Large Language Models with Human Feedback Through Natural Language](#2023-11-27-3)

  - [4. Machine Translation for Ge'ez Language](#2023-11-27-4)

  - [5. Controlled Text Generation via Language Model Arithmetic](#2023-11-27-5)

  - [6. DP-NMT: Scalable Differentially-Private Machine Translation](#2023-11-27-6)

  - [7. Evaluating GPT-4's Vision Capabilities on Brazilian University Admission Exams](#2023-11-27-7)

  - [8. MLLM-Bench, Evaluating Multi-modal LLMs using GPT-4V](#2023-11-27-8)

  - [9. Efficient Transformer Knowledge Distillation: A Performance Review](#2023-11-27-9)

  - [10. Language Model Inversion](#2023-11-27-10)

  - [11. tinyCLAP: Distilling Constrastive Language-Audio Pretrained Models](#2023-11-27-11)

  - [12. Prompt Risk Control: A Rigorous Framework for Responsible Deployment of Large Language Models](#2023-11-27-12)

- [2023-11-23](#2023-11-23)
  - [1. LM-Cocktail: Resilient Tuning of Language Models via Model Merging](#2022-11-23-1)
  - [2. Machine Translation to Control Formality Features in the Target Language](#2022-11-23-2)
  - [3. Mitigating Large Language Model Hallucinations via Autonomous Knowledge Graph-based Retrofitting](#2022-11-23-3)
  - [4. Automatic Instruction Optimization for Open-source LLM Instruction Tuning](#2022-11-23-4)
  - [5. On the Calibration of Large Language Models and Alignment](#2022-11-23-5)
  - [6. GAIA: a benchmark for General AI Assistants](#2022-11-23-6)
  - [7. AS-LLM: When Algorithm Selection Meets Large Language Model](#2022-11-23-7)
  - [8. LIMIT: Less Is More for Instruction Tuning Across Evaluation Paradigms](#2022-11-23-8)

- [2022-10-20](#2022-10-20)
  - [1. RankT5: Fine-Tuning T5 for Text Ranking with Ranking Losses](#2022-10-20-1)

- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



# 2023-11-29

[Return to Index](#Index)



<h2 id="2023-11-29-1">1. A Benchmark for Evaluating Machine Translation Metrics on Dialects Without Standard Orthography
</h2>

Title: [A Benchmark for Evaluating Machine Translation Metrics on Dialects Without Standard Orthography](https://arxiv.org/abs/2311.16865)

Authors: [Noëmi Aepli](https://arxiv.org/search/cs?searchtype=author&query=Aepli,+N), [Chantal Amrhein](https://arxiv.org/search/cs?searchtype=author&query=Amrhein,+C), [Florian Schottmann](https://arxiv.org/search/cs?searchtype=author&query=Schottmann,+F), [Rico Sennrich](https://arxiv.org/search/cs?searchtype=author&query=Sennrich,+R)

> For sensible progress in natural language processing, it is important that we are aware of the limitations of the evaluation metrics we use. In this work, we evaluate how robust metrics are to non-standardized dialects, i.e. spelling differences in language varieties that do not have a standard orthography. To investigate this, we collect a dataset of human translations and human judgments for automatic machine translations from English to two Swiss German dialects. We further create a challenge set for dialect variation and benchmark existing metrics' performances. Our results show that existing metrics cannot reliably evaluate Swiss German text generation outputs, especially on segment level. We propose initial design adaptations that increase robustness in the face of non-standardized dialects, although there remains much room for further improvement. The dataset, code, and models are available here: [this https URL](https://github.com/textshuttle/dialect_eval)

| Comments:    | WMT 2023 Research Paper                                      |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**                         |
| ACM classes: | I.2.7                                                        |
| Cite as:     | [arXiv:2311.16865](https://arxiv.org/abs/2311.16865) [cs.CL] |
|              | (or [arXiv:2311.16865v1](https://arxiv.org/abs/2311.16865v1) [cs.CL] for this version) |
|              | https://doi.org/10.48550/arXiv.2311.16865Focus to learn more |





<h2 id="2023-11-29-2">2. CharacterGLM: Customizing Chinese Conversational AI Characters with Large Language Models
</h2>

Title: [CharacterGLM: Customizing Chinese Conversational AI Characters with Large Language Models](https://arxiv.org/abs/2311.16832)

Authors: [Jinfeng Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou,+J), [Zhuang Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen,+Z), [Dazhen Wan](https://arxiv.org/search/cs?searchtype=author&query=Wan,+D), [Bosi Wen](https://arxiv.org/search/cs?searchtype=author&query=Wen,+B), [Yi Song](https://arxiv.org/search/cs?searchtype=author&query=Song,+Y), [Jifan Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu,+J), [Yongkang Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang,+Y), [Libiao Peng](https://arxiv.org/search/cs?searchtype=author&query=Peng,+L), [Jiaming Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang,+J), [Xiyao Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao,+X), [Sahand Sabour](https://arxiv.org/search/cs?searchtype=author&query=Sabour,+S), [Xiaohan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+X), [Wenjing Hou](https://arxiv.org/search/cs?searchtype=author&query=Hou,+W), [Yijia Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+Y), [Yuxiao Dong](https://arxiv.org/search/cs?searchtype=author&query=Dong,+Y), [Jie Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang,+J), [Minlie Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang,+M)

> In this paper, we present CharacterGLM, a series of models built upon ChatGLM, with model sizes ranging from 6B to 66B parameters. Our CharacterGLM is designed for generating Character-based Dialogues (CharacterDial), which aims to equip a conversational AI system with character customization for satisfying people's inherent social desires and emotional needs. On top of CharacterGLM, we can customize various AI characters or social agents by configuring their attributes (identities, interests, viewpoints, experiences, achievements, social relationships, etc.) and behaviors (linguistic features, emotional expressions, interaction patterns, etc.). Our model outperforms most mainstream close-source large langauge models, including the GPT series, especially in terms of consistency, human-likeness, and engagement according to manual evaluations. We will release our 6B version of CharacterGLM and a subset of training data to facilitate further research development in the direction of character-based dialogue generation.

| Comments: | Work in progress                                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | [arXiv:2311.16832](https://arxiv.org/abs/2311.16832) [cs.CL] |
|           | (or [arXiv:2311.16832v1](https://arxiv.org/abs/2311.16832v1) [cs.CL] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.16832Focus to learn more |





<h2 id="2023-11-29-3">3. Evaluating Optimal Reference Translations
</h2>

Title: [Evaluating Optimal Reference Translations](https://arxiv.org/abs/2311.16787)

Authors: [Vilém Zouhar](https://arxiv.org/search/cs?searchtype=author&query=Zouhar,+V), [Věra Kloudová](https://arxiv.org/search/cs?searchtype=author&query=Kloudová,+V), [Martin Popel](https://arxiv.org/search/cs?searchtype=author&query=Popel,+M), [Ondřej Bojar](https://arxiv.org/search/cs?searchtype=author&query=Bojar,+O)

> The overall translation quality reached by current machine translation (MT) systems for high-resourced language pairs is remarkably good. Standard methods of evaluation are not suitable nor intended to uncover the many translation errors and quality deficiencies that still persist. Furthermore, the quality of standard reference translations is commonly questioned and comparable quality levels have been reached by MT alone in several language pairs. Navigating further research in these high-resource settings is thus difficult. In this article, we propose a methodology for creating more reliable document-level human reference translations, called "optimal reference translations," with the simple aim to raise the bar of what should be deemed "human translation quality." We evaluate the obtained document-level optimal reference translations in comparison with "standard" ones, confirming a significant quality increase and also documenting the relationship between evaluation and translation editing.

| Comments: | To appear in Natural Language Engineering 2024               |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:2311.16787](https://arxiv.org/abs/2311.16787) [cs.CL] |
|           | (or [arXiv:2311.16787v1](https://arxiv.org/abs/2311.16787v1) [cs.CL] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.16787Focus to learn more |





<h2 id="2023-11-29-4">4. Reducing Gender Bias in Machine Translation through Counterfactual Data Generation
</h2>

Title: [Reducing Gender Bias in Machine Translation through Counterfactual Data Generation](https://arxiv.org/abs/2311.16362)

Authors: [Ranjita Naik](https://arxiv.org/search/cs?searchtype=author&query=Naik,+R), [Spencer Rarrick](https://arxiv.org/search/cs?searchtype=author&query=Rarrick,+S), [Vishal Chowdhary](https://arxiv.org/search/cs?searchtype=author&query=Chowdhary,+V)

> Recent advances in neural methods have led to substantial improvement in the quality of Neural Machine Translation (NMT) systems. However, these systems frequently produce translations with inaccurate gender (Stanovsky et al., 2019), which can be traced to bias in training data. Saunders and Byrne (2020) tackle this problem with a handcrafted dataset containing balanced gendered profession words. By using this data to fine-tune an existing NMT model, they show that gender bias can be significantly mitigated, albeit at the expense of translation quality due to catastrophic forgetting. They recover some of the lost quality with modified training objectives or additional models at inference. We find, however, that simply supplementing the handcrafted dataset with a random sample from the base model training corpus is enough to significantly reduce the catastrophic forgetting. We also propose a novel domain-adaptation technique that leverages in-domain data created with the counterfactual data generation techniques proposed by Zmigrod et al. (2019) to further improve accuracy on the WinoMT challenge test set without significant loss in translation quality. We show its effectiveness in NMT systems from English into three morphologically rich languages French, Spanish, and Italian. The relevant dataset and code will be available at Github.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2311.16362](https://arxiv.org/abs/2311.16362) [cs.CL] |
|           | (or [arXiv:2311.16362v1](https://arxiv.org/abs/2311.16362v1) [cs.CL] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.16362Focus to learn more |





<h2 id="2023-11-29-5">5. MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training
</h2>

Title: [MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training](https://arxiv.org/abs/2311.17049)

Authors: [Pavan Kumar Anasosalu Vasu](https://arxiv.org/search/cs?searchtype=author&query=Vasu,+P+K+A), [Hadi Pouransari](https://arxiv.org/search/cs?searchtype=author&query=Pouransari,+H), [Fartash Faghri](https://arxiv.org/search/cs?searchtype=author&query=Faghri,+F), [Raviteja Vemulapalli](https://arxiv.org/search/cs?searchtype=author&query=Vemulapalli,+R), [Oncel Tuzel](https://arxiv.org/search/cs?searchtype=author&query=Tuzel,+O)

> Contrastive pretraining of image-text foundation models, such as CLIP, demonstrated excellent zero-shot performance and improved robustness on a wide range of downstream tasks. However, these models utilize large transformer-based encoders with significant memory and latency overhead which pose challenges for deployment on mobile devices. In this work, we introduce MobileCLIP -- a new family of efficient image-text models optimized for runtime performance along with a novel and efficient training approach, namely multi-modal reinforced training. The proposed training approach leverages knowledge transfer from an image captioning model and an ensemble of strong CLIP encoders to improve the accuracy of efficient models. Our approach avoids train-time compute overhead by storing the additional knowledge in a reinforced dataset. MobileCLIP sets a new state-of-the-art latency-accuracy tradeoff for zero-shot classification and retrieval tasks on several datasets. Our MobileCLIP-S2 variant is 2.3× faster while more accurate compared to previous best CLIP model based on ViT-B/16. We further demonstrate the effectiveness of our multi-modal reinforced training by training a CLIP model based on ViT-B/16 image backbone and achieving +2.9% average performance improvement on 38 evaluation benchmarks compared to the previous best. Moreover, we show that the proposed approach achieves 10×-1000× improved learning efficiency when compared with non-reinforced CLIP training.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2311.17049](https://arxiv.org/abs/2311.17049) [cs.CV] |
|           | (or [arXiv:2311.17049v1](https://arxiv.org/abs/2311.17049v1) [cs.CV] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.17049Focus to learn more |





<h2 id="2023-11-29-6">6. LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models
</h2>

Title: [LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models](https://arxiv.org/abs/2311.17043)

Authors: [Yanwei Li](https://arxiv.org/search/cs?searchtype=author&query=Li,+Y), [Chengyao Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+C), [Jiaya Jia](https://arxiv.org/search/cs?searchtype=author&query=Jia,+J)

> In this work, we present a novel method to tackle the token generation challenge in Vision Language Models (VLMs) for video and image understanding, called LLaMA-VID. Current VLMs, while proficient in tasks like image captioning and visual question answering, face computational burdens when processing long videos due to the excessive visual tokens. LLaMA-VID addresses this issue by representing each frame with two distinct tokens, namely context token and content token. The context token encodes the overall image context based on user input, whereas the content token encapsulates visual cues in each frame. This dual-token strategy significantly reduces the overload of long videos while preserving critical information. Generally, LLaMA-VID empowers existing frameworks to support hour-long videos and pushes their upper limit with an extra context token. It is proved to surpass previous methods on most of video- or image-based benchmarks. Code is available [this https URL](https://github.com/dvlab-research/LLaMA-VID)}{[this https URL](https://github.com/dvlab-research/LLaMA-VID)

| Comments: | Code is available at [this https URL](https://github.com/dvlab-research/LLaMA-VID) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| Cite as:  | [arXiv:2311.17043](https://arxiv.org/abs/2311.17043) [cs.CV] |
|           | (or [arXiv:2311.17043v1](https://arxiv.org/abs/2311.17043v1) [cs.CV] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.17043Focus to learn more |





<h2 id="2023-11-29-7">7. Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding
</h2>

Title: [Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding](https://arxiv.org/abs/2311.16922)

Authors: [Sicong Leng](https://arxiv.org/search/cs?searchtype=author&query=Leng,+S), [Hang Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+H), [Guanzheng Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen,+G), [Xin Li](https://arxiv.org/search/cs?searchtype=author&query=Li,+X), [Shijian Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu,+S), [Chunyan Miao](https://arxiv.org/search/cs?searchtype=author&query=Miao,+C), [Lidong Bing](https://arxiv.org/search/cs?searchtype=author&query=Bing,+L)

> Large Vision-Language Models (LVLMs) have advanced considerably, intertwining visual recognition and language understanding to generate content that is not only coherent but also contextually attuned. Despite their success, LVLMs still suffer from the issue of object hallucinations, where models generate plausible yet incorrect outputs that include objects that do not exist in the images. To mitigate this issue, we introduce Visual Contrastive Decoding (VCD), a simple and training-free method that contrasts output distributions derived from original and distorted visual inputs. The proposed VCD effectively reduces the over-reliance on statistical bias and unimodal priors, two essential causes of object hallucinations. This adjustment ensures the generated content is closely grounded to visual inputs, resulting in contextually accurate outputs. Our experiments show that VCD, without either additional training or the usage of external tools, significantly mitigates the object hallucination issue across different LVLM families. Beyond mitigating object hallucinations, VCD also excels in general LVLM benchmarks, highlighting its wide-ranging applicability.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2311.16922](https://arxiv.org/abs/2311.16922) [cs.CV] |
|           | (or [arXiv:2311.16922v1](https://arxiv.org/abs/2311.16922v1) [cs.CV] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.16922Focus to learn more |





<h2 id="2023-11-29-8">8. RELIC: Investigating Large Language Model Responses using Self-Consistency
</h2>

Title: [RELIC: Investigating Large Language Model Responses using Self-Consistency](https://arxiv.org/abs/2311.16842)

Authors: [Furui Cheng](https://arxiv.org/search/cs?searchtype=author&query=Cheng,+F), [Vilém Zouhar](https://arxiv.org/search/cs?searchtype=author&query=Zouhar,+V), [Simran Arora](https://arxiv.org/search/cs?searchtype=author&query=Arora,+S), [Mrinmaya Sachan](https://arxiv.org/search/cs?searchtype=author&query=Sachan,+M), [Hendrik Strobelt](https://arxiv.org/search/cs?searchtype=author&query=Strobelt,+H), [Mennatallah El-Assady](https://arxiv.org/search/cs?searchtype=author&query=El-Assady,+M)

> Large Language Models (LLMs) are notorious for blending fact with fiction and generating non-factual content, known as hallucinations. To tackle this challenge, we propose an interactive system that helps users obtain insights into the reliability of the generated text. Our approach is based on the idea that the self-consistency of multiple samples generated by the same LLM relates to its confidence in individual claims in the generated texts. Using this idea, we design RELIC, an interactive system that enables users to investigate and verify semantic-level variations in multiple long-form responses. This allows users to recognize potentially inaccurate information in the generated text and make necessary corrections. From a user study with ten participants, we demonstrate that our approach helps users better verify the reliability of the generated text. We further summarize the design implications and lessons learned from this research for inspiring future studies on reliable human-LLM interactions.

| Subjects: | **Human-Computer Interaction (cs.HC)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2311.16842](https://arxiv.org/abs/2311.16842) [cs.HC] |
|           | (or [arXiv:2311.16842v1](https://arxiv.org/abs/2311.16842v1) [cs.HC] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.16842Focus to learn more |





<h2 id="2023-11-29-9">9. ChartLlama: A Multimodal LLM for Chart Understanding and Generation
</h2>

Title: [ChartLlama: A Multimodal LLM for Chart Understanding and Generation](https://arxiv.org/abs/2311.16483)

Authors: [Yucheng Han](https://arxiv.org/search/cs?searchtype=author&query=Han,+Y), [Chi Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+C), [Xin Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen,+X), [Xu Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang,+X), [Zhibin Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+Z), [Gang Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu,+G), [Bin Fu](https://arxiv.org/search/cs?searchtype=author&query=Fu,+B), [Hanwang Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+H)

> Multi-modal large language models have demonstrated impressive performances on most vision-language tasks. However, the model generally lacks the understanding capabilities for specific domain data, particularly when it comes to interpreting chart figures. This is mainly due to the lack of relevant multi-modal instruction tuning datasets. In this article, we create a high-quality instruction-tuning dataset leveraging GPT-4. We develop a multi-step data generation process in which different steps are responsible for generating tabular data, creating chart figures, and designing instruction tuning data separately. Our method's flexibility enables us to generate diverse, high-quality instruction-tuning data consistently and efficiently while maintaining a low resource expenditure. Additionally, it allows us to incorporate a wider variety of chart and task types not yet featured in existing datasets. Next, we introduce ChartLlama, a multi-modal large language model that we've trained using our created dataset. ChartLlama outperforms all prior methods in ChartQA, Chart-to-text, and Chart-extraction evaluation benchmarks. Additionally, ChartLlama significantly improves upon the baseline in our specially compiled chart dataset, which includes new chart and task types. The results of ChartLlama confirm the value and huge potential of our proposed data generation method in enhancing chart comprehension.

| Comments: | Code and model on [this https URL](https://tingxueronghua.github.io/ChartLlama/) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| Cite as:  | [arXiv:2311.16483](https://arxiv.org/abs/2311.16483) [cs.CV] |
|           | (or [arXiv:2311.16483v1](https://arxiv.org/abs/2311.16483v1) [cs.CV] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.16483Focus to learn more |





<h2 id="2023-11-29-10">10. ChatTraffc: Text-to-Traffic Generation via Diffusion Model
</h2>

Title: [ChatTraffc: Text-to-Traffic Generation via Diffusion Model](https://arxiv.org/abs/2311.16203)

Authors: [Chengyang Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+C), [Yong Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+Y), [Qitan Shao](https://arxiv.org/search/cs?searchtype=author&query=Shao,+Q), [Bo Li](https://arxiv.org/search/cs?searchtype=author&query=Li,+B), [Yisheng Lv](https://arxiv.org/search/cs?searchtype=author&query=Lv,+Y), [Xinglin Piao](https://arxiv.org/search/cs?searchtype=author&query=Piao,+X), [Baocai Yin](https://arxiv.org/search/cs?searchtype=author&query=Yin,+B)

> Traffic prediction is one of the most significant foundations in Intelligent Transportation Systems (ITS). Traditional traffic prediction methods rely only on historical traffic data to predict traffic trends and face two main challenges. 1) insensitivity to unusual events. 2) poor performance in long-term prediction. In this work, we explore how generative models combined with text describing the traffic system can be applied for traffic generation and name the task Text-to-Traffic Generation (TTG). The key challenge of the TTG task is how to associate text with the spatial structure of the road network and traffic data for generating traffic situations. To this end, we propose ChatTraffic, the first diffusion model for text-to-traffic generation. To guarantee the consistency between synthetic and real data, we augment a diffusion model with the Graph Convolutional Network (GCN) to extract spatial correlations of traffic data. In addition, we construct a large dataset containing text-traffic pairs for the TTG task. We benchmarked our model qualitatively and quantitatively on the released dataset. The experimental results indicate that ChatTraffic can generate realistic traffic situations from the text. Our code and dataset are available at [this https URL](https://github.com/ChyaZhang/ChatTraffic).

| Subjects: | **Machine Learning (cs.LG)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2311.16203](https://arxiv.org/abs/2311.16203) [cs.LG] |
|           | (or [arXiv:2311.16203v1](https://arxiv.org/abs/2311.16203v1) [cs.LG] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.16203Focus to learn more |





<h2 id="2023-11-29-11">11. Pre-trained Language Models Do Not Help Auto-regressive Text-to-Image Generation
</h2>

Title: [Pre-trained Language Models Do Not Help Auto-regressive Text-to-Image Generation](https://arxiv.org/abs/2311.16201)

Authors: [Yuhui Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+Y), [Brandon McKinzie](https://arxiv.org/search/cs?searchtype=author&query=McKinzie,+B), [Zhe Gan](https://arxiv.org/search/cs?searchtype=author&query=Gan,+Z), [Vaishaal Shankar](https://arxiv.org/search/cs?searchtype=author&query=Shankar,+V), [Alexander Toshev](https://arxiv.org/search/cs?searchtype=author&query=Toshev,+A)

> Recent advances in image tokenizers, such as VQ-VAE, have enabled text-to-image generation using auto-regressive methods, similar to language modeling. However, these methods have yet to leverage pre-trained language models, despite their adaptability to various downstream tasks. In this work, we explore this gap by adapting a pre-trained language model for auto-regressive text-to-image generation, and find that pre-trained language models offer limited help. We provide a two-fold explanation by analyzing tokens from each modality. First, we demonstrate that image tokens possess significantly different semantics compared to text tokens, rendering pre-trained language models no more effective in modeling them than randomly initialized ones. Second, the text tokens in the image-text datasets are too simple compared to normal language model pre-training data, which causes the catastrophic degradation of language models' capability.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2311.16201](https://arxiv.org/abs/2311.16201) [cs.CV] |
|           | (or [arXiv:2311.16201v1](https://arxiv.org/abs/2311.16201v1) [cs.CV] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.16201Focus to learn more |






# 2023-11-28

[Return to Index](#Index)



<h2 id="2023-11-28-1">1. DUnE: Dataset for Unified Editing
</h2>

Title: [DUnE: Dataset for Unified Editing](https://arxiv.org/abs/2311.16087)

Authors: [Afra Feyza Akyürek](https://arxiv.org/search/cs?searchtype=author&query=Akyürek,+A+F), [Eric Pan](https://arxiv.org/search/cs?searchtype=author&query=Pan,+E), [Garry Kuwanto](https://arxiv.org/search/cs?searchtype=author&query=Kuwanto,+G), [Derry Wijaya](https://arxiv.org/search/cs?searchtype=author&query=Wijaya,+D)

> Even the most advanced language models remain susceptible to errors necessitating to modify these models without initiating a comprehensive retraining process. Model editing refers to the modification of a model's knowledge or representations in a manner that produces the desired outcomes. Prior research primarily centered around editing factual data e.g. "Messi plays for Inter Miami" confining the definition of an edit to a knowledge triplet i.e. (subject, object, relation). However, as the applications of language models expand, so do the diverse ways in which we wish to edit and refine their outputs. In this study, we broaden the scope of the editing problem to include an array of editing cases such as debiasing and rectifying reasoning errors and define an edit as any natural language expression that solicits a change in the model's outputs. We are introducing DUnE-an editing benchmark where edits are natural language sentences and propose that DUnE presents a challenging yet relevant task. To substantiate this claim, we conduct an extensive series of experiments testing various editing approaches to address DUnE, demonstrating their respective strengths and weaknesses. We show that retrieval-augmented language modeling can outperform specialized editing techniques and neither set of approaches has fully solved the generalized editing problem covered by our benchmark.

| Comments: | Accepted at EMNLP 2023                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:2311.16087](https://arxiv.org/abs/2311.16087) [cs.CL] |
|           | (or [arXiv:2311.16087v1](https://arxiv.org/abs/2311.16087v1) [cs.CL] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.16087Focus to learn more |





<h2 id="2023-11-28-2">2. MEDITRON-70B: Scaling Medical Pretraining for Large Language Models
</h2>

Title: [MEDITRON-70B: Scaling Medical Pretraining for Large Language Models](https://arxiv.org/abs/2311.16079)

Authors: [Zeming Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen,+Z), [Alejandro Hernández Cano](https://arxiv.org/search/cs?searchtype=author&query=Cano,+A+H), [Angelika Romanou](https://arxiv.org/search/cs?searchtype=author&query=Romanou,+A), [Antoine Bonnet](https://arxiv.org/search/cs?searchtype=author&query=Bonnet,+A), [Kyle Matoba](https://arxiv.org/search/cs?searchtype=author&query=Matoba,+K), [Francesco Salvi](https://arxiv.org/search/cs?searchtype=author&query=Salvi,+F), [Matteo Pagliardini](https://arxiv.org/search/cs?searchtype=author&query=Pagliardini,+M), [Simin Fan](https://arxiv.org/search/cs?searchtype=author&query=Fan,+S), [Andreas Köpf](https://arxiv.org/search/cs?searchtype=author&query=Köpf,+A), [Amirkeivan Mohtashami](https://arxiv.org/search/cs?searchtype=author&query=Mohtashami,+A), [Alexandre Sallinen](https://arxiv.org/search/cs?searchtype=author&query=Sallinen,+A), [Alireza Sakhaeirad](https://arxiv.org/search/cs?searchtype=author&query=Sakhaeirad,+A), [Vinitra Swamy](https://arxiv.org/search/cs?searchtype=author&query=Swamy,+V), [Igor Krawczuk](https://arxiv.org/search/cs?searchtype=author&query=Krawczuk,+I), [Deniz Bayazit](https://arxiv.org/search/cs?searchtype=author&query=Bayazit,+D), [Axel Marmet](https://arxiv.org/search/cs?searchtype=author&query=Marmet,+A), [Syrielle Montariol](https://arxiv.org/search/cs?searchtype=author&query=Montariol,+S), [Mary-Anne Hartley](https://arxiv.org/search/cs?searchtype=author&query=Hartley,+M), [Martin Jaggi](https://arxiv.org/search/cs?searchtype=author&query=Jaggi,+M), [Antoine Bosselut](https://arxiv.org/search/cs?searchtype=author&query=Bosselut,+A)

> Large language models (LLMs) can potentially democratize access to medical knowledge. While many efforts have been made to harness and improve LLMs' medical knowledge and reasoning capacities, the resulting models are either closed-source (e.g., PaLM, GPT-4) or limited in scale (<= 13B parameters), which restricts their abilities. In this work, we improve access to large-scale medical LLMs by releasing MEDITRON: a suite of open-source LLMs with 7B and 70B parameters adapted to the medical domain. MEDITRON builds on Llama-2 (through our adaptation of Nvidia's Megatron-LM distributed trainer), and extends pretraining on a comprehensively curated medical corpus, including selected PubMed articles, abstracts, and internationally-recognized medical guidelines. Evaluations using four major medical benchmarks show significant performance gains over several state-of-the-art baselines before and after task-specific finetuning. Overall, MEDITRON achieves a 6% absolute performance gain over the best public baseline in its parameter class and 3% over the strongest baseline we finetuned from Llama-2. Compared to closed-source LLMs, MEDITRON-70B outperforms GPT-3.5 and Med-PaLM and is within 5% of GPT-4 and 10% of Med-PaLM-2. We release our code for curating the medical pretraining corpus and the MEDITRON model weights to drive open-source development of more capable medical LLMs.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2311.16079](https://arxiv.org/abs/2311.16079) [cs.CL] |
|           | (or [arXiv:2311.16079v1](https://arxiv.org/abs/2311.16079v1) [cs.CL] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.16079Focus to learn more |







<h2 id="2023-11-28-3">3. A Quantitative Approach to Understand Self-Supervised Models as Cross-lingual Feature Extractors
</h2>

Title: [A Quantitative Approach to Understand Self-Supervised Models as Cross-lingual Feature Extractors](https://arxiv.org/abs/2311.15954)

Authors: [Shuyue Stella Li](https://arxiv.org/search/cs?searchtype=author&query=Li,+S+S), [Beining Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu,+B), [Xiangyu Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+X), [Hexin Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+H), [Wenhan Chao](https://arxiv.org/search/cs?searchtype=author&query=Chao,+W), [Leibny Paola Garcia](https://arxiv.org/search/cs?searchtype=author&query=Garcia,+L+P)

> In this work, we study the features extracted by English self-supervised learning (SSL) models in cross-lingual contexts and propose a new metric to predict the quality of feature representations. Using automatic speech recognition (ASR) as a downstream task, we analyze the effect of model size, training objectives, and model architecture on the models' performance as a feature extractor for a set of topologically diverse corpora. We develop a novel metric, the Phonetic-Syntax Ratio (PSR), to measure the phonetic and synthetic information in the extracted representations using deep generalized canonical correlation analysis. Results show the contrastive loss in the wav2vec2.0 objective facilitates more effective cross-lingual feature extraction. There is a positive correlation between PSR scores and ASR performance, suggesting that phonetic information extracted by monolingual SSL models can be used for downstream tasks in cross-lingual settings. The proposed metric is an effective indicator of the quality of the representations and can be useful for model selection.

| Comments: | 12 pages, 5 figures, 4 tables                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Audio and Speech Processing (eess.AS) |
| Cite as:  | [arXiv:2311.15954](https://arxiv.org/abs/2311.15954) [cs.CL] |
|           | (or [arXiv:2311.15954v1](https://arxiv.org/abs/2311.15954v1) [cs.CL] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.15954Focus to learn more |







<h2 id="2023-11-28-4">4. WorldSense: A Synthetic Benchmark for Grounded Reasoning in Large Language Models
</h2>

Title: [WorldSense: A Synthetic Benchmark for Grounded Reasoning in Large Language Models](https://arxiv.org/abs/2311.15930)

Authors: [Youssef Benchekroun](https://arxiv.org/search/cs?searchtype=author&query=Benchekroun,+Y), [Megi Dervishi](https://arxiv.org/search/cs?searchtype=author&query=Dervishi,+M), [Mark Ibrahim](https://arxiv.org/search/cs?searchtype=author&query=Ibrahim,+M), [Jean-Baptiste Gaya](https://arxiv.org/search/cs?searchtype=author&query=Gaya,+J), [Xavier Martinet](https://arxiv.org/search/cs?searchtype=author&query=Martinet,+X), [Grégoire Mialon](https://arxiv.org/search/cs?searchtype=author&query=Mialon,+G), [Thomas Scialom](https://arxiv.org/search/cs?searchtype=author&query=Scialom,+T), [Emmanuel Dupoux](https://arxiv.org/search/cs?searchtype=author&query=Dupoux,+E), [Dieuwke Hupkes](https://arxiv.org/search/cs?searchtype=author&query=Hupkes,+D), [Pascal Vincent](https://arxiv.org/search/cs?searchtype=author&query=Vincent,+P)

> We propose WorldSense, a benchmark designed to assess the extent to which LLMs are consistently able to sustain tacit world models, by testing how they draw simple inferences from descriptions of simple arrangements of entities. Worldsense is a synthetic benchmark with three problem types, each with their own trivial control, which explicitly avoids bias by decorrelating the abstract structure of problems from the vocabulary and expressions, and by decorrelating all problem subparts with the correct response. We run our benchmark on three state-of-the-art chat-LLMs (GPT3.5, GPT4 and Llama2-chat) and show that these models make errors even with as few as three objects. Furthermore, they have quite heavy response biases, preferring certain responses irrespective of the question. Errors persist even with chain-of-thought prompting and in-context learning. Lastly, we show that while finetuning on similar problems does result in substantial improvements -- within- and out-of-distribution -- the finetuned models do not generalise beyond a constraint problem space.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2311.15930](https://arxiv.org/abs/2311.15930) [cs.CL] |
|           | (or [arXiv:2311.15930v1](https://arxiv.org/abs/2311.15930v1) [cs.CL] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.15930Focus to learn more |







<h2 id="2023-11-28-5">5. YUAN 2.0: A Large Language Model with Localized Filtering-based Attention
</h2>

Title: [YUAN 2.0: A Large Language Model with Localized Filtering-based Attention](https://arxiv.org/abs/2311.15786)

Authors: [Shaohua Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu,+S), [Xudong Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao,+X), [Shenling Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+S), [Jiangang Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo,+J), [Lingjun Li](https://arxiv.org/search/cs?searchtype=author&query=Li,+L), [Xi Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen,+X), [Bing Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao,+B), [Wei Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+W), [Tong Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu,+T), [Rongguo Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+R), [Jiahua Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+J), [Chao Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+C)

> In this work, the Localized Filtering-based Attention (LFA) is introduced to incorporate prior knowledge of local dependencies of natural language into Attention. Based on LFA, we develop and release Yuan 2.0, a large language model with parameters ranging from 2.1 billion to 102.6 billion. A data filtering and generation method is presented to build pretraining and fine-tuning dataset in high quality. A distributed training method with non-uniform pipeline parallel, data parallel, and optimizer parallel is proposed, which greatly reduces the bandwidth requirements of intra-node communication, and achieves good performance in large-scale distributed training. Yuan 2.0 models display impressive ability in code generation, math problem-solving, and chat compared with existing models. The latest version of YUAN 2.0, including model weights and source code, is accessible at Github.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Human-Computer Interaction (cs.HC) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2311.15786](https://arxiv.org/abs/2311.15786) [cs.CL] |
|           | (or [arXiv:2311.15786v1](https://arxiv.org/abs/2311.15786v1) [cs.CL] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.15786Focus to learn more |







<h2 id="2023-11-28-6">6. Knowledge Unlearning for LLMs: Tasks, Methods, and Challenges
</h2>

Title: [Knowledge Unlearning for LLMs: Tasks, Methods, and Challenges](https://arxiv.org/abs/2311.15766)

Authors: [Nianwen Si](https://arxiv.org/search/cs?searchtype=author&query=Si,+N), [Hao Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+H), [Heyu Chang](https://arxiv.org/search/cs?searchtype=author&query=Chang,+H), [Wenlin Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+W), [Dan Qu](https://arxiv.org/search/cs?searchtype=author&query=Qu,+D), [Weiqiang Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+W)

> In recent years, large language models (LLMs) have spurred a new research paradigm in natural language processing. Despite their excellent capability in knowledge-based question answering and reasoning, their potential to retain faulty or even harmful knowledge poses risks of malicious application. The challenge of mitigating this issue and transforming these models into purer assistants is crucial for their widespread applicability. Unfortunately, Retraining LLMs repeatedly to eliminate undesirable knowledge is impractical due to their immense parameters. Knowledge unlearning, derived from analogous studies on machine unlearning, presents a promising avenue to address this concern and is notably advantageous in the context of LLMs. It allows for the removal of harmful knowledge in an efficient manner, without affecting unrelated knowledge in the model. To this end, we provide a survey of knowledge unlearning in the era of LLMs. Firstly, we formally define the knowledge unlearning problem and distinguish it from related works. Subsequently, we categorize existing knowledge unlearning methods into three classes: those based on parameter optimization, parameter merging, and in-context learning, and introduce details of these unlearning methods. We further present evaluation datasets used in existing methods, and finally conclude this survey by presenting the ongoing challenges and future directions.

| Comments: | Work in progress                                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:2311.15766](https://arxiv.org/abs/2311.15766) [cs.CL] |
|           | (or [arXiv:2311.15766v1](https://arxiv.org/abs/2311.15766v1) [cs.CL] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.15766Focus to learn more |







<h2 id="2023-11-28-7">7. Towards Vision Enhancing LLMs: Empowering Multimodal Knowledge Storage and Sharing in LLMs
</h2>

Title: [Towards Vision Enhancing LLMs: Empowering Multimodal Knowledge Storage and Sharing in LLMs](https://arxiv.org/abs/2311.15759)

Authors: [Yunxin Li](https://arxiv.org/search/cs?searchtype=author&query=Li,+Y), [Baotian Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu,+B), [Wei Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+W), [Xiaochun Cao](https://arxiv.org/search/cs?searchtype=author&query=Cao,+X), [Min Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+M)

> Recent advancements in multimodal large language models (MLLMs) have achieved significant multimodal generation capabilities, akin to GPT-4. These models predominantly map visual information into language representation space, leveraging the vast knowledge and powerful text generation abilities of LLMs to produce multimodal instruction-following responses. We could term this method as LLMs for Vision because of its employing LLMs for visual-language understanding, yet observe that these MLLMs neglect the potential of harnessing visual knowledge to enhance overall capabilities of LLMs, which could be regraded as Vision Enhancing LLMs. In this paper, we propose an approach called MKS2, aimed at enhancing LLMs through empowering Multimodal Knowledge Storage and Sharing in LLMs. Specifically, we introduce the Modular Visual Memory, a component integrated into the internal blocks of LLMs, designed to store open-world visual information efficiently. Additionally, we present a soft Mixtures-of-Multimodal Experts architecture in LLMs to invoke multimodal knowledge collaboration during generation. Our comprehensive experiments demonstrate that MKS2 substantially augments the reasoning capabilities of LLMs in contexts necessitating physical or commonsense knowledge. It also delivers competitive results on multimodal benchmarks.

| Comments: | 12 pages, 4 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV) |
| Cite as:  | [arXiv:2311.15759](https://arxiv.org/abs/2311.15759) [cs.CL] |
|           | (or [arXiv:2311.15759v1](https://arxiv.org/abs/2311.15759v1) [cs.CL] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.15759Focus to learn more |







<h2 id="2023-11-28-8">8. MoDS: Model-oriented Data Selection for Instruction Tuning
</h2>

Title: [MoDS: Model-oriented Data Selection for Instruction Tuning](https://arxiv.org/abs/2311.15653)

Authors: [Qianlong Du](https://arxiv.org/search/cs?searchtype=author&query=Du,+Q), [Chengqing Zong](https://arxiv.org/search/cs?searchtype=author&query=Zong,+C), [Jiajun Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+J)

> Instruction tuning has become the de facto method to equip large language models (LLMs) with the ability of following user instructions. Usually, hundreds of thousands or millions of instruction-following pairs are employed to fine-tune the foundation LLMs. Recently, some studies show that a small number of high-quality instruction data is enough. However, how to select appropriate instruction data for a given LLM is still an open problem. To address this problem, in this paper we present a model-oriented data selection (MoDS) approach, which selects instruction data based on a new criteria considering three aspects: quality, coverage and necessity. First, our approach utilizes a quality evaluation model to filter out the high-quality subset from the original instruction dataset, and then designs an algorithm to further select from the high-quality subset a seed instruction dataset with good coverage. The seed dataset is applied to fine-tune the foundation LLM to obtain an initial instruction-following LLM. Finally, we develop a necessity evaluation model to find out the instruction data which are performed badly in the initial instruction-following LLM and consider them necessary instructions to further improve the LLMs. In this way, we can get a small high-quality, broad-coverage and high-necessity subset from the original instruction datasets. Experimental results show that, the model fine-tuned with 4,000 instruction pairs selected by our approach could perform better than the model fine-tuned with the full original dataset which includes 214k instruction data.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2311.15653](https://arxiv.org/abs/2311.15653) [cs.CL] |
|           | (or [arXiv:2311.15653v1](https://arxiv.org/abs/2311.15653v1) [cs.CL] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.15653Focus to learn more |







<h2 id="2023-11-28-9">9. LongStory: Coherent, Complete and Length Controlled Long story Generation
</h2>

Title: [LongStory: Coherent, Complete and Length Controlled Long story Generation](https://arxiv.org/abs/2311.15208)

Authors: [Kyeongman Park](https://arxiv.org/search/cs?searchtype=author&query=Park,+K), [Nakyeong Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang,+N), [Kyomin Jung](https://arxiv.org/search/cs?searchtype=author&query=Jung,+K)

> A human author can write any length of story without losing coherence. Also, they always bring the story to a proper ending, an ability that current language models lack. In this work, we present the LongStory for coherent, complete, and length-controlled long story generation. LongStory introduces two novel methodologies: (1) the long and short-term contexts weight calibrator (CWC) and (2) long story structural positions (LSP). The CWC adjusts weights for long-term context Memory and short-term context Cheating, acknowledging their distinct roles. The LSP employs discourse tokens to convey the structural positions of a long story. Trained on three datasets with varied average story lengths, LongStory outperforms other baselines, including the strong story generator Plotmachine, in coherence, completeness, relevance, and repetitiveness. We also perform zero-shot tests on each dataset to assess the model's ability to predict outcomes beyond its training data and validate our methodology by comparing its performance with variants of our model.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2311.15208](https://arxiv.org/abs/2311.15208) [cs.CL] |
|           | (or [arXiv:2311.15208v1](https://arxiv.org/abs/2311.15208v1) [cs.CL] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.15208Focus to learn more |







<h2 id="2023-11-28-10">10. Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion
</h2>

Title: [Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion](https://arxiv.org/abs/2311.15106)

Authors: [Bernal Jimenez Gutierrez](https://arxiv.org/search/cs?searchtype=author&query=Gutierrez,+B+J), [Yuqing Mao](https://arxiv.org/search/cs?searchtype=author&query=Mao,+Y), [Vinh Nguyen](https://arxiv.org/search/cs?searchtype=author&query=Nguyen,+V), [Kin Wah Fung](https://arxiv.org/search/cs?searchtype=author&query=Fung,+K+W), [Yu Su](https://arxiv.org/search/cs?searchtype=author&query=Su,+Y), [Olivier Bodenreider](https://arxiv.org/search/cs?searchtype=author&query=Bodenreider,+O)

> As the immense opportunities enabled by large language models become more apparent, NLP systems will be increasingly expected to excel in real-world settings. However, in many instances, powerful models alone will not yield translational NLP solutions, especially if the formulated problem is not well aligned with the real-world task. In this work, we study the case of UMLS vocabulary insertion, an important real-world task in which hundreds of thousands of new terms, referred to as atoms, are added to the UMLS, one of the most comprehensive open-source biomedical knowledge bases. Previous work aimed to develop an automated NLP system to make this time-consuming, costly, and error-prone task more efficient. Nevertheless, practical progress in this direction has been difficult to achieve due to a problem formulation and evaluation gap between research output and the real-world task. In order to address this gap, we introduce a new formulation for UMLS vocabulary insertion which mirrors the real-world task, datasets which faithfully represent it and several strong baselines we developed through re-purposing existing solutions. Additionally, we propose an effective rule-enhanced biomedical language model which enables important new model behavior, outperforms all strong baselines and provides measurable qualitative improvements to editors who carry out the UVI task. We hope this case study provides insight into the considerable importance of problem formulation for the success of translational NLP solutions.

| Comments: | EMNLP 2023 Findings; Code is available at [this https URL](https://github.com/OSU-NLP-Group/UMLS-Vocabulary-Insertion) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:2311.15106](https://arxiv.org/abs/2311.15106) [cs.CL] |
|           | (or [arXiv:2311.15106v1](https://arxiv.org/abs/2311.15106v1) [cs.CL] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.15106Focus to learn more |







<h2 id="2023-11-28-11">11. Offensive Language Identification in Transliterated and Code-Mixed Bangla
</h2>

Title: [Offensive Language Identification in Transliterated and Code-Mixed Bangla](https://arxiv.org/abs/2311.15023)

Authors: [Md Nishat Raihan](https://arxiv.org/search/cs?searchtype=author&query=Raihan,+M+N), [Umma Hani Tanmoy](https://arxiv.org/search/cs?searchtype=author&query=Tanmoy,+U+H), [Anika Binte Islam](https://arxiv.org/search/cs?searchtype=author&query=Islam,+A+B), [Kai North](https://arxiv.org/search/cs?searchtype=author&query=North,+K), [Tharindu Ranasinghe](https://arxiv.org/search/cs?searchtype=author&query=Ranasinghe,+T), [Antonios Anastasopoulos](https://arxiv.org/search/cs?searchtype=author&query=Anastasopoulos,+A), [Marcos Zampieri](https://arxiv.org/search/cs?searchtype=author&query=Zampieri,+M)

> Identifying offensive content in social media is vital for creating safe online communities. Several recent studies have addressed this problem by creating datasets for various languages. In this paper, we explore offensive language identification in texts with transliterations and code-mixing, linguistic phenomena common in multilingual societies, and a known challenge for NLP systems. We introduce TB-OLID, a transliterated Bangla offensive language dataset containing 5,000 manually annotated comments. We train and fine-tune machine learning models on TB-OLID, and we evaluate their results on this dataset. Our results show that English pre-trained transformer-based models, such as fBERT and HateBERT achieve the best performance on this dataset.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2311.15023](https://arxiv.org/abs/2311.15023) [cs.CL] |
|           | (or [arXiv:2311.15023v1](https://arxiv.org/abs/2311.15023v1) [cs.CL] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.15023Focus to learn more |







<h2 id="2023-11-28-12">12. Vector-Quantized Prompt Learning for Paraphrase Generation
</h2>

Title: [Vector-Quantized Prompt Learning for Paraphrase Generation](https://arxiv.org/abs/2311.14949)

Authors: [Haotian Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo,+H), [Yixin Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+Y), [Peidong Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+P), [Xianggen Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+X)

> Deep generative modeling of natural languages has achieved many successes, such as producing fluent sentences and translating from one language into another. However, the development of generative modeling techniques for paraphrase generation still lags behind largely due to the challenges in addressing the complex conflicts between expression diversity and semantic preservation. This paper proposes to generate diverse and high-quality paraphrases by exploiting the pre-trained models with instance-dependent prompts. To learn generalizable prompts, we assume that the number of abstract transforming patterns of paraphrase generation (governed by prompts) is finite and usually not large. Therefore, we present vector-quantized prompts as the cues to control the generation of pre-trained models. Extensive experiments demonstrate that the proposed method achieves new state-of-art results on three benchmark datasets, including Quora, Wikianswers, and MSCOCO. We will release all the code upon acceptance.

| Comments: | EMNLP Findings, 2023                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:2311.14949](https://arxiv.org/abs/2311.14949) [cs.CL] |
|           | (or [arXiv:2311.14949v1](https://arxiv.org/abs/2311.14949v1) [cs.CL] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.14949Focus to learn more |





<h2 id="2023-11-28-13">13. How Many Unicorns Are in This Image? A Safety Evaluation Benchmark for Vision LLMs
</h2>

Title: [How Many Unicorns Are in This Image? A Safety Evaluation Benchmark for Vision LLMs](https://arxiv.org/abs/2311.16101)

Authors: [Haoqin Tu](https://arxiv.org/search/cs?searchtype=author&query=Tu,+H), [Chenhang Cui](https://arxiv.org/search/cs?searchtype=author&query=Cui,+C), [Zijun Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+Z), [Yiyang Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou,+Y), [Bingchen Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao,+B), [Junlin Han](https://arxiv.org/search/cs?searchtype=author&query=Han,+J), [Wangchunshu Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou,+W), [Huaxiu Yao](https://arxiv.org/search/cs?searchtype=author&query=Yao,+H), [Cihang Xie](https://arxiv.org/search/cs?searchtype=author&query=Xie,+C)

> This work focuses on the potential of Vision LLMs (VLLMs) in visual reasoning. Different from prior studies, we shift our focus from evaluating standard performance to introducing a comprehensive safety evaluation suite, covering both out-of-distribution (OOD) generalization and adversarial robustness. For the OOD evaluation, we present two novel VQA datasets, each with one variant, designed to test model performance under challenging conditions. In exploring adversarial robustness, we propose a straightforward attack strategy for misleading VLLMs to produce visual-unrelated responses. Moreover, we assess the efficacy of two jailbreaking strategies, targeting either the vision or language component of VLLMs. Our evaluation of 21 diverse models, ranging from open-source VLLMs to GPT-4V, yields interesting observations: 1) Current VLLMs struggle with OOD texts but not images, unless the visual information is limited; and 2) These VLLMs can be easily misled by deceiving vision encoders only, and their vision-language training often compromise safety protocols. We release this safety evaluation suite at [this https URL](https://github.com/UCSC-VLAA/vllm-safety-benchmark).

| Comments: | H.T., C.C., and Z.W. contribute equally. Work done during H.T. and Z.W.'s internship at UCSC, and C.C. and Y.Z.'s internship at UNC |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | [arXiv:2311.16101](https://arxiv.org/abs/2311.16101) [cs.CV] |
|           | (or [arXiv:2311.16101v1](https://arxiv.org/abs/2311.16101v1) [cs.CV] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.16101Focus to learn more |





<h2 id="2023-11-28-14">14. Data Generation for Post-OCR correction of Cyrillic handwriting
</h2>

Title: [Data Generation for Post-OCR correction of Cyrillic handwriting](https://arxiv.org/abs/2311.15896)

Authors: [Evgenii Davydkin](https://arxiv.org/search/cs?searchtype=author&query=Davydkin,+E), [Aleksandr Markelov](https://arxiv.org/search/cs?searchtype=author&query=Markelov,+A), [Egor Iuldashev](https://arxiv.org/search/cs?searchtype=author&query=Iuldashev,+E), [Anton Dudkin](https://arxiv.org/search/cs?searchtype=author&query=Dudkin,+A), [Ivan Krivorotov](https://arxiv.org/search/cs?searchtype=author&query=Krivorotov,+I)

> This paper introduces a novel approach to post-Optical Character Recognition Correction (POC) for handwritten Cyrillic text, addressing a significant gap in current research methodologies. This gap is due to the lack of large text corporas that provide OCR errors for further training of language-based POC models, which are demanding in terms of corpora size. Our study primarily focuses on the development and application of a synthetic handwriting generation engine based on Bézier curves. Such an engine generates highly realistic handwritten text in any amounts, which we utilize to create a substantial dataset by transforming Russian text corpora sourced from the internet. We apply a Handwritten Text Recognition (HTR) model to this dataset to identify OCR errors, forming the basis for our POC model training. The correction model is trained on a 90-symbol input context, utilizing a pre-trained T5 architecture with a seq2seq correction task. We evaluate our approach on HWR200 and School_notebooks_RU datasets as they provide significant challenges in the HTR domain. Furthermore, POC can be used to highlight errors for teachers, evaluating student performance. This can be done simply by comparing sentences before and after correction, displaying differences in text. Our primary contribution lies in the innovative use of Bézier curves for Cyrillic text generation and subsequent error correction using a specialized POC model. We validate our approach by presenting Word Accuracy Rate (WAR) and Character Accuracy Rate (CAR) results, both with and without post-OCR correction, using real open corporas of handwritten Cyrillic text. These results, coupled with our methodology, are designed to be reproducible, paving the way for further advancements in the field of OCR and handwritten text analysis. Paper contributions can be found in [this https URL](https://github.com/dbrainio/CyrillicHandwritingPOC)

| Comments: | 17 pages, 27 figures, 6 tables, 26 references                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| Cite as:  | [arXiv:2311.15896](https://arxiv.org/abs/2311.15896) [cs.CV] |
|           | (or [arXiv:2311.15896v1](https://arxiv.org/abs/2311.15896v1) [cs.CV] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.15896Focus to learn more |





<h2 id="2023-11-28-15">15. Can Vision-Language Models Think from a First-Person Perspective?
</h2>

Title: [Can Vision-Language Models Think from a First-Person Perspective?](https://arxiv.org/abs/2311.15596)

Authors: [Sijie Cheng](https://arxiv.org/search/cs?searchtype=author&query=Cheng,+S), [Zhicheng Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo,+Z), [Jingwen Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu,+J), [Kechen Fang](https://arxiv.org/search/cs?searchtype=author&query=Fang,+K), [Peng Li](https://arxiv.org/search/cs?searchtype=author&query=Li,+P), [Huaping Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+H), [Yang Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+Y)

> Vision-language models (VLMs) have recently shown promising results in traditional downstream tasks. Evaluation studies have emerged to assess their abilities, with the majority focusing on the third-person perspective, and only a few addressing specific tasks from the first-person perspective. However, the capability of VLMs to "think" from a first-person perspective, a crucial attribute for advancing autonomous agents and robotics, remains largely unexplored. To bridge this research gap, we introduce EgoThink, a novel visual question-answering benchmark that encompasses six core capabilities with twelve detailed dimensions. The benchmark is constructed using selected clips from egocentric videos, with manually annotated question-answer pairs containing first-person information. To comprehensively assess VLMs, we evaluate eighteen popular VLMs on EgoThink. Moreover, given the open-ended format of the answers, we use GPT-4 as the automatic judge to compute single-answer grading. Experimental results indicate that although GPT-4V leads in numerous dimensions, all evaluated VLMs still possess considerable potential for improvement in first-person perspective tasks. Meanwhile, enlarging the number of trainable parameters has the most significant impact on model performance on EgoThink. In conclusion, EgoThink serves as a valuable addition to existing evaluation benchmarks for VLMs, providing an indispensable resource for future research in the realm of embodied artificial intelligence and robotics.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2311.15596](https://arxiv.org/abs/2311.15596) [cs.CV] |
|           | (or [arXiv:2311.15596v1](https://arxiv.org/abs/2311.15596v1) [cs.CV] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.15596Focus to learn more |





<h2 id="2023-11-28-16">16. ChatGPT and Beyond: The Generative AI Revolution in Education
</h2>

Title: [ChatGPT and Beyond: The Generative AI Revolution in Education](https://arxiv.org/abs/2311.15198)

Authors: [Mohammad AL-Smadi](https://arxiv.org/search/cs?searchtype=author&query=AL-Smadi,+M)

> The wide adoption and usage of generative artificial intelligence (AI) models, particularly ChatGPT, has sparked a surge in research exploring their potential applications in the educational landscape. This survey examines academic literature published between November, 2022, and July, 2023, specifically targeting high-impact research from Scopus-indexed Q1 and Q2 journals. This survey delves into the practical applications and implications of generative AI models across a diverse range of educational contexts. Through a comprehensive and rigorous evaluation of recent academic literature, this survey seeks to illuminate the evolving role of generative AI models, particularly ChatGPT, in education. By shedding light on the potential benefits, challenges, and emerging trends in this dynamic field, the survey endeavors to contribute to the understanding of the nexus between artificial intelligence and education. The findings of this review will empower educators, researchers, and policymakers to make informed decisions about the integration of AI technologies into learning environments.

| Subjects: | **Computers and Society (cs.CY)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2311.15198](https://arxiv.org/abs/2311.15198) [cs.CY] |
|           | (or [arXiv:2311.15198v1](https://arxiv.org/abs/2311.15198v1) [cs.CY] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.15198Focus to learn more |







# 2023-11-27

[Return to Index](#Index)



<h2 id="2023-11-27-1">1. Calibrated Language Models Must Hallucinate
</h2>

Title: [Calibrated Language Models Must Hallucinate](https://arxiv.org/abs/2311.14648)

Authors: [Adam Tauman Kalai](https://arxiv.org/search/cs?searchtype=author&query=Kalai,+A+T), [Santosh S. Vempala](https://arxiv.org/search/cs?searchtype=author&query=Vempala,+S+S)

> Recent language models have a mysterious tendency to generate false but plausible-sounding text. Such "hallucinations" are an obstacle to the usability of language-based AI systems and can harm people who rely upon their outputs. This work shows shows that there is an inherent statistical reason that pretrained language models hallucinate certain types of facts, having nothing to do with the transformer LM architecture or data quality. For "arbitrary" facts whose veracity cannot be determined from the training data, we show that hallucination is necessary for language models that satisfy a statistical calibration condition appropriate for generative language models. Specifically, if the maximum probability of any fact is bounded, we show that the probability of generating a hallucination is close to the fraction of facts that occur exactly once in the training data (a "Good-Turing" estimate), even assuming ideal training data without errors.
> One conclusion is that models pretrained to be sufficiently good predictors (i.e., calibrated) may require post-training to mitigate hallucinations on the type of arbitrary facts that tend to appear once in the training set. However, our analysis also suggests that there is no statistical reason that pretraining will lead to hallucination on facts that tend to appear more than once in the training data (like references to publications such as articles and books, whose hallucinations have been particularly notable and problematic) or on systematic facts (like arithmetic calculations). Therefore, different architectures and learning algorithms may mitigate these latter types of hallucinations.





<h2 id="2023-11-27-2">2. GPT Struct Me: Probing GPT Models on Narrative Entity Extraction
</h2>

Title: [GPT Struct Me: Probing GPT Models on Narrative Entity Extraction](https://arxiv.org/abs/2311.14583)

Authors: [Hugo Sousa](https://arxiv.org/search/cs?searchtype=author&query=Sousa,+H), [Nuno Guimarães](https://arxiv.org/search/cs?searchtype=author&query=Guimarães,+N), [Alípio Jorge](https://arxiv.org/search/cs?searchtype=author&query=Jorge,+A), [Ricardo Campos](https://arxiv.org/search/cs?searchtype=author&query=Campos,+R)

> The importance of systems that can extract structured information from textual data becomes increasingly pronounced given the ever-increasing volume of text produced on a daily basis. Having a system that can effectively extract such information in an interoperable manner would be an asset for several domains, be it finance, health, or legal. Recent developments in natural language processing led to the production of powerful language models that can, to some degree, mimic human intelligence. Such effectiveness raises a pertinent question: Can these models be leveraged for the extraction of structured information? In this work, we address this question by evaluating the capabilities of two state-of-the-art language models -- GPT-3 and GPT-3.5, commonly known as ChatGPT -- in the extraction of narrative entities, namely events, participants, and temporal expressions. This study is conducted on the Text2Story Lusa dataset, a collection of 119 Portuguese news articles whose annotation framework includes a set of entity structures along with several tags and attribute values. We first select the best prompt template through an ablation study over prompt components that provide varying degrees of information on a subset of documents of the dataset. Subsequently, we use the best templates to evaluate the effectiveness of the models on the remaining documents. The results obtained indicate that GPT models are competitive with out-of-the-box baseline systems, presenting an all-in-one alternative for practitioners with limited resources. By studying the strengths and limitations of these models in the context of information extraction, we offer insights that can guide future improvements and avenues to explore in this field.

| Subjects:    | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Information Retrieval (cs.IR) |
| ------------ | ------------------------------------------------------------ |
| Cite as:     | [arXiv:2311.14583](https://arxiv.org/abs/2311.14583) [cs.CL] |
|              | (or [arXiv:2311.14583v1](https://arxiv.org/abs/2311.14583v1) [cs.CL] for this version) |
| Related DOI: | https://doi.org/10.1109/WI-IAT59888.2023.00063Focus to learn more |





<h2 id="2023-11-27-3">3. Data-Efficient Alignment of Large Language Models with Human Feedback Through Natural Language
</h2>

Title: [Data-Efficient Alignment of Large Language Models with Human Feedback Through Natural Language](https://arxiv.org/abs/2311.14543)

Authors: [Di Jin](https://arxiv.org/search/cs?searchtype=author&query=Jin,+D), [Shikib Mehri](https://arxiv.org/search/cs?searchtype=author&query=Mehri,+S), [Devamanyu Hazarika](https://arxiv.org/search/cs?searchtype=author&query=Hazarika,+D), [Aishwarya Padmakumar](https://arxiv.org/search/cs?searchtype=author&query=Padmakumar,+A), [Sungjin Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee,+S), [Yang Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+Y), [Mahdi Namazifar](https://arxiv.org/search/cs?searchtype=author&query=Namazifar,+M)

> Learning from human feedback is a prominent technique to align the output of large language models (LLMs) with human expectations. Reinforcement learning from human feedback (RLHF) leverages human preference signals that are in the form of ranking of response pairs to perform this alignment. However, human preference on LLM outputs can come in much richer forms including natural language, which may provide detailed feedback on strengths and weaknesses of a given response. In this work we investigate data efficiency of modeling human feedback that is in natural language. Specifically, we fine-tune an open-source LLM, e.g., Falcon-40B-Instruct, on a relatively small amount (1000 records or even less) of human feedback in natural language in the form of critiques and revisions of responses. We show that this model is able to improve the quality of responses from even some of the strongest LLMs such as ChatGPT, BARD, and Vicuna, through critique and revision of those responses. For instance, through one iteration of revision of ChatGPT responses, the revised responses have 56.6% win rate over the original ones, and this win rate can be further improved to 65.9% after applying the revision for five iterations.

| Comments: | Accepted by Workshop on Instruction Tuning and Instruction Following at NeurIPS 2023, Submitted to AAAI 2024 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | [arXiv:2311.14543](https://arxiv.org/abs/2311.14543) [cs.CL] |
|           | (or [arXiv:2311.14543v1](https://arxiv.org/abs/2311.14543v1) [cs.CL] for this version) |





<h2 id="2023-11-27-4">4. Machine Translation for Ge'ez Language
</h2>

Title: [Machine Translation for Ge'ez Language](https://arxiv.org/abs/2311.14530)

Authors: [Aman Kassahun Wassie](https://arxiv.org/search/cs?searchtype=author&query=Wassie,+A+K)

> Machine translation (MT) for low-resource languages such as Ge'ez, an ancient language that is no longer spoken in daily life, faces challenges such as out-of-vocabulary words, domain mismatches, and lack of sufficient labeled training data. In this work, we explore various methods to improve Ge'ez MT, including transfer-learning from related languages, optimizing shared vocabulary and token segmentation approaches, finetuning large pre-trained models, and using large language models (LLMs) for few-shot translation with fuzzy matches. We develop a multilingual neural machine translation (MNMT) model based on languages relatedness, which brings an average performance improvement of about 4 BLEU compared to standard bilingual models. We also attempt to finetune the NLLB-200 model, one of the most advanced translation models available today, but find that it performs poorly with only 4k training samples for Ge'ez. Furthermore, we experiment with using GPT-3.5, a state-of-the-art LLM, for few-shot translation with fuzzy matches, which leverages embedding similarity-based retrieval to find context examples from a parallel corpus. We observe that GPT-3.5 achieves a remarkable BLEU score of 9.2 with no initial knowledge of Ge'ez, but still lower than the MNMT baseline of 15.2. Our work provides insights into the potential and limitations of different approaches for low-resource and ancient language MT.





<h2 id="2023-11-27-5">5. Controlled Text Generation via Language Model Arithmetic
</h2>

Title: [Controlled Text Generation via Language Model Arithmetic](https://arxiv.org/abs/2311.14479)

Authors: [Jasper Dekoninck](https://arxiv.org/search/cs?searchtype=author&query=Dekoninck,+J), [Marc Fischer](https://arxiv.org/search/cs?searchtype=author&query=Fischer,+M), [Luca Beurer-Kellner](https://arxiv.org/search/cs?searchtype=author&query=Beurer-Kellner,+L), [Martin Vechev](https://arxiv.org/search/cs?searchtype=author&query=Vechev,+M)

> As Large Language Models (LLMs) are deployed more widely, customization with respect to vocabulary, style and character becomes more important. In this work we introduce model arithmetic, a novel inference framework for composing and biasing LLMs without the need for model (re)training or highly specific datasets. In addition, the framework allows for more precise control of generated text than direct prompting and prior controlled text generation (CTG) techniques. Using model arithmetic, we can express prior CTG techniques as simple formulas and naturally extend them to new and more effective formulations. Further, we show that speculative sampling, a technique for efficient LLM sampling, extends to our setting. This enables highly efficient text generation with multiple composed models with only marginal overhead over a single model. Our empirical evaluation demonstrates that model arithmetic allows fine-grained control of generated text while outperforming state-of-the-art on the task of toxicity reduction.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2311.14479](https://arxiv.org/abs/2311.14479) [cs.CL] |
|           | (or [arXiv:2311.14479v1](https://arxiv.org/abs/2311.14479v1) [cs.CL] for this version) |





<h2 id="2023-11-27-6">6. DP-NMT: Scalable Differentially-Private Machine Translation
</h2>

Title: [DP-NMT: Scalable Differentially-Private Machine Translation](https://arxiv.org/abs/2311.14465)

Authors: [Timour Igamberdiev](https://arxiv.org/search/cs?searchtype=author&query=Igamberdiev,+T), [Doan Nam Long Vu](https://arxiv.org/search/cs?searchtype=author&query=Vu,+D+N+L), [Felix Künnecke](https://arxiv.org/search/cs?searchtype=author&query=Künnecke,+F), [Zhuo Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu,+Z), [Jannik Holmer](https://arxiv.org/search/cs?searchtype=author&query=Holmer,+J), [Ivan Habernal](https://arxiv.org/search/cs?searchtype=author&query=Habernal,+I)

> Neural machine translation (NMT) is a widely popular text generation task, yet there is a considerable research gap in the development of privacy-preserving NMT models, despite significant data privacy concerns for NMT systems. Differentially private stochastic gradient descent (DP-SGD) is a popular method for training machine learning models with concrete privacy guarantees; however, the implementation specifics of training a model with DP-SGD are not always clarified in existing models, with differing software libraries used and code bases not always being public, leading to reproducibility issues. To tackle this, we introduce DP-NMT, an open-source framework for carrying out research on privacy-preserving NMT with DP-SGD, bringing together numerous models, datasets, and evaluation metrics in one systematic software package. Our goal is to provide a platform for researchers to advance the development of privacy-preserving NMT systems, keeping the specific details of the DP-SGD algorithm transparent and intuitive to implement. We run a set of experiments on datasets from both general and privacy-related domains to demonstrate our framework in use. We make our framework publicly available and welcome feedback from the community.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2311.14465](https://arxiv.org/abs/2311.14465) [cs.CL] |
|           | (or [arXiv:2311.14465v1](https://arxiv.org/abs/2311.14465v1) [cs.CL] for this version) |





<h2 id="2023-11-27-7">7. Evaluating GPT-4's Vision Capabilities on Brazilian University Admission Exams
</h2>

Title: [Evaluating GPT-4's Vision Capabilities on Brazilian University Admission Exams](https://arxiv.org/abs/2311.14169)

Authors: [Ramon Pires](https://arxiv.org/search/cs?searchtype=author&query=Pires,+R), [Thales Sales Almeida](https://arxiv.org/search/cs?searchtype=author&query=Almeida,+T+S), [Hugo Abonizio](https://arxiv.org/search/cs?searchtype=author&query=Abonizio,+H), [Rodrigo Nogueira](https://arxiv.org/search/cs?searchtype=author&query=Nogueira,+R)

> Recent advancements in language models have showcased human-comparable performance in academic entrance exams. However, existing studies often overlook questions that require the integration of visual comprehension, thus compromising the full spectrum and complexity inherent in real-world scenarios. To address this gap, we present a comprehensive framework to evaluate language models on entrance exams, which incorporates both textual and visual elements. We evaluate the two most recent editions of Exame Nacional do Ensino Médio (ENEM), the main standardized entrance examination adopted by Brazilian universities. Our study not only reaffirms the capabilities of GPT-4 as the state of the art for handling complex multidisciplinary questions, but also pioneers in offering a realistic assessment of multimodal language models on Portuguese examinations. One of the highlights is that text captions transcribing visual content outperform the direct use of images, suggesting that the vision model has room for improvement. Yet, despite improvements afforded by images or captions, mathematical questions remain a challenge for these state-of-the-art models. The code and data used on experiments are available at [this https URL](https://github.com/piresramon/gpt-4-enem).

| Comments: | arXiv admin note: substantial text overlap with [arXiv:2303.17003](https://arxiv.org/abs/2303.17003) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | [arXiv:2311.14169](https://arxiv.org/abs/2311.14169) [cs.CL] |
|           | (or [arXiv:2311.14169v1](https://arxiv.org/abs/2311.14169v1) [cs.CL] for this version) |





<h2 id="2023-11-27-8">8. MLLM-Bench, Evaluating Multi-modal LLMs using GPT-4V
</h2>

Title: [MLLM-Bench, Evaluating Multi-modal LLMs using GPT-4V](https://arxiv.org/abs/2311.13951)

Authors: [Wentao Ge](https://arxiv.org/search/cs?searchtype=author&query=Ge,+W), [Shunian Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen,+S), [Guiming Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen,+G), [Junying Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen,+J), [Zhihong Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen,+Z), [Shuo Yan](https://arxiv.org/search/cs?searchtype=author&query=Yan,+S), [Chenghao Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu,+C), [Ziyue Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin,+Z), [Wenya Xie](https://arxiv.org/search/cs?searchtype=author&query=Xie,+W), [Xidong Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+X), [Anningzhe Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao,+A), [Zhiyi Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+Z), [Jianquan Li](https://arxiv.org/search/cs?searchtype=author&query=Li,+J), [Xiang Wan](https://arxiv.org/search/cs?searchtype=author&query=Wan,+X), [Benyou Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+B)

> In the pursuit of Artificial General Intelligence (AGI), the integration of vision in language models has marked a significant milestone. The advent of vision-language models (MLLMs) like GPT-4V have expanded AI applications, aligning with the multi-modal capabilities of the human brain. However, evaluating the efficacy of MLLMs poses a substantial challenge due to the subjective nature of tasks that lack definitive answers. Existing automatic evaluation methodologies on multi-modal large language models rely on objective queries that have standard answers, inadequately addressing the nuances of creative and associative multi-modal tasks. To address this, we introduce MLLM-Bench, an innovative benchmark inspired by Vicuna, spanning a diverse array of scenarios, including Perception, Understanding, Applying, Analyzing, Evaluating, and Creation along with the ethical consideration. MLLM-Bench is designed to reflect user experience more accurately and provide a more holistic assessment of model performance. Comparative evaluations indicate a significant performance gap between existing open-source models and GPT-4V. We posit that MLLM-Bench will catalyze progress in the open-source community towards developing user-centric vision-language models that meet a broad spectrum of real-world applications. See online leaderboard in \url{[this https URL](https://mllm-bench.llmzoo.com/)}.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2311.13951](https://arxiv.org/abs/2311.13951) [cs.CL] |
|           | (or [arXiv:2311.13951v1](https://arxiv.org/abs/2311.13951v1) [cs.CL] for this version) |





<h2 id="2023-11-27-9">9. Efficient Transformer Knowledge Distillation: A Performance Review
</h2>

Title: [Efficient Transformer Knowledge Distillation: A Performance Review](https://arxiv.org/abs/2311.13657)

Authors: [Nathan Brown](https://arxiv.org/search/cs?searchtype=author&query=Brown,+N), [Ashton Williamson](https://arxiv.org/search/cs?searchtype=author&query=Williamson,+A), [Tahj Anderson](https://arxiv.org/search/cs?searchtype=author&query=Anderson,+T), [Logan Lawrence](https://arxiv.org/search/cs?searchtype=author&query=Lawrence,+L)

> As pretrained transformer language models continue to achieve state-of-the-art performance, the Natural Language Processing community has pushed for advances in model compression and efficient attention mechanisms to address high computational requirements and limited input sequence length. Despite these separate efforts, no investigation has been done into the intersection of these two fields. In this work, we provide an evaluation of model compression via knowledge distillation on efficient attention transformers. We provide cost-performance trade-offs for the compression of state-of-the-art efficient attention architectures and the gains made in performance in comparison to their full attention counterparts. Furthermore, we introduce a new long-context Named Entity Recognition dataset, GONERD, to train and test the performance of NER models on long sequences. We find that distilled efficient attention transformers can preserve a significant amount of original model performance, preserving up to 98.6% across short-context tasks (GLUE, SQUAD, CoNLL-2003), up to 94.6% across long-context Question-and-Answering tasks (HotpotQA, TriviaQA), and up to 98.8% on long-context Named Entity Recognition (GONERD), while decreasing inference times by up to 57.8%. We find that, for most models on most tasks, performing knowledge distillation is an effective method to yield high-performing efficient attention models with low costs.

| Comments: | Accepted to EMNLP 2023. 12 pages, 1 figure, 11 tables. Models and data available at [this https URL](https://huggingface.co/giant-oak) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | [arXiv:2311.13657](https://arxiv.org/abs/2311.13657) [cs.CL] |
|           | (or [arXiv:2311.13657v1](https://arxiv.org/abs/2311.13657v1) [cs.CL] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.13657Focus to learn more |





<h2 id="2023-11-27-10">10. Language Model Inversion
</h2>

Title: [Language Model Inversion](https://arxiv.org/abs/2311.13647)

Authors: [John X. Morris](https://arxiv.org/search/cs?searchtype=author&query=Morris,+J+X), [Wenting Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao,+W), [Justin T. Chiu](https://arxiv.org/search/cs?searchtype=author&query=Chiu,+J+T), [Vitaly Shmatikov](https://arxiv.org/search/cs?searchtype=author&query=Shmatikov,+V), [Alexander M. Rush](https://arxiv.org/search/cs?searchtype=author&query=Rush,+A+M)

> Language models produce a distribution over the next token; can we use this information to recover the prompt tokens? We consider the problem of language model inversion and show that next-token probabilities contain a surprising amount of information about the preceding text. Often we can recover the text in cases where it is hidden from the user, motivating a method for recovering unknown prompts given only the model's current distribution output. We consider a variety of model access scenarios, and show how even without predictions for every token in the vocabulary we can recover the probability vector through search. On Llama-2 7b, our inversion method reconstructs prompts with a BLEU of 59and token-level F1 of 78 and recovers 27% of prompts exactly. Code for reproducing all experiments is available at [this http URL](http://github.com/jxmorris12/vec2text).

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2311.13647](https://arxiv.org/abs/2311.13647) [cs.CL] |
|           | (or [arXiv:2311.13647v1](https://arxiv.org/abs/2311.13647v1) [cs.CL] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.13647Focus to learn more |





<h2 id="2023-11-27-11">11. tinyCLAP: Distilling Constrastive Language-Audio Pretrained Models
</h2>

Title: [tinyCLAP: Distilling Constrastive Language-Audio Pretrained Models](https://arxiv.org/abs/2311.14517)

Authors: [Francesco Paissan](https://arxiv.org/search/cs?searchtype=author&query=Paissan,+F), [Elisabetta Farella](https://arxiv.org/search/cs?searchtype=author&query=Farella,+E)

> Contrastive Language-Audio Pretraining (CLAP) became of crucial importance in the field of audio and speech processing. Its employment ranges from sound event detection to text-to-audio generation. However, one of the main limitations is the considerable amount of data required in the training process and the overall computational complexity during inference. This paper investigates how we can reduce the complexity of contrastive language-audio pre-trained models, yielding an efficient model that we call tinyCLAP. We derive an unimodal distillation loss from first principles and explore how the dimensionality of the shared, multimodal latent space can be reduced via pruning. TinyCLAP uses only 6% of the original Microsoft CLAP parameters with a minimal reduction (less than 5%) in zero-shot classification performance across the three sound event detection datasets on which it was tested

| Subjects: | **Sound (cs.SD)**; Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2311.14517](https://arxiv.org/abs/2311.14517) [cs.SD] |
|           | (or [arXiv:2311.14517v1](https://arxiv.org/abs/2311.14517v1) [cs.SD] for this version) |





<h2 id="2023-11-27-12">12. Prompt Risk Control: A Rigorous Framework for Responsible Deployment of Large Language Models
</h2>

Title: [Prompt Risk Control: A Rigorous Framework for Responsible Deployment of Large Language Models](https://arxiv.org/abs/2311.13628)

Authors: [Thomas P. Zollo](https://arxiv.org/search/cs?searchtype=author&query=Zollo,+T+P), [Todd Morrill](https://arxiv.org/search/cs?searchtype=author&query=Morrill,+T), [Zhun Deng](https://arxiv.org/search/cs?searchtype=author&query=Deng,+Z), [Jake C. Snell](https://arxiv.org/search/cs?searchtype=author&query=Snell,+J+C), [Toniann Pitassi](https://arxiv.org/search/cs?searchtype=author&query=Pitassi,+T), [Richard Zemel](https://arxiv.org/search/cs?searchtype=author&query=Zemel,+R)

> The recent explosion in the capabilities of large language models has led to a wave of interest in how best to prompt a model to perform a given task. While it may be tempting to simply choose a prompt based on average performance on a validation set, this can lead to a deployment where unexpectedly poor responses are generated, especially for the worst-off users. To mitigate this prospect, we propose Prompt Risk Control, a lightweight framework for selecting a prompt based on rigorous upper bounds on families of informative risk measures. We offer methods for producing bounds on a diverse set of metrics, including quantities that measure worst-case responses and disparities in generation quality across the population of users. In addition, we extend the underlying statistical bounding techniques to accommodate the possibility of distribution shifts in deployment. Experiments on applications such as open-ended chat, medical question summarization, and code generation highlight how such a framework can foster responsible deployment by reducing the risk of the worst outcomes.

| Comments: | 33 pages, 10 figures, and accepted to the Socially Responsible Language Modelling Research (SoLaR) workshop at NeurIPS 2023 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL) |
| Cite as:  | [arXiv:2311.13628](https://arxiv.org/abs/2311.13628) [cs.LG] |
|           | (or [arXiv:2311.13628v1](https://arxiv.org/abs/2311.13628v1) [cs.LG] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.13628Focus to learn more |



# 2023-11-23

[Return to Index](#Index)



<h2 id="2022-11-23-1">1. LM-Cocktail: Resilient Tuning of Language Models via Model Merging
</h2>


Title: [LM-Cocktail: Resilient Tuning of Language Models via Model Merging](https://arxiv.org/abs/2311.13534)

Authors: [Shitao Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao,+S), [Zheng Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+Z), [Peitian Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+P), [Xingrun Xing](https://arxiv.org/search/cs?searchtype=author&query=Xing,+X)

> The pre-trained language models are continually fine-tuned to better support downstream applications. However, this operation may result in significant performance degeneration on general tasks beyond the targeted domain. To overcome this problem, we propose a novel method which enables the fine-tuned model to stay resilient in general perspectives. Our method is conducted in the form of model merging (namely LM-Cocktail), where the fine-tuned language model is merged with the pre-trained base model or the peer models from other domains through weighted average. Despite simplicity, LM-Cocktail is surprisingly effective: the resulted model is able to achieve a strong empirical performance in the whole scope of general tasks while preserving a superior capacity in its targeted domain. We conduct comprehensive experiments with LLama and BGE model on popular benchmarks, including FLAN, MMLU, MTEB, whose results validate the efficacy of our proposed method. The code and checkpoints are available at [this https URL](https://github.com/FlagOpen/FlagEmbedding).

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Information Retrieval (cs.IR) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2311.13534](https://arxiv.org/abs/2311.13534) [cs.CL] |
|           | (or [arXiv:2311.13534v1](https://arxiv.org/abs/2311.13534v1) [cs.CL] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.13534Focus to learn more |





<h2 id="2022-11-23-2">2. Machine Translation to Control Formality Features in the Target Language
</h2>


Title: [Machine Translation to Control Formality Features in the Target Language](https://arxiv.org/abs/2311.13475)

Authors: [Harshita Tyagi](https://arxiv.org/search/cs?searchtype=author&query=Tyagi,+H), [Prashasta Jung](https://arxiv.org/search/cs?searchtype=author&query=Jung,+P), [Hyowon Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee,+H)

> Formality plays a significant role in language communication, especially in low-resource languages such as Hindi, Japanese and Korean. These languages utilise formal and informal expressions to convey messages based on social contexts and relationships. When a language translation technique is used to translate from a source language that does not pertain the formality (e.g. English) to a target language that does, there is a missing information on formality that could be a challenge in producing an accurate outcome. This research explores how this issue should be resolved when machine learning methods are used to translate from English to languages with formality, using Hindi as the example data. This was done by training a bilingual model in a formality-controlled setting and comparing its performance with a pre-trained multilingual model in a similar setting. Since there are not a lot of training data with ground truth, automated annotation techniques were employed to increase the data size. The primary modeling approach involved leveraging transformer models, which have demonstrated effectiveness in various natural language processing tasks. We evaluate the official formality accuracy(ACC) by comparing the predicted masked tokens with the ground truth. This metric provides a quantitative measure of how well the translations align with the desired outputs. Our study showcases a versatile translation strategy that considers the nuances of formality in the target language, catering to diverse language communication needs and scenarios.

| Comments: | 9 pages, based on DCU MCM Practicum 2022/2023                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Human-Computer Interaction (cs.HC); Machine Learning (cs.LG) |
| Cite as:  | [arXiv:2311.13475](https://arxiv.org/abs/2311.13475) [cs.CL] |
|           | (or [arXiv:2311.13475v1](https://arxiv.org/abs/2311.13475v1) [cs.CL] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.13475Focus to learn more |





<h2 id="2022-11-23-3">3. Mitigating Large Language Model Hallucinations via Autonomous Knowledge Graph-based Retrofitting
</h2>


Title: [Mitigating Large Language Model Hallucinations via Autonomous Knowledge Graph-based Retrofitting](https://arxiv.org/abs/2311.13314)

Authors: [Xinyan Guan](https://arxiv.org/search/cs?searchtype=author&query=Guan,+X), [Yanjiang Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+Y), [Hongyu Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin,+H), [Yaojie Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu,+Y), [Ben He](https://arxiv.org/search/cs?searchtype=author&query=He,+B), [Xianpei Han](https://arxiv.org/search/cs?searchtype=author&query=Han,+X), [Le Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun,+L)

> Incorporating factual knowledge in knowledge graph is regarded as a promising approach for mitigating the hallucination of large language models (LLMs). Existing methods usually only use the user's input to query the knowledge graph, thus failing to address the factual hallucination generated by LLMs during its reasoning process. To address this problem, this paper proposes Knowledge Graph-based Retrofitting (KGR), a new framework that incorporates LLMs with KGs to mitigate factual hallucination during the reasoning process by retrofitting the initial draft responses of LLMs based on the factual knowledge stored in KGs. Specifically, KGR leverages LLMs to extract, select, validate, and retrofit factual statements within the model-generated responses, which enables an autonomous knowledge verifying and refining procedure without any additional manual efforts. Experiments show that KGR can significantly improve the performance of LLMs on factual QA benchmarks especially when involving complex reasoning processes, which demonstrates the necessity and effectiveness of KGR in mitigating hallucination and enhancing the reliability of LLMs.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2311.13314](https://arxiv.org/abs/2311.13314) [cs.CL] |
|           | (or [arXiv:2311.13314v1](https://arxiv.org/abs/2311.13314v1) [cs.CL] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.13314Focus to learn more |





<h2 id="2022-11-23-4">4. Automatic Instruction Optimization for Open-source LLM Instruction Tuning
</h2>
Title: [Automatic Instruction Optimization for Open-source LLM Instruction Tuning](https://arxiv.org/abs/2311.13246)

Authors: [Yilun Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+Y), [Shimin Tao](https://arxiv.org/search/cs?searchtype=author&query=Tao,+S), [Xiaofeng Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao,+X), [Ming Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu,+M), [Wenbing Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma,+W), [Junhao Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu,+J), [Chang Su](https://arxiv.org/search/cs?searchtype=author&query=Su,+C), [Yutai Hou](https://arxiv.org/search/cs?searchtype=author&query=Hou,+Y), [Miao Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+M), [Min Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+M), [Hongxia Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma,+H), [Li Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+L), [Hao Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang,+H), [Yanfei Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang,+Y)

> Instruction tuning is crucial for enabling Language Learning Models (LLMs) in responding to human instructions. The quality of instruction pairs used for tuning greatly affects the performance of LLMs. However, the manual creation of high-quality instruction datasets is costly, leading to the adoption of automatic generation of instruction pairs by LLMs as a popular alternative in the training of open-source LLMs. To ensure the high quality of LLM-generated instruction datasets, several approaches have been proposed. Nevertheless, existing methods either compromise dataset integrity by filtering a large proportion of samples, or are unsuitable for industrial applications. In this paper, instead of discarding low-quality samples, we propose CoachLM, a novel approach to enhance the quality of instruction datasets through automatic revisions on samples in the dataset. CoachLM is trained from the samples revised by human experts and significantly increases the proportion of high-quality samples in the dataset from 17.7% to 78.9%. The effectiveness of CoachLM is further assessed on various real-world instruction test sets. The results show that CoachLM improves the instruction-following capabilities of the instruction-tuned LLM by an average of 29.9%, which even surpasses larger LLMs with nearly twice the number of parameters. Furthermore, CoachLM is successfully deployed in a data management system for LLMs at Huawei, resulting in an efficiency improvement of up to 20% in the cleaning of 40k real-world instruction pairs. We release the training data and code of CoachLM ([this https URL](https://github.com/lunyiliu/CoachLM)).

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2311.13246](https://arxiv.org/abs/2311.13246) [cs.CL] |
|           | (or [arXiv:2311.13246v1](https://arxiv.org/abs/2311.13246v1) [cs.CL] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.13246Focus to learn more |





<h2 id="2022-11-23-5">5. On the Calibration of Large Language Models and Alignment
</h2>


Title: [On the Calibration of Large Language Models and Alignment](https://arxiv.org/abs/2311.13240)

Authors: [Chiwei Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu,+C), [Benfeng Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu,+B), [Quan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+Q), [Yongdong Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+Y), [Zhendong Mao](https://arxiv.org/search/cs?searchtype=author&query=Mao,+Z)

> As large language models attract increasing attention and find widespread application, concurrent challenges of reliability also arise at the same time. Confidence calibration, an effective analysis method for gauging the reliability of deep models, serves as a crucial tool for assessing and improving their reliability. However, such investigation has been comparatively underexplored. In this work, we conduct a systematic examination of the calibration of aligned language models throughout the entire construction process, including pretraining and alignment training. At each stage, we investigate how different training settings, such as parameter scales and training data, affect model calibration. To thoroughly assess model calibration, we evaluate models on three most concerned aspects: generation, factuality and understanding. Our work sheds light on whether popular LLMs are well-calibrated and how the training process influences model calibration.

| Comments: | to be published in findings of EMNLP-2023                    |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | [arXiv:2311.13240](https://arxiv.org/abs/2311.13240) [cs.CL] |
|           | (or [arXiv:2311.13240v1](https://arxiv.org/abs/2311.13240v1) [cs.CL] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.13240Focus to learn more |





<h2 id="2022-11-23-6">6. GAIA: a benchmark for General AI Assistants
</h2>


Title: [GAIA: a benchmark for General AI Assistants](https://arxiv.org/abs/2311.12983)

Authors: [Grégoire Mialon](https://arxiv.org/search/cs?searchtype=author&query=Mialon,+G), [Clémentine Fourrier](https://arxiv.org/search/cs?searchtype=author&query=Fourrier,+C), [Craig Swift](https://arxiv.org/search/cs?searchtype=author&query=Swift,+C), [Thomas Wolf](https://arxiv.org/search/cs?searchtype=author&query=Wolf,+T), [Yann LeCun](https://arxiv.org/search/cs?searchtype=author&query=LeCun,+Y), [Thomas Scialom](https://arxiv.org/search/cs?searchtype=author&query=Scialom,+T)

> We introduce GAIA, a benchmark for General AI Assistants that, if solved, would represent a milestone in AI research. GAIA proposes real-world questions that require a set of fundamental abilities such as reasoning, multi-modality handling, web browsing, and generally tool-use proficiency. GAIA questions are conceptually simple for humans yet challenging for most advanced AIs: we show that human respondents obtain 92\% vs. 15\% for GPT-4 equipped with plugins. This notable performance disparity contrasts with the recent trend of LLMs outperforming humans on tasks requiring professional skills in e.g. law or chemistry. GAIA's philosophy departs from the current trend in AI benchmarks suggesting to target tasks that are ever more difficult for humans. We posit that the advent of Artificial General Intelligence (AGI) hinges on a system's capability to exhibit similar robustness as the average human does on such questions. Using GAIA's methodology, we devise 466 questions and their answer. We release our questions while retaining answers to 300 of them to power a leader-board available at [this https URL](https://huggingface.co/gaia-benchmark).

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2311.12983](https://arxiv.org/abs/2311.12983) [cs.CL] |
|           | (or [arXiv:2311.12983v1](https://arxiv.org/abs/2311.12983v1) [cs.CL] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.12983Focus to learn more |





<h2 id="2022-11-23-7">7. AS-LLM: When Algorithm Selection Meets Large Language Model
</h2>


Title: [AS-LLM: When Algorithm Selection Meets Large Language Model](https://arxiv.org/abs/2311.13184)

Authors: [Xingyu Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu,+X), [Yan Zhong](https://arxiv.org/search/cs?searchtype=author&query=Zhong,+Y), [Jibin Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu,+J), [Kay Chen Tan](https://arxiv.org/search/cs?searchtype=author&query=Tan,+K+C)

> Algorithm selection aims to identify the most suitable algorithm for solving a specific problem before execution, which has become a critical process of the AutoML. Current mainstream algorithm selection techniques rely heavily on feature representations of various problems and employ the performance of each algorithm as supervised information. However, there is a significant research gap concerning the consideration of algorithm features. This gap is primarily attributed to the inherent complexity of algorithms, making it particularly challenging to find a universally effective feature extraction method that is applicable across a diverse range of algorithms. Unfortunately, neglecting this aspect undoubtedly impacts the accuracy of algorithm selection and indirectly necessitates an increased volume of problem data for training purposes. This paper takes a significant stride towards addressing this gap by proposing an approach that integrates algorithm representation into the algorithm selection process. Specifically, our proposed model employs distinct modules to extract representations of both problems and algorithms, where the algorithm representation leverages the capabilities of pre-trained LLMs in the realm of code comprehension. Following the extraction of embedding vectors for both algorithms and problems, the most suitable algorithm is determined through calculations of matching degrees. Our experiments not only validate the effectiveness of the proposed model but also showcase the performance of different embedded pre-trained LLMs, which suggests that the proposed algorithm selection framework holds the potential to serve as a baseline task for evaluating the code representation capabilities of LLMs.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | [arXiv:2311.13184](https://arxiv.org/abs/2311.13184) [cs.LG] |
|           | (or [arXiv:2311.13184v1](https://arxiv.org/abs/2311.13184v1) [cs.LG] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.13184Focus to learn more |





<h2 id="2022-11-23-8">8. LIMIT: Less Is More for Instruction Tuning Across Evaluation Paradigms
</h2>


Title: [LIMIT: Less Is More for Instruction Tuning Across Evaluation Paradigms](https://arxiv.org/abs/2311.13133)

Authors: [Aditi Jha](https://arxiv.org/search/cs?searchtype=author&query=Jha,+A), [Sam Havens](https://arxiv.org/search/cs?searchtype=author&query=Havens,+S), [Jeremey Dohmann](https://arxiv.org/search/cs?searchtype=author&query=Dohmann,+J), [Alex Trott](https://arxiv.org/search/cs?searchtype=author&query=Trott,+A), [Jacob Portes](https://arxiv.org/search/cs?searchtype=author&query=Portes,+J)

> Large Language Models are traditionally finetuned on large instruction datasets. However recent studies suggest that small, high-quality datasets can suffice for general purpose instruction following. This lack of consensus surrounding finetuning best practices is in part due to rapidly diverging approaches to LLM evaluation. In this study, we ask whether a small amount of diverse finetuning samples can improve performance on both traditional perplexity-based NLP benchmarks, and on open-ended, model-based evaluation. We finetune open-source MPT-7B and MPT-30B models on instruction finetuning datasets of various sizes ranging from 1k to 60k samples. We find that subsets of 1k-6k instruction finetuning samples are sufficient to achieve good performance on both (1) traditional NLP benchmarks and (2) model-based evaluation. Finally, we show that mixing textbook-style and open-ended QA finetuning datasets optimizes performance on both evaluation paradigms.

| Comments: | 36 pages, 12 figures, NeurIPS 2023 Workshop on Instruction Tuning and Instruction Following |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL) |
| Cite as:  | [arXiv:2311.13133](https://arxiv.org/abs/2311.13133) [cs.LG] |
|           | (or [arXiv:2311.13133v1](https://arxiv.org/abs/2311.13133v1) [cs.LG] for this version) |
|           | https://doi.org/10.48550/arXiv.2311.13133Focus to learn more |





# 2022-10-20

[Return to Index](#Index)





<h2 id="2022-10-20-1">1. RankT5: Fine-Tuning T5 for Text Ranking with Ranking Losses
</h2>


Title: [RankT5: Fine-Tuning T5 for Text Ranking with Ranking Losses](https://arxiv.org/abs/2210.10634)

Authors: [Honglei Zhuang](https://arxiv.org/search/cs?searchtype=author&query=Zhuang%2C+H), [Zhen Qin](https://arxiv.org/search/cs?searchtype=author&query=Qin%2C+Z), [Rolf Jagerman](https://arxiv.org/search/cs?searchtype=author&query=Jagerman%2C+R), [Kai Hui](https://arxiv.org/search/cs?searchtype=author&query=Hui%2C+K), [Ji Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+J), [Jing Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+J), [Jianmo Ni](https://arxiv.org/search/cs?searchtype=author&query=Ni%2C+J), [Xuanhui Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Michael Bendersky](https://arxiv.org/search/cs?searchtype=author&query=Bendersky%2C+M)

> Recently, substantial progress has been made in text ranking based on pretrained language models such as BERT. However, there are limited studies on how to leverage more powerful sequence-to-sequence models such as T5. Existing attempts usually formulate text ranking as classification and rely on postprocessing to obtain a ranked list. In this paper, we propose RankT5 and study two T5-based ranking model structures, an encoder-decoder and an encoder-only one, so that they not only can directly output ranking scores for each query-document pair, but also can be fine-tuned with "pairwise" or "listwise" ranking losses to optimize ranking performances. Our experiments show that the proposed models with ranking losses can achieve substantial ranking performance gains on different public text ranking data sets. Moreover, when fine-tuned with listwise ranking losses, the ranking model appears to have better zero-shot ranking performance on out-of-domain data sets compared to the model fine-tuned with classification losses.

| Comments: | 13 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Information Retrieval (cs.IR)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2210.10634](https://arxiv.org/abs/2210.10634) [cs.IR]** |
|           | (or **[arXiv:2210.10634v1](https://arxiv.org/abs/2210.10634v1) [cs.IR]** for this version) |
|           | https://doi.org/10.48550/arXiv.2210.10634Focus to learn more |







