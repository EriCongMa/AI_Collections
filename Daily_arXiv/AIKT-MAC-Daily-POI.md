# MA C.'s Daily Paper Of Interest - November, 2022

# Index

- [2022-11-23](#2022-11-23)
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







