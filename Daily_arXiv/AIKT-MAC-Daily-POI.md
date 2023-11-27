# MA C.'s Daily Paper Of Interest - November, 2022

# Index

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







