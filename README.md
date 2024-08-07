# KoCommonGEN v2
KoCommonGEN v2: A Benchmark for Navigating Korean Commonsense Reasoning Challenges in Large Language Models (ACL-findings 2024)

*Jaehyung Seo, Jaewook Lee, Chanjun Park, SeongTae Hong, Seungjun Lee and Heuiseok Lim* 

🏫 [NLP & AI Lab](https://blpkorea.cafe24.com/wp/), **Korea University** 

---

### 🌠 Overview


### 🛠️ Installation

This repository partially adopts the evaluation methods of version 0.3.0 of [EleutherAI/lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.3.0) for the evaluation of KoCommonGEN v2

```bash
$ git clone https://github.com/J-Seo/KoCommonGEN-V2.git
```

```bash
# python_requires >=3.9
$ cd KoCommonGEN_v2
$ pip install -r requirements.txt 
```
### 🚀 Usage

The maximum number of few-shot examples currently uploaded is 5. Users can freely add more to increase *--num_fewshot*

```bash
$ sh test.sh
```

```bash
## test.sh
python3 main.py \ 
--model hf-causal-experimental \
--model_args pretrained="nlpai-lab/KULLM3" \
--task ko_commongen_v2 \
--device cuda:1 \
--num_fewshot 2 \
--batch_size 1 \
--output nlpai-lab/KULLM3 &
```

### ✔️ Results

An example of the evaluation of [KULLM3](https://github.com/nlpai-lab/KULLM)

hf-causal-experimental (pretrained=nlpai-lab/KULLM3), provide_description: False, num_fewshot: 2, batch_size: 1
|     Task      |Version| Metric |Value |   |Stderr|
|---------------|------:|--------|-----:|---|-----:|
|ko_commongen_v2|      0|acc     |0.5797|±  |0.0170|
|               |       |acc_norm|0.6033|±  |0.0168|

As mentioned in the paper, it is possible to evaluate various models.

### 🇰🇷🇺🇸🇯🇵🇨🇳🇪🇸 Code-switching 

The multilingual dataset consists of 99 samples for numerical commonsense reasoning, which were created relying on machine translation.

The dataset can be found at the following path: `lm_eval/datasets/ko_commongen_v2/shuffled_$LANG$_1.0.jsonl`.

(The code-switching data relies on machine translation, which may result in some inaccuracies.)

If you intend to use it for evaluation, you should modify the prompt and file path in `lm_eval/tasks/ko_commongen_v2.py`.


### 📖 Citation

```
@inproceedings{seo2024Kocommongenv2,
    title = "KoCommonGEN v2: A Benchmark for Navigating Korean Commonsense Reasoning Challenges in Large Language Models",
    author = "Jaehyung Seo and Jaewook Lee and Chanjun Park and SeongTae Hong and Seungjun Lee and Heuiseok Lim",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month = August,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "TBD",
    doi = "TBD",
    pages = "TBD"}
```
### 🚨 Warning!

This dataset contains some instances of toxic speech.


### 🙏 Acknowledgement

We sincerely appreciate the dedication of Sung Kim, Chanjun Park, and Sanghoon Kim from **Upstage AI** in managing one of the benchmark datasets for the
[Open Ko-LLM LeaderBoard](https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard). 











