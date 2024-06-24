import datasets
from .. import base

# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.

TODO: Write a Short Description of the task.

Homepage: TODO: Add the URL to the task's Homepage here.
"""
import os
from lm_eval.base import MultipleChoiceTask
import numpy as np

# TODO: Add the BibTeX citation for the task.
_CITATION = """"""


class Commongen(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = None
    DATASET_NAME = "ko_commongen_v2"

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        # get the dataset folder
        # it is located on ../datasets/ko_commongen_v2
        self.ko_commongen_v2_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'ko_commongen_v2')   
        test_path = os.path.join(self.ko_commongen_v2_path, 'korean_commongen_v2_847_shuffled.jsonl')
        train_path = os.path.join(self.ko_commongen_v2_path, 'v2_5shot.jsonl')

        test = datasets.load_dataset('json', data_files=test_path)
        print(test['train'][0])

        train = datasets.load_dataset('json', data_files=train_path)
        self.dataset = datasets.DatasetDict({'train': train['train'], 'test': test['train']})

        print(self.dataset['test'][0])

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(map(self._process_doc, self.dataset['train']))
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs(): return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        if self.has_test_docs(): return map(self._process_doc, self.dataset['test'])
    
    def _format_subject(self, subject):
        #words = subject.split("_")
        #return " ".join(words)
        return subject

    # """다음은 주어진 개념정보인 concept set: 에 존재하는 형태소를 조합해서 상식에 부합하는 문장을 만드는 작업이다. """
    # """concept set: 의 형태소를 조합하여 만든 4개의 예시 중에서 가장 상식적으로 타당한 문장을 포함한 선택지의 번호를 선택하라."""
    # """The following task involves combining morphemes from the concept set: to create a sentence that is consistent with commonsense."""
    # """Choose the number of the option that contains the most logically valid sentence among the four examples created by combining morphemes from the concept set: """
    #"""以下任务是结合给定概念信息 concept set: 中的形态素，创造出符合常识的句子。"""
    #"""从通过组合 concept set: 中的形态素创造的四个例子中，选择包含最符合常识和合理的句子的选项编号。"""
    #"""次は、与えられた概念情報 concept set: に存在する形態素を組み合わせて、常識に合う文を作る作業です。"""
    #""" concept set: の形態素を組み合わせて作った4つの例の中から、最も常識的で妥当な文を含む選択肢の番号を選んでください。"""
    #"""La siguiente tarea consiste en combinar morfemas existentes en el conjunto de conceptos dado, concept set:, para crear una oración que concuerde con el sentido común."""
    #"""Elige el número de la opción que incluya la oración más coherente y válida entre los cuatro ejemplos creados combinando morfemas del concept set: ."""

    def fewshot_context(self, doc, num_fewshot, **kwargs):
        subject = self.DATASET_NAME
        description = (
            """다음은 주어진 개념정보인 concept set: 에 존재하는 형태소를 조합해서 상식에 부합하는 문장을 만드는 작업이다. """
            """concept set: 의 형태소를 조합하여 만든 4개의 예시 중에서 가장 상식적으로 타당한 문장을 포함한 선택지의 번호를 선택하라."""
        )
        kwargs["description"] = description
        return super().fewshot_context(doc=doc, num_fewshot=num_fewshot, **kwargs)
    
    def _process_doc(self, doc):
        query = (
            #"""다음은 주어진 개념정보인 concept set: 에 존재하는 형태소를 조합해서 상식에 부합하는 문장을 만드는 작업이다. """
            #"""concept set: 의 형태소를 조합하여 만든 4개의 예시 중에서 가장 상식적으로 타당한 문장을 포함한 선택지의 번호를 선택하라.\n\n"""
            f"""concept set: {{{doc['concept_set'].replace("#", ", ")}}}\n""")        
        query += "\n".join([f"{i+1}. {doc[str(i+1)]}" for i in range(4)])
        query += "\n정답:"
        # query += "\nAnswer:"
        # query += "\n请回答:"
        # query += "\n正答:"
        # query += "\nContesta:"

        out_doc = {
            "query": query,
            "choices": [f"{i+1}. {doc[str(i+1)]}" for i in range(4)],
            # "choices": [f"{doc[str(i+1)]}" for i in range(4)],
            # "choices": [f'{str(i+1)}. ' + doc['{i}'.format(i=i + 1)] for i in range(4)],  # The list of choices.
            # "choices": [str(i+1) for i in range(4)],  # The list of choices.
            "gold": doc['gold']-1,  # The integer used to index into the correct element of `"choices"`.
        }
        return out_doc

    def doc_to_text(self, doc):
        #ret = f"{doc['query']}\n1. {doc['choices'][0]}\n2. {doc['choices'][1]}\n3. {doc['choices'][2]}\n4. {doc['choices'][3]}\nAnswer:"
        return doc['query']

    #def doc_to_target(self, doc):
    #    #return " " + doc["choices"][(doc["gold"]) - 1]
    #    return " " + ['A', 'B', 'C', 'D'][(doc["gold"]) - 1]

    """
    def process_results(self, doc, results):
        gold = doc["gold"] - 1 # 레이블이 1부터 4로 되어 있음.

        acc = 1.0 if np.argmax(results) == gold else 0.0
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        return {
            "acc": acc,
            "acc_norm": acc_norm,
        }
    """


