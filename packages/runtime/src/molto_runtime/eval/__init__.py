# SPDX-License-Identifier: Apache-2.0
"""Accuracy evaluation benchmarks for LLMs.

Provides benchmarks across knowledge, commonsense reasoning, math,
coding, safety, and bias categories with deterministic sampling
for fair model comparison.
"""

from molto_runtime.eval.arc import ARCChallengeBenchmark
from molto_runtime.eval.base import BaseBenchmark, BenchmarkResult, QuestionResult
from molto_runtime.eval.bbq import BBQBenchmark
from molto_runtime.eval.cmmlu import CMMLUBenchmark
from molto_runtime.eval.gsm8k import GSM8KBenchmark
from molto_runtime.eval.hellaswag import HellaSwagBenchmark
from molto_runtime.eval.humaneval import HumanEvalBenchmark
from molto_runtime.eval.jmmlu import JMMLUBenchmark
from molto_runtime.eval.kmmlu import KMMLUBenchmark
from molto_runtime.eval.livecodebench import LiveCodeBenchBenchmark
from molto_runtime.eval.mathqa import MathQABenchmark
from molto_runtime.eval.mbpp import MBPPBenchmark
from molto_runtime.eval.mmlu import MMLUBenchmark
from molto_runtime.eval.mmlu_pro import MMLUProBenchmark
from molto_runtime.eval.safetybench import SafetyBenchBenchmark
from molto_runtime.eval.truthfulqa import TruthfulQABenchmark
from molto_runtime.eval.winogrande import WinograndeBenchmark

BENCHMARKS: dict[str, type[BaseBenchmark]] = {
    "mmlu": MMLUBenchmark,
    "mmlu_pro": MMLUProBenchmark,
    "kmmlu": KMMLUBenchmark,
    "cmmlu": CMMLUBenchmark,
    "jmmlu": JMMLUBenchmark,
    "hellaswag": HellaSwagBenchmark,
    "truthfulqa": TruthfulQABenchmark,
    "arc_challenge": ARCChallengeBenchmark,
    "winogrande": WinograndeBenchmark,
    "gsm8k": GSM8KBenchmark,
    "mathqa": MathQABenchmark,
    "humaneval": HumanEvalBenchmark,
    "mbpp": MBPPBenchmark,
    "livecodebench": LiveCodeBenchBenchmark,
    "bbq": BBQBenchmark,
    "safetybench": SafetyBenchBenchmark,
}

__all__ = [
    "BENCHMARKS",
    "BaseBenchmark",
    "BenchmarkResult",
    "QuestionResult",
    "MMLUBenchmark",
    "MMLUProBenchmark",
    "HellaSwagBenchmark",
    "TruthfulQABenchmark",
    "ARCChallengeBenchmark",
    "WinograndeBenchmark",
    "GSM8KBenchmark",
    "MathQABenchmark",
    "HumanEvalBenchmark",
    "MBPPBenchmark",
    "LiveCodeBenchBenchmark",
    "BBQBenchmark",
    "SafetyBenchBenchmark",
]
