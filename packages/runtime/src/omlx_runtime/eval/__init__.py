# SPDX-License-Identifier: Apache-2.0
"""Accuracy evaluation benchmarks for LLMs.

Provides benchmarks across knowledge, commonsense reasoning, math,
coding, safety, and bias categories with deterministic sampling
for fair model comparison.
"""

from omlx_runtime.eval.arc import ARCChallengeBenchmark
from omlx_runtime.eval.base import BaseBenchmark, BenchmarkResult, QuestionResult
from omlx_runtime.eval.bbq import BBQBenchmark
from omlx_runtime.eval.cmmlu import CMMLUBenchmark
from omlx_runtime.eval.gsm8k import GSM8KBenchmark
from omlx_runtime.eval.hellaswag import HellaSwagBenchmark
from omlx_runtime.eval.humaneval import HumanEvalBenchmark
from omlx_runtime.eval.jmmlu import JMMLUBenchmark
from omlx_runtime.eval.kmmlu import KMMLUBenchmark
from omlx_runtime.eval.livecodebench import LiveCodeBenchBenchmark
from omlx_runtime.eval.mathqa import MathQABenchmark
from omlx_runtime.eval.mbpp import MBPPBenchmark
from omlx_runtime.eval.mmlu import MMLUBenchmark
from omlx_runtime.eval.mmlu_pro import MMLUProBenchmark
from omlx_runtime.eval.safetybench import SafetyBenchBenchmark
from omlx_runtime.eval.truthfulqa import TruthfulQABenchmark
from omlx_runtime.eval.winogrande import WinograndeBenchmark

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
