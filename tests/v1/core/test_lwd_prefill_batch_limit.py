# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the DeepSeek V4 edge prefill batch limit.

picker(_lwd_pick_prefill_batch)经 AST 抽取在 SimpleNamespace 上裸跑;
云内 DP 的目标过滤(_lwd_route/_lwd_route_of/_lwd_engine_load)以
NS 字段补位:空路由表 + 缺省目标 (0, 0) 退化为旧全局口径。
"""

import ast
import logging
import unittest
from pathlib import Path
import time
from types import SimpleNamespace as NS


class TestPrefillBatchLimit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = (
            Path(__file__).resolve().parents[3]
            / "vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_scheduler.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        scheduler = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "LwdEdgeScheduler"
        )
        method = next(
            node
            for node in scheduler.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_lwd_pick_prefill_batch"
        )
        namespace = {
            "logger": logging.getLogger("lwd.picker.test"),
            "time": time,
        }
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        cls.pick = staticmethod(namespace[method.name])

    @staticmethod
    def request(rid, prompt=8196, computed=0):
        return NS(
            request_id=rid, num_prompt_tokens=prompt, num_computed_tokens=computed
        )

    def runner(self, model_type="deepseek_v4", running=(), waiting=(),
               skipped=(), awaiting=(), route=None):
        route = dict(route or {})

        def route_of(rid):
            return route.get(rid, (0, 0))

        def engine_load(cloud_id, engine):
            prefill = sum(
                1 for r in running if route_of(r.request_id) == (cloud_id, engine)
            )
            decode = sum(1 for rid in awaiting if route_of(rid) == (cloud_id, engine))
            return prefill, decode

        return NS(
            vllm_config=NS(model_config=NS(hf_text_config=NS(model_type=model_type))),
            _lwd_mixed=True,
            _lwd_awaiting=dict.fromkeys(awaiting, 0.0),
            # 多 DP 路由账(空表 = 全部归 (0, 0),与单云单引擎语义一致)
            # 与 per-目标负载访问器
            _lwd_route=route,
            _lwd_route_of=route_of,
            _lwd_engine_load=engine_load,
            _lwd_default_cloud_id=lambda: 0,
            # 网格未学到恒等重切;无 mamba 冲突台账,预检恒不冲突
            _mamba_block_aligned_split=lambda request, n, *a, **k: n,
            _lwd_mamba_dep_conflict=lambda *a, **k: False,
            requests={},
            _lwd_spec_factor=2,
            _lwd_chunk_unit=8196,
            max_num_scheduled_tokens=20000,
            max_num_running_reqs=60,
            running=list(running),
            waiting=list(waiting),
            skipped_waiting=list(skipped),
        )

    def test_dsv4_waiting_batch_contains_only_first_request(self):
        runner = self.runner(waiting=[self.request("a"), self.request("b")])
        picked, expected, _ = self.pick(runner)
        self.assertEqual(picked, ["a"])
        self.assertEqual(expected, {"a": 8196})
        self.assertEqual(len(runner.waiting), 2)

    def test_continuation_excludes_other_continuations_and_new_requests(self):
        runner = self.runner(
            running=[self.request("a", computed=8000), self.request("b")],
            waiting=[self.request("c")],
        )
        self.assertEqual(self.pick(runner)[:2], (["a"], {"a": 196}))

    def test_decode_does_not_consume_prefill_slot_and_skipped_has_priority(self):
        runner = self.runner(
            running=[self.request("decode", computed=8200)],
            skipped=[self.request("skipped")],
            waiting=[self.request("new")],
        )
        self.assertEqual(self.pick(runner)[:2], (["skipped"], {"skipped": 8196}))

    def test_budget_truncation_and_full_sequence_slots(self):
        runner = self.runner(
            running=[self.request("a", computed=1000)], waiting=[self.request("b")]
        )
        runner.max_num_scheduled_tokens = 102
        runner.max_num_running_reqs = 1
        self.assertEqual(self.pick(runner), (["a"], {"a": 100}, 100))
        runner.running[0].num_computed_tokens = 8196
        self.assertEqual(self.pick(runner), ([], {}, 100))
        runner.max_num_scheduled_tokens = 2
        self.assertEqual(self.pick(runner), ([], {}, 0))

    def test_other_models_keep_multi_request_batches(self):
        for model_type in ("deepseek_v3", "qwen3_5"):
            with self.subTest(model_type=model_type):
                runner = self.runner(
                    model_type,
                    waiting=[
                        self.request("a"),
                        self.request("b"),
                    ],
                )
                self.assertEqual(
                    self.pick(runner)[:2], (["a", "b"], {"a": 8196, "b": 8196})
                )

    def test_threshold_boundary_and_short_batches(self):
        for length in (7999, 8000, 8001, 8062, 8192):
            with self.subTest(length=length):
                runner = self.runner(
                    waiting=[
                        self.request("a", prompt=length),
                        self.request("b", prompt=1),
                    ]
                )
                expected = {"a": length}
                if length < 8000:
                    expected["b"] = 1
                self.assertEqual(self.pick(runner)[:2], (list(expected), expected))

    def test_short_then_long_preserves_fcfs(self):
        runner = self.runner(
            waiting=[
                self.request("short", prompt=10),
                self.request("long", prompt=8000),
                self.request("later", prompt=10),
            ]
        )
        self.assertEqual(self.pick(runner)[:2], (["short"], {"short": 10}))

    def test_long_continuation_blocks_new_short_requests(self):
        runner = self.runner(
            running=[self.request("long", prompt=8000, computed=7998)],
            waiting=[self.request("short", prompt=10)],
        )
        self.assertEqual(self.pick(runner)[:2], (["long"], {"long": 2}))

    def test_pending_long_continuation_is_not_bypassed(self):
        runner = self.runner(
            running=[
                self.request("short", prompt=20, computed=10),
                self.request("long", prompt=8000, computed=7998),
            ],
            waiting=[self.request("new", prompt=10)],
        )
        self.assertEqual(self.pick(runner)[:2], (["short"], {"short": 10}))

    def test_short_continuations_and_waiting_can_batch(self):
        runner = self.runner(
            running=[self.request("a", prompt=20, computed=10)],
            waiting=[self.request("b", prompt=10), self.request("c", prompt=10)],
        )
        self.assertEqual(
            self.pick(runner)[:2], (["a", "b", "c"], {"a": 10, "b": 10, "c": 10})
        )

    def test_route_filter_picks_only_target_engine(self):
        """云内 DP:picker 只选路由到目标 (cloud, engine) 的请求,
        其他引擎的请求既不续传也不开新(批目标单一性)。"""
        runner = self.runner(
            model_type="qwen3_5",
            running=[self.request("c0", prompt=20, computed=10)],
            waiting=[self.request("w0", prompt=10), self.request("w1", prompt=10)],
            route={"c0": (0, 1), "w1": (0, 1)},
        )
        # 目标 (0, 0):只有 w0 在路由表上归 (0, 0)
        picked, expected, _ = self.pick(runner, (0, 0))
        self.assertEqual((picked, expected), (["w0"], {"w0": 10}))
        # 目标 (0, 1):c0 续传优先,w1 随后
        picked, expected, _ = self.pick(runner, (0, 1))
        self.assertEqual((picked, expected), (["c0", "w1"], {"c0": 10, "w1": 10}))

    def test_engine_load_drives_per_target_cap(self):
        """cap 按目标在途(running+awaiting 中路由到它的)核减,
        其他引擎的在途不占本目标预算。"""
        runner = self.runner(
            model_type="qwen3_5",
            waiting=[self.request("w", prompt=100)],
            awaiting=["a0", "a1"],
            route={"a1": (0, 1)},
        )
        # 目标 (0, 0):仅 a0 在途 -> cap = 20000 - 1*2
        _, _, cap0 = self.pick(runner, (0, 0))
        self.assertEqual(cap0, 20000 - 2)
        # 目标 (0, 1):仅 a1 在途 -> 同值(验证按目标过滤而非全局)
        _, _, cap1 = self.pick(runner, (0, 1))
        self.assertEqual(cap1, 20000 - 2)


if __name__ == "__main__":
    unittest.main()
