# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import dataclasses
import json
import pathlib

import pytest

from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LWD_MESSAGE_VERSION,
    LwdAbortNotify,
    LwdC2eNotify,
    LwdCloudParamsNotify,
    LwdPrefixReadyNotify,
    LwdRangeItem,
    LwdRangeNotify,
    LwdRefillNotify,
    LwdRequestNotify,
    lwd_decode_cloud_to_edge,
    lwd_decode_edge_to_cloud,
    lwd_decode_structured_outputs,
    lwd_encode_message,
    lwd_encode_structured_outputs,
)

_SAMPLING_PARAMS_FILE = (
    pathlib.Path(__file__).parents[3] / "vllm" / "sampling_params.py"
)


@dataclasses.dataclass
class _WireStructuredOutputsParams:
    """Local mirror of `StructuredOutputsParams`.

    The wire envelope only depends on the dataclass shape (public init=True
    fields plus the private init=False ones); mirroring it keeps these tests
    import-free of vllm while the AST test below and the live-import test at
    the end pin the mirror to the real class.
    """

    json: object = None
    regex: object = None
    choice: object = None
    grammar: object = None
    json_object: object = None
    disable_any_whitespace: object = None
    disable_additional_properties: object = None
    whitespace_pattern: object = None
    structural_tag: object = None
    _backend: object = dataclasses.field(default=None, init=False)
    _backend_was_auto: object = dataclasses.field(default=None, init=False)


def _mirror_fields() -> dict[str, bool]:
    return {f.name: f.init for f in dataclasses.fields(_WireStructuredOutputsParams)}


def _real_fields_from_source() -> dict[str, bool]:
    """Parse (field name -> init flag) from the real dataclass source."""

    def _init_flag(value: ast.expr | None) -> bool:
        if not isinstance(value, ast.Call):
            return True
        func = value.func
        is_field_call = isinstance(func, ast.Name) and func.id == "field"
        is_field_call |= isinstance(func, ast.Attribute) and func.attr == "field"
        if not is_field_call:
            return True
        for kw in value.keywords:
            if kw.arg == "init":
                # `init=False` -> the constructor rejects the keyword.
                return bool(getattr(kw.value, "value", True))
        return True

    tree = ast.parse(_SAMPLING_PARAMS_FILE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "StructuredOutputsParams":
            fields: dict[str, bool] = {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    fields[stmt.target.id] = _init_flag(stmt.value)
            return fields
    raise AssertionError("StructuredOutputsParams not found in sampling_params.py")


def test_request_notify_round_trips_full_prompt_mrope_positions() -> None:
    assert LWD_MESSAGE_VERSION == 5
    payload = bytes(range(24))
    decoded = lwd_decode_edge_to_cloud(
        lwd_encode_message(
            LwdRequestNotify(
                request_id="r1",
                num_prompt_tokens=1,
                prefix_nonce=7,
                skip_reading_prefix_cache=True,
                mrope_positions=payload,
            )
        )
    )
    assert isinstance(decoded, LwdRequestNotify)
    assert decoded.prefix_nonce == 7
    assert decoded.skip_reading_prefix_cache
    assert decoded.mrope_positions == payload
    assert decoded.structured_outputs == ""


def test_request_notify_round_trips_structured_outputs() -> None:
    wire = lwd_encode_structured_outputs(_WireStructuredOutputsParams(regex="[0-9]+"))
    decoded = lwd_decode_edge_to_cloud(
        lwd_encode_message(
            LwdRequestNotify(
                request_id="r1",
                num_prompt_tokens=1,
                structured_outputs=wire,
            )
        )
    )
    assert isinstance(decoded, LwdRequestNotify)
    assert decoded.structured_outputs == wire


def test_prefix_messages_round_trip() -> None:
    params = LwdCloudParamsNotify(
        max_num_batched_tokens=128,
        max_num_seqs=4,
        num_speculative_tokens=0,
        prefix_caching=True,
        hash_block_size=16,
        prefix_alignment_tokens=32,
        prefix_alignment_required=True,
        prefix_use_eagle=True,
    )
    for message in (
        params,
        LwdPrefixReadyNotify("r1", 32, 7),
        LwdRefillNotify("r1", 16, 32, prefix_nonce=7),
        LwdC2eNotify(
            hidden_num_elements=0,
            top_id_ths=[[]],
            num_accepted_tokens=[1],
            req_ids=["r1"],
            prefix_nonces=[7],
        ),
    ):
        assert lwd_decode_cloud_to_edge(lwd_encode_message(message)) == message


def test_range_item_round_trips_prefix_nonce() -> None:
    message = LwdRangeNotify(
        seqno=3,
        items=[LwdRangeItem("r1", 16, 8, is_refill=True, prefix_nonce=7)],
    )

    decoded = lwd_decode_edge_to_cloud(lwd_encode_message(message))

    assert isinstance(decoded, LwdRangeNotify)
    assert decoded.items[0].prefix_nonce == 7

    abort = lwd_decode_edge_to_cloud(
        lwd_encode_message(LwdAbortNotify("r1", prefix_nonce=7))
    )
    assert isinstance(abort, LwdAbortNotify)
    assert abort.prefix_nonce == 7


def test_structured_outputs_envelope_keeps_private_fields_out() -> None:
    params = _WireStructuredOutputsParams(
        json={"type": "object"},
        whitespace_pattern=" ",
    )
    params._backend = "xgrammar"
    params._backend_was_auto = True

    wire = lwd_encode_structured_outputs(params)
    envelope = json.loads(wire)

    # Private fields never ride the field dict: params_cls(**fields) rejects
    # them as constructor keywords, so the backend travels beside it.
    assert set(envelope) == {"fields", "backend"}
    assert "_backend" not in envelope["fields"]
    assert "_backend_was_auto" not in envelope["fields"]
    assert envelope["fields"]["json"] == {"type": "object"}
    assert envelope["backend"] == "xgrammar"


def test_structured_outputs_round_trip_reattaches_backend() -> None:
    params = _WireStructuredOutputsParams(
        json={"type": "object"},
        disable_additional_properties=True,
        whitespace_pattern=" ",
    )
    params._backend = "guidance"

    decoded = lwd_decode_structured_outputs(
        lwd_encode_structured_outputs(params), _WireStructuredOutputsParams
    )

    assert isinstance(decoded, _WireStructuredOutputsParams)
    assert decoded.json == {"type": "object"}
    assert decoded.disable_additional_properties is True
    assert decoded.whitespace_pattern == " "
    assert decoded._backend == "guidance"
    # Cloud-side rebuild: only the backend matters, the auto flag is dropped.
    assert decoded._backend_was_auto is None


def test_structured_outputs_plain_payload_decodes_to_none() -> None:
    assert lwd_decode_structured_outputs("", _WireStructuredOutputsParams) is None


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        json.dumps({"json": {}}),  # flat dict, not the envelope
        json.dumps({"fields": "nope", "backend": None}),
    ],
)
def test_structured_outputs_rejects_malformed_payload(payload: str) -> None:
    with pytest.raises(ValueError, match="malformed structured_outputs"):
        lwd_decode_structured_outputs(payload, _WireStructuredOutputsParams)


def test_structured_outputs_rejects_unknown_fields() -> None:
    wire = json.dumps({"fields": {"json": {}, "bogus": 1}, "backend": None})
    with pytest.raises(ValueError, match="do not fit"):
        lwd_decode_structured_outputs(wire, _WireStructuredOutputsParams)


def test_structured_outputs_mirror_matches_real_class() -> None:
    assert _mirror_fields() == _real_fields_from_source()


def test_structured_outputs_live_round_trip() -> None:
    sampling_params = pytest.importorskip("vllm.sampling_params")
    real_cls = sampling_params.StructuredOutputsParams
    params = real_cls(regex="[0-9]+")
    params._backend = "xgrammar"

    decoded = lwd_decode_structured_outputs(
        lwd_encode_structured_outputs(params), real_cls
    )

    assert decoded.regex == "[0-9]+"
    assert decoded._backend == "xgrammar"
