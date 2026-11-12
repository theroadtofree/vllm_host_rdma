# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
    lwd_encode_message,
)


def test_request_notify_round_trips_full_prompt_mrope_positions() -> None:
    assert LWD_MESSAGE_VERSION == 3
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
