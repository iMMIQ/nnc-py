# Scheduled Residual Tile Streaming Design

## Goal

Extend scheduled-native O3 tile streaming so `conv + add + relu` style residual blocks can execute correctly under a 1MB `fast_mem` budget without relying on detached home-tensor execution for the streamed group.

## Problem

The current scheduled-native runtime only streams safe single-input conv groups:

- `CONV2D`
- `FUSED_CONV_RELU`
- optional trailing `RELU`

Residual groups still fall back to home execution. This preserves correctness, but it leaves large detached activation buffers in place and prevents scheduled-native runtime from making real progress on small-memory `resnet18`.

## Scope

This round adds scheduled-native streaming for:

- `conv -> add`
- `conv -> add -> relu`
- `conv -> fused_add_relu`
- residual blocks whose skip input is produced by an earlier node and consumed tile-by-tile

This round may also cover blocks with `downsample` on the residual path, but correctness takes priority over fully streaming the downsample producer itself.

## Non-Goals

- Remove all detached home tensors in one pass
- Convert `maxpool`, `avgpool`, or `fc` to streaming in the same change
- Rewrite the legacy `tile_regions_v3` runtime plan

## Design

### Scheduled Group Coverage

Scheduled-native helper generation should accept the same group shapes already proven useful by `tile_regions_v3` planning:

- root `CONV2D` or `FUSED_CONV_RELU`
- terminal `ADD`, `RELU`, or `FUSED_ADD_RELU`

The generated helper must own the entire group execution and replace all compute work inside the group with a single tile-streaming entrypoint.

### Buffer Model

The helper must use explicit live tile buffers rather than assuming one reusable shared offset is always safe. The minimum safe set is:

- conv input tile
- conv output tile
- residual/add input tile
- terminal/output tile

Buffers may alias only when the helper proves there is no overlapping live range.

### Execution Semantics

For each output tile:

1. Stage the conv input tile from the root input tensor
2. Run conv or fused-conv-relu into a dedicated conv output tile
3. Stage the residual/add peer tile from its source tensor
4. Run add or fused-add-relu on tile-shaped tensor views
5. Commit the terminal tile into the group output tensor

If a group contains only conv output as the terminal result, the helper may commit conv output directly.

### Tensor Storage

Only tensors fully covered by scheduled tile streaming should lose detached home backing in `tensors.c`. Partially covered groups must keep safe home storage.

## Verification

This change must add failing tests first for:

- scheduled codegen of `conv + add + relu` under 1MB
- runtime correctness of a streamed residual group vs ONNX Runtime
- a small residual/downsample pattern if support is enabled in this round

It must also re-verify 1MB `resnet18` compile and runtime correctness.
