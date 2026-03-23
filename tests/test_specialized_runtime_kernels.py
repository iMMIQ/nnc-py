from pathlib import Path


def _extract_function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    brace_start = source.index("{", start)
    depth = 0
    for idx in range(brace_start, len(source)):
        char = source[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[brace_start + 1:idx]
    raise AssertionError(f"Failed to extract body for {signature}")


def test_conv3x3_specialized_entrypoint_is_not_generic_wrapper() -> None:
    source = Path("/home/ayd/code/nnc-py/runtime/x86/ops.c").read_text()
    body = _extract_function_body(source, "void nnc_conv3x3_s1(")

    assert "nnc_conv(" not in body


def test_conv1x1_specialized_entrypoint_is_not_generic_wrapper() -> None:
    source = Path("/home/ayd/code/nnc-py/runtime/x86/ops.c").read_text()
    body = _extract_function_body(source, "void nnc_conv1x1(")

    assert "nnc_conv(" not in body


def test_conv_relu3x3_specialized_entrypoint_is_not_generic_wrapper() -> None:
    source = Path("/home/ayd/code/nnc-py/runtime/x86/ops.c").read_text()
    body = _extract_function_body(source, "void nnc_conv_relu3x3_s1(")

    assert "nnc_conv_relu(" not in body


def test_conv_relu1x1_specialized_entrypoint_is_not_generic_wrapper() -> None:
    source = Path("/home/ayd/code/nnc-py/runtime/x86/ops.c").read_text()
    body = _extract_function_body(source, "void nnc_conv_relu1x1(")

    assert "nnc_conv_relu(" not in body


def test_conv7x7_specialized_entrypoint_is_not_generic_wrapper() -> None:
    source = Path("/home/ayd/code/nnc-py/runtime/x86/ops.c").read_text()
    body = _extract_function_body(source, "void nnc_conv7x7_s2(")

    assert "nnc_conv(" not in body


def test_conv_relu7x7_specialized_entrypoint_is_not_generic_wrapper() -> None:
    source = Path("/home/ayd/code/nnc-py/runtime/x86/ops.c").read_text()
    body = _extract_function_body(source, "void nnc_conv_relu7x7_s2(")

    assert "nnc_conv_relu(" not in body


def test_gemm_nt_specialized_entrypoint_is_not_generic_wrapper() -> None:
    source = Path("/home/ayd/code/nnc-py/runtime/x86/ops.c").read_text()
    body = _extract_function_body(source, "void nnc_gemm_nt(")

    assert "nnc_gemm(" not in body
