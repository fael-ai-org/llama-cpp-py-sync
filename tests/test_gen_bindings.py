"""Header preprocessing must preserve the generated public function ABI."""

import pytest
from cffi import FFI

from scripts.gen_bindings import generate_cdef
from scripts.validate_cffi_surface import (
    _extract_cdef_function_signatures,
    _extract_header_function_signatures,
)


@pytest.mark.parametrize("directive", [
    "#define LLAMA_DRY_MAX_SEQ_BREAKERS 64",
    "  #define LLAMA_DRY_MAX_SEQ_BREAKERS (32 + 32)",
    "#define LLAMA_DRY_MAX_SEQ_BREAKERS \\\n    (32 + 32)",
    "#define LLAMA_DRY_MAX_SEQ_BREAKERS (32 + \\\r\n    32)",
])
def test_numeric_macro_does_not_change_following_sampler_signature(tmp_path, directive):
    # b12039 introduced this numeric define immediately before the DRY API.
    header = """
    LLAMA_API void llama_before(void);
    """ + directive + """
    LLAMA_API struct llama_sampler * llama_sampler_init_dry(
        const struct llama_vocab * vocab,
        float dry_multiplier, float dry_base, int32_t dry_allowed_length,
        int32_t dry_penalty_last_n, // last n tokens (0 = disable penalty)
        const char ** seq_breakers, size_t num_breakers);
    LLAMA_API void llama_after(void);
    #define TRAILING_VALUE 7"""
    path = tmp_path / "llama.h"
    path.write_bytes(header.encode("utf-8"))

    cdef = generate_cdef({"llama.h": path})

    assert "#" not in cdef
    signatures = _extract_cdef_function_signatures(cdef)
    assert signatures == _extract_header_function_signatures(header)
    assert set(signatures) == {"llama_before", "llama_sampler_init_dry", "llama_after"}
    ffi = FFI()
    ffi.cdef(cdef)
