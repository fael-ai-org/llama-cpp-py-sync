# Test-only artifact signing

The Windows wheel jobs create an ephemeral self-signed Authenticode
certificate on the GitHub runner after the LLaMA/ggml native DLLs have been
assembled and before the wheel is built. The certificate's private key stays
in the runner certificate store and is removed at the end of the signing step.
Only the public `.cer` file is uploaded as a separate test artifact.

The workflow signs LLaMA-owned `llama*.dll`, `ggml*.dll`, `mtmd*.dll`, `.pyd`,
and executable files. Microsoft, CUDA, and Vulkan redistributables are not
re-signed; their vendor identity must remain intact.

Linux wheel jobs create a short-lived GPG key and publish detached `.asc`
signatures, a per-variant SHA-256 manifest and signature, and the corresponding
public test key. This is artifact verification only; Linux does not
automatically show a trusted publisher for these signatures.

macOS wheel jobs apply an ad-hoc code signature to LLaMA-owned Mach-O files
before packaging. Ad-hoc signing detects later modification but does not
establish an Apple-trusted Developer ID identity or replace notarization.

After signing, packaging, retagging, and smoke testing, every final Linux,
macOS, and Windows wheel receives a keyless GitHub artifact attestation. The
attestation binds the wheel digest to this repository, workflow, commit, and
triggering event. Consumers can verify a downloaded wheel with:

```text
gh attestation verify <wheel> --repo <owner>/llama-cpp-py-sync
```

The workflow receives only the narrow `id-token: write` and
`attestations: write` permissions needed by GitHub's Sigstore-backed attestation
service; there is no persistent attestation private key.

These signatures are deliberately test-only. A clean Windows installation
will normally still show `Unknown publisher` or a SmartScreen warning because
the certificate has no trusted chain or reputation. No private certificate,
PFX, password, or key file belongs in this repository. The ignore rules are
only an accidental-staging guard.

The ephemeral Windows certificate and Linux GPG key deliberately use generic
`TEST ONLY` identities. Personal names, company publisher names, and contact
addresses are not embedded in the test-signing scripts. A self-signed subject
name is not an authenticated identity and must never be presented as one.
