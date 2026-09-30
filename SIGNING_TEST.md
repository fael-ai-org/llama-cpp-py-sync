<!-- FAEL-DOCUMENT-METADATA
{
  "schema_version": 1,
  "repository": "llama-cpp-py-sync",
  "path": "SIGNING_TEST.md",
  "owner": "llama-cpp-py-sync-build",
  "lifecycle": "active",
  "canonical": true,
  "canonical_path": null,
  "review_status": "current",
  "last_reviewed": "2026-10-01",
  "review_due": "2026-12-30",
  "notes": "Repository-local platform test signing, keyless wheel attestations and wheel-only release uploads; Linux GPG test signing removed.",
  "fingerprint": "fael-doc-v1:sha256:5fbe52678603cfeef9ef8a4c618ceaf3671457fb7fabd851f1a5a6240155ae5a"
}
FAEL-DOCUMENT-METADATA -->

# Test-only artifact signing

The Windows wheel jobs create an ephemeral self-signed Authenticode
certificate on the GitHub runner after the LLaMA/ggml native DLLs have been
assembled and before the wheel is built. The certificate's private key stays
in the runner certificate store and is removed at the end of the signing step.
Only the public `.cer` file is uploaded as a separate test artifact.

The workflow signs LLaMA-owned `llama*.dll`, `ggml*.dll`, `mtmd*.dll`, `.pyd`,
and executable files. Microsoft, CUDA, and Vulkan redistributables are not
re-signed; their vendor identity must remain intact.

Linux wheel jobs do not generate GPG test keys, detached signatures, or checksum
manifests. GitHub releases publish only wheels. Linux wheels retain the keyless
GitHub artifact attestations described below.

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

The ephemeral Windows certificate deliberately uses a generic
`TEST ONLY` identity. Personal names, company publisher names, and contact
addresses are not embedded in the test-signing scripts. A self-signed subject
name is not an authenticated identity and must never be presented as one.
