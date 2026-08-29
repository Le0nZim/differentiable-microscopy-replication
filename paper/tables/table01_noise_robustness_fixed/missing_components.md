# Missing / data-blocked components — Table 1

- **Paper's inverted photon-count ordering (pc=10 better than pc=10000) is NOT reproduced.** Our v3 gives the physically-expected SNR ordering (more photons -> lower MSE). The supplement's own normalization equations (variance alpha_norm/k + gamma/k^2, read noise sigma_read/k) imply higher k should *reduce* noise, so our ordering is physically more natural. This blocks only the exact magnitude/ordering match, not the qualitative robustness claims.
- Per-cell magnitudes differ ~2-3x from the paper (same order of magnitude).

