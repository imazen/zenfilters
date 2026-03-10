# zenfilters development commands

# Run all tests (default features)
test:
    cargo test

# Run all tests including experimental
test-all:
    cargo test --features experimental

# Run clippy on everything
clippy:
    cargo clippy --features experimental --examples

# Format and check
fmt:
    cargo fmt

# Local CI sanity check
ci: fmt clippy test-all

# Run parity evaluation (20 samples, rule-based only)
parity samples="20":
    ZEN_SAMPLES={{samples}} cargo run --release --features experimental --example darktable_parity

# Run training (64 clusters, cached phases)
train:
    cargo run --release --features experimental --example train_autotune

# Run comparison image generation
compare samples="32":
    ZEN_SAMPLES={{samples}} cargo run --release --features experimental --example compare_autotune

# Quick DNG parity sweep over sigmoid params
sigmoid-sweep:
    @for c in 1.2 1.4 1.6; do \
        for s in 0.50 0.58 0.65; do \
            result=$$(ZEN_BASE_CONTRAST=$$c ZEN_BASE_SKEW=$$s ZEN_SAMPLES=8 \
                cargo run --release --features experimental --example darktable_parity 2>/dev/null | \
                grep "^MEAN" | awk '{print $$2}'); \
            echo "c=$$c s=$$s => base=$$result"; \
        done; \
    done
