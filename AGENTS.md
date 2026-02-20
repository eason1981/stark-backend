# AGENTS.md - Development Guide for OpenVM STARK Backend

This file provides essential information for agentic coding agents working on this Rust workspace implementing a modular STARK proof system.

## Build System & Commands

### Core Commands
```bash
# Build all default workspace crates
cargo build --release

# Build all workspace members (including CUDA crates)
cargo build --all

# Build with specific features
cargo build --features "parallel,mimalloc,prometheus"

# Build CUDA crates (requires CUDA toolkit)
cargo build --all --features "touchemall"

# Clean build artifacts
cargo clean
```

### Testing Commands
```bash
# Run all tests using nextest (preferred test runner)
cargo nextest run

# Run tests for specific crate
cargo nextest run -p stark-backend-v2

# Run a SINGLE TEST by name (partial match)
cargo nextest run test_name

# Run a SINGLE TEST with exact path
cargo nextest run path::to::module::test_name

# Run tests in a specific file
cargo test --lib path::to::module

# Run tests with output visible (print println! statements)
cargo nextest run --no-capture

# Run tests in release mode (faster)
cargo nextest run --release

# Run CUDA tests (requires GPU)
cargo test --all --features "touchemall"
```

### Linting & Formatting
```bash
# Format Rust code (requires nightly)
cargo +nightly fmt --all

# Check formatting without making changes
cargo +nightly fmt --all --check

# Run clippy with warnings as errors
cargo clippy --all-targets --tests --features "default mimalloc parallel prometheus" -- -D warnings

# Spell check
codespell --skip Cargo.lock --ignore-words-file .codespellignore --path "crates/"

# Security audit
cargo audit
```

## Code Style Guidelines

### Formatting (rustfmt.toml)
- Edition: 2021, Line limit: 100 | Import: Crate level, StdExternalCrate
- Comments wrapped at 100, field init shorthand

### Import Organization (alphabetical within groups)
```rust
use std::collections::HashMap;
use std::sync::Arc;

use p3_field::Field;
use p3_matrix::Matrix;

use crate::chip::Chip;
use crate::prover::ProverBackend;
```

### Naming Conventions
- Types: `PascalCase` | Functions: `snake_case` | Constants: `SCREAMING_SNAKE_CASE`
- Modules/Fields/Variables: `snake_case` | Boolean getters: `is_`, `has_`, `does_`

### Error Handling
- Use `Result<T, Error>` for fallible operations; prefer `thiserror` for custom errors
- Use `?` for propagation; avoid `unwrap()` in production (use `expect()` with messages)
- Include context using `context()` or `with_context()`

### Generic Type Parameters
- Use descriptive single letters: `T` for types, `F` for fields, `PB` for backend
- Use constraint syntax: `T: Clone + Send + Sync`; prefer `where` clauses for complex bounds

### Code Patterns
```rust
// Prefer early returns
fn example(input: &str) -> Result<Output, Error> {
    if input.is_empty() { return Err(Error::EmptyInput); }
}

// Config structs
#[derive(Debug, Clone, Default)]
pub struct Config { pub field_size: usize, pub security_level: u32 }

// Error types with thiserror
#[derive(Debug, Error)]
pub enum ProverError {
    #[error("Invalid trace length: {0}")] InvalidTraceLength(usize),
    #[error("Chip not found: {0}")] ChipNotFound(String),
    #[error("GPU operation failed: {source}")] GpuError { #[from] source: CudaError },
}
```

## Project Structure
```
stark-backend/
├── crates/
│   ├── stark-backend/       # Core STARK proving system
│   ├── stark-sdk/          # Low-level SDK and utilities
│   ├── stark-backend-v2/   # Next generation backend
│   │   └── derive/         # Derive macros for v2
│   ├── cuda-common/        # Shared CUDA utilities
│   ├── cuda-backend/       # CUDA implementation
│   └── cuda-builder/       # CUDA build utilities
├── benchmarks/fields/      # Performance benchmarks
└── scripts/                # Utility scripts
```

### Default Crates
`stark-backend`, `stark-sdk`, `stark-backend-v2`, `stark-backend-v2/derive`

### Feature Flags
- `parallel` (default): Multi-threading via rayon | `mimalloc`: Alt memory allocator
- `metrics`: Performance metrics | `prometheus`: Prometheus export
- `touchemall`: CUDA features (requires CUDA toolkit) | `test-utils`: Testing utilities

## Testing Guidelines
- Unit tests: `#[test]` in source files | Integration tests: `tests/` directories
- Use `nextest` as test runner | Test naming: `test_<functionality>_<scenario>`
- Use `test-case` crate: `#[case::name(params)]` | Use `test-log`: `#[tokio::test]`

## Prerequisites & CI
- Rust 1.83 (pinned in rust-toolchain.toml) | CUDA 12.8+ (optional)
- CI required: `cargo +nightly fmt --all --check` && `cargo clippy ... -- -D warnings` && `cargo nextest run`

This codebase is performance-critical. Always run the full linting and test suite before submitting changes.
