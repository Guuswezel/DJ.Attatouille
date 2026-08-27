FROM rust:1.88-slim AS builder
WORKDIR /app
COPY Cargo.toml Cargo.toml
RUN mkdir src && printf 'fn main() {}' > src/main.rs && cargo build --release
COPY src src
RUN touch src/main.rs && cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates wget && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/dj-attatouille-api /usr/local/bin/dj-attatouille-api
EXPOSE 8080
CMD ["dj-attatouille-api"]
