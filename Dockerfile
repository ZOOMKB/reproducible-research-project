FROM python:3.12-slim

ARG QUARTO_VERSION=1.9.37
ARG TARGETARCH

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RPY2_CFFI_MODE=ABI \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl make tar \
    && curl -fsSL -o quarto.tar.gz \
        "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-${TARGETARCH}.tar.gz" \
    && mkdir -p /opt/quarto \
    && tar -xzf quarto.tar.gz -C /opt/quarto --strip-components=1 \
    && ln -s /opt/quarto/bin/quarto /usr/local/bin/quarto \
    && rm -f quarto.tar.gz \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base r-base-dev \
    cmake \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libtirpc-dev \
    && rm -rf /var/lib/apt/lists/*

RUN R -e " \
    options(repos=c(CRAN='https://cloud.r-project.org')); \
    install.packages(c('fs', 'nloptr'), dependencies=TRUE); \
    install.packages('rugarch', dependencies=TRUE); \
    if (!requireNamespace('rugarch', quietly=TRUE)) stop('rugarch install failed') \
"

COPY --from=ghcr.io/astral-sh/uv:0.9.3 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --group r-bridge

COPY . .

CMD ["make", "reproduce"]
