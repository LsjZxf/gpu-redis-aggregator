FROM golang:1.23 AS builder
WORKDIR /src
COPY go.mod .
COPY go.sum .
COPY main.go .
RUN go mod tidy
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o /out/gpu-redis-aggregator main.go

FROM debian:12-slim
WORKDIR /app
COPY --from=builder /out/gpu-redis-aggregator /app/gpu-redis-aggregator
EXPOSE 8080
ENTRYPOINT ["/app/gpu-redis-aggregator"]