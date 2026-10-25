# syntax=docker/dockerfile:1
# Ingress container of the hosted backend (infra/gcp): Caddy with the route filter baked in.
#   docker build infra/gcp/proxy
FROM caddy:2.11.7-alpine@sha256:d8542f48d34a9cf4e4c11a478865229840e87e4c96ea3f439101f31a5d35f75f
COPY Caddyfile /etc/caddy/Caddyfile
EXPOSE 8080
