# syntax=docker/dockerfile:1
# Ingress container of the hosted backend (infra/gcp): Caddy with the route filter baked in.
#   docker build infra/gcp/proxy
FROM caddy:2.11.7-alpine@sha256:d8542f48d34a9cf4e4c11a478865229840e87e4c96ea3f439101f31a5d35f75f
COPY Caddyfile /etc/caddy/Caddyfile
# Not root: port 8080 needs no privilege. Caddy writes its instance id and autosaved config to
# /data and /config.
RUN addgroup -S caddy \
    && adduser -S -G caddy -H -s /sbin/nologin caddy \
    && chown -R caddy:caddy /data /config
USER caddy
EXPOSE 8080
