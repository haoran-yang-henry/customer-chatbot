# syntax=docker/dockerfile:1
# Keycloak for the hosted backend (infra/gcp): production mode, built ahead of time so a cold
# start only starts the server. Build context: infra/
#   docker build -f infra/gcp/keycloak/Dockerfile infra
ARG KEYCLOAK_IMAGE=quay.io/keycloak/keycloak:26.3.5@sha256:357829ec7c4693397533035092ad13b0644bcc95ded311f33a3738c4d9e9bdba
ARG PYTHON_IMAGE=python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

FROM ${PYTHON_IMAGE} AS realm
COPY keycloak/nordelektro-realm.json gcp/keycloak/realm.py /src/
RUN python /src/realm.py /src/nordelektro-realm.json > /realm.json

FROM ${KEYCLOAK_IMAGE} AS build
# Build options: one instance (Cloud Run keeps at most one), so no cluster cache.
ENV KC_DB=postgres \
    KC_HEALTH_ENABLED=true \
    KC_CACHE=local
RUN /opt/keycloak/bin/kc.sh build

FROM ${KEYCLOAK_IMAGE}
COPY --from=build /opt/keycloak/ /opt/keycloak/
COPY --from=realm --chown=keycloak:keycloak /realm.json /opt/keycloak/data/import/nordelektro-realm.json
# The base image's keycloak user, stated: never root.
USER 1000
ENTRYPOINT ["/opt/keycloak/bin/kc.sh"]
# The realm is imported once, when the database has none; `make gcp-realm` re-imports it.
CMD ["start", "--optimized", "--import-realm"]
