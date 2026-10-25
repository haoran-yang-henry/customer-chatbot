# One Cloud Run service with three containers sharing localhost, as the tunnel demo runs them:
# the proxy (the only ingress, the route filter), the API and Keycloak. One origin serves API and
# login, as the frontend expects. At most one instance: Keycloak keeps its caches in memory.

locals {
  # Model calls only: the API and the jobs reach the internet directly, Cloud SQL via the VPC.
  app_env = merge(var.model_settings, {
    CA_LOG_FORMAT            = "json"
    CA_AUTH_ISSUER           = local.issuer
    CA_AUTH_JWKS_URL         = "http://localhost:8081/realms/nordelektro/protocol/openid-connect/certs"
    CA_AUTH_REQUIRED         = "true" # every message costs model calls: logged-in demo users only
    CA_CORS_ORIGINS          = var.pages_origin
    CA_RATE_LIMIT_PER_MINUTE = "10"
    CA_DB_POOL_SIZE          = "5"
    CA_DB_MAX_OVERFLOW       = "5"
  })
  app_secrets = merge(
    {
      CA_DATABASE_URL   = "database-url"
      CA_LLM_API_KEY    = "llm-api-key"
      CA_OPENAI_API_KEY = "openai-api-key"
    },
    var.langfuse_enabled ? { LANGFUSE_SECRET_KEY = "langfuse-secret-key" } : {},
  )
  keycloak_env = {
    KC_HOSTNAME                 = local.url
    KC_HTTP_ENABLED             = "true"
    KC_HTTP_PORT                = "8081"
    KC_PROXY_HEADERS            = "xforwarded"
    KC_PROXY_TRUSTED_ADDRESSES  = "127.0.0.1/32,::1/128"
    KC_DB_URL                   = "jdbc:postgresql://${local.db_host}:5432/keycloak?sslmode=require"
    KC_DB_USERNAME              = google_sql_user.keycloak.name
    KC_DB_POOL_MAX_SIZE         = "5"
    KC_BOOTSTRAP_ADMIN_USERNAME = "admin" # only for the first start; the console is not routed
    JAVA_OPTS_KC_HEAP           = "-Xms256m -Xmx640m"
  }
  keycloak_secrets = {
    KC_DB_PASSWORD              = "keycloak-db-password"
    KC_BOOTSTRAP_ADMIN_PASSWORD = "keycloak-admin-password"
    DEMO_PASSWORD               = "demo-password" # the demo users' password, on realm import
  }
}

resource "google_cloud_run_v2_service" "agent" {
  name                 = local.service
  location             = var.region
  ingress              = "INGRESS_TRAFFIC_ALL"
  invoker_iam_disabled = true # public: the proxy and the login protect what is behind it
  deletion_protection  = false

  template {
    service_account                  = google_service_account.runtime.email
    execution_environment            = "EXECUTION_ENVIRONMENT_GEN2"
    timeout                          = "300s"
    max_instance_request_concurrency = 20

    scaling {
      min_instance_count = 0 # nothing to pay while nobody chats; a cold start takes ~15 s
      max_instance_count = 1
    }

    vpc_access {
      network_interfaces {
        network    = google_compute_network.vpc.id
        subnetwork = google_compute_subnetwork.run.id
      }
      egress = "PRIVATE_RANGES_ONLY"
    }

    containers {
      name       = "proxy"
      image      = "${local.registry}/proxy:${var.image_tag}"
      depends_on = ["app", "keycloak"]
      ports {
        container_port = 8080
      }
      resources {
        limits = { cpu = "1", memory = "256Mi" }
      }
    }

    containers {
      name  = "app"
      image = "${local.registry}/app:${var.image_tag}"
      resources {
        limits            = { cpu = "1", memory = "1Gi" }
        startup_cpu_boost = true
      }
      startup_probe {
        http_get {
          path = "/health/live"
          port = 8000
        }
        period_seconds    = 2
        failure_threshold = 30
      }
      dynamic "env" {
        for_each = local.app_env
        content {
          name  = env.key
          value = env.value
        }
      }
      dynamic "env" {
        for_each = local.app_secrets
        content {
          name = env.key
          value_source {
            secret_key_ref {
              secret  = env.value
              version = "latest"
            }
          }
        }
      }
    }

    containers {
      name  = "keycloak"
      image = "${local.registry}/keycloak:${var.image_tag}"
      resources {
        limits            = { cpu = "1", memory = "1Gi" }
        startup_cpu_boost = true
      }
      startup_probe {
        http_get {
          path = "/health/started"
          port = 9000
        }
        initial_delay_seconds = 5
        period_seconds        = 2
        failure_threshold     = 60
      }
      dynamic "env" {
        for_each = local.keycloak_env
        content {
          name  = env.key
          value = env.value
        }
      }
      dynamic "env" {
        for_each = local.keycloak_secrets
        content {
          name = env.key
          value_source {
            secret_key_ref {
              secret  = env.value
              version = "latest"
            }
          }
        }
      }
    }
  }

  depends_on = [
    google_secret_manager_secret_iam_member.runtime,
    google_sql_database.app,
    google_sql_database.keycloak,
    # Every secret a container reads needs a version before a revision can start.
    google_secret_manager_secret_version.generated,
    google_secret_manager_secret_version.llm_api_key,
    google_secret_manager_secret_version.openai_api_key,
    google_secret_manager_secret_version.langfuse_secret_key,
  ]
}

# --- jobs: what `make migrate seed ingest` and `make retention` do locally ------------------

resource "google_cloud_run_v2_job" "jobs" {
  for_each = {
    # After every deploy (deploy.py): schema, demo orders, knowledge index; all idempotent.
    setup = ["sh", "-c", "alembic upgrade head && customer-agent-seed && customer-agent-ingest"]
    # Daily (scheduler below): conversations past the retention period.
    retention = ["customer-agent-retention"]
  }
  name                = each.key
  location            = var.region
  deletion_protection = false

  template {
    task_count = 1
    template {
      service_account = google_service_account.runtime.email
      timeout         = "900s"
      max_retries     = 0

      vpc_access {
        network_interfaces {
          network    = google_compute_network.vpc.id
          subnetwork = google_compute_subnetwork.run.id
        }
        egress = "PRIVATE_RANGES_ONLY"
      }

      containers {
        image   = "${local.registry}/app:${var.image_tag}"
        command = each.value
        resources {
          limits = { cpu = "1", memory = "1Gi" }
        }
        dynamic "env" {
          for_each = merge(var.model_settings, { CA_LOG_FORMAT = "json" })
          content {
            name  = env.key
            value = env.value
          }
        }
        dynamic "env" {
          for_each = { CA_DATABASE_URL = "database-url", CA_OPENAI_API_KEY = "openai-api-key" }
          content {
            name = env.key
            value_source {
              secret_key_ref {
                secret  = env.value
                version = "latest"
              }
            }
          }
        }
      }
    }
  }

  depends_on = [
    google_secret_manager_secret_iam_member.runtime,
    google_sql_database.app,
    # Every secret a container reads needs a version before a revision can start.
    google_secret_manager_secret_version.generated,
    google_secret_manager_secret_version.llm_api_key,
    google_secret_manager_secret_version.openai_api_key,
    google_secret_manager_secret_version.langfuse_secret_key,
  ]
}

resource "google_cloud_run_v2_job_iam_member" "scheduler" {
  name     = google_cloud_run_v2_job.jobs["retention"].name
  location = var.region
  role     = "roles/run.invoker"
  member   = google_service_account.scheduler.member
}

resource "google_cloud_scheduler_job" "retention" {
  name      = "retention-daily"
  region    = var.region
  schedule  = "0 3 * * *"
  time_zone = "Europe/Berlin"
  http_target {
    http_method = "POST"
    uri         = "https://run.googleapis.com/v2/projects/${var.project_id}/locations/${var.region}/jobs/${google_cloud_run_v2_job.jobs["retention"].name}:run"
    oauth_token {
      service_account_email = google_service_account.scheduler.email
    }
  }
  depends_on = [google_cloud_run_v2_job_iam_member.scheduler]
}
