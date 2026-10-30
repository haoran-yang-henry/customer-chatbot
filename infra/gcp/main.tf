# The hosted backend: one Cloud Run service (proxy, API, Keycloak) and one Cloud SQL instance,
# reached over its private IP. See README.md in this directory.

data "google_project" "this" {
  project_id = var.project_id
}

locals {
  service  = "customer-agent"
  url      = "https://${local.service}-${data.google_project.this.number}.${var.region}.run.app"
  issuer   = "${local.url}/realms/nordelektro"
  registry = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

resource "google_project_service" "apis" {
  for_each = toset([
    "artifactregistry.googleapis.com",
    "billingbudgets.googleapis.com",
    "cloudscheduler.googleapis.com",
    "compute.googleapis.com",
    "iam.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "servicenetworking.googleapis.com",
    "sqladmin.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false
}

resource "google_billing_budget" "monthly" {
  billing_account = var.billing_account
  display_name    = "customer-agent"
  budget_filter {
    projects = ["projects/${data.google_project.this.number}"]
  }
  amount {
    specified_amount {
      units = tostring(var.monthly_budget)
    }
  }
  dynamic "threshold_rules" {
    for_each = [0.5, 0.9, 1.0]
    content {
      threshold_percent = threshold_rules.value
    }
  }
  depends_on = [google_project_service.apis]
}

resource "google_artifact_registry_repository" "images" {
  repository_id = "images"
  location      = var.region
  format        = "DOCKER"
  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions {
      keep_count = 5
    }
  }
  cleanup_policies {
    id     = "delete-old"
    action = "DELETE"
    condition {
      older_than = "2592000s" # 30 days, unless among the five most recent
    }
  }
  depends_on = [google_project_service.apis]
}

# --- network: Cloud Run reaches Cloud SQL's private IP through direct VPC egress -------------

resource "google_compute_network" "vpc" {
  name                    = "agent"
  auto_create_subnetworks = false
  depends_on              = [google_project_service.apis]
}

resource "google_compute_subnetwork" "run" {
  name                     = "run"
  network                  = google_compute_network.vpc.id
  region                   = var.region
  ip_cidr_range            = "10.8.0.0/26"
  private_ip_google_access = true
}

resource "google_compute_global_address" "sql" {
  name          = "sql-peering"
  network       = google_compute_network.vpc.id
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 20
}

resource "google_service_networking_connection" "sql" {
  network                 = google_compute_network.vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.sql.name]
  # Removing the peering right after the instance fails; it goes with the network.
  deletion_policy = "ABANDON"
}

# --- database ---------------------------------------------------------------------------------

resource "google_sql_database_instance" "db" {
  name                = "agent-db"
  database_version    = "POSTGRES_16"
  region              = var.region
  deletion_protection = false # `make gcp-destroy` removes everything

  settings {
    edition           = "ENTERPRISE"
    tier              = "db-f1-micro" # shared core, 0.6 GB: enough for a demo
    availability_type = "ZONAL"
    disk_size         = 10
    disk_autoresize   = true

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.vpc.id
      ssl_mode        = "ENCRYPTED_ONLY"
    }
    backup_configuration {
      enabled = true
    }
    database_flags {
      name  = "max_connections" # API pool, checkpoint pool, Keycloak and a job at once
      value = "50"
    }
  }

  lifecycle {
    # `make gcp-pause` stops the instance between demos.
    ignore_changes = [settings[0].activation_policy]
  }
  depends_on = [google_service_networking_connection.sql]
}

locals {
  db_host = google_sql_database_instance.db.private_ip_address
}

resource "google_sql_database" "app" {
  name     = "customer_agent"
  instance = google_sql_database_instance.db.name
}

resource "google_sql_database" "keycloak" {
  name     = "keycloak"
  instance = google_sql_database_instance.db.name
}

resource "random_password" "app_db" {
  length  = 32
  special = false
}

resource "random_password" "keycloak_db" {
  length  = 32
  special = false
}

resource "google_sql_user" "app" {
  name     = "agent"
  instance = google_sql_database_instance.db.name
  password = random_password.app_db.result
}

resource "google_sql_user" "keycloak" {
  name     = "keycloak"
  instance = google_sql_database_instance.db.name
  password = random_password.keycloak_db.result
}

# --- secrets ----------------------------------------------------------------------------------

resource "random_password" "keycloak_admin" {
  length  = 32
  special = false
}

resource "random_password" "demo" {
  length  = 20
  special = false
}

locals {
  # Generated here; they live in Secret Manager and in the local state (keep it private).
  generated_secrets = {
    "database-url"            = "postgresql+psycopg://agent:${random_password.app_db.result}@${local.db_host}:5432/customer_agent?sslmode=require"
    "keycloak-db-password"    = random_password.keycloak_db.result
    "keycloak-admin-password" = random_password.keycloak_admin.result
    "demo-password"           = random_password.demo.result
  }
  provided_secrets = concat(["llm-api-key", "openai-api-key"], var.langfuse_enabled ? ["langfuse-secret-key"] : [])
}

resource "google_secret_manager_secret" "secrets" {
  for_each  = toset(concat(keys(local.generated_secrets), local.provided_secrets))
  secret_id = each.value
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "generated" {
  for_each    = local.generated_secrets
  secret      = google_secret_manager_secret.secrets[each.key].id
  secret_data = each.value
}

resource "google_secret_manager_secret_version" "llm_api_key" {
  secret                 = google_secret_manager_secret.secrets["llm-api-key"].id
  secret_data_wo         = var.llm_api_key
  secret_data_wo_version = var.secrets_version
}

resource "google_secret_manager_secret_version" "openai_api_key" {
  secret                 = google_secret_manager_secret.secrets["openai-api-key"].id
  secret_data_wo         = var.openai_api_key
  secret_data_wo_version = var.secrets_version
}

resource "google_secret_manager_secret_version" "langfuse_secret_key" {
  count                  = var.langfuse_enabled ? 1 : 0
  secret                 = google_secret_manager_secret.secrets["langfuse-secret-key"].id
  secret_data_wo         = var.langfuse_secret_key
  secret_data_wo_version = var.secrets_version
}

# --- identities -------------------------------------------------------------------------------

resource "google_service_account" "runtime" {
  account_id   = "agent-runtime"
  display_name = "Customer agent: service and jobs"
  depends_on   = [google_project_service.apis]
}

resource "google_secret_manager_secret_iam_member" "runtime" {
  for_each  = google_secret_manager_secret.secrets
  secret_id = each.value.id
  role      = "roles/secretmanager.secretAccessor"
  member    = google_service_account.runtime.member
}

resource "google_service_account" "scheduler" {
  account_id   = "agent-scheduler"
  display_name = "Customer agent: starts the daily retention job"
  depends_on   = [google_project_service.apis]
}
