# Continuous deployment: the deploy job in .github/workflows/ci.yml signs in through Workload
# Identity Federation (GitHub's OIDC token for this repository's main branch, no stored key) as
# the deployer. The deployer reads everything here and changes only images, revisions and job
# runs: IAM, secrets and the rest change through `make gcp-deploy` with the owner's login.

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github"
  display_name              = "GitHub Actions"
  depends_on                = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "customer-chatbot"
  display_name                       = "customer-chatbot, main"
  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }
  # Pull requests and other branches get no token.
  attribute_condition = "assertion.repository_id == '${var.github_repository_id}' && assertion.ref == 'refs/heads/main'"
  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account" "deployer" {
  account_id   = "deployer"
  display_name = "GitHub Actions: deploys the main branch"
  depends_on   = [google_project_service.apis]
}

resource "google_service_account_iam_member" "deployer_github" {
  service_account_id = google_service_account.deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repository}"
}

resource "google_project_iam_member" "deployer" {
  for_each = toset([
    "roles/viewer",                            # plan: read every resource here
    "roles/iam.securityReviewer",              # plan: read their IAM policies
    "roles/iam.workloadIdentityPoolViewer",    # plan: read the pool above
    "roles/serviceusage.serviceUsageConsumer", # API calls billed to this project
    "roles/run.developer",                     # new revisions, job runs
    "roles/compute.networkUser",               # revisions with direct VPC egress
  ])
  project = var.project_id
  role    = each.value
  member  = google_service_account.deployer.member
}

# Plan reads the budget, which belongs to the billing account.
resource "google_billing_account_iam_member" "deployer" {
  billing_account_id = var.billing_account
  role               = "roles/billing.viewer"
  member             = google_service_account.deployer.member
}

# The service and the jobs run as the runtime account.
resource "google_service_account_iam_member" "deployer_runtime" {
  service_account_id = google_service_account.runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = google_service_account.deployer.member
}

resource "google_artifact_registry_repository_iam_member" "deployer" {
  repository = google_artifact_registry_repository.images.name
  location   = var.region
  role       = "roles/artifactregistry.writer"
  member     = google_service_account.deployer.member
}

# The state, its lock and the saved deploy settings.
resource "google_storage_bucket_iam_member" "deployer_state" {
  bucket = var.state_bucket
  role   = "roles/storage.objectAdmin"
  member = google_service_account.deployer.member
}

# Plan reads the generated secrets' values; the state holds them anyway. Not the model keys.
resource "google_secret_manager_secret_iam_member" "deployer" {
  for_each  = toset(keys(local.generated_secrets))
  secret_id = google_secret_manager_secret.secrets[each.key].id
  role      = "roles/secretmanager.secretAccessor"
  member    = google_service_account.deployer.member
}
