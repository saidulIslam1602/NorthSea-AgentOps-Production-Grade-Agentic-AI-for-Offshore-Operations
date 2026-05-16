terraform {
  required_version = ">= 1.6.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.110.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6.0"
    }
  }

  backend "azurerm" {
    resource_group_name  = "rg-northsea-infra"
    storage_account_name = "stnorthseaterraform"
    container_name       = "tfstate"
    key                  = "northsea-agentops.tfstate"
  }
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
}

# ─── Variables ────────────────────────────────────────────────────────────────

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "norwayeast"
}

variable "openai_api_key" {
  description = "OpenAI API key"
  type        = string
  sensitive   = true
}

variable "db_admin_password" {
  description = "PostgreSQL admin password"
  type        = string
  sensitive   = true
}

# ─── Resource Group ───────────────────────────────────────────────────────────

resource "azurerm_resource_group" "main" {
  name     = "rg-northsea-agentops-${var.environment}"
  location = var.location

  tags = {
    project     = "northsea-agentops"
    environment = var.environment
    managed_by  = "terraform"
  }
}

# ─── Log Analytics ────────────────────────────────────────────────────────────

resource "azurerm_log_analytics_workspace" "main" {
  name                = "log-northsea-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

# ─── Container Apps Environment ───────────────────────────────────────────────

resource "azurerm_container_app_environment" "main" {
  name                       = "cae-northsea-${var.environment}"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
}

# ─── PostgreSQL Flexible Server ───────────────────────────────────────────────

resource "azurerm_postgresql_flexible_server" "main" {
  name                   = "psql-northsea-${var.environment}"
  resource_group_name    = azurerm_resource_group.main.name
  location               = azurerm_resource_group.main.location
  version                = "16"
  administrator_login    = "northsea_admin"
  administrator_password = var.db_admin_password
  storage_mb             = 32768
  sku_name               = var.environment == "prod" ? "GP_Standard_D4s_v3" : "B_Standard_B2ms"
  zone                   = "1"

  authentication {
    active_directory_auth_enabled = false
    password_auth_enabled         = true
  }
}

resource "azurerm_postgresql_flexible_server_database" "main" {
  name      = "northsea_agentops"
  server_id = azurerm_postgresql_flexible_server.main.id
  collation = "en_US.utf8"
  charset   = "utf8"
}

resource "azurerm_postgresql_flexible_server_configuration" "pgvector" {
  name      = "azure.extensions"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "VECTOR,UUID-OSSP,PG_TRGM"
}

resource "azurerm_postgresql_flexible_server_firewall_rule" "allow_azure" {
  name             = "AllowAzureServices"
  server_id        = azurerm_postgresql_flexible_server.main.id
  start_ip_address = "0.0.0.0"
  end_ip_address   = "0.0.0.0"
}

# ─── Event Hubs ───────────────────────────────────────────────────────────────

resource "azurerm_eventhub_namespace" "main" {
  name                = "evhns-northsea-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "Standard"
  capacity            = 1
}

resource "azurerm_eventhub" "telemetry" {
  name              = "well-telemetry"
  namespace_id      = azurerm_eventhub_namespace.main.id
  partition_count   = 4
  message_retention = 7
}

resource "azurerm_eventhub" "anomalies" {
  name              = "well-anomalies"
  namespace_id      = azurerm_eventhub_namespace.main.id
  partition_count   = 2
  message_retention = 7
}

resource "azurerm_eventhub" "escalations" {
  name              = "agent-escalations"
  namespace_id      = azurerm_eventhub_namespace.main.id
  partition_count   = 2
  message_retention = 30
}

# ─── Key Vault ────────────────────────────────────────────────────────────────

data "azurerm_client_config" "current" {}

resource "azurerm_key_vault" "main" {
  name                = "kv-northsea-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"

  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id

    secret_permissions = ["Get", "List", "Set", "Delete"]
  }
}

resource "azurerm_key_vault_secret" "openai_api_key" {
  name         = "openai-api-key"
  value        = var.openai_api_key
  key_vault_id = azurerm_key_vault.main.id
}

# ─── Container App — API ──────────────────────────────────────────────────────

resource "azurerm_container_app" "api" {
  name                         = "ca-northsea-api-${var.environment}"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"

  identity {
    type = "SystemAssigned"
  }

  template {
    min_replicas = var.environment == "prod" ? 2 : 1
    max_replicas = var.environment == "prod" ? 10 : 3

    container {
      name   = "northsea-api"
      image  = "ghcr.io/your-org/northsea-agentops:latest"
      cpu    = var.environment == "prod" ? 2.0 : 0.5
      memory = var.environment == "prod" ? "4Gi" : "1Gi"

      env {
        name  = "APP_ENV"
        value = var.environment
      }
      env {
        name  = "DATABASE_URL"
        value = "postgresql+psycopg://northsea_admin:${var.db_admin_password}@${azurerm_postgresql_flexible_server.main.fqdn}:5432/northsea_agentops"
      }
      env {
        name        = "OPENAI_API_KEY"
        secret_name = "openai-api-key"
      }
      env {
        name  = "KAFKA_BOOTSTRAP_SERVERS"
        value = "${azurerm_eventhub_namespace.main.name}.servicebus.windows.net:9093"
      }

      liveness_probe {
        transport               = "HTTP"
        path                    = "/health"
        port                    = 8000
        initial_delay           = 15
        period_seconds          = 30
        failure_count_threshold = 3
      }

      readiness_probe {
        transport = "HTTP"
        path      = "/health"
        port      = 8000
      }
    }
  }

  secret {
    name  = "openai-api-key"
    value = var.openai_api_key
  }

  ingress {
    external_enabled = true
    target_port      = 8000

    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }
}

# ─── Outputs ──────────────────────────────────────────────────────────────────

output "api_url" {
  description = "Container App API URL"
  value       = "https://${azurerm_container_app.api.ingress[0].fqdn}"
}

output "postgres_host" {
  description = "PostgreSQL server FQDN"
  value       = azurerm_postgresql_flexible_server.main.fqdn
}

output "event_hub_namespace" {
  description = "Event Hub namespace"
  value       = azurerm_eventhub_namespace.main.name
}

output "key_vault_uri" {
  description = "Key Vault URI"
  value       = azurerm_key_vault.main.vault_uri
}
