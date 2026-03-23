#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  install_linux_service.sh [options]

Options:
  --install-root PATH       Application install root. Default: /opt/mlp-license-server
  --config-dir PATH         Config directory. Default: /etc/mlp-license-server
  --data-dir PATH           Data directory. Default: /var/lib/mlp-license-server
  --log-dir PATH            Log directory. Default: /var/log/mlp-license-server
  --service-user NAME       Service account user. Default: mlp-license-server
  --service-group NAME      Service account group. Default: mlp-license-server
  --service-name NAME       systemd unit base name. Default: mlp-license-server
  --service-command CMD     Full command used to start the HTTP service
  --enable                  Enable the service after install
  --start                   Start or restart the service after install
  --help                    Show this help text
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_template="${script_dir}/../shared/config.linux.toml.sample"
env_template="${script_dir}/mlp-license-server.env.sample"
unit_template="${script_dir}/mlp-license-server.service.template"

install_root="/opt/mlp-license-server"
config_dir="/etc/mlp-license-server"
data_dir="/var/lib/mlp-license-server"
log_dir="/var/log/mlp-license-server"
service_user="mlp-license-server"
service_group="mlp-license-server"
service_name="mlp-license-server"
service_command="/opt/mlp-license-server/.venv/bin/python -m uvicorn license_server.main:app --host 0.0.0.0 --port 8090"
enable_service=0
start_service=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-root)
      install_root="$2"
      shift 2
      ;;
    --config-dir)
      config_dir="$2"
      shift 2
      ;;
    --data-dir)
      data_dir="$2"
      shift 2
      ;;
    --log-dir)
      log_dir="$2"
      shift 2
      ;;
    --service-user)
      service_user="$2"
      shift 2
      ;;
    --service-group)
      service_group="$2"
      shift 2
      ;;
    --service-name)
      service_name="$2"
      shift 2
      ;;
    --service-command)
      service_command="$2"
      shift 2
      ;;
    --enable)
      enable_service=1
      shift
      ;;
    --start)
      start_service=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ $EUID -ne 0 ]]; then
  echo "Run this script as root." >&2
  exit 1
fi

if [[ ! -f "$config_template" ]]; then
  echo "Missing config template: $config_template" >&2
  exit 1
fi
if [[ ! -f "$env_template" ]]; then
  echo "Missing environment template: $env_template" >&2
  exit 1
fi
if [[ ! -f "$unit_template" ]]; then
  echo "Missing unit template: $unit_template" >&2
  exit 1
fi

if ! getent group "$service_group" >/dev/null 2>&1; then
  groupadd --system "$service_group"
fi
if ! id -u "$service_user" >/dev/null 2>&1; then
  useradd --system --home-dir "$data_dir" --shell /usr/sbin/nologin --gid "$service_group" "$service_user"
fi

install -d -m 0755 "$install_root" "$config_dir"
install -d -m 0750 -o "$service_user" -g "$service_group" "$data_dir" "$log_dir"

config_path="${config_dir}/config.toml"
env_path="${config_dir}/service.env"
unit_path="/etc/systemd/system/${service_name}.service"

escape_sed_replacement() {
  printf '%s' "$1" | sed -e 's/[|&]/\\&/g'
}

render_file() {
  local template="$1"
  local destination="$2"

  sed \
    -e "s|__SERVICE_USER__|$(escape_sed_replacement "${service_user}")|g" \
    -e "s|__SERVICE_GROUP__|$(escape_sed_replacement "${service_group}")|g" \
    -e "s|__WORKING_DIRECTORY__|$(escape_sed_replacement "${install_root}")|g" \
    -e "s|__ENV_FILE__|$(escape_sed_replacement "${env_path}")|g" \
    -e "s|__CONFIG_PATH__|$(escape_sed_replacement "${config_path}")|g" \
    -e "s|__DATA_DIR__|$(escape_sed_replacement "${data_dir}")|g" \
    -e "s|__LOG_DIR__|$(escape_sed_replacement "${log_dir}")|g" \
    -e "s|__SERVICE_COMMAND__|$(escape_sed_replacement "${service_command}")|g" \
    "$template" > "$destination"
}

if [[ ! -f "$config_path" ]]; then
  render_file "$config_template" "$config_path"
  chmod 0640 "$config_path"
fi

render_file "$env_template" "$env_path"
chmod 0640 "$env_path"

render_file "$unit_template" "$unit_path"
chmod 0644 "$unit_path"

systemctl daemon-reload

if [[ $enable_service -eq 1 ]]; then
  systemctl enable "$service_name"
fi

if [[ $start_service -eq 1 ]]; then
  systemctl restart "$service_name"
fi

cat <<EOF
Prepared Linux service assets:
  systemd unit: $unit_path
  app config:   $config_path
  env file:     $env_path

Current example service command assumption:
  $service_command

If the runtime entrypoint changes later, update $env_path and restart the service.
EOF
