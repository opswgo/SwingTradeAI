#!/usr/bin/env bash
# Environment variables for Divinator project

if [[ -z "$APP_HOME" || -z "$DB_HOME" ]]; then
  cd ..
  source ./build-setvars.zsh
  cd API
fi

# Update the following DATABASE_URL with your Postgres credentials and database name.
if [[ $(basename "$PWD") == "API" ]]; then
  # Ensure virtual environment is activated
  if [[ -z "$VIRTUAL_ENV" ]]; then
    source ./build-activate_venv.sh
  fi
fi

export DATABASE_URL="postgres://postgres:mysecretpassword@localhost:5432/divinator_db"
echo "Database startup script finished."
echo $DATABASE_URL
