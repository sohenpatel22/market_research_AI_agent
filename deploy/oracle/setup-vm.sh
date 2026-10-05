#!/usr/bin/env bash
# One-time VM setup (Ubuntu on an Always Free Ampere A1 instance): Docker and the host firewall rule for port 80.
set -euo pipefail

sudo apt-get update
sudo apt-get install -y docker.io curl
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

# Oracle's Ubuntu images ship with iptables rules that reject everything except SSH.
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo apt-get install -y iptables-persistent
sudo netfilter-persistent save

echo "Done. Log out and back in so the docker group applies."
