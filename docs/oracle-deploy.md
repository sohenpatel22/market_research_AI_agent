# Deploying to Oracle Cloud Always Free

The app runs as one container on an Always Free Ampere A1 (Arm) VM, with the Neon database as before.
The `deploy-oracle.yml` workflow builds the arm64 image, pushes it to GitHub Container Registry and restarts it on the VM.

Oracle asks for a credit or debit card at signup. Stay on resources marked Always Free and do not upgrade the
account to Pay As You Go. Check the Always Free limits in the console before creating anything.

## 1. Create the VM (Oracle console)

1. Sign up at https://www.oracle.com/cloud/free/ and choose a home region (it cannot be changed later).
2. Compute > Instances > Create instance.
   - Image: Canonical Ubuntu 22.04 or 24.04 (aarch64).
   - Shape: Ampere `VM.Standard.A1.Flex`, 2 OCPU and 12 GB memory is plenty (the Always Free total is 4 OCPU / 24 GB).
   - Boot volume: the default (50 GB) is fine.
   - Add your SSH public key, or let Oracle generate a key pair and download the private key.
3. If the shape is "out of capacity", retry later or try another availability domain.
4. Networking: open the instance's subnet security list (or network security group) and add an ingress rule for
   TCP port 80 from `0.0.0.0/0`.

## 2. Prepare the VM (once)

```bash
ssh -i <key> ubuntu@<public-ip> 'bash -s' < deploy/oracle/setup-vm.sh
```

This installs Docker and opens port 80 in the VM's own firewall.

## 3. GitHub configuration

Create an environment named `oracle` (Settings > Environments) and add these repository secrets:

| Secret | Value |
|---|---|
| `ORACLE_HOST` | the VM's public IP |
| `ORACLE_USER` | `ubuntu` (optional, this is the default) |
| `ORACLE_SSH_KEY` | the private key, including the BEGIN/END lines |

The database and LLM secrets (`DATABASE_URL`, `DEEPSEEK_API_KEY`, `LANGFUSE_*`) are reused from the existing setup.

## 4. Deploy

Run the **Deploy to Oracle Cloud** workflow from the Actions tab. The first run takes a while because the arm64
image is built under emulation; later runs reuse the layer cache. When it finishes, the app is at
`http://<public-ip>/`.

## Notes

- Traffic is plain HTTP on the IP. For HTTPS, put a domain in front with a free Cloudflare tunnel or a Caddy container.
- Oracle can reclaim Always Free instances that stay idle for an extended period, so check the console occasionally.
- The image is public-pull only through the workflow's temporary token; the package can stay private.
