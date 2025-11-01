Here’s your text rewritten and formatted in clean, professional Markdown:

---

## Updating Freqtrade to the Newest Version

To ensure Freqtrade updates and reinstalls properly to the latest version:

1. **Update the version number**

   * Change the Freqtrade version (e.g., from `2025.10` to the newer version when available)
     in both:

     * `docker-compose.yml`
     * `Dockerfile.technical`

2. **Clean up old Docker data**

   ```bash
   docker system prune -a
   ```

   > Confirm when prompted to remove all unused images, containers, and volumes.

3. **Rebuild the Docker containers**

   ```bash
   docker compose build
   ```

---

Would you like me to include a short code snippet example showing how to change the version line in those files?
