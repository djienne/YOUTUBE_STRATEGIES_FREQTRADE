## Updating Freqtrade to the Newest Version

If you want to update and/or reinstall Freqtrade properly to the latest version or any version you want:

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
