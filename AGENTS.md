# Publication repository

This checkout holds projects and source code to share publicly. It is not the
operational workspace for the author's bots. People downloading the projects
are responsible for making them work on their own machines.

## Protect local projects and running bots

- Sibling projects outside this repository, including the local `donchain`
  project, are read-only sources unless the user explicitly asks to change them.
- Copy the requested source files into this repository and edit only the copies.
- Do not modify a source project's code, configuration, credentials, data,
  databases, dependencies or Git state as part of preparing a publication.
- Do not build images, create validation containers, start or stop services,
  restart bots, deploy changes, submit orders or change live settings unless
  the user explicitly requests that operational action. Preparing, testing or
  publishing code is not authorization to operate a bot.
- Prefer static checks of the publication copy. Do not spend time making every
  project run locally or upgrading its environment merely to share its source.
  Never run live-account test scripts as publication checks.

## Share source, not private runtime state

- Keep one project per folder under `Strategies`, mirroring its top-level source
  project's name and layout. Keep strategy code, parameter exports and configs
  from the same source together; do not mix spot and futures siblings.
- Public configurations must retain source settings except for explicit private
  value redactions. Generic examples in `Configs` are separate from project configs.
- Keep the root README as a neutral project index; put strategy-specific details
  in that project's own README.
- Include the useful project code, parameter files, sanitized configuration
  examples, dependency/build files and concise instructions.
- Prefer browsable source directories to ZIP bundles. Preserve distinct code
  variants when unpacking; do not assume two similarly named files are equal.
- Exclude real credentials, private keys, personal wallet/account identifiers,
  Telegram IDs, local override files, trade databases, logs and caches. Keep
  generated/downloaded runtime data local. Public research inputs and figures
  may remain when useful to understand an analysis.
- Existing public JWT example values may remain; the user explicitly permits
  them. Do not copy private authentication values from a local running project.
- Review the exact file set before committing. Never blanket-add the many
  unrelated local/untracked directories in this checkout.
- History rewriting or a force push requires explicit user authorization for
  that task. When authorized, preserve a local recovery copy, verify the result,
  and use an explicit force-with-lease against the observed remote revision.
