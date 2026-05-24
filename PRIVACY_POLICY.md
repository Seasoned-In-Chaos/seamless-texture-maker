# SEAMS - Privacy Policy

**Last updated: May 2026**

## Overview

SEAMS ("Seamless Texture Maker") is a desktop application created by Shubham Panchasara. We respect your privacy and are committed to protecting your personal data.

## Data We Collect

**We do not collect, store, or transmit any personal data.**

SEAMS operates entirely on your local machine. The application:

- Does NOT require an account or login
- Does NOT collect personal information (name, email, location)
- Does NOT track usage analytics or telemetry
- Does NOT use cookies or tracking technologies
- Does NOT communicate with external servers, except for:
  - Checking for application updates (a single read-only request to the GitHub Releases API, which sends only your current version number in the User-Agent header)

## Local Data Storage

SEAMS stores the following data locally on your device:

| Data | Location | Purpose |
|------|----------|---------|
| Settings/preferences | `%APPDATA%\SeamlessTextureMaker\settings.json` | Remembers your window size, tool settings, and preferences |
| Log files | `%APPDATA%\SeamlessTextureMaker\logs\` | Application diagnostic logs (rotated, max 2MB each, 5 files max) |
| Image cache | `%APPDATA%\SeamlessTextureMaker\cache\` | Processing cache for performance |
| Numba JIT cache | `%APPDATA%\SeamlessTextureMaker\numba_cache\` | Pre-compiled JIT kernels for faster startup |

All local data can be deleted at any time by removing the `%APPDATA%\SeamlessTextureMaker` folder. The uninstaller offers to remove this data.

## Third-Party Services

SEAMS does not integrate with any third-party services, advertising networks, or analytics platforms.

## Updates

The application checks for updates by making a single request to:

`https://api.github.com/repos/Seasoned-In-Chaos/seamless-texture-maker/releases/latest`

This request includes only a User-Agent header with the app name and version. No personal data is sent. You can disable update checks by declining the update notification when prompted.

## Children's Privacy

SEAMS does not knowingly collect data from children. Since we collect no personal data at all, our application is suitable for users of all ages.

## Changes to This Policy

We may update this privacy policy from time to time. Changes will be reflected in the "Last updated" date above.

## Contact

If you have questions about this privacy policy, please contact:

- GitHub: [Seasoned-In-Chaos/seamless-texture-maker](https://github.com/Seasoned-In-Chaos/seamless-texture-maker)
