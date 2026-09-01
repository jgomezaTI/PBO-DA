# Overleaf synchronization

This project can be synchronized with an Overleaf template through its Git
integration. Never store the token in this repository, in `.env`, in the README, or
inside a remote URL.

## Overleaf setup

1. Open the template project.
2. Open **Integrations / Git** from the project menu and copy the Git URL.
3. Generate a Git authentication token from Account Settings (or from that dialog).

Overleaf uses the username `git` and the token as the password. The token grants
access to every project permitted for your account and must remain private. See the
[official Git token documentation](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/git/git-integration-authentication-tokens).

## Connect this repository

From the project root, replace `<GIT-URL>` with the URL copied from Overleaf:

```powershell
git remote add overleaf <GIT-URL>
git remote -v
```

If the template already contains files that must be brought into the local repository:

```powershell
git pull overleaf master --allow-unrelated-histories --rebase=false
```

After reviewing and confirming the changes locally:

```powershell
git add .
git commit -m "Synchronize research template"
git push overleaf main:master
```

When Git requests credentials, use:

```text
Username: git
Password: <Overleaf token>
```

Overleaf's Git bridge uses a single linear history and a remote `master` branch;
the last command maps the local `main` branch to `master`.

## What is needed to connect it here

To add the remote in this checkout, the Overleaf project's Git URL is required. Do
not paste the token into chat: enter it interactively on the first `pull`/`push`, or
use a local credential manager. If the template is not synchronized yet, download
it as a ZIP and place the `.tex`, `.bib`, and resource files in a separate folder
before deciding how to integrate them with the Python code.
