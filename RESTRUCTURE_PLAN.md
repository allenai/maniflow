# ManiFlow Repository Fork and Restructure Plan

## Overview
This document outlines the steps to fork the ManiFlow_Policy repository, rename it to `maniflow`, and flatten the redundant directory structure.

## Current Structure (Problems)
```
ManiFlow_Policy/              # Repo with mixed case + redundant suffix
├── ManiFlow/                 # Unnecessary extra directory
│   ├── setup.py             # Setup.py not at repo root
│   ├── maniflow/            # The actual package (3 levels deep!)
│   │   ├── model/
│   │   ├── policy/
│   │   └── ...
│   └── manifold/            # Possibly abandoned package
├── scripts/
├── third_party/
└── ...
```

## Target Structure (Clean)
```
maniflow/                     # Clean repo name
├── setup.py                 # At repo root (standard)
├── maniflow/                # The actual package (only 2 levels!)
│   ├── model/
│   ├── policy/
│   └── ...
├── scripts/
├── third_party/
└── ...
```

---

## Phase 1: Fork and Setup

**CHOOSE ONE APPROACH:** Remote Update Method OR Fresh Clone Method

---

### APPROACH A: Fresh Clone Method (SIMPLER - RECOMMENDED)

Use this if you have only a few local commits and want a clean start.

#### Step 1A: Commit Your Local Changes
```bash
cd /Users/roseh/repositories/ManiFlow_Policy

# Check what you have
git status

# Commit everything
git add -A
git commit -m "WIP: my changes before fork"

# Note the commit hash (you'll need this)
git log -1
# Copy the commit hash (first 7-8 characters is enough)
```

#### Step 2A: Fork on GitHub
1. Go to https://github.com/geyan21/ManiFlow_Policy
2. Click "Fork" button
3. **Change repository name to `maniflow`**
4. Create the fork

#### Step 3A: Clone Your Fork Fresh
```bash
cd /Users/roseh/repositories

# Rename old directory as backup
mv ManiFlow_Policy ManiFlow_Policy_backup

# Clone your fork with the new name
git clone git@github.com:yourUsername/maniflow.git

# Add upstream for future updates
cd maniflow
git remote add upstream git@github.com:geyan21/ManiFlow_Policy.git
```

#### Step 4A: Cherry-Pick Your Changes
```bash
cd /Users/roseh/repositories/maniflow

# Add your old repo as a temporary remote
git remote add old-repo /Users/roseh/repositories/ManiFlow_Policy_backup

# Fetch from it
git fetch old-repo

# Cherry-pick your commit (use the hash from Step 1A)
git cherry-pick <commit-hash>

# If you had multiple commits, cherry-pick each one:
# git cherry-pick <commit-hash-1>
# git cherry-pick <commit-hash-2>

# Remove the temporary remote
git remote remove old-repo

# Push to your fork
git push origin main
```

Now skip to **Phase 2** for restructuring!

---

### APPROACH B: Remote Update Method (Keeps existing clone)

Use this if you have many local commits or prefer to keep your existing directory.

#### Step 1B: Fork on GitHub
1. Go to https://github.com/geyan21/ManiFlow_Policy
2. Click "Fork" button
3. **Change repository name to `maniflow`**
4. Create the fork

#### Step 2B: Update Local Git Remotes
```bash
cd /Users/roseh/repositories/ManiFlow_Policy

# Rename current remote to track upstream
git remote rename origin upstream

# Add your fork as the new origin (replace 'yourUsername')
git remote add origin git@github.com:yourUsername/maniflow.git

# Verify remotes
git remote -v
# Should show:
#   origin    git@github.com:yourUsername/maniflow.git
#   upstream  git@github.com:geyan21/ManiFlow_Policy.git
```

#### Step 3B: Commit Any Pending Changes
```bash
# Check status
git status

# Add and commit any changes
git add -A
git commit -m "Save work before restructure"

# Push to your fork
git push -u origin main
```

---

## Phase 2: Restructure the Repository

### Step 4: Flatten the Directory Structure

**Execute these commands to restructure:**

```bash
cd /Users/roseh/repositories/ManiFlow_Policy

# Move maniflow package to repo root
git mv ManiFlow/maniflow ./

# Move setup.py to repo root
git mv ManiFlow/setup.py ./

# Check if manifold is needed (appears mostly empty)
# If not needed, remove it
git rm -rf ManiFlow/manifold

# Remove now-empty ManiFlow directory
git rm -rf ManiFlow

# Verify the structure
ls -la
# Should see: maniflow/ setup.py scripts/ third_party/ ...
```

### Step 5: Update setup.py (if needed)

The setup.py should already be fine since it uses `find_packages()`, but verify:

```python
# File: setup.py
from setuptools import setup, find_packages

setup(
    name='maniflow',
    packages=find_packages(),
)
```

If it looks good, no changes needed!

### Step 6: Check for Import Issues

Search for any imports that might reference the old structure:

```bash
# Search for problematic imports
grep -r "from ManiFlow" . --include="*.py"
grep -r "import ManiFlow" . --include="*.py"

# These should return nothing or only third_party references
```

If you find any imports like `from ManiFlow.maniflow import ...`, change them to `from maniflow import ...`

### Step 7: Update Configuration Files

Check if any config files reference the old path structure:

```bash
# Check yaml configs
grep -r "ManiFlow" ManiFlow/maniflow/config/ --include="*.yaml"

# Check any scripts
grep -r "ManiFlow" scripts/ --include="*.py" --include="*.sh"
```

Update any paths that point to `ManiFlow/maniflow/...` to just `maniflow/...`

### Step 8: Commit the Restructure

```bash
# Stage all changes
git add -A

# Commit with clear message
git commit -m "Restructure: flatten directory hierarchy, remove redundant ManiFlow/ wrapper"

# Push to your fork
git push origin main
```

---

## Phase 3: Rename Local Directory (Optional but Recommended)

```bash
# Go up to parent directory
cd /Users/roseh/repositories

# Rename the local directory to match the repo name
mv ManiFlow_Policy maniflow

# Enter the renamed directory
cd maniflow

# Verify everything still works
git status
git remote -v
```

---

## Phase 4: Test Everything Works

### Step 9: Test Installation

```bash
cd /Users/roseh/repositories/maniflow

# Uninstall old version (if installed)
pip uninstall maniflow -y

# Install from new location
pip install -e .

# Verify import works
python -c "import maniflow; print(maniflow.__file__)"
# Should print: /Users/roseh/repositories/maniflow/maniflow/__init__.py
```

### Step 10: Test Your Code

```bash
# Try running your test files
python -m pytest test_environment.py

# Or run whatever test command you normally use
```

---

## Phase 5: Update Documentation

### Step 11: Update README References

Update any references in `README.md` that mention:
- Installation paths
- Directory structure
- Repository name

### Step 12: Update Your Notes

Update any personal documentation (like `questions.md`, `lightning_conversion_notes.md`) that reference the old structure.

---

## Troubleshooting

### If imports break:
1. Check `sys.path` to ensure the new location is included
2. Reinstall the package: `pip install -e .`
3. Search for hardcoded paths: `grep -r "ManiFlow_Policy" . --include="*.py"`

### If git push fails:
```bash
# Make sure you've set up SSH keys with GitHub
ssh -T git@github.com

# Or use HTTPS instead
git remote set-url origin https://github.com/yourUsername/maniflow.git
```

### To pull future updates from upstream:
```bash
# Fetch from original repo
git fetch upstream

# Merge into your main branch
git checkout main
git merge upstream/main

# Resolve any conflicts if the original repo structure changes
```

---

## Summary of Changes

- ✅ Repository renamed: `ManiFlow_Policy` → `maniflow`
- ✅ Directory flattened: `ManiFlow/maniflow/` → `maniflow/`
- ✅ setup.py moved to repo root (standard location)
- ✅ Removed redundant wrapper directory
- ✅ Local directory renamed to match repo name
- ✅ Git remotes configured for fork workflow

---

## Commands Quick Reference

### Quick Reference: Fresh Clone Method (RECOMMENDED)

```bash
# Save your work
cd /Users/roseh/repositories/ManiFlow_Policy
git add -A
git commit -m "WIP: my changes before fork"
COMMIT_HASH=$(git rev-parse --short HEAD)
echo "Your commit hash: $COMMIT_HASH"

# Fork on GitHub as 'maniflow', then:
cd /Users/roseh/repositories
mv ManiFlow_Policy ManiFlow_Policy_backup
git clone git@github.com:yourUsername/maniflow.git
cd maniflow

# Cherry-pick your changes
git remote add old-repo ../ManiFlow_Policy_backup
git fetch old-repo
git cherry-pick $COMMIT_HASH
git remote remove old-repo
git push origin main

# Restructure
git mv ManiFlow/maniflow ./
git mv ManiFlow/setup.py ./
git rm -rf ManiFlow
git add -A
git commit -m "Restructure: flatten directory hierarchy"
git push origin main

# Test
pip install -e .
python -c "import maniflow; print('Success!')"

# Cleanup backup after confirming everything works
rm -rf ../ManiFlow_Policy_backup
```

### Quick Reference: Remote Update Method

```bash
# Complete workflow in one go (after forking on GitHub):

cd /Users/roseh/repositories/ManiFlow_Policy

# Update remotes
git remote rename origin upstream
git remote add origin git@github.com:yourUsername/maniflow.git

# Restructure
git mv ManiFlow/maniflow ./
git mv ManiFlow/setup.py ./
git rm -rf ManiFlow
git add -A
git commit -m "Restructure: flatten directory hierarchy"
git push -u origin main

# Rename local directory
cd ..
mv ManiFlow_Policy maniflow
cd maniflow

# Reinstall package
pip uninstall maniflow -y
pip install -e .

# Test
python -c "import maniflow; print('Success!')"
```

---

## Ready to Execute?

When you're ready, feed this back to me with a message like:

> "Execute the restructure plan using Approach A (fresh clone)"

Or:

> "Execute the restructure plan using Approach B (remote update)"

Or if you want to do it step-by-step:

> "I've forked and cloned. Start with Phase 2 of RESTRUCTURE_PLAN.md (restructuring the directories)"

