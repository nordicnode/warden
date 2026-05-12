#!/usr/bin/env bash
# install_language_servers.sh
# Installs all Language Servers required by CodeForge MCP Server.
# Run this from the root of the project.

set -e

echo "============================================================"
echo "    CodeForge MCP - Language Server Installation Script     "
echo "============================================================"
echo ""

# 1. Python: Pyright
echo "➔ Installing Python Language Server (pyright)..."
if command -v uv &> /dev/null; then
    uv pip install pyright
elif [ -n "$VIRTUAL_ENV" ]; then
    pip install pyright
else
    echo "  [!] Not in a virtual environment and 'uv' not found. Skipping pyright."
    echo "      Run 'uv pip install pyright' inside your venv."
fi

# 2. TypeScript / JavaScript
echo "➔ Installing TypeScript Language Server..."
if command -v npm &> /dev/null; then
    npm install -g typescript typescript-language-server
else
    echo "  [!] 'npm' not found. Please install Node.js and run:"
    echo "      npm install -g typescript typescript-language-server"
fi

# Helper to detect package manager for system-level dependencies
install_sys_pkg() {
    local pkg=$1
    if command -v pacman &> /dev/null; then
        sudo pacman -S --noconfirm --needed "$pkg"
    elif command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y "$pkg"
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y "$pkg"
    elif command -v brew &> /dev/null; then
        brew install "$pkg"
    else
        echo "  [!] Could not detect package manager to install: $pkg"
    fi
}

# 3. Rust: rust-analyzer
echo "➔ Installing Rust Language Server (rust-analyzer)..."
if command -v rustup &> /dev/null; then
    rustup component add rust-analyzer
elif command -v rust-analyzer &> /dev/null; then
    echo "  rust-analyzer is already installed globally."
else
    echo "  [!] rust-analyzer not found. Attempting to install via system packages..."
    install_sys_pkg rust-analyzer
fi

# 4. C / C++: clangd
echo "➔ Installing C/C++ Language Server (clangd)..."
if command -v clangd &> /dev/null; then
    echo "  clangd is already installed."
else
    echo "  [!] clangd not found. Attempting to install via system packages..."
    install_sys_pkg clang
fi

# 5. Go: gopls
echo "➔ Installing Go Language Server (gopls)..."
if ! command -v go &> /dev/null; then
    echo "  [!] 'go' not found. Attempting to install via system packages..."
    install_sys_pkg go
fi

if command -v go &> /dev/null; then
    # Install gopls (will place it in ~/go/bin by default)
    go install golang.org/x/tools/gopls@latest
    
    # Check if ~/go/bin is in PATH, if not warn the user
    if [[ ":$PATH:" != *":$HOME/go/bin:"* ]]; then
        echo "  [WARNING] 'gopls' was installed to $HOME/go/bin, but it is not in your PATH."
        echo "            Please add 'export PATH=\$PATH:\$HOME/go/bin' to your ~/.bashrc or ~/.zshrc."
    fi
else
    echo "  [!] Failed to install Go. Cannot install gopls."
fi

echo ""
echo "============================================================"
echo "                       Installation Complete!               "
echo "============================================================"
echo "To verify the installations, run:"
echo "  pyright-langserver --version"
echo "  typescript-language-server --version"
echo "  rust-analyzer --version"
echo "  clangd --version"
echo "  gopls version"
