#!/usr/bin/env python3
"""
Setup script for BudgetWise AI with Gemini Integration
© 2025 Mohammed Arfath
"""

import subprocess
import sys
import os
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def install_requirements():
    """Install required packages"""
    print_header("📦 Installing Required Packages")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages")
        return False

def setup_env_file():
    """Setup environment file"""
    print_header("🔧 Setting Up Environment")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("⚠️  .env file already exists")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            print("Skipping .env setup")
            return True
    
    if env_example.exists():
        # Copy example file
        with open(env_example, 'r') as f:
            content = f.read()
        
        print("\n📋 To use AI Chat features, you need a Gemini API key")
        print("🔗 Get your free API key from: https://makersuite.google.com/app/apikey\n")
        
        api_key = input("Enter your Gemini API key (or press Enter to skip): ").strip()
        
        if api_key:
            content = content.replace("your_gemini_api_key_here", api_key)
            print("✅ API key configured")
        else:
            print("ℹ️  You can add your API key later in the .env file")
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ .env file created successfully!")
        return True
    else:
        print("⚠️  .env.example not found, creating basic .env file")
        with open(env_file, 'w') as f:
            f.write("# BudgetWise AI Configuration\n")
            f.write("# © 2025 Mohammed Arfath\n\n")
            f.write("GEMINI_API_KEY=your_gemini_api_key_here\n")
        print("✅ Basic .env file created")
        return True

def verify_installation():
    """Verify key packages are installed"""
    print_header("🔍 Verifying Installation")
    
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'google.generativeai'
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not installed")
            all_installed = False
    
    return all_installed

def main():
    """Main setup function"""
    print_header("🚀 BudgetWise AI Setup")
    print("© 2025 Mohammed Arfath")
    print("Setting up your AI-powered expense forecasting tool\n")
    
    # Install requirements
    if not install_requirements():
        print("\n⚠️  Package installation failed. Please run manually:")
        print("   pip install -r requirements.txt")
        return
    
    # Setup environment
    setup_env_file()
    
    # Verify installation
    if verify_installation():
        print_header("✨ Setup Complete!")
        print("🎉 BudgetWise AI is ready to use!")
        print("\n📖 Next steps:")
        print("   1. If you skipped API key setup, add it to .env file")
        print("   2. Run: streamlit run app/budgetwise_app.py")
        print("   3. Navigate to '🤖 AI Chat' in the app sidebar")
        print("\n📚 For detailed AI Chat setup, see: docs/AI_CHAT_GUIDE.md")
    else:
        print_header("⚠️  Setup Incomplete")
        print("Some packages failed to install. Please check the errors above.")

if __name__ == "__main__":
    main()
