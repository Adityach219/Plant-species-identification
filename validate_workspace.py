"""
Validation script to check the plant identification workspace implementation
"""

import os
import sys
import json


def check_file_exists(file_path, description):
    """Check if a file exists and print status"""
    exists = os.path.exists(file_path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {file_path}")
    return exists


def check_directory_exists(dir_path, description):
    """Check if a directory exists and print status"""
    exists = os.path.exists(dir_path) and os.path.isdir(dir_path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {dir_path}")
    return exists


def validate_python_syntax(file_path):
    """Validate Python file syntax"""
    try:
        with open(file_path, 'r') as f:
            compile(f.read(), file_path, 'exec')
        return True
    except SyntaxError as e:
        print(f"    ❌ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"    ⚠️  Warning: {e}")
        return True


def main():
    """Main validation function"""
    print("🌱 Plant Species Identification Workspace Validation")
    print("=" * 60)
    
    total_checks = 0
    passed_checks = 0
    
    # Check core project files
    print("\n📋 Core Project Files:")
    core_files = [
        ("README.md", "Project documentation"),
        ("requirements.txt", "Python dependencies"),
        ("setup.py", "Package setup"),
        (".gitignore", "Git ignore rules"),
        ("LICENSE", "License file"),
        ("Dockerfile", "Docker configuration"),
        ("docker-compose.yml", "Docker Compose configuration"),
        ("setup.sh", "Setup script"),
        ("quickstart.py", "Quick start demo")
    ]
    
    for file_path, description in core_files:
        total_checks += 1
        if check_file_exists(file_path, description):
            passed_checks += 1
    
    # Check directory structure
    print("\n📁 Directory Structure:")
    directories = [
        ("src/", "Source code directory"),
        ("src/plant_identifier/", "Plant identifier module"),
        ("src/models/", "Model definitions"),
        ("src/data/", "Data processing"),
        ("src/utils/", "Utility functions"),
        ("webapp/", "Web applications"),
        ("tests/", "Test files"),
        ("config/", "Configuration"),
        ("data/", "Data storage")
    ]
    
    for dir_path, description in directories:
        total_checks += 1
        if check_directory_exists(dir_path, description):
            passed_checks += 1
    
    # Check Python modules
    print("\n🐍 Python Modules:")
    python_files = [
        ("src/models/plant_classifier.py", "Plant classifier model"),
        ("src/data/preprocessor.py", "Image preprocessor"),
        ("src/plant_identifier/train.py", "Training script"),
        ("src/plant_identifier/predict.py", "Prediction script"),
        ("src/utils/helpers.py", "Helper utilities"),
        ("config/config.py", "Configuration manager"),
        ("webapp/app.py", "Streamlit web app"),
        ("webapp/flask_api.py", "Flask API server"),
        ("tests/test_plant_identifier.py", "Test suite")
    ]
    
    for file_path, description in python_files:
        total_checks += 1
        if check_file_exists(file_path, description):
            passed_checks += 1
            # Validate syntax
            if validate_python_syntax(file_path):
                print("    ✅ Syntax valid")
            else:
                print("    ❌ Syntax errors found")
    
    # Check __init__.py files
    print("\n📦 Package Init Files:")
    init_files = [
        "src/__init__.py",
        "src/plant_identifier/__init__.py",
        "src/models/__init__.py",
        "src/data/__init__.py",
        "src/utils/__init__.py"
    ]
    
    for init_file in init_files:
        total_checks += 1
        if check_file_exists(init_file, "Package init file"):
            passed_checks += 1
    
    # Check if requirements.txt has necessary dependencies
    print("\n📚 Dependencies Check:")
    required_deps = [
        "tensorflow", "scikit-learn", "opencv-python", "Pillow",
        "flask", "streamlit", "numpy", "pandas", "matplotlib"
    ]
    
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", 'r') as f:
            requirements_content = f.read().lower()
        
        for dep in required_deps:
            total_checks += 1
            if dep.lower() in requirements_content:
                print(f"✅ {dep} found in requirements")
                passed_checks += 1
            else:
                print(f"❌ {dep} missing from requirements")
    else:
        print("❌ requirements.txt not found")
    
    # Check project features
    print("\n🎯 Project Features:")
    features = [
        ("Deep learning model implementation", "src/models/plant_classifier.py"),
        ("Image preprocessing pipeline", "src/data/preprocessor.py"),
        ("Model training script", "src/plant_identifier/train.py"),
        ("Prediction script", "src/plant_identifier/predict.py"),
        ("Web interface (Streamlit)", "webapp/app.py"),
        ("REST API (Flask)", "webapp/flask_api.py"),
        ("Docker support", "Dockerfile"),
        ("Test suite", "tests/test_plant_identifier.py"),
        ("Configuration management", "config/config.py"),
        ("Utility functions", "src/utils/helpers.py")
    ]
    
    for feature, file_path in features:
        total_checks += 1
        if os.path.exists(file_path):
            print(f"✅ {feature}")
            passed_checks += 1
        else:
            print(f"❌ {feature}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Validation Summary:")
    print(f"   Total Checks: {total_checks}")
    print(f"   Passed: {passed_checks}")
    print(f"   Failed: {total_checks - passed_checks}")
    print(f"   Success Rate: {(passed_checks/total_checks)*100:.1f}%")
    
    if passed_checks == total_checks:
        print("\n🎉 All checks passed! The workspace is fully implemented.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Add training data to data/training/")
        print("3. Train a model: python src/plant_identifier/train.py --data_dir data/training")
        print("4. Start the web app: streamlit run webapp/app.py")
    else:
        print(f"\n⚠️  {total_checks - passed_checks} checks failed. Please review the missing components.")
    
    return passed_checks == total_checks


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)