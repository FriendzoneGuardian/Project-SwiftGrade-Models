import importlib
import warnings
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Store warnings
captured_warnings = {}

# Define categorized packages
packages = {
    "1. Data Sciences": {
        # Essentials: Core libraries for data manipulation and analysis
        "1a. Essentials": 
        [
            ('numpy', ''),        # Fundamental package for numerical computations
            ('scipy', ''),        # Library for scientific and technical computing
            ('pandas', ''),       # Data structures and data analysis tools
        ],
        # Auxiliaries: Additional tools for specific tasks
        "1b. Auxiliaries": 
        [
            ('statsmodels', ''),  # Statistical modeling and hypothesis testing
            ('sklearn', ''),      # Machine learning library
            ('scrapy', ''),       # Web crawling and web scraping framework
            ('beautifulsoup4', ''), # Library for parsing HTML and XML documents
            ('requests', ''),     # Simple HTTP library for making requests
        ],
        # Visualization tools for data science
        "1c. Data Visualization Tools": 
        [
            ('matplotlib', ''),   # Plotting library for creating static, animated, and interactive visualizations
            ('seaborn', ''),      # Statistical data visualization based on matplotlib
            ('bokeh', ''),        # Interactive visualizations for modern web browsers
            ('plotly', ''),       # Interactive graphing library for Python
            ('altair', ''),       # Declarative statistical visualization library
            ('missingno', ''),    # Missing data visualization module
            ('kaleido', ''),      # Static image export for web-based visualization libraries
            ('wordcloud', ''),    # Library for generating word clouds
        ],
    },
    "2. Machine Learning": {
        # Libraries and GPU Checker (for GPU support)
        "GPU Check:": 
        [
            ('GPUtil', ''),        # GPU utility library for Python
            ('cuda', ''),         # NVIDIA CUDA library for GPU computing
            ('tensorflow-gpu', ''), # TensorFlow with GPU support
            ('torch.cuda', ''), # PyTorch with CUDA support
        ],
        

        # Libraries for machine learning
        "2a. Libraries": 
        [
            ('openpyxl', ''), ('xlrd', '')
        ],
        # Tools for Computer Vision
        "2b. Computer Vision": 
        [
            ('cv2', 'OpenCV'), ('PIL', 'Pillow'), 
            ('imageio', ''), ('scikit-image', '')
        ],
        # Tools for Natural Language
        "2c. Natural Language": 
        [
            ('spacy', ''), ('nltk', ''), 
            ('transformers', ''), ('gensim', '')
        ],
        # The Big Guns of Machine Learning
        "2d. The Big Guns": 
        [
            ('tensorflow', ''), ('torch', 'pyTorch'), 
            ('fastai', ''), ('xgboost', ''), 
            ('lightgbm', ''), ('catboost', ''),
            ('keras', '')
        ]
    },
    "3. Weeb Packages": {
        # Weeb packages for waifus and anime
        "3a. Weeb Tools": 
        [
            ('waifu2x', ''), ('deepdanbooru', ''), 
            ('booru', ''), ('animegan', ''), ('rembg', '')
        ]
    }
}

# Header
print(Fore.CYAN + Style.BRIGHT + "=" * 60)
print(Fore.MAGENTA + Style.BRIGHT + "PYTHON PACKAGE CHECKSUM - SYSTEM PACKAGE AUDIT".center(60))
print(Fore.CYAN + Style.BRIGHT + "=" * 60)

# Loop through categories
for category, items in packages.items():
    print(Fore.YELLOW + Style.BRIGHT + f"\n{category}")
    print(Fore.CYAN + "-" * 60)

    # Subcategory format
    if isinstance(items, dict):
        for subcat, subitems in items.items():
            print(Fore.LIGHTYELLOW_EX + f"  {subcat}")
            for pkg, alias in subitems:
                try:
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        module = importlib.import_module(pkg)
                        version = getattr(module, '__version__', 'Unknown')

                        # Capture any warnings issued
                        if w:
                            captured_warnings[pkg] = [str(warn.message) for warn in w]

                        print(Fore.GREEN + f"    {pkg:<20} | Installed | Version: {version}")
                except ImportError:
                    print(Fore.RED + f"    {pkg:<20} | Not Installed |")
    else:
        # Regular category
        for pkg, alias in items:
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    module = importlib.import_module(pkg)
                    version = getattr(module, '__version__', 'Unknown')
                    
                    if w:
                        captured_warnings[pkg] = [str(warn.message) for warn in w]

                    print(Fore.GREEN + f"  {pkg:<20} | Installed | Version: {version}")
            except ImportError:
                print(Fore.RED + f"  {pkg:<20} | Not Installed|")




# Warnings Summary
if captured_warnings:
    print(Fore.YELLOW + Style.BRIGHT + "\nWarnings Summary:")
    print(Fore.LIGHTYELLOW_EX + "-" * 60)
    for pkg, warns in captured_warnings.items():
        print(Fore.YELLOW + f"  {pkg}:")
        for msg in warns:
            print(Fore.LIGHTWHITE_EX + f"    ⚠ {msg}")
else:
    print(Fore.GREEN + "\nNo warnings captured during import.")

# Footer
print(Fore.CYAN + Style.BRIGHT + "\n" + "=" * 60)
print(Fore.MAGENTA + Style.BRIGHT + "CHECKSUM COMPLETED!".center(60))
print(Fore.CYAN + Style.BRIGHT + "=" * 60)