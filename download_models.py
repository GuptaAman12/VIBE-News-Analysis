import nltk

print("Starting download of required NLTK models...")

# List of packages your main script needs
packages = ['stopwords', 'punkt', 'vader_lexicon']

for package in packages:
    try:
        # Check if the package is already downloaded
        nltk.data.find(f'corpora/{package}' if package == 'stopwords' else f'tokenizers/{package}' if package == 'punkt' else f'sentiment/{package}.zip')
        print(f"✅ Package '{package}' is already downloaded and up-to-date.")
    except LookupError:
        # If not found, download it
        print(f"⬇️  Downloading package '{package}'...")
        nltk.download(package)
        print(f"👍 Download of '{package}' complete.")

print("\nAll necessary NLTK models are ready!")
