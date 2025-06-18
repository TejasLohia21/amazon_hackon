Of course. Here is a comprehensive `README.md` file for your project. It explains the purpose, architecture, setup, and usage of the visual review verification pipeline you've built, incorporating the context from your architecture diagram.

---

# Multimodal Review Authenticity Auditor

This project implements a Python pipeline to audit the authenticity of product reviews by cross-referencing textual claims with visual evidence from user-submitted images and videos. It leverages a two-step Large Language Model (LLM) process to extract and verify claims, followed by a CLIP-based model to perform the visual-text matching.

This module serves as a practical implementation of the "Text Filter for removing texts which don't match image" and "Contrastive Language-Image Pretraining" components envisioned in the broader system architecture[1].



## Features

-   **LLM-Powered Claim Extraction:** Uses a Large Language Model (via GroqCloud API) to intelligently extract sentences from review text that describe visually verifiable features (e.g., "the screen was cracked").
-   **Two-Step LLM Verification:** Employs a second LLM call to filter out hallucinated or non-visual claims, ensuring only relevant statements are passed to the vision model.
-   **CLIP-Based Visual Matching:** Utilizes OpenAI's CLIP model (via Hugging Face `transformers`) to calculate the semantic similarity between textual claims and visual content.
-   **Automatic Media Handling:** Detects whether the input is an image or video and processes it accordingly. Videos are automatically sampled into frames.
-   **Robust and Modular Design:** Built with an object-oriented structure (`ReviewPipelineWrapper`) for easy integration, reusability, and maintenance.
-   **Resilient Error Handling:** Gracefully handles common issues like corrupted media files or API errors without crashing.

## Architecture Workflow

The pipeline follows a sequential, multi-stage process:

1.  **Input:** Takes a review text string and a local file path to an associated image or video.
2.  **Claim Extraction:** The review text is sent to an LLM to identify potential visual claims.
3.  **Claim Verification:** The extracted claims are sent back to the LLM for a second pass to confirm they are truly visual in nature and not hallucinations.
4.  **Media Processing:** The system identifies the media file type. If it's a video, it is sampled into individual frames.
5.  **Visual Verification:** Each verified claim is compared against the image (or each video frame) using CLIP. The pipeline checks if *any* frame supports the claim.
6.  **Output:** Returns a dictionary mapping each verified claim to a boolean value indicating whether it was visually supported (`True`) or not (`False`).

## Installation

Follow these steps to set up the project environment.

**1. Clone the Repository (or Save the Code)**
Save your main script as `main.py`.

**2. Create a Virtual Environment**
It is highly recommended to use a virtual environment to manage dependencies.
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Create `requirements.txt`**
Create a file named `requirements.txt` in your project directory with the following content:
```
requests
moviepy
imageio
transformers
torch
Pillow
```

**4. Install Dependencies**
Install all the required packages using pip.
```bash
pip install -r requirements.txt
```

## Configuration

Before running the script, you must set your **GroqCloud API key**. Open `main.py` and find the `if __name__ == "__main__":` block. Replace the placeholder with your actual key:

```python
# In main.py, inside the if __name__ == "__main__" block
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"
```

## Usage

The core logic is encapsulated in the `ReviewPipelineWrapper` class. Instantiate this class once with your API key, then use its `.run()` method to process reviews efficiently.

### Example

Here is how to use the wrapper in your main script:

```python
from main import ReviewPipelineWrapper

if __name__ == "__main__":
    # --- Configuration ---
    GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"
    REVIEW_TEXT = "This phone is great, but it arrived with a noticeable crack on the screen."
    IMAGE_PATH = "/path/to/your/image.png"

    # --- Execution ---
    # 1. Create a single, reusable pipeline object
    #    (Models are loaded only once during this step)
    auditor = ReviewPipelineWrapper(api_key=GROQ_API_KEY)

    # 2. Run the pipeline on a review
    results = auditor.run(
        review_text=REVIEW_TEXT,
        media_path=IMAGE_PATH
    )

    # --- Display Results ---
    print("\n--- Final Pipeline Results ---")
    if "error" in results:
        print(f"Pipeline failed: {results['error']}")
    else:
        print("Success! Verification complete:")
        for claim, is_supported in results.items():
            print(f"  - Claim: '{claim}' -> Supported: {is_supported}")
```

### Using as a Module

You can easily import the `ReviewPipelineWrapper` into other parts of your application (e.g., a web server or a batch processing script).

```python
# In another file, e.g., app.py
from main import ReviewPipelineWrapper

# Create a single auditor instance for your application
auditor = ReviewPipelineWrapper(api_key="YOUR_API_KEY")

# Process reviews as they come in
results = auditor.run(review_text="...", media_path="...")
print(results)
```

## License

This project is licensed under the MIT License.

[1] https://pplx-res.cloudinary.com/image/private/user_uploads/52308478/d39ef78f-5fa3-4efe-b63f-f85bab14ac7b/image.jpg
