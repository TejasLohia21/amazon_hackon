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

## Example

Before running the script, you must set your **GroqCloud API key**. Open `run.py`. Replace the placeholder with your actual key:

```python
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"
```

## Usage

The core logic is encapsulated in the `ReviewPipelineWrapper` class. Instantiate this class once with your API key, then use its `.run()` method to process reviews efficiently.

### Example


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

