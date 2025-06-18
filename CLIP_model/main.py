import os
import requests
from typing import List, Dict
# from moviepy.editor import VideoFileClip
import imageio
from filetype import FileTypeChecker
from videosampler import VideoSampler
from transformers import CLIPProcessor, CLIPModel as HuggingFaceCLIPModel
from PIL import Image
import torch



# --- LLM Integration Classes (Corrected) ---

class LLMClient:
    """A generic client for interacting with an LLM chat completions API."""
    def __init__(self, api_key: str, model: str = "llama3-8b-8192", endpoint: str = "https://api.groq.com/openai/v1/chat/completions"):
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _make_request(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens
        }
        response = requests.post(self.endpoint, json=payload, headers=self.headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()

    def extract_visual_claims(self, review_text: str) -> List[str]:
        system_prompt = "You are an expert assistant. Extract all sentences from the user's review that describe a physical, visually verifiable feature, defect, or state (e.g., 'the screen was cracked', 'the color was wrong', 'the box was torn'). Respond with each claim on a new line. If no such claims exist, respond with 'NONE'."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": review_text}
        ]
        response_text = self._make_request(messages, max_tokens=256)
        if "NONE" in response_text:
            return []
        return [line.strip() for line in response_text.split('\n') if line.strip()]

    def verify_claim(self, claim: str) -> bool:
        system_prompt = "Analyze the user's claim. Determine if it describes something that can be visually verified in an image or video. Respond with only 'YES' or 'NO'."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": claim}
        ]
        response_text = self._make_request(messages, max_tokens=10)
        return "YES" in response_text.upper()


# --- Main Pipeline Components ---

class VisualClaimExtractor:
    """Uses an LLMClient to extract claims from reviews."""
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def extract_from_review(self, review_text: str) -> List[str]:
        return self.llm_client.extract_visual_claims(review_text)

class VisualClaimVerifier:
    """Uses an LLMClient to verify and filter extracted claims."""
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def filter_claims(self, claims: List[str]) -> List[str]:
        return [claim for claim in claims if self.llm_client.verify_claim(claim)]
    
# --- CLIP Model (Robust Version) ---
from PIL import Image, UnidentifiedImageError
from transformers import CLIPProcessor, CLIPModel as HuggingFaceCLIPModel
import torch

class CLIPModel:
    """A robust implementation of the CLIP model that handles image errors."""
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = HuggingFaceCLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print(f"CLIP model loaded on device: {self.device}")

    def check_text_in_image(self, text: str, image_path: str, threshold: float = 0.2) -> bool:
        """
        Checks if the text is supported by the image, handling potential file errors.
        """
        try:
            # Attempt to open the image file
            image = Image.open(image_path)
        except (FileNotFoundError, UnidentifiedImageError) as e:
            # If the file is not found or is corrupted/invalid, log the error and return False
            print(f"[ERROR] Could not process image '{image_path}'. Reason: {e}")
            return False

        # Process the image and text for CLIP
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Get the similarity score from the model
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probability = logits_per_image.softmax(dim=1)
            score = probability[0][0].item()

        print(f"[CLIP] Similarity score for '{text}': {score:.4f}")
        return score > threshold


# --- Main Orchestrator ---

class ReviewVisualVerifier:
    """The main pipeline orchestrator."""
    def __init__(self, claim_extractor: VisualClaimExtractor, claim_verifier: VisualClaimVerifier, clip_model: CLIPModel):
        self.claim_extractor = claim_extractor
        self.claim_verifier = claim_verifier
        self.clip_model = clip_model

    def process_review(self, review_text: str, media_path: str, output_dir="frames", fps=0.5) -> Dict[str, bool]:
        print("--- Starting Visual Verification Pipeline ---")
        # 1. Extract potential visual claims from review text.
        initial_claims = self.claim_extractor.extract_from_review(review_text)
        print(f"1. Extracted Claims: {initial_claims}")

        # 2. Filter out hallucinated/non-visual claims.
        verified_claims = self.claim_verifier.filter_claims(initial_claims)
        print(f"2. Verified Visual Claims: {verified_claims}")
        if not verified_claims:
            print("No visually verifiable claims found. Exiting.")
            return {}

        # 3. Detect media type and prepare images/frames.
        media_type = FileTypeChecker.detect_media_type(media_path)
        image_paths = []
        if media_type == 'image':
            image_paths = [media_path]
            print(f"3. Detected media type: Image")
        elif media_type == 'video':
            print(f"3. Detected media type: Video. Sampling frames...")
            sampler = VideoSampler(media_path, output_dir, fps)
            image_paths = sampler.sample_frames()
            print(f"   Extracted {len(image_paths)} frames.")
        else:
            raise ValueError(f"Unsupported media type for file: {media_path}")

        # 4. Check each verified claim against the image(s) using CLIP.
        print("4. Verifying claims against media with CLIP model...")
        results = {}
        for claim in verified_claims:
            # Check if ANY of the images/frames support the claim.
            is_supported = any(self.clip_model.check_text_in_image(claim, img_path) for img_path in image_paths)
            results[claim] = is_supported
            print(f"   - Claim: '{claim}' -> Supported: {is_supported}")
        
        print("--- Pipeline Finished ---")
        return results

# --- Usage Example ---

if __name__ == "__main__":
    # NOTE: Using a dummy key will result in an authentication error.
    GROQ_API_KEY = "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" 

    # Instantiate your dependencies
    llm_client = LLMClient(api_key=GROQ_API_KEY)
    claim_extractor = VisualClaimExtractor(llm_client)
    claim_verifier = VisualClaimVerifier(llm_client)
    clip_model_stub = CLIPModel() # Using the placeholder

    # Main orchestrator
    pipeline = ReviewVisualVerifier(
        claim_extractor=claim_extractor,
        claim_verifier=claim_verifier,
        clip_model=clip_model_stub
    )

    # --- Run with an example review and image ---
    review_text = "This phone is great, but it arrived with a noticeable crack on the screen. The box was fine, though."
    # Create dummy files for testing
    dummy_image_path = "/Users/tejasmacipad/Downloads/images.jpeg"
    with open(dummy_image_path, 'w') as f:
        f.write("dummy image content")

    try:
        verification_results = pipeline.process_review(review_text, dummy_image_path)
        print("\n--- Final Results ---")
        print(verification_results)
    except requests.exceptions.HTTPError as e:
        print(f"\nAPI Error: {e}. Please check your API key and network connection.")
    except FileNotFoundError as e:
        print(f"\nFile Error: {e}. Please check your media file path.")
    finally:
        # Clean up dummy file
        if os.path.exists(dummy_image_path):
            os.remove(dummy_image_path)


