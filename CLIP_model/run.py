from run import ReviewPipelineWrapper

API_KEY = "" # Replace with your actual key
REVIEW_1 = "The camera lens was shattered on arrival."
IMAGE_1 = "/path/to/review1_image.jpg"
REVIEW_2 = "The t-shirt color was completely wrong."
IMAGE_2 = "/path/to/review2_image.png"

auditor = ReviewPipelineWrapper(api_key=API_KEY)

results1 = auditor.run(REVIEW_1, IMAGE_1)
print(f"Results for Review 1: {results1}")

results2 = auditor.run(REVIEW_2, IMAGE_2)
print(f"Results for Review 2: {results2}")
