# Multimodal Review Authenticity Verification System

This repository contains the source code and documentation for a sophisticated, multi-modal AI system designed to verify the authenticity of online reviews. By analyzing text, images, videos, and user metadata in concert, the model provides a robust defense against fraudulent, AI-generated, and misleading content, ensuring a more trustworthy e-commerce ecosystem.

![System Architecture](https://github.com/user-attachments/assets/bf9dfb38-f249-430e-aa14-40edd6779fa3)
![Confusion Matrix on Test Set](https://github.com/user-attachments/assets/d851aaf3-e1ee-477f-8c45-d55b9b27bc9c)

- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Features
*   **Multimodal Analysis**: Integrates text, visual, and user data streams for comprehensive verification, a technique proven to enhance system robustness against attacks[1].
*   **AI-Generated Text Detection**: Employs a specialized "PAN Voight-Kampff" checker to identify and flag reviews written by generative AI[2][3].
*   **Visual Content Verification**: Analyzes images to ensure they are relevant and authentic, using similarity search to detect reused or stock photos[2][4].
*   **Cross-Modal Consistency Check**: Filters out textual claims that are not supported by the provided visual evidence using contrastive learning models[2].
*   **User Behavior Profiling**: Assesses user trustworthiness by analyzing return history and offers a perks system to incentivize authentic contributions[2].
*   **Advanced NLP**: Leverages state-of-the-art models like RoBERTa and BERT for deep semantic understanding, including sentiment analysis, aspect extraction, and categorization[2].

## System Architecture

The system is designed as a multi-pipeline architecture that processes different data modalities in parallel before fusing them for a final decision. This approach is common in advanced authentication systems to increase accuracy and security[5][6].

### 1. Input Layer
The model ingests three primary sources of data for each review:
*   **Text Reviews**: The user-submitted written text.
*   **Images/Videos Content**: All associated media files.
*   **User Certification Status**: Metadata related to the user's account and history[2].

### 2. Text Analysis Pipeline
This pipeline performs an in-depth analysis of the review text through several stages:
*   **Preprocessing**: Standard text cleaning procedures, including tokenization and lemmatization, are applied.
*   **AI Authorship Detection**: The text is analyzed by a **PAN Voight-Kampff Generative AI Detection** module. This component is designed to solve the specific problem of determining whether a text was written by a human or a machine[2][3].
*   **Semantic Embedding**: A **RoBERTa** model generates rich, contextual vector embeddings from the text.
*   **Feature Optimization**: The **Tunicate Swarm Algorithm (TSA)** is applied to the embeddings. This is a metaheuristic optimization algorithm used for feature selection, refining the data to improve the performance of subsequent classification tasks.
*   **Cross-Modal Filtering**: A **Contrastive Language-Image Pretraining (CLIP)** based filter removes textual claims that do not semantically match the content of the associated image and product details. This ensures that the text being analyzed is relevant to the visual evidence[2].
*   **Parallel Analysis Streams**: The filtered embeddings are fed into four parallel models for comprehensive analysis:
    *   **Categorical Analysis**: A **BERT** model classifies the review into predefined categories.
    *   **Aspect Extraction**: A **Neural Attention Model** identifies and extracts key product features or topics discussed in the review (e.g., "battery life," "screen quality").
    *   **Temporal Analysis**: This module extracts and interprets any time-related information within the text.
    *   **Sentiment Analysis**: A fine-tuned **RoBERTa** model determines the overall sentiment of the review[2].

### 3. Visual & User Analysis Pipelines
These pipelines run in parallel to the text analysis:
*   **Image Analysis**: The system uses **EfficientNet-B3**, a highly efficient convolutional neural network, for detailed image analysis[2].
*   **Similar Object Search**: A **FAISS**-based system performs a high-speed similarity search on the product image against a database, also considering seller and product details. This is critical for identifying duplicate images or photos inconsistent with the product listing[2][4].
*   **User Behavior Analysis**: The model analyzes user return history, comparing spending against returns to identify suspicious patterns. It includes a mechanism to reward trustworthy users with perks[2].
*   **Pricing Model**: An auxiliary model identifies and shows low-priced objects if the product and user are the same, potentially flagging unusual purchasing behavior[2].

### 4. Fusion and Final Output
*   **Late Fusion**: The high-level feature outputs from all parallel pipelines are integrated using a **Multi-Layer Perceptron (MLP)**. This late fusion strategy combines the processed insights from each modality.
*   **Boolean Output**: The MLP generates a final boolean value (`True` or `False`) indicating whether the review is deemed authentic[2].

## Technology Stack

| Category                | Technologies                                      |
| ----------------------- | ------------------------------------------------- |
| **AI / Machine Learning** | `PyTorch`, `Transformers`, `scikit-learn`         |
| **NLP Models**          | `RoBERTa`, `BERT`, `spaCy`                          |
| **Computer Vision**     | `OpenCV`, `Pillow`, `EfficientNet`                  |
| **Similarity Search**   | `FAISS`                                           |
| **Backend & Deployment**  | `Python`, `Flask`/`Django`, `Docker`, `NGINX`       |
| **Data Storage**        | `PostgreSQL`, `MinIO` (for media files)           |

## Installation

Follow these steps to set up the project locally.

**1. Clone the Repository**
```bash
git clone https://github.com/your-username/multimodal-review-authenticity.git
cd multimodal-review-authenticity
```

**2. Create and Activate a Virtual Environment**
It is highly recommended to use a virtual environment.
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Install Dependencies**
Create a `requirements.txt` file with the necessary packages and install them.
```
# requirements.txt
torch
transformers
scikit-learn
opencv-python-headless
faiss-cpu # or faiss-gpu for CUDA support
spacy
numpy
Pillow
```

Install using pip:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```


## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss proposed changes or additions.

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Commit your changes (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/YourFeature`).
5.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
