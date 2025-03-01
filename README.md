
lothing Recommendation & Virtual Try-On System

## Overview
This project provides a chat-based interface that recommends clothing items based on user queries and allows users to visualize themselves wearing the recommended items. The system leverages a vector database for clothing item retrieval and a virtual try-on feature using ComfyUI and Stable Diffusion.

## Data Source
We use the clothing dataset available on Kaggle:
[Clothing Dataset](https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full/data)

## Technologies Used
- **Vector Database**: Astra DB
- **Data Processing**: Unstructured
- **Chat Interface**: LangFlow
- **Image Processing & Virtual Try-On**: ComfyUI, Stable Diffusion (DreamShaper model), Segment Anything Model (SAM)
- **Infrastructure**: Modal (for remote compute container on A100 GPU 40VRAM)

## Workflow
### 1. Data Ingestion & Vectorization
- We process the Kaggle dataset using **Unstructured**.
- The processed data is stored in **Astra DB** as a vector database for similarity search.

### 2. Chat-Based Clothing Recommendation
- Users interact with a chat interface powered by **LangFlow**.
- The chatbot suggests clothing items based on the user's input and **ChatGPT's** outfit recommendations.
- A suitable clothing item is retrieved from the **vector database** based on the suggestion.

### 3. Virtual Try-On Feature
- Users upload an image of themselves.
- The system replaces their original outfit with the recommended clothing item.
- Steps performed with **ComfyUI** on a remote container:
  1. **Segment Anything Model (SAM)**: Extracts a layer mask from the text input.
  2. **VAE Encoding**: Encodes the input image and layer mask.
  3. **KSampler Processing**:
     - Runs 5 steps of decoding with a configuration (`cfg`) of 4.0.
     - Uses **CLIP positive prompt text** (same as SAM input) for enhancement.
  4. **VAE Decoding**: Generates the final output image with the user wearing the recommended clothing item.

## Results
The system presents users with:
- An image of the recommended clothing item.
- A transformed image where they appear to be wearing the suggested clothing.

## Additional Resources
[Google Drive with Project Files](https://drive.google.com/drive/folders/14wBLooJTVKv14Yy-vatcOyuBT59Z-gCB?usp=sharing)

## Future Improvements
- Enhancing clothing fit realism using diffusion models.
- Expanding clothing dataset for better recommendations.
- Improving chat interaction with more personalized responses.

## Contributors
- Zach Kysar, Mikhail Ocampo, Logesh Rajendran



#

https://drive.google.com/drive/folders/14wBLooJTVKv14Yy-vatcOyuBT59Z-gCB?usp=sharing
