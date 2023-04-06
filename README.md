# lora_tag_helper
Simple application to help tag and categorize images for LoRA generation

The workflow is as follows:

00. Clone the repository and `pip install -r requirements.txt` or use a venv.
0. Get automatic captions for your images via a service like the tagger extension for stable-diffusion-webui (Eventually I plan to include this capability in this script, but it's not available yet) 
1. Put your images in the dataset directory, or subfolders therein
2. Run `tag_helper.py`
3. Open the dataset (Ctrl+O) (if you placed it in the one in the repo, it's the default)
4. Add info for your images. You can use tab and (Ctrl+F/Ctrl+B) to navigate quickly.
    a. If you have multiple specific features to add, put their descriptions under different feature tags
    b. The summary should be a short description of the portions of the image relevant to your training
5. Generate a LoRA subset (Ctrl+L)
6. Select a name for your LoRA
7. (optional) review generated captions
8. Point your LoRA trainer to the generated subset
9. Done! Copy the generated LoRA to your stable-diffusion-webui's models/Lora folder.

Note: This program uses .json files to track its own tagging info for each image, and expects you to use .txt files for the image captions used in training.


There are now two versions of the requirements.txt. If you don't have/want to use automatic tagging, then use the requirements_no_ai.txt file, which will greatly reduce the size of your dependencies. In that case, the app will show an error when it first queries the availability of these libraries, but should continue normally after that except for the lack of AI tagging.
