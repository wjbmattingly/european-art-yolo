from huggingface_hub import HfApi, create_repo

def upload_model_to_hf(
    local_path: str = "runs/detect/ms_yolo11n2/weights/best.pt",
    repo_id: str = "wjbmattingly/european-art-yolov11",
    model_name: str = "european-art-yolov11n.pt",
    repo_type: str = "model",
    private: bool = False
):
    """
    Upload a model to Hugging Face Hub
    
    Args:
        local_path (str): Path to the local model file
        repo_id (str): Hugging Face repository ID (username/repo-name)
        model_name (str): Desired name for the uploaded model file
        repo_type (str): Type of repository ('model', 'dataset', etc.)
        private (bool): Whether to create a private repository
    """
    api = HfApi()
    
    # Create the repository if it doesn't exist
    try:
        create_repo(repo_id=repo_id, repo_type=repo_type, private=private)
    except Exception as e:
        print(f"Repository already exists or error occurred: {e}")
    
    # Upload the model
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=model_name,
        repo_id=repo_id,
        repo_type=repo_type
    )
    
    print(f"Model uploaded successfully to {repo_id}")

# Example usage:
if __name__ == "__main__":
    upload_model_to_hf(
        local_path="runs/detect/ms_yolo11n2/weights/best.pt",
        repo_id="wjbmattingly/european-art-yolov11",
        model_name="european-art-yolov11n.pt",
        private=False
    )
