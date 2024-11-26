import wandb
import os

def initialize_wandb(project_name, run_name, api_key=None):
    """
    Khởi tạo wandb với thông tin đăng nhập và dự án.
    :param project_name: Tên dự án trên wandb
    :param run_name: Tên phiên làm việc
    :param api_key: API key của wandb (tùy chọn)
    """
    # Đăng nhập bằng API key nếu được cung cấp
    if api_key:
        wandb.login(key=api_key)
    elif "WANDB_API_KEY" in os.environ:
        # Lấy API key từ biến môi trường nếu không truyền trực tiếp
        wandb.login(key=os.getenv("WANDB_API_KEY"))
    else:
        raise ValueError("API key cho wandb không được cung cấp hoặc không có trong biến môi trường.")

    # Khởi tạo phiên làm việc
    wandb.init(project=project_name, name=run_name)

def log_metrics(metrics):
    """
    Ghi lại các chỉ số (metrics) vào wandb.
    :param metrics: Dictionary các chỉ số (key-value)
    """
    wandb.log(metrics)

def finish_wandb():
    """
    Kết thúc phiên làm việc wandb.
    """
    wandb.finish()
