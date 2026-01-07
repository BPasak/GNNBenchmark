import platform
import psutil
import GPUtil

def get_system_specs():
    specs = {
        "CPU": platform.processor(),
        "Cores": psutil.cpu_count(logical = True),
        "RAM": f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB"
    }

    gpus = GPUtil.getGPUs()
    gpu_overview = []
    for gpu in gpus:
        gpu_overview.append(
            {
                "id": str(gpu.id),
                "name": str(gpu.name)
            }
        )

    specs["GPUs"] = gpu_overview

    return specs