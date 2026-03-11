import re
import math

def count_params_from_log(log_path):
    total_params = 0
    
    with open(log_path, "r") as f:
        for line in f:
            match = re.search(r"shape:\s*\((.*?)\)", line)
            if match:
                shape_str = match.group(1)
                dims = [int(x.strip()) for x in shape_str.split(",") if x.strip()]
                param_count = math.prod(dims) if dims else 0
                total_params += param_count

    return total_params


log_file = "grad_params.log"
total = count_params_from_log(log_file)
print(f"Total trainable parameters: {total:,}")