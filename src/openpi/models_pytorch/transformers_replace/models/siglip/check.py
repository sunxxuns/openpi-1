import transformers

def check_whether_transformers_replace_is_installed_correctly():
    version = transformers.__version__.split(".")
    major, minor = int(version[0]), int(version[1])
    return major >= 4 and minor >= 53