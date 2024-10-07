import re
from lib import load_model_and_encoders, predict_compatibility

def extract_system_info(system_info_str):
    cpu_match = re.search(r'CPU Brand: (.+)', system_info_str)
    gpu_match = re.search(r'Driver: NVIDIA Corporation (.+)/PCIe/SSE2', system_info_str)
    os_match = re.search(r'"(.+)" \(64 bit\)', system_info_str)
    kernel_match = re.search(r'Kernel Version: (.+)', system_info_str)
    ram_match = re.search(r'RAM: (\d+) Mb', system_info_str)

    cpu = cpu_match.group(1) if cpu_match else None
    gpu = gpu_match.group(1) if gpu_match else None
    distribution = os_match.group(1) if os_match else None
    kernel = kernel_match.group(1) if kernel_match else None
    ram = f"{int(ram_match.group(1)) // 1024} GB" if ram_match else None

    return {
        'cpu': cpu,
        'gpu': gpu,
        'distribution': distribution,
        'kernel': kernel,
        'ram': ram
    }

def main():
    system_info = """
    Computer Information:
    Manufacturer: Gigabyte Technology Co., Ltd.
    Model: X570S AORUS MASTER
    Form Factor: Desktop
    No Touch Input Detected
    Processor Information:
    CPU Vendor: AuthenticAMD
    CPU Brand: AMD Ryzen 9 5950X 16-Core Processor
    CPU Family: 0x19
    CPU Model: 0x21
    CPU Stepping: 0x2
    CPU Type: 0x0
    Speed: 5084 MHz
    32 logical processors
    16 physical processors
    Hyper-threading: Supported
    FCMOV: Supported
    SSE2: Supported
    SSE3: Supported
    SSSE3: Supported
    SSE4a: Supported
    SSE41: Supported
    SSE42: Supported
    AES: Supported
    AVX: Supported
    AVX2: Supported
    AVX512F: Unsupported
    AVX512PF: Unsupported
    AVX512ER: Unsupported
    AVX512CD: Unsupported
    AVX512VNNI: Unsupported
    SHA: Supported
    CMPXCHG16B: Supported
    LAHF/SAHF: Supported
    PrefetchW: Unsupported
    Operating System Version:
    "Arch Linux" (64 bit)
    Kernel Name: Linux
    Kernel Version: 6.11.1-zen1-1-zen
    X Server Vendor: The X.Org Foundation
    X Server Release: 12101013
    X Window Manager: GNOME Shell
    Steam Runtime Version: steam-runtime_0.20240916.101793
    Client Information:
    Version: 1728009005
    Browser GPU Acceleration Status: Disabled
    Browser Canvas: Disabled
    Browser Canvas out-of-process rasterization: Disabled
    Browser Direct Rendering Display Compositor: Disabled
    Browser Compositing: Disabled
    Browser Multiple Raster Threads: Disabled
    Browser OpenGL: Disabled
    Browser Rasterization: Disabled
    Browser Raw Draw: Disabled
    Browser Skia Graphite: Disabled
    Browser Video Decode: Disabled
    Browser Video Encode: Disabled
    Browser Vulkan: Disabled
    Browser WebGL: Disabled
    Browser WebGL2: Disabled
    Browser WebGPU: Disabled
    Browser WebNN: Disabled
    Video Card:
    Driver: NVIDIA Corporation NVIDIA GeForce RTX 2080 Ti/PCIe/SSE2
    Driver Version: 4.6.0 NVIDIA 560.35.03
    Desktop Color Depth: 24 bits per pixel
    Monitor Refresh Rate: 99 Hz
    VendorID: 0x10de
    DeviceID: 0x1e07
    Revision Not Detected
    Number of Monitors: 3
    Number of Logical Video Cards: 1
    Primary Display Resolution: 3440 x 1440
    Desktop Resolution: 8720 x 2560
    Primary Display Size: 32.28" x 13.62" (35.04" diag), 82.0cm x 34.6cm (89.0cm diag)
    Primary VRAM: 11264 MB
    Sound card:
    Audio device: USB Mixer
    Memory:
    RAM: 31983 Mb
    VR Hardware:
    VR Headset: None detected
    Miscellaneous:
    UI Language: English
    LANG: de_DE.UTF-8
    Total Hard Disk Space Available: 1429495 MB
    Largest Free Hard Disk Block: 329148 MB
    Storage:
    Number of SSDs: 6
    SSD sizes: 500G,500G,500G,250G,250G,240G
    Number of HDDs: 0
    Number of removable drives: 0
    """

    system_info = extract_system_info(system_info)

    print("Extrahierte Systeminformationen:")
    for key, value in system_info.items():
        print(f"{key}: {value}")

    print("\nLade Modell und Encoder...")
    model, le_title, le_gpu_manufacturer, le_gpu_model, le_distribution, le_cpu, le_ram, le_kernel, scaler = load_model_and_encoders()

    print("\nBeispielvorhersagen:")
    games = [

        "Stardew Valley",
        "The Witcher 3: Wild Hunt",
        "Cyberpunk 2077",
        "New World",
        "No Man's Sky",
        "Destiny 2"
    ]

    for game in games:
        result = predict_compatibility(
            game,
            system_info['gpu'],
            system_info['distribution'],
            system_info['cpu'],
            system_info['ram'],
            system_info['kernel'],
            model, le_title, le_gpu_manufacturer, le_gpu_model,
            le_distribution, le_cpu, le_ram, le_kernel, scaler
        )

        print(f"\nVorhersage für {result['title']}:")
        print(f"Kompatibilitätslevel: {result['compatibility_level']}")
        print(f"Qualitätsscore: {result['quality_score']:.2f}")
        if result['specs']:
            print("Spezifikationen:")
            print(f"  {', '.join(result['specs'])}")
        if result['unknown_labels']:
            print(f"Unbekannte Labels: {', '.join(result['unknown_labels'])}")
        if result['partial_info']:
            print(f"Teilweise Informationen: {', '.join(result['partial_info'])}")
        print("--------------------")

if __name__ == "__main__":
    main()
