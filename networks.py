from model import CliffordNet
    
    
def gen_shifts(n):
    return [1 << i for i in range(n)]

def gen_shifts_fibonacci(n):
    a, b = 1, 2
    for _ in range(n):
        yield a
        a, b = b, a + b
        
def cliffordnet_12_2(num_classes=100, patch_size=2, embed_dim=128, enable_cuda=False):
    # Nano: shifts=[1, 2]
    shifts = gen_shifts(2)
    return CliffordNet(
        enable_cuda=enable_cuda,
        num_classes=num_classes,
        patch_size=patch_size,
        embed_dim=embed_dim, 
        cli_mode='full', 
        ctx_mode='diff',
        shifts=shifts, 
        depth=12,
        drop_path_rate=0.3
    )    

def cliffordnet_12_3(num_classes=100, patch_size=1, embed_dim=160, enable_cuda=False):
    # Nano: shifts=[1, 2]
    shifts = gen_shifts(3)
    return CliffordNet(
        enable_cuda=enable_cuda,
        num_classes=num_classes,
        patch_size=patch_size,
        embed_dim=embed_dim, 
        cli_mode='full', 
        ctx_mode='diff',
        shifts=shifts, 
        depth=12,
        drop_path_rate=0.3 
    )    

def cliffordnet_12_5(num_classes=100, patch_size=2, embed_dim=128, enable_cuda=False):
    # Lite: shifts=[1, 2, 4, 8, 16]
    shifts = gen_shifts(5)
    return CliffordNet(
        enable_cuda=enable_cuda,
        num_classes=num_classes,
        patch_size=patch_size,
        embed_dim=embed_dim, 
        cli_mode='full', 
        ctx_mode='diff',
        shifts=shifts, 
        depth=12,
        drop_path_rate=0.3 
    )    

def cliffordnet_18_5(num_classes=100, patch_size=2, embed_dim=128, enable_cuda=False):
    # Lite: shifts=[1, 2, 4, 8, 16]
    shifts = gen_shifts(5)
    return CliffordNet(
        enable_cuda=enable_cuda,
        num_classes=num_classes,
        patch_size=patch_size,
        embed_dim=embed_dim, 
        cli_mode='full', 
        ctx_mode='diff',
        shifts=shifts, 
        depth=18,
        drop_path_rate=0.3 
    )  

def cliffordnet_32_3(num_classes=100, patch_size=2, embed_dim=128, enable_cuda=False):
    # Small: 32 layers
    # Shifts: [1, 2, 4]
    shifts = gen_shifts(3)
    return CliffordNet(
        enable_cuda=enable_cuda,
        num_classes=num_classes,
        patch_size=patch_size,
        embed_dim=embed_dim, 
        cli_mode='full', 
        ctx_mode='diff',
        shifts=shifts, 
        depth=32,
        drop_path_rate=0.3 
    )     

def cliffordnet_32_5(num_classes=100, patch_size=2, embed_dim=128, enable_cuda=False):
    # Small: 32 layers
    # Shifts: [1, 2, 4, 8, 16]
    shifts = gen_shifts(5)
    return CliffordNet(
        enable_cuda=enable_cuda,
        num_classes=num_classes,
        patch_size=patch_size,
        embed_dim=embed_dim, 
        cli_mode='inner', 
        ctx_mode='diff',
        shifts=shifts, 
        depth=32,
        drop_path_rate=0.3 
    )     
 
def cliffordnet_64_5(num_classes=100, patch_size=2, embed_dim=128, enable_cuda=False):
    # Deep: 64 layers
    # Shifts: [1, 2, 4, 8, 16]
    shifts = gen_shifts(5)
    return CliffordNet(
        enable_cuda=enable_cuda,
        num_classes=num_classes,
        patch_size=patch_size,
        embed_dim=embed_dim, 
        cli_mode='inner', 
        ctx_mode='diff',
        shifts=shifts, 
        depth=64,
        drop_path_rate=0.4 
    )     
