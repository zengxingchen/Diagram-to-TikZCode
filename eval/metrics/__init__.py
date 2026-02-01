"""
Evaluation Metrics Module
Provides a unified interface for importing metrics to avoid name conflicts.
"""

# Export all metric classes
__all__ = [
    'CrystalBLEUMetric',
    'TextEditDistance', 
    'DreamSimMetric',
    'SigLIPSimilarity',
    'KIDMetric',
    'EdgeBasedMSE',
    'PixelMSE',
    'LPIPS',
    'SSIM',
    'PSNRMetric'
]

# Use __getattr__ to implement precise lazy imports
def __getattr__(name):
    if name == 'CrystalBLEUMetric':
        from .crystal_bleu import CrystalBLEU as CrystalBLEUMetric
        return CrystalBLEUMetric
    elif name == 'TextEditDistance':
        from .text_edit_distance import TexEditDistance as TextEditDistance
        return TextEditDistance
    elif name == 'DreamSimMetric':
        from .dreamsim import DreamSim as DreamSimMetric
        return DreamSimMetric
    elif name == 'SigLIPSimilarity':
        from .siglip_similarity import ImageSim as SigLIPSimilarity
        return SigLIPSimilarity
    elif name == 'KIDMetric':
        from .kid import KernelInceptionDistance as KIDMetric
        return KIDMetric
    elif name == 'EdgeBasedMSE':
        from .edge_mse import MSE as EdgeBasedMSE
        return EdgeBasedMSE
    elif name == 'PixelMSE':
        from .pixel_mse import MSE_withoutcanny as PixelMSE
        return PixelMSE
    elif name == 'LPIPS':
        from .lpips import LPIPSMetric as LPIPS
        return LPIPS
    elif name == 'SSIM':
        from .ssim import StructuralSIM as SSIM
        return SSIM
    elif name == 'PSNRMetric':
        from .psnr import PSNR as PSNRMetric
        return PSNRMetric
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
