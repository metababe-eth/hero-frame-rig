"""Conservative local edge ownership; no object IDs, ROIs, or model weights."""
import torch.nn.functional as F


def expand(x, radius):
    x=F.max_pool2d(x,(1,2*radius+1),1,(0,radius))
    return F.max_pool2d(x,(2*radius+1,1),1,(radius,0))


def ownership(depth, image, strength):
    """Return a nearby solid owner's disparity and a feathered confidence.

    A 3px erosion rejects isolated near-depth noise. A bounded search finds
    nearer solid support across a depth jump. RGB high-pass evidence restricts
    reassignment to visible thin structures. Smooth confidence, rather than a
    depth lerp through the focus plane, is used when blending their treatment.
    """
    h,w=depth.shape[-2:]
    radius=max(3,min(12,round(min(h,w)/70)))
    solid=-F.max_pool2d(-depth,3,1,1)
    owner=expand(solid,radius)
    jump=((owner-depth-.08)/.10).clamp(0,1)
    luma=image.mean(1,keepdim=True)
    mean=F.avg_pool2d(F.pad(luma,(2,2,2,2),mode='reflect'),5,1)
    thin=((luma-mean).abs()-.008).clamp_min(0)/.04
    evidence=jump*thin.clamp(0,1)
    # Cover both sides and the immediate blur footprint, with no hard mask.
    # This is bounded local evidence, not semantic material identification.
    band=radius*2
    spread=expand(evidence,band)
    feather=max(2,radius//2)
    confidence=F.avg_pool2d(F.pad(spread,(feather,feather,feather,feather),mode='reflect'),2*feather+1,1)
    owner=expand(solid,band+radius)
    confidence=(confidence*(2*strength)).clamp(0,1)
    return owner,confidence
