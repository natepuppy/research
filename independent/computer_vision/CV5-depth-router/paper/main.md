# Let a Small Depth Model Answer the Spatial Questions

*Draft skeleton, written before the pilot. Numbers in brackets are filled from `results/`. Sections marked [3080] are the paper-scale experiments.*

**Abstract.** Spatial questions are where a vision-language model and a geometric specialist should fail differently, and the difference has not been used. We pair Depth Anything V2 Small, 25 million parameters, with an off-the-shelf detector, and we compare its answers to a vision-language model's on the same items. The claim is complementary error. Geometry is right when the depth gap is large and the detector saw the object. The vision-language model is right when the question is semantic or the depth map is flat. In a controlled draw of that rule, exactly one system is correct on [xx%] of items, the depth gap on geometry-only successes exceeds the gap on vision-language-only successes by [0.xx], and a threshold fit on 400 items and then frozen lifts accuracy from [0.xx] to [0.xx], keeping [xx%] of the gain of a threshold fit on the test set itself. [3080] On What'sUp, a spatial benchmark and a counting slice, the same split [holds / does not hold] for a frozen depth model against Qwen3-VL, and the router [beats / ties] the better of the two. If it holds, spatial questions should be routed, not answered by the larger model alone, and the router is a threshold rather than a trained policy.

## 1. Introduction

A repeated CVPR 2026 result is a small specialist beating a large generalist on the specialist's own slice. Spatial language is an obvious slice and an unmeasured one. Monocular depth is cheap, detectors are cheap, and both already run faster than a vision-language model. What is missing is the map of who is right.

The map has two regions. Where the depth gap is large and the detector has the category, geometry should win, because the vision-language model is known to be weak on exactly these relations. Where the depth map is flat, or the category is one the detector does not have, geometry has nothing to say and the vision-language model should win. A router that knows which region it is in, from the depth gap, the object size and the detector's confidence, can call the right one. Fitting that router on a few hundred labeled items is the whole training budget.

Contributions. (1) The disagreement map: who is right as a function of depth gap [pilot; 3080]. (2) The frozen router, and how much of an oracle threshold's gain survives a fit on a few hundred items [pilot; 3080]. (3) [3080] The measurement with Depth Anything V2 Small against Qwen3-VL on What'sUp, a spatial benchmark and a counting slice.

## 2. Related work

Depth Anything V2. What'sUp (Kamath et al.). DrivePI and the specialist-versus-generalist results at CVPR 2026. BenchDepth and FoundationGeo, which evaluate depth rather than route it. SpatialVLM and its successors, which train a spatial model. This paper does not.

## 3. Setting

**Pilot.** Items with a depth gap, a size, an occlusion and a semantic flag. Detector confidence is high only when the flag is off, so the router never sees the flag directly. Geometry is right with probability 0.92 on the easy slice (gap above 0.30, size above 0.20, occlusion below 0.80, detector saw it) and 0.12 elsewhere. The vision-language model is right with probability 0.84 on semantic items, 0.70 on flat depth, and 0.38 otherwise. Router: three thresholds, grid-searched on 400 validation items. 4,000 test items, 8 seeds.

**[3080].** Depth Anything V2 Small plus a detector. Qwen3-VL-2B and 4B, and one larger 4-bit model that fits the card. What'sUp (820 images), a few thousand spatial questions, a counting slice, two prompts. About 20 GPU-hours. The router is fit once, on a few hundred hand-labeled or benchmark-labeled items.

Pre-registered. K1: exactly one system correct on ≥ 30% of items, and the median depth gap of geometry-only successes exceeds that of vision-language-only successes by ≥ 0.15, in ≥ 6 of 8 seeds. K2: the router beats the better system by ≥ 0.05, in ≥ 6 of 8 seeds. K3, reported: the validation fit keeps ≥ 80% of the test-oracle threshold's gain. SUPPORTED if K1 and K2.

## 4. Results

[pilot] Accuracy of geometry, the vision-language model, the router and the oracle threshold. The fraction of items where exactly one is correct. Median depth gap in the two exclusive-success sets. The thresholds the validation fit chose.

[3080] The same, plus the disagreement plot against depth gap, object size and occlusion. A note on items where the depth model itself is wrong for a geometric reason, which the router will inherit.

## 5. Discussion

The pilot draws the split it then recovers, so it tests the router rather than the split. The vision-language experiment is where the split can fail to exist. If both systems are wrong together, or the router merely ties the better one, the paper stops: there is no complementary error, and calling a depth model is engineering advice.

Limitations. The pilot's depth is a scalar gap, not a depth map, and its detector confidence is a function of the semantic flag. Real detector confidence is noisier, and K3 is the check on whether a few hundred labels still pin the threshold under that noise. The 3080 version has to show the router's errors, not only its accuracy, because a router that calls geometry on a bad depth map fails confidently.

## 6. [3080] Planned experiments

(a) Depth and detector over the image sets. (b) Vision-language forwards, two prompts, three models. (c) The disagreement plot. (d) The router fit on a few hundred items, evaluated on the rest, against each system alone and against an oracle threshold.

## 7. Conclusion

[After the pilot.]

## References

Depth Anything V2; What'sUp, Kamath et al.; DrivePI; BenchDepth; FoundationGeo; SpatialVLM.
