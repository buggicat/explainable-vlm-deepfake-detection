# Forensic Techniques Reference (auto-generated)

READ THIS BEFORE ANALYZING. The following is a curated reference of standard
image-forensics techniques, distilled from a literature corpus of 270+ papers
(`multimedia_forensics_techniques/chunks.jsonl`). Use it to (a) recall the
canonical name of each technique, (b) map your observations to that taxonomy,
and (c) recall typical parameters / thresholds. You are not required to use
every technique — choose what fits the image.

## Key methodological priors

- **Bias-free evaluation** (B-Free, CVPR 2025): real and fake images must be
  content-matched to isolate generation artifacts from dataset bias. High
  accuracy on standard benchmarks often reflects content/format/resolution
  shortcuts rather than true generation cues.
- **Frequency-domain robustness** (Seeing What Matters, 2025): mid-high
  *diagonal* spatial frequencies are the most robust forensic cues — they
  survive JPEG and codec compression. Horizontal/vertical bands and very-high
  frequencies are contaminated by compression artifacts and are weaker
  discriminators.
- **Self-conditioned reconstruction signal**: passing a real image through a
  diffusion model's inpainting pipeline (empty mask, generic prompt) injects
  the architecture's fingerprint while preserving content; the residual
  between real and self-conditioned versions reveals subtle low-frequency
  artifacts.

## Techniques (54 total)

### IMAGE & VIDEO PHYSICS-BASED FORENSICS

**1. Illumination consistency analysis**
> Color is impacted by scene objects, illumination, and the in-camera processing pipeline. *(source: 2206.10737v1.pdf, p. 1)*
> 1 Gradient-Based Illumination Description for Image Forgery Detection Falko Matern, Christian Riess, Senior Member, IEEE, Marc Stamminger Abstract—The goal of blind image forensics is to determine authenticity and origin of an image without using an explicitly embedded security… *(source: 2019-Matern-GBI.pdf, p. 1)*

**2. Specular highlight consistency**
> trary content, while physics-based methods oftentimes require speciﬁc image content [15], [18], and they also oftentimes require manual annotations by an analyst [15], [21], [22]. *(source: 2206.10737v1.pdf, p. 1)*
> 1 Gradient-Based Illumination Description for Image Forgery Detection Falko Matern, Christian Riess, Senior Member, IEEE, Marc Stamminger Abstract—The goal of blind image forensics is to determine authenticity and origin of an image without using an explicitly embedded security… *(source: 2019-Matern-GBI.pdf, p. 1)*

**3. Shadow geometry and temporal shadow consistency**
> The proposed method shares with physics-based methods the beneﬁt of being remarkably robust to compression and downsampling, which is illustrated by the three variants of the same picture in horizontal direction and their associated heatmaps. *(source: 2206.10737v1.pdf, p. 1)*
> 1 Gradient-Based Illumination Description for Image Forgery Detection Falko Matern, Christian Riess, Senior Member, IEEE, Marc Stamminger Abstract—The goal of blind image forensics is to determine authenticity and origin of an image without using an explicitly embedded security… *(source: 2019-Matern-GBI.pdf, p. 1)*

**4. 3D facial geometry consistency**
> trary content, while physics-based methods oftentimes require speciﬁc image content [15], [18], and they also oftentimes require manual annotations by an analyst [15], [21], [22]. *(source: 2206.10737v1.pdf, p. 1)*
> 1 Gradient-Based Illumination Description for Image Forgery Detection Falko Matern, Christian Riess, Senior Member, IEEE, Marc Stamminger Abstract—The goal of blind image forensics is to determine authenticity and origin of an image without using an explicitly embedded security… *(source: 2019-Matern-GBI.pdf, p. 1)*

**5. Perspective and camera model consistency**
> Color is impacted by scene objects, illumination, and the in-camera processing pipeline. *(source: 2206.10737v1.pdf, p. 1)*
> Physics-based methods explain image inconsistencies using an analytic model, and are more robust to common image processing operations such as resizing or recompression. *(source: 2019-Matern-GBI.pdf, p. 1)*

**6. Image formation model violations**
> We pro- pose a forensic descriptor to characterize the color formation in an image. *(source: 2206.10737v1.pdf, p. 1)*
> 1 Gradient-Based Illumination Description for Image Forgery Detection Falko Matern, Christian Riess, Senior Member, IEEE, Marc Stamminger Abstract—The goal of blind image forensics is to determine authenticity and origin of an image without using an explicitly embedded security… *(source: 2019-Matern-GBI.pdf, p. 1)*

### FREQUENCY & TRANSFORM-DOMAIN FORENSICS

**7. Fourier magnitude spectrum analysis (FFT)**
> Ablation Variations Accuracy (%) Baseline - 90.35 Features added to the learned ones Magnitude 82.36 FFT 88.18 LBP 88.42 Magnitude and FFT 81.20 Magnitude and LBP 81.67 FFT and LBP 91.60 Magnitude, FFT and LBP 81.56 Table 4. *(source: 2406.04932v1.pdf, p. 7)*
> The high-pass filter Bh(·) can be de- fined by: Bh(fi,j) = fi,j, otherwise, 0, if |i| < W/4, |j| < H/4 (4) The Thirty-Eighth AAAI Conference on Artiﬁcial Intelligence (AAAI-24) 5054 High-Frequency Representation of Image(HFRI) Residual Conv block FFT iFFT Frequency Conv… *(source: 28310-Article Text-32364-1-2-20240324.pdf, p. 3)*

**8. Radial and angular frequency profiling**
> In fact, frequency analysis is a com- mon and important way in digital image processing and has been widely applied to various tasks in computer vi- sion [44, 47, 41, 18]. *(source: 2103.01856v3.pdf, p. 3)*
> 3.1 FAD: Frequency-Aware Decomposition Towards the frequency-aware image decomposition, former studies usually ap- ply hand-crafted ﬁlter banks [10,22] in the spatial domain, thus fail to cover the complete frequency domain. *(source: 2007.09355v1.pdf, p. 6)*

**9. Discrete Cosine Transform (DCT) statistics**
> The popular choices in the literature include Discrete Cosine Transform (DCT) [8], Wavelet Transform (WT) [9], Fourier Transform (FT) [10], and Singular Value Decomposition (SVD) [11]. *(source: Shrinking_the_Semantic_Gap_Spatial_Pooling_of_Local_Moment_Invariants_for_Copy-Move_Forgery_Detection.pdf, p. 2)*
> Preliminaries We transform images into the frequency domain using the discrete cosine transform (DCT). *(source: 2003.08685v2.pdf, p. 2)*

**10. Wavelet domain analysis**
> Therefore, a multi-level wavelet-powered feature enhancement module (MWFEM) is proposed to promote our network focusing on local artifacts from the spatio-frequency domain. *(source: 3503161.3547832.pdf, p. 5)*
> In the multi-domain probability estimation module, since the wavelet coefﬁcients corresponding to the effective image information are larger than those corresponding to lossy noise, and residual processing can enhance the detail of image edge structure weakened by lossy noise,… *(source: Multi-domain_Probability_Estimation_Network_for_Forgery_Detection_over_Online_Social_Network_Shared_Images.pdf, p. 2)*

**11. High-pass / band-pass residual analysis**
> represent the combined low and high pass filters. *(source: Forgery-aware_Adaptive_Transformer_for_Generalizable_Synthetic_Image_Detection.pdf, p. 4)*
> Some studies use high-pass ﬁlters [57,15,29,48], Gabor ﬁlters [10,22] etc. *(source: 2007.09355v1.pdf, p. 5)*

### SENSOR & NOISE-BASED FORENSICS

**12. PRNU (Photo-Response Non-Uniformity)**
> The noise-based Deepfake detection approaches up to date are mostly relying on the Photo Response Non-Uniformity (PRNU), a noise pattern created by small factory defects in the light-sensitive sen- sors of a digital camera (Lukas, Fridrich, and Goljan 2006). *(source: 26701-Article Text-30764-1-2-20230626.pdf, p. 2)*
> noise (PRNU), and N1 # 2 is an additive noise term. *(source: farid-sigproc09.pdf, p. 6)*

**13. PRNU proxy / noise residual stationarity**
> Weever and Wilczek (de Weever and Wilczek 2020) made several ex- periments calculating the correlation of the PRNU noise and found out that none of the PRNU noise analyses had resulted in a definite proof of real or fake. *(source: 26701-Article Text-30764-1-2-20230626.pdf, p. 2)*
> noise (PRNU), and N1 # 2 is an additive noise term. *(source: farid-sigproc09.pdf, p. 6)*

**14. Demosaicing artifact analysis**
> When using different algorithms for demosaicing and analysis, detection is still possible as long as one is using a close enough demosaicing method – hence the necessity of trying several algorithms to select the best-matching one. *(source: WIFS22_Double_Demosaicing.pdf, p. 3)*

**15. Sensor pipeline consistency checks**
> 3) CMOS Image Sensor Based PUF: To avoid sophisticated statistical image processing techniques, CMOS image sensor based PUF [26] uses the innate ﬁxed pattern noise (FPN) of active pixel sensor array for camera or imaging device identiﬁcation. *(source: A_PUF-Based_Data-Device_Hash_for_Tampered_Image_Detection_and_Source_Camera_Identification.pdf, p. 5)*
> • We present a novel idea of extracting the face- background pairs via the Siamese structure towards Deepfake noise features analyses. *(source: 26701-Article Text-30764-1-2-20230626.pdf, p. 2)*

### COMPRESSION & ENCODING FORENSICS

**16. Error Level Analysis (ELA)**
> For this purpose, we intro- duced four modules −i) the SRMConv [64] layer, ii) the BayarConv [3] layer, iii) the classic convolution layer termed as RGBConv, and iv) our proposed Error Level Analysis (ELA) Module. *(source: 2112.04298v3.pdf, p. 8)*
> Error Level Analysis (ELA) [34] ﬁts in this category and creates a heatmap by recompressing the image and visualising the difference. *(source: An_Adaptive_Neural_Network_for_Unsupervised_Mosaic_Consistency_Analysis_in_Image_Forensics.pdf, p. 1)*

**17. JPEG quantization table analysis**
> JPEG Compression Trace Extractor JPEG Compression Trace In the JPEG compression pipeline, each 8 × 8 image block f(i, j) of an image I will be transformed by F(u, v) = DCT(f(i, j)). *(source: 25095-Article Text-29158-1-2-20230626.pdf, p. 2)*
> The quantization tables can be extracted from the encoded JPEG image or blindly estimated from the image, as described in [6] . *(source: farid-sigproc09.pdf, p. 3)*

**18. Double compression detection**
> INTRODUCTION Detection of double JPEG (DJPEG) compression plays a major role in image forensics since double compression reveals important information about the past history of an image [1], [2]. *(source: 2102.01439v2.pdf, p. 1)*
> double compression artifacts. *(source: 28571-Article Text-32625-1-2-20240324.pdf, p. 1)*

**19. Video codec artifact analysis**
> Due to the complexity of video codecs, a number of techniques have been proposed for various settings of a codec where speciﬁc video encoding features are turned on or oﬀ. *(source: 2105.06361v2.pdf, p. 2)*
> Explaining Deepfake Detection by Analysing Image Matching 7 3.3 Vulnerability of artifact representations to video compression Hypothesis 3: Implicitly learned artifact visual concepts through the FST- Matching in the raw training set are vulnerable to the video compression. *(source: 2207.09679v1.pdf, p. 6)*

### SPATIAL MANIPULATION FORENSICS

**20. Splicing boundary detection**
> On the other hand, to encourage the model to concen- trate on the boundary of tampered regions, we propose a boundary-guided tampering contrastive loss based on a boundary-guided sampling strategy in favor of distinguishing tampered and non-tampered region. *(source: Attentive_and_Contrastive_Image_Manipulation_Localization_With_Boundary_Guidance.pdf, p. 3)*
> 3.2.2 Soft Boundary Supervision Local Details Region Groundtruth Absolute Boundary Groundtruth Soft Boundary Groundtruth G G ݃ Figure 7. *(source: Sun_SAFL-Net_Semantic-Agnostic_Feature_Learning_Network_with_Auxiliary_Plugins_for_Image_ICCV_2023_paper.pdf, p. 5)*

**21. Copy-move forgery detection**
> Chen, “Robust and accurate detection of image copy-move forgery using PCET-SVD and histogram of block similarity measures,” J. *(source: Shrinking_the_Semantic_Gap_Spatial_Pooling_of_Local_Moment_Invariants_for_Copy-Move_Forgery_Detection.pdf, p. 15)*

**22. Texture inconsistency analysis**
> Then, we assume the facial texture T as the composition of common texture and identity texture, where the common texture Tcom is the texture patterns shared by all the people, as shown in Figure 1(d), and the identity texture Tid is the discriminative ﬁne-grained texture… *(source: 2011.09737v1.pdf, p. 3)*
> Then, these frames are simultaneously passed through the attribute analysis model, gaze analysis model, and texture analysis model to extract features (batch size × N_frames, channel, height, width). *(source: Where_Deepfakes_Gaze_at_SpatialTemporal_Gaze_Inconsistency_Analysis_for_Video_Face_Forgery_Detection.pdf, p. 3)*

**23. Blending and alpha-matte anomalies**
> al [41] formu- late the synthetic image detection as an identification prob- lem and achieve generalizable detection through language- guided contrastive learning. *(source: 2403.17465v4.pdf, p. 3)*
> Motivation Artifacts in forged face images can be roughly divided into two types: spatial-related (e.g., generative artifacts, blending, and etc.) and temporal-related artifacts (e.g., flick- ering and discontinuity). *(source: 2307.08317v1.pdf, p. 3)*

### TEMPORAL & MOTION FORENSICS (VIDEO)

**24. Optical flow consistency**
> For instance, artifact-specific approaches focus on detecting unnatural areas in deepfake human faces by leveraging edges and optical flow. *(source: 2406.15921v2.pdf, p. 1)*
> [19,28,36,48, 74] where the authenticity is somehow represented by hid- den signals in pristine videos. *(source: 2212.14033v1.pdf, p. 1)*

**25. Temporal coherence analysis**
> Exploring Temporal Coherence for More General Video Face Forgery Detection Yinglin Zheng 1 Jianmin Bao 2, Dong Chen 2, Ming Zeng 1*, Fang Wen 2 1 School of Informatics, Xiamen University 2 Microsoft Research Asia {zhengyinglin@stu., zengming@}xmu.edu.cn, {jianbao, doch,… *(source: 2108.06693v1.pdf, p. 1)*
> Table 4: Initial List of Analytics for Detecting Media Manipulation (Part 1) Name Why Where What Paper Title MesoNet deepfake, *(source: 3706598.3713711.pdf, p. 22)*

**26. Physiological signal analysis**
> [19,28,36,48, 74] where the authenticity is somehow represented by hid- den signals in pristine videos. *(source: 2212.14033v1.pdf, p. 1)*
> Because of the continuous improvement on the detection accuracy by facial physiological signals, video face forgery detection based on facial physiological signal analysis has received more and more attention, which has become an important research branch in the field of face… *(source: Where_Deepfakes_Gaze_at_SpatialTemporal_Gaze_Inconsistency_Analysis_for_Video_Face_Forgery_Detection.pdf, p. 1)*

**27. Head pose and motion dynamics**
> Our approach • analyzes the motion patterns in real and fake videos, combining traditional and deep methods; • proposes a novel, robust, and generalizable deepfake source detector based on motion cues; and • improves both source detection and fake detection us- ing motion… *(source: 2212.14033v1.pdf, p. 1)*
> (a) Flowchart of the FFE-Net; and (b) Illustration of the deconvolution (Transposed convolution) operation with kernel size of 3×3 and stride of 2×2. *(source: Preventing_DeepFake_Attacks_on_Speaker_Authentication_by_Dynamic_Lip_Movement_Analysis.pdf, p. 5)*

### AUDIO & AUDIO-VISUAL FORENSICS

**28. Lip-sync consistency**
> inconsistencies such as a lack of lip-sync, unnatural facial and lip appearance/movements or asymmetry between facial regions such as the left and right eyes (see Fig. *(source: 3394171.3413700.pdf, p. 2)*
> segments of target events. *(source: AVoiD-DF_Audio-Visual_Joint_Learning_for_Detecting_Deepfake.pdf, p. 3)*

**29. Audio-visual embedding alignment**
> A novel audio-visual transformer [39] frame- work has been proposed to localize audio-visual events with audio features jointly observed over visual features. *(source: AVoiD-DF_Audio-Visual_Joint_Learning_for_Detecting_Deepfake.pdf, p. 3)*
> in real video face context, the audio-visual correspondence is deeply intuitive since there is an intrinsic correlation between the mouth articulations (visemes) and the speech units (phonemes) [2, 10, 50, 72], as well as an alignment of emotional nuances embedded in the facial… *(source: 2406.02951v1.pdf, p. 1)*

**30. Speech prosody and voice biometrics**
> Contrastive learning [41] has also been exploited for action recognition and video understanding by transferring knowledge across heterogeneous modalities between audio-visual. *(source: AVoiD-DF_Audio-Visual_Joint_Learning_for_Detecting_Deepfake.pdf, p. 3)*
> in real video face context, the audio-visual correspondence is deeply intuitive since there is an intrinsic correlation between the mouth articulations (visemes) and the speech units (phonemes) [2, 10, 50, 72], as well as an alignment of emotional nuances embedded in the facial… *(source: 2406.02951v1.pdf, p. 1)*

**31. Audio spectral artifact analysis**
> A novel audio-visual transformer [39] frame- work has been proposed to localize audio-visual events with audio features jointly observed over visual features. *(source: AVoiD-DF_Audio-Visual_Joint_Learning_for_Detecting_Deepfake.pdf, p. 3)*
> Such inherent audio-visual correspondence, for example, in audio-driven emotion, is challenging to faithfully replicate in deepfake videos. *(source: 2406.02951v1.pdf, p. 1)*

### METADATA & PROVENANCE FORENSICS

**32. EXIF and container metadata consistency**
> Consequently, metadata has a higher de- gree of opacity than pixel data, which makes metadata-based media forensics more reliable and its corresponding attacks more challenging. *(source: 2105.06361v2.pdf, p. 2)*
> However, social media platforms, image hosting ser- vices and commercial applications are forced to strip the metadata (EXIF) and camera id for various reasons [76]. *(source: 2203.07824v1.pdf, p. 1)*

**33. Metadata–content contradiction analysis**
> Compared to pixel-level analysis, metadata-based methods possess unique advantages. *(source: 2105.06361v2.pdf, p. 2)*

**34. Provenance graph reasoning**
> Note that in-camera color processing is different from other camera provenance-based approaches since it does not require data with provenance labels, which *(source: 2206.10737v1.pdf, p. 2)*
> A cryptographically signed hash of the media and metadata is then stored on a trust list. *(source: Farid_Creating.pdf, p. 24)*

### GENERATIVE MODEL FINGERPRINTING

**35. GAN fingerprint detection**
> dataset To study the fingerprints of more ad- vanced generative models, we collect samples from state-of- the-art models such as NVAE [58], Efficient VDVAE [19], VQ-GAN [10], StyleGAN2 [27], Denoising Diffusion GAN (DDGAN) [63], DDPM++ [22], NCSN++ [55] and Latent Score-based… *(source: 2402.10401v2.pdf, p. 12)*
> (2021) use shallow fingerprints for image generators in the form of gi(z) = g0(z) + ϕi where g0(·) is an unfingerprinted model and show that ϕis have 1 arXiv:2304.09752v2 [cs.CV] 26 May 2023 Attributing Image Generative Models using Latent Fingerprints Fingerprinted latent var. *(source: 2304.09752v2.pdf, p. 1)*

**36. Diffusion model latent inversion analysis**
> To study this problem, we design a latent inversion based method called LATENTTRACER to trace the gen- erated images of the inspected model by checking if the examined images can be well-reconstructed with an inverted latent input. *(source: 2405.13360v1.pdf, p. 1)*

**37. Autoencoder reconstruction error analysis**
> The reconstruction error ∆AEi(x) is defined as the distance between an image x and its reconstruction ˜x obtained from passing it through the encoder Ei and decoder Di of an LDM’s AE. *(source: AEROBLADE_Training-Free_Detection_of_Latent_Diffusion_Images_Using_Autoencoder_Reconstruction_Error.pdf, p. 3)*
> Error-Guided Feature Refinement Existing methods exploit the reconstruction error as the only feature to detect the generated image, which ignores the relationships between the reconstruction error and the raw image. *(source: 2403.17465v4.pdf, p. 4)*

**38. Model source attribution**
> Synthetic Image Source Attribution Researchers have developed many algorithms for synthetic image attribution [1, 5, 49]. *(source: 2308.11557v1.pdf, p. 3)*
> Attribution network The goal of our attribution network is to predict the source generative model of an observed image. *(source: 2402.10401v2.pdf, p. 5)*

**39. Watermark and active signal detection**
> 3.4 Watermark Detection The well-trained watermark extractor can extract watermark information from test images. *(source: 0037.pdf, p. 5)*

### LEARNING-BASED DETECTION

**40. CNN-based manipulation classifiers**
> [73]   CNN-based  Wang et al. *(source: Hierarchical_Fine-Grained_Image_Forgery_Detection_and_Localization.pdf, p. 3)*
> 2021) and the CNN-based detection methods. *(source: 25095-Article Text-29158-1-2-20230626.pdf, p. 2)*

**41. Transformer-based detectors**
> Our parts-based detectors are below the dotted line. *(source: 2109.10688v1.pdf, p. 8)*
> datasets, including all pseudo- deepfake based detectors, video-based models, one-shot techniques and transformer-based models, while being ri- valed on CDFv2 only by the video transformer based detec- tor LTTD [27] and the PCL-I2G technique [77]. *(source: 2211.11296v2.pdf, p. 7)*

**42. Graph neural network forensic models**
> [16] represented the image as a graph structure and introduces a vision graph neural network (GNN) architecture to extract graph-level features for visual tasks including image recognition and object detection. *(source: Using_Graph_Neural_Networks_to_Improve_Generalization_Capability_of_the_Models_for_Deepfake_Detection.pdf, p. 3)*
> The Graph Neural Network Model. *(source: 20233-Article Text-24246-1-2-20220628.pdf, p. 9)*

**43. Prototype-based temporal models**
> The prototype layer p computes the squared ℓ2 dis- tance between each the prototype vectors pj and each spa- tial/temporal patch (of shape (1, 1, C)) within the input fea- ture maps z. *(source: 2006.15473v2.pdf, p. 3)*
> Prototype-Based Enhancement Module Although MIL enables the extraction of forgery features by aligning prominent frames with video-level attributions, it may overlook the subtle forgery features that heavily inﬂuence generalization in deepfake detection. *(source: Bi-Stream_Coteaching_Network_for_Weakly-Supervised_Deepfake_Localization_in_Videos.pdf, p. 4)*

**44. Self-supervised and contrastive anomaly detection**
> One biggest difference between supervised con- trastive learning and self-supervised contrastive learning is that the former leverages task labels so positive samples for each anchor point is more informative. *(source: 3581783.3612377.pdf, p. 5)*
> [16] Alban Siffer, Pierre-Alain Fouque, Alexandre Termier, and Christine Largouet, “Anomaly detection in streams with extreme value theory,” in Proc. *(source: Facial_Region-Based_Ensembling_for_Unsupervised_Temporal_Deepfake_Localization.pdf, p. 6)*

### EXPLAINABILITY METHODS

**45. SHAP**
> something a fake that actually wasn’t one because we would lose more credibility that way than vice versa.” Explainability was frequently ranked highly, often second or third in importance. *(source: 3706598.3713711.pdf, p. 10)*

**46. LIME**
> something a fake that actually wasn’t one because we would lose more credibility that way than vice versa.” Explainability was frequently ranked highly, often second or third in importance. *(source: 3706598.3713711.pdf, p. 10)*

**47. Saliency maps**
> materials how our human saliency maps can be incor- porated into the attention mechanism, and show that CY- BORG allows for a better gain in accuracy than using hu- man saliency in the attention mechanism. *(source: 2112.00686v3.pdf, p. 2)*

**48. Attention map inspection**
> Three major factors were considered during the con- struction and development of our attention map; i) explain- ability, ii) usefulness, and iii) modularity. *(source: Dang_On_the_Detection_of_Digital_Face_Manipulation_CVPR_2020_paper.pdf, p. 3)*

**49. Prototype and example-based explanations**
> Textual explanations were widely favored (P1, P4, P16, P17, P18, P19, P20, P27, P29) because they were seen as “easy to understand,” “simple,” and “less technical.” Many participants also noted their usefulness for direct inclusion in reports. *(source: 3706598.3713711.pdf, p. 10)*
> The prototype layer p computes the squared ℓ2 dis- tance between each the prototype vectors pj and each spa- tial/temporal patch (of shape (1, 1, C)) within the input fea- ture maps z. *(source: 2006.15473v2.pdf, p. 3)*

### HYBRID & META-FORENSIC TECHNIQUES

**50. Physics-constrained explainability**
> data Ds meta to improve the model’s generalization. *(source: 16367-Article Text-19861-1-2-20210518.pdf, p. 4)*
> The relationship between speed and explainability was also subtle, with some suggesting that “if the speed is signifcantly slow, then it matters more when compared to explainability.” Explainability was also seen as a potential mitigator for false positives and negatives… *(source: 3706598.3713711.pdf, p. 10)*

**51. Explanation–evidence alignment metrics**
> data Ds meta to improve the model’s generalization. *(source: 16367-Article Text-19861-1-2-20210518.pdf, p. 4)*
> [26] Detection efforts typically focus on developing methods that seek evidence of manipulation and present that evidence in the form of a numerical output or a visualization to alert an analyst that the media needs further analysis. *(source: CSI-DEEPFAKE-THREATS.PDF, p. 6)*

**52. Human-in-the-loop forensic validation**
> As a result, low-quality local forensic traces can cause forensic algorithms to make incorrect decision. *(source: 2211.15775v2.pdf, p. 2)*
> data Ds meta to improve the model’s generalization. *(source: 16367-Article Text-19861-1-2-20210518.pdf, p. 4)*

**53. Adversarial robustness evaluation of explanations**
> data Ds meta to improve the model’s generalization. *(source: 16367-Article Text-19861-1-2-20210518.pdf, p. 4)*

**54. Cross-modal consistency reasoning**
> Cross-modal Consistency Learning. *(source: 2206.05741v3.pdf, p. 7)*

