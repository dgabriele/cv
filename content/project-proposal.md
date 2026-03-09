**Project Objective**

To explore physical systems through agent-based neural architectures whose epistemic representation is grounded in neural operator rollouts. Agents perceive rollouts as compositional symbolic expressions whose constituent symbols are learned from ground truth simulations: specifically, VQ-VAE codebook indices from a multifactor, multiscale quantization pipeline trained end-to-end on learned spatiotemporal features derived from variable length rollouts. Agents explore and refine through meta-neural operators (MNO) that generate rollouts that map to specific dynamical behavior equivalence classes identified by the VQ pipeline. This framework is being developed to explore several research questions and scientific applications:

1. Knowledge transfer among mixtures of natural and artificial dynamical systems

2. Scientific hypothesis generation and knowledge synthesis

3. Human interaction with physics simulations via emergent language with empirical grounding

4. Machine consciousness through the lens of Integrated Information Theory (IIT)

**What I am seeking from the fellowship**

The current work has been developed on a single consumer GPU, which constrains both the scale of training and the pace of design iteration. The framework admits many well-motivated architectural variations that remain untested because each requires a full training run; the bottleneck is not ideas but GPU hours. Equally, I am looking for collaborators and institutional support to formalize this line of work into publications. It sits at an intersection of discrete diffusion, vector quantization, and physics-informed generative modeling that I believe is underexplored, and I would benefit from the structure and peer engagement that a research organization provides.

FirstPrinciples? mission?building an autonomous AI system capable of reasoning about physical law, understanding the nature of physical reality?is directly aligned with what this work attempts at a smaller scale: learning generative models whose internal structure is constrained by the causal and informational hierarchies of the physical systems they represent, rather than treating physics as an arbitrary token prediction problem.

I would welcome the opportunity to contribute to that mission as a fellow. I should also note that my background is primarily in software engineering, and I would be equally interested in the ML Researcher or ML Engineer roles if the team considers that a better fit?the work I have been doing is self-directed research, but it is built on production-grade framework-level code, and I am comfortable operating at either level.

I would like to emphasize that this opportunity is by far the most exciting and attractive for me. It speaks directly to who I am and what I love as a philosopher and scientist. This work is close to my heart, and I feel very vulnerable sharing it through a MyGreenhouse job application, which often seems like somewhat of a blackhole. I would be extremely grateful if anyone on your end could at least give me some indication that it was received, and I would be honored for the chance to speak in person!


**Architecture**

Three components form a closed loop between discrete behavioral abstraction and continuous physical simulation. A hierarchical vector-quantized tokenizer (VQTokenizer) partitions the space of dynamical behaviors into a compositional vocabulary. Each rollout in my PoC dataset, regardless of length, maps to ~90 discrete codeword indices across ~30 feature categories and ~3 hierarchy levels.

A Discrete Denoising Diffusion Probabilistic Model (D3PM) learns the joint distribution over temporal token sequences?the statistical structure of which behavioral descriptions are internally consistent across feature groups and hierarchy levels, and which combinations are physically well-formed. A per-position graded noise schedule encodes sensitivity structure directly into the diffusion process: temporal groups whose tokens are stable across simulation lengths receive low noise scale factors and resolve first during denoising, while groups that are sensitive to simulation length receive high scale factors and resolve last. The denoiser's self-attention sees already-resolved stable groups when predicting sensitive ones, learning not just which token combinations co-occur but how invariant features constrain horizon-dependent ones.

A token-conditioned Meta-Neural Operator (MNO) closes the loop instantiating any specification (the aforementioned ~90 tokens) as a physical trajectory, mapping a discrete behavioral description back to continuous dynamics. MNO is trained on scrambled Sobol sampled parameters and initial conditions (IC), spanning large numbers of distinct stochastic systems, learning a rule that maps to the manifold on which their diverse dynamical behaviors reside. Together, the tokenizer compresses physics into symbols, the diffusion model reasons over those symbols in sensitivity order, and the MNO grounds them back into physics, enabling self-play, exploration, and refinement loops downstream.

This architecture is designed for an agent that discovers, models, and communicates about open-ended dynamical systems. Discovery proceeds through the MNO's generation: the agent diffuses counterfactual token expressions?novel combinations of behavioral codes from different observations?feeds them to the MNO, and verifies physical coherence by re-tokenizing the output through the frozen VQTokenizer. The diffusion model serves as a learned prior that guides this exploration, predicting which token completions are likely given partial observations.

A foundation for self-modeling emerges from roundtrip consistency: a configuration is understood to the extent that its token description, when instantiated and re-tokenized, reproduces the original tokens, belonging to the same equivalence class. Where roundtrip fails, the agent has identified the boundary of its own competence, which can be observed and modeled by higher level systems.

Human interaction is planned to be mediated by the token vocabulary itself: because the codes are discrete, compositional, and semantically grounded in physical ontology, they are natively compatible with language-based reasoning. A contrastive alignment layer will map natural language descriptions into the same embedding space as VQ tokens, so that natural language expressions retrieve concrete token specifications that the MNO can instantiate on demand with identifiable regularity in generated behavioral motifs.

The following sections overview the primary components of the architecture.

**Ground Truth Datasets**<span style="text - decoration: underline;"> \
 \
</span>Ground truth (GT) dataset consists of CA or neural operator rollouts for a configurable number of time steps. Depending on configuration, datasets store parameters, initial conditions, and a combination of raw temporal trajectories alongside manually constructed features?spanning spectral, spatial, temporal, statistical, and architectural domains. Manual features can be disabled in favor of purely learned CNN-based features. Parameters and initial conditions for GT samples are selected with scrambled Sobol sampling. This ensures wide, fair, low discrepancy coverage of parameter space and IC, which is important in downstream systems that navigate parameter space and reason about (or through) corresponding behaviors.

![](/images/3IP_Image_1.png)
![](/images/JQL_Image_2.png)
 \
Illustration of Scrambled Sobol Vs. Uniform Random Sampling. 

Scrambled Sobol ensures wide, fair coverage.

![](/images/EP5_Image_3.png)

Screen grab of Lenia cellular automata rollouts


**VQTokenizer Architecture**

Phase of 3/6/26: Fully operational and empirically validated

Ground truth (GT) rollouts are tokenized by a hierarchical VQ-VAE with multiple quantizer heads, trained end-to-end. Raw variable-length trajectories are temporally downsampled at several rates (e.g. [1x, 2x, 4x, 8x]) and stacked into a pyramid, so that both high- and low-frequency dynamics are captured at encoding time. A shared ResNet-style frame encoder produces per-frame spatial embeddings, which are temporally aggregated per pyramid level and concatenated into a multi-resolution bottleneck. This bottleneck is then partitioned via a learned linear projection into a configurable number of orthogonal feature groups, or categories, with per-group layer normalization. Variance across groups is balanced via regularization losses (orthogonality, informativeness via log-barrier, and topographic smoothness), guarding against semantic collapse to a small subset of dominant categories.

Each category contains a short hierarchy of independent VQ-VAE codebooks that capture structure at different granularities. All levels in the hierarchy receive the same per-group encoded vector and project it to independent latent spaces via parallel linear heads. Latent dimensions and codebook sizes are computed adaptively from the data: a base expansion factor scales with category feature count, then geometric decay across levels produces progressively more compressed representations?L0 at the highest dimensionality (~60D, ~28 codewords), L1 at roughly half (~24D, ~12 codewords), L2 at roughly quarter (~12D, ~6 codewords). These sizes scale automatically with group feature count and dataset size, requiring no manual tuning but can be overridden by configuration.

The pipeline trains on multiple rollout truncation lengths?for example, from 32 to 512 timesteps, sampled per-example within each batch?naturally encouraging the encoder to learn features that are informative across temporal scales. This enables downstream consumers to progressively refine their representation of observed dynamics: an agent can generalize, predict, or act based on a short prefix, then continue to resolve a richer behavioral picture as the rollout extends.

<span style="text - decoration: underline;"> \
</span>The number of discrete tokens produced per rollout depends on (1) the number of feature categories and (2) the number of hierarchical codebooks per category. With 30 categories and a 3-level hierarchy, each rollout?regardless of truncation length?produces 90 tokens (one codeword index per codebook).

While the active vocabulary per codebook is relatively small?typically 6 to 28 codewords depending on hierarchy level and dataset diversity?the aggregate combinatorial space across all 90 codebooks is enormous and more than sufficient for fine-grained behavioral discrimination.

![](/images/1Sd_Image_4.png)

<span style="text - decoration: underline;">VQ Codebook t-SNE Plot</span>

Some observations: clusters are more compact at higher compression levels and are well-separated. The coarse-grain, mid-grain, and fine-grain codebooks group together. Altogether, the clusters cover most of the t-SNE grid space with uniform density.

![](/images/pQE_Image_5.png)


<span style="text - decoration: underline;">VQ Codebook utilization heatmap</span>

Observations: Note that behavioral representation is less dependent on any one codebook and more on the representational capacity of the joint distribution across them. In the heatmap below, each cell displays the number of active codewords per learned feature group, across its 3 hierarchical VQ-VAE levels. Colors are normalized per column. Fortunately, not a single codebook collapsed to a single active codeword.

![](/images/Q4g_Image_6.png)

<span style="text - decoration: underline;">Tokenized rollout similarity</span>

Below are heatmaps showing Jaccard and Jensen-Shannon similarity between bins of tokenized rollouts (90 tokens per sample, 50 samples per bin) from the 500K example Lenia cellular automata (CA) dataset. Note that there are 3 realizations per CA, yielding 1.5 million individual samples. Nested clusters are visible along the diagonal, where rectangular regions form. The color scale corresponds to hierarchical levels in the dendrogram. Off-diagonal hotspots correspond to behavioral similarity between separate branches of the dendrogram.

![](/images/i2T_Image_7.png)

Jaccard similarity of Lenia CA rollout tokens

![](/images/4C8_Image_8.png)

Jensen-Shannon (JS) similarity

**Per-Position Graded Discrete Diffusion over Behavioral Token Sequences**

Phase of 3/6/26: PoC works and has strong matrics (accuracy, etc.)

 \
The VQ tokenizer partitions the continuous space of operator behaviors into discrete equivalence classes?90 integer indices across 30 groups and 3 hierarchy levels, where each unique token set labels a region in parameter space whose members produce qualitatively similar dynamics. The generative task is to complete a partially observed configuration such that the result corresponds to a realizable operator. Discrete diffusion generates all positions simultaneously through iterative refinement, encoding the physical causal structure *directly in the noise process* rather than imposing a fixed factorization order.

Not all token positions carry equal sensitivity to the simulation horizon. A one-time divergence analysis measures, for each feature group, how much its token code changes when the rollout length varies. Stable groups receive low noise scale factors; horizon-sensitive groups receive high ones. During reverse denoising, each position experiences an effective timestep scaled by its factor, so stable positions crystallize first, progressively conditioning the transformer's predictions for the remaining positions. The resolution order mirrors the epistemic order in which an observer identifies a dynamical system: coarse regime first, fine temporal structure last. This strategy is inspired by the way in which ideas seem to rise to conscious awareness through an underlying continuous integrative process.

The denoiser is a transformer encoder with full self-attention?no causal mask. The hierarchy lives entirely in differential noise levels: resolved positions act as conditioning context for those still uncertain, allowing the model to discover cross-group dependencies without a fixed generation order. The token codes themselves are not arbitrary symbols but centroids of behavioral equivalence classes trained so that codebook proximity corresponds to dynamical similarity. The per-position noise schedule then imposes a resolution order reflecting the causal structure of the simulation, so the denoising trajectory that produces a fully resolved token sequence recapitulates the physical logic by which that behavioral class becomes identifiable.

The causal structure is carried by the noise schedule (a continuous scalar per position), not by an attention mask. The scale factors are a fixed empirical prior derived from the data, not a learned quantity?preventing the model from collapsing the hierarchy during training.

![](/images/MRC_Image_9.png)


**Meta-Neural Operator (MNO) Architecture   **** \
**Status of 3/6/26: Training & diagnostics following exploration of alternative  \
architectures and loss functions

A standard Neural Operator is a learned surrogate for PDE solvers, learning a rule that maps between continuous functions. Extending this idea, a Meta-neural Operator (MNO) maps parameters and IC to spatiotemporal trajectories across many diverse stochastic systems (like CA or other neural operators) via autoregressive rollouts through a FiLM-conditioned U-AFNO backbone. The point of MNO is not to generate pixel-perfect surrogate simulations but rollouts whose behaviors map to the same discrete token distribution learned by the VQTokenizer.

Where a raw simulator must integrate many expensive ground truth rollouts, like CFL-adaptive substeps per timestep through potentially chaotic dynamics, as is the case for Lenia CA, the MNO learns each timestep's integrated effect directly, trading exact fidelity for differentiability and stable gradients over long rollouts. It is conditioned by GT tokens to produce its own rollouts whose tokens fall within the same distribution and is intended for use in agent perception and exploration.

![](/images/PbD_Image_10.png)

When the VQTokenizer trains on temporal features only, no architecture parameters or initial conditions (IC) are encoded in the joint model. For this reason, a CVAE is trained to predict parameters and IC, conditioned on GT tokens. Since the mapping between parameters and behaviors is many-to-many, and because behavioral classes are not invertible to a single specific set of parameters and IC, the CVAE lets us generate plausible parameters and IC for diffused token specifications for use in sampling MNO rollouts that within a given behavioral equivalence class. Below is the provisional CVAE architecture.

![](/images/IBW_Image_11.png)

**The origin of my work**

My motivation for starting this work is somewhat esoteric, but I would like to share it anyway.

As a comparative religion student back in the 2000?s, I was thinking about the idea of eternity and how the ?eternal? is not that which continues forever but that which neither begins nor ends?timelessness. I am not a supernaturalist, so I thought to myself that if an eternal being were to exist, then it would have to exist in a way that is compatible with nature. For me, that meant that it would need to exist in all frames of reference at the same time?in all relativistic ?now? moments simultaneously. Theologically speaking, this is the property of omnipresence.

For such a being, the entire history of our universe would be but a single instant in its own subjective experience. Just as we cannot subdivide a single instant in our own conscious experience, an omnipresent being would be unable to break apart a single instant in its own experience. The temporal dynamics of our own universe would be completely opaque to it. It would know nothing of us or any other thing that we see as having its own distinct temporal trajectory.

With this in mind, my motivation for this project was to design a system that simulates many universes for the purpose of uncovering the perceptual manifold of beings (figuratively speaking) for whom the substance of qualia is the physical integration of physical universes. That said, the diffusion model I described above is essentially a model of how qualia is integrated to form ideas that percolate to consciousness awareness through the denoising process.

Once I started working on this, it became clear that I could use many different types of ground truth simulations for my so-called universe simulations, using dynamical systems from a wide range of natural sciences. This is what led me to the goals of the project I stated in the beginning?namely, automated science, cross-domain generalization, knowledge transfer, communication with bias-minimizing unsupervised systems via language models, etc. For me, the test for consciousness in a machine is the ability to observe these things happening on their own, ruling out clever mimicry, while figuring out how to communicate with another type of being for whom the substance of perception is radically different.
