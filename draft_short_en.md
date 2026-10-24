# Detecting Spurious Features via Intervention-Based Performance Sensitivity

## Abstract

Models can achieve high in-distribution accuracy while relying on non-causal, spurious features. Prior work has mainly focused on discovering suspicious features or slices. We study the next step: given a candidate feature, how can we verify whether the model actually depends on it for predictive performance? We propose `PerformanceCheck`, a post-hoc confirmation procedure that measures performance sensitivity to targeted input interventions. For each candidate feature, we apply a destructive but structure-preserving intervention and define the resulting performance drop, `Δ`, as a proxy for functional necessity. The method is black-box with respect to the model: it only requires a trained oracle, evaluation data, and an intervention operator.

We evaluate `PerformanceCheck` on three benchmarks with 30 seeds each: GridWorld, ColoredMNIST with IRM-trained oracles, and Waterbirds. On GridWorld, `PerformanceCheck` achieves `85/120` top-1 detections and `106/120` top-2 detections, compared with `14/120` and `21/120` for SHAP. On ColoredMNIST, `PerformanceCheck` achieves `102/120`, while SHAP reaches `120/120` under our aggregation rule; we show that this apparent advantage is a structural artifact caused by the benchmark’s R/G encoding and the aggregation itself. On Waterbirds, SHAP flags background in `30/30` seeds, while `PerformanceCheck` detects stronger functional background reliance in `18/30` seeds and separates candidate existence from practical performance impact. Across these settings, intervention-based performance sensitivity provides information that is complementary to attribution-based discovery and is especially useful as a confirmation step before remediation.

## 1. Introduction

Shortcut learning under distribution shift has been reported across domains: vision models rely on background or color cues, language models exploit superficial patterns, and policies in control settings may latch onto incidental correlations. Training-time approaches such as IRM, GroupDRO, and counterfactual invariance attempt to suppress such dependencies during learning, but they require modified objectives, environment structure, or group annotations. In many practical cases, however, the immediate problem is post-hoc diagnosis: given a trained model, which suspected features actually matter for its predictive performance?

Existing post-hoc methods only partially answer this question. Attribution methods such as SHAP, LIME, and Integrated Gradients quantify importance through output-level decompositions or local approximations. They are useful for explanation, but they do not directly measure what practitioners often care about: how much performance degrades when a specific feature is broken. Prior work has also shown that attribution can fail systematically for unknown spurious correlations. This motivates a confirmation-oriented perspective.

Our contribution is to formalize this missing confirmation step. We do not aim to replace candidate discovery methods such as Neural PCA, Domino, SpRAy, or group discovery methods. Instead, we assume that suspicious candidates are already available and ask whether the model functionally relies on them. `PerformanceCheck` answers this by applying targeted input interventions and measuring the resulting performance drop. The output is a ranked list of candidates by intervention-induced performance sensitivity, which can then be used to prioritize remediation.

This framing also makes the method easy to place inside a broader pipeline. Candidate generators, including prior slice-discovery or shift-diagnosis frameworks, can feed features into `PerformanceCheck`; `PerformanceCheck` then confirms which candidates are most damaging when perturbed; and practitioners can prioritize augmentation, data collection, or retraining based on the resulting ranking. The present paper validates the confirmation stage, not a fully automated end-to-end system.

We evaluate the method on three settings that stress different aspects of spurious detection. GridWorld provides a synthetic benchmark with explicit feature control. ColoredMNIST highlights how benchmark encoding can interact with attribution aggregation rules. Waterbirds offers a more realistic vision setting where background serves as a spatially localized spurious cue. Across all three, our central question is the same: can intervention-based performance sensitivity provide a more operational signal of feature reliance than output-distribution attribution alone?

## 2. Related Work

Work on spurious correlations broadly falls into two categories. The first modifies training, for example through invariant risk minimization, distributionally robust optimization, or counterfactual constraints. These methods are important, but they target mitigation during learning rather than diagnosis after training.

The second category concerns post-hoc analysis. Attribution methods such as SHAP, LIME, and Integrated Gradients explain predictions through feature importance scores. They are not designed specifically as operational tests of “what breaks performance if removed,” and their scores can diverge from intervention-based sensitivity when features are strongly correlated. Recent work has also explored automatic discovery of suspicious components or slices, for example through representation-space analysis, clustering of explanation maps, or slice discovery with foundation-model embeddings. We view these as candidate generators.

Our method is complementary to both lines. It does not change training and it does not claim to discover candidate features from scratch. Instead, it formalizes a post-discovery confirmation step based on input intervention and direct performance measurement.

## 3. Method

Let `f` be a trained oracle, `D` an evaluation dataset, and `F` a set of candidate features. For each candidate `i in F`, we define an intervention operator `T_i` that destroys the target feature while preserving other information as much as possible. We then define intervention-based performance sensitivity as

`Δ_i = P(f, D) - P(f, T_i(D))`,

where `P` is an appropriate scalar performance metric such as accuracy. Larger `Δ_i` indicates that the model loses more performance when candidate `i` is broken.

We interpret `Δ_i` as a proxy for functional necessity, not as a causal effect estimate. This distinction matters. The goal is not to prove that a feature is intrinsically causal or spurious in the strongest sense; rather, it is to rank candidate features by how much they matter to the trained model’s behavior under targeted perturbation.

### 3.1 Admissible Interventions

An ideal intervention should satisfy two desiderata: `(a)` destroy information in the target feature, and `(b)` preserve the marginal structure of non-target features as much as possible. In practice, condition `(b)` is approximate, and the quality of the intervention affects the trustworthiness of `Δ`.

We therefore distinguish structure-preserving interventions from generic destructive corruptions. For GridWorld, we resample a single feature from its empirical marginal. For ColoredMNIST, we intervene on color channels or neutralize color while preserving shape. For Waterbirds, we use two interventions: `background_replace`, which swaps only the background using a class-consistent donor pool while keeping the foreground mask fixed, and `foreground_noise`, which injects Gaussian noise into the foreground region. These are imperfect, but they are designed to perturb the targeted feature without collapsing the rest of the input into unrelated OOD noise.

### 3.2 Decision Rule

The operational output of `PerformanceCheck` is a ranked candidate list. Our primary metric is therefore `recall@k` based on whether the target spurious candidate appears in the top `k`. In settings with effectively two candidates, such as Waterbirds foreground vs. background, the pairwise comparison `Δ_bg > Δ_fg` is simply the `recall@1` case.

## 4. Experiments

We use three benchmarks and compare `PerformanceCheck` with SHAP under matched seeds and matched oracles.

GridWorld is a 10-dimensional synthetic setting with one spurious feature `c`, two causal features, and seven noise features. We vary the spurious correlation strength `p in {0.0, 0.1, 0.2, 0.3}` and evaluate 30 seeds per setting.

ColoredMNIST follows the standard correlated-color construction with IRM-trained oracles. The key difficulty here is that the benchmark’s R/G encoding can interact with attribution aggregation in nontrivial ways.

Waterbirds uses a frozen ResNet-18 backbone with a linear classifier head, trained on the standard background-biased bird benchmark. We evaluate 30 seeds, use a class-consistent donor pool of 200 backgrounds per class for `background_replace`, and compute repeated intervention averages with `n_eval = 300`.

All main comparisons use the same seeds `0..29` and the same trained oracle for `PerformanceCheck` and SHAP.

| Benchmark | Oracle / setting | Main target | Metric |
| --- | --- | --- | --- |
| GridWorld | 10D synthetic, 30 seeds x 4 `p` values | spurious feature `c` | `recall@k` |
| ColoredMNIST | IRM-trained CNN, 30 seeds x 4 `p` values | color vs. structure | `recall@1` |
| Waterbirds | frozen ResNet-18 + linear head, 30 seeds | background vs. foreground | `recall@1` |

Table 1: Summary of the three evaluation settings.

## 5. Results

### 5.1 Q1: Does performance sensitivity track spurious-correlation strength?

In GridWorld, `PerformanceCheck` maintains high detection in strong-correlation regimes (small `p`) and degrades gradually as `p` increases and the correlation weakens. Across all 120 trials, it achieves `85/120` top-1 detections and `106/120` top-2 detections. SHAP achieves `14/120` top-1 and `21/120` top-2. Paired McNemar tests confirm that the gap is significant for both top-1 (`p = 8.38 x 10^-16`) and top-2 (`p = 2.04 x 10^-4`).

This result supports the idea that direct intervention on candidate features can recover operational dependence even when attribution of the model output (mean absolute Shapley values of the Q-values) favors correlated non-spurious dimensions.

![Figure 1: GridWorld comparison. `PerformanceCheck` maintains high detection in strong-correlation regimes (small `p`), while SHAP struggles to rank the spurious feature highly in the same setting.](experiments/results/exp016_shap_psweep_30seeds/results.png)

### 5.2 Q2: Can the method track dependence switches under IRM?

In ColoredMNIST, `PerformanceCheck` achieves `102/120 (85%)`, while SHAP reaches `120/120`. However, under our explicit R/G aggregation rule, SHAP’s perfect score is not evidence of better confirmation. Instead, the benchmark encoding mechanically induces a proportional relationship between color and structure attribution scores, causing SHAP to continue reporting color dependence even when the oracle has shifted toward structure. A paired McNemar test still shows a significant difference (`p = 7.63 x 10^-6`).

The point here is not that SHAP is “bad” in general, but that attribution scores can be dominated by encoding-aggregation interactions that are orthogonal to the operational question of what breaks performance.

| Setting | PerformanceCheck | SHAP | Paired test |
| --- | --- | --- | --- |
| GridWorld top-1 | `85/120 (71%)` | `14/120 (12%)` | McNemar `p = 8.38 x 10^-16` |
| GridWorld top-2 | `106/120 (88%)` | `21/120 (18%)` | McNemar `p = 2.04 x 10^-4` |
| ColoredMNIST | `102/120 (85%)` | `120/120 (100%)` | McNemar `p = 7.63 x 10^-6` |
| Waterbirds | `18/30 (60%)` | `30/30 (100%)` | sign gap `p = 1.86 x 10^-9`* |

Table 2: Headline comparisons. `*` In Waterbirds, the significance test is on repeated-intervention mean gaps `bg_mean_delta - fg_mean_delta`, while `18/30` is the single-seed ranking outcome.

![Figure 2: ColoredMNIST comparison. Under the aggregation rule used here, SHAP’s perfect score follows the encoding structure, while `PerformanceCheck` tracks the oracle’s shift toward structure dependence at `p = 0.3`.](experiments/results/exp017_cmnist_shap_psweep_30seeds/fig_psweep.png)

### 5.3 Q3: How do candidate detection and confirmation interact in Waterbirds?

In Waterbirds, SHAP flags background in all `30/30` seeds, while `PerformanceCheck` ranks background above foreground in `18/30` seeds. This means both methods agree that background is a suspicious candidate, but they do not assign the same strength of functional reliance. `PerformanceCheck` is more selective.

This difference becomes clearer in the specialized-baseline comparison. Representation-space proxies such as SpecScore and PCA consistently indicate background as a suspicious component, while `PerformanceCheck` distinguishes seeds in which background perturbation harms performance more strongly from those in which it does not dominate foreground perturbation. The average gap `bg_mean_delta - fg_mean_delta` is positive in all 30 seeds under repeated interventions, but the single-seed ranking outcome still varies, which is exactly why the confirmation step is informative.

![Figure 3: Waterbirds comparison. Background is consistently flagged as a suspicious candidate, but intervention-based confirmation shows variation in how strongly that background actually affects performance.](experiments/results/exp018_waterbirds_30seeds/fig_scatter.png)

### 5.4 Robustness to Mask Quality

We also evaluate mask sensitivity in Waterbirds. A single-seed local sweep shows that small erosions and dilations preserve the foreground-vs-background ordering. A 30-seed radius sweep shows that mean `Δ_bg` and `Δ_fg` change continuously with mask radius rather than collapsing abruptly near the reference setting. This supports the claim that the result is not an artifact of one arbitrary segmentation boundary.

![Figure 4: Waterbirds mask sensitivity. Small mask perturbations preserve the foreground-versus-background ordering, and the 30-seed radius sweep changes continuously rather than collapsing abruptly.](experiments/results/exp020_waterbirds_mask_sensitivity/fig_mask_sensitivity.png)

## 6. Discussion

The main practical role of `PerformanceCheck` is not to declare a final truth label for each feature. Its role is to help prioritize remediation. A useful workflow is: `(1)` generate suspicious candidates using discovery methods or attribution; `(2)` run `PerformanceCheck` to confirm which of those candidates actually damage performance when perturbed; `(3)` prioritize augmentation, data collection, or retraining using the highest-`Δ` candidates.

This is where the method connects naturally to prior work. Candidate-discovery methods, or earlier shift-diagnosis frameworks, can provide the front end of the pipeline. `PerformanceCheck` then supplies a model-specific confirmation signal that is closer to the deployment question: which candidate, if broken, most harms predictive performance?

Our results also clarify how to interpret disagreement. In ColoredMNIST, disagreement reveals an aggregation artifact. In Waterbirds, disagreement reflects a meaningful distinction between “candidate existence” and “functional strength.” In this sense, attribution and `PerformanceCheck` are best read as complementary rather than mutually exclusive.

## 7. Limitations

Our method depends on intervention quality. When candidate features are not spatially localized or otherwise isolable, it may be difficult to design interventions that destroy the target while preserving the rest of the input. High-dimensional settings also raise computational costs because a full scan scales roughly with the number of candidates times the number of evaluation samples. In such regimes, candidate pruning before intervention becomes useful, but this reintroduces dependence on the candidate generator.

Our ColoredMNIST finding is specific to the aggregation rule used in this paper and should not be overgeneralized to SHAP as a whole. Our Waterbirds results are based on a frozen ResNet-18 setup with validation accuracy roughly between `0.719` and `0.825`; stronger backbones may change the absolute rates. Finally, the present paper validates the confirmation step, not a fully automated discovery-to-remediation system.

## 8. Conclusion

We introduced `PerformanceCheck`, a post-hoc confirmation procedure that tests whether discovered spurious candidates actually matter for predictive performance. Across GridWorld, ColoredMNIST, and Waterbirds, intervention-based performance sensitivity provides a useful signal that differs from, and often complements, attribution-based candidate discovery. The method is especially valuable when the downstream goal is not only to explain a model, but to decide which suspicious features should be addressed first.
