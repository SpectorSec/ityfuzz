use std::collections::HashMap;

use libafl_bolts::{
    impl_serdeany,
    rands::{Rand, StdRand},
};
use serde::{Deserialize, Serialize};

use crate::evm::{
    abi::{ABIAddressToInstanceMap, BoxedABI},
    input::{CampaignSequence, ConciseEVMInput, EVMInputTy, StepLinkage},
    leak_class::LeakClass,
    middlewares::cmp_linearity::TaintDim,
    guidance::{Guidance, GuidanceMetadata},
    types::{EVMAddress, EVMU256},
};

/// Vault/prime selectors: functions that accept assets and change protocol
/// state.
const PRIME_SELECTORS: &[[u8; 4]] = &[
    [0x47, 0xe7, 0xef, 0x34], // receiveWithPermit
    [0x6e, 0x55, 0x3f, 0x65], // deposit(uint256)
    [0xaa, 0x45, 0xde, 0x31], // mint
    [0x36, 0x63, 0x09, 0xb5], // stake
    [0xa3, 0x14, 0x6b, 0xd2], // addLiquidity
    [0x02, 0x2c, 0x0d, 0x9f], // deposit
];

/// Exploit trigger selectors: functions that extract value or manipulate state.
const EXPLOIT_SELECTORS: &[[u8; 4]] = &[
    [0x44, 0x1a, 0x3e, 0x70], // withdraw(uint256)
    [0xdb, 0x00, 0x6b, 0x75], // redeem
    [0x4e, 0x71, 0xd9, 0x2d], // sync
    [0xa9, 0x05, 0x9c, 0xbb], // liquidate(address,uint256,address)
    [0x4e, 0x84, 0x73, 0xcb], // skim
    [0x85, 0x38, 0x28, 0xb6], // donate
];

/// Function-NAME substrings that indicate a trigger/exploit function. TESTING
/// ONLY — gated behind `campaign_generic_fallback`, off by default. Name
/// matching is NOT machine truth: substrings also hit getters (`claimable`,
/// `withdrawable`) and miss attacker-renamed functions. Production stays on
/// exact-selector truth.
#[cfg(feature = "campaign_generic_fallback")]
const EXPLOIT_NAME_PATTERNS: &[&str] = &[
    "withdraw",
    "redeem",
    "claim",
    "harvest",
    "exit",
    "unstake",
    "unlock",
    "collect",
    "skim",
    "sync",
    "liquidate",
    "drain",
    "payout",
    "sweep",
    "borrow",
    "cashout",
    "release",
    "settle",
];

/// Lowercased function name (portion before `(`) from the global signature
/// registry, if known. Returns `None` when signatures were not registered.
#[cfg(feature = "campaign_generic_fallback")]
fn fn_name_lc(abi: &BoxedABI) -> Option<String> {
    abi.get_func_signature()
        .map(|sig| sig.split('(').next().unwrap_or("").to_ascii_lowercase())
}

/// Pre-filtered campaign target cache, initialized once during corpus setup.
/// Replaces the O(N) ABI registry scan with an O(1) read-only lookup.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CampaignTargetCache {
    pub prime_targets: Vec<(EVMAddress, [u8; 4], BoxedABI)>,
    pub exploit_targets: Vec<(EVMAddress, [u8; 4], BoxedABI)>,
    pub borrowable_tokens: Vec<EVMAddress>,
    /// Fallback campaign targets: contracts the selector allowlist didn't match
    /// but that look campaignable (>= 2 functions incl. a trigger-named one).
    /// Each entry is (address, prime_fn_abi, exploit_fn_abi) — the exploit is
    /// the trigger-named function (pinned so the executor probe calls it),
    /// the prime is a different (benign) function. Forms a same-contract
    /// prime->exploit chain.
    pub generic_targets: Vec<(EVMAddress, Option<BoxedABI>, Option<BoxedABI>)>,
    /// Feature 015: contracts exposing a reflexive-skew liquidity primitive
    /// (`add_liquidity` / `remove_liquidity_imbalance`). Scanned independently
    /// of the prime/exploit allowlists so promotion can fire without
    /// polluting normal discovery; consulted ONLY on the
    /// `--reflexive-lever` path, so when the feature is off this
    /// field is computed once but never read — off-path behavior is
    /// byte-identical.
    #[serde(default)]
    pub reflexive_targets: Vec<(EVMAddress, [u8; 4], BoxedABI)>,
}

impl_serdeany!(CampaignTargetCache);

/// Feature 015 Phase 2 — per-step boundary offsets into the campaign's ordered
/// `erc20_transfers` log, written by the campaign executor when
/// `CampaignSequence.aposteriori` is set. `offsets[i]` is the length of the
/// transfer log BEFORE step `i` executed, with a trailing entry for the total
/// after the last step — so step `i`'s transfers are the slice
/// `erc20_transfers[offsets[i]..offsets[i+1]]`. This is the ONLY new
/// instrumentation the a-posteriori path needs: the atomic campaign's
/// staged-state chaining already accumulates the transfer log in order across
/// steps, so recording the offsets suffices to attribute an attacker-inflow
/// delta to the belly call that produced it. `offsets.len() == steps.len()+1`.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct CampaignInflowBoundaries {
    pub offsets: Vec<usize>,
}

impl_serdeany!(CampaignInflowBoundaries);

/// Feature 015 Phase 2 — the ledger-moving belly call discovered a-posteriori.
/// The feedback attributes per-step attacker inflow via
/// `CampaignInflowBoundaries`, and records the single highest-inflow step here
/// (one lever/frame — protects the 3.5GB ceiling against over-promotion). The
/// mutator reads this and pins the matching campaign step into
/// `CampaignSequence.promoted` so Locate+Amplify (the ledger-secant) tunes it.
/// Keyed by `(contract, selector)` so the pin re-fires whenever that call
/// recurs in a freshly sampled campaign, despite clone-per-iteration corpus
/// semantics. `best_inflow` is a high-water mark: only a strictly larger delta
/// replaces the incumbent candidate.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct PromotionCandidate {
    pub contract: EVMAddress,
    pub selector: [u8; 4],
    pub best_inflow: u128,
    /// Feature 020 — WHY this candidate was promoted. The a-posteriori path is
    /// value-inflow, so it records `Value`; 019-C, when it lands, records
    /// `Permission` for a routed permission leak. `#[serde(default)]` →
    /// pre-020 corpora (no `kind`) deserialize as `Value` (Default), keeping
    /// serialized-corpus round-trip byte-compatible.
    #[serde(default)]
    pub kind: LeakClass,
    /// Feature 021 (Taint↔Promotion Weld) — PROOF the candidate was
    /// attacker-delivered rather than a source-less state change. The
    /// a-posteriori producer stamps the taint verdict for the
    /// promoted step's execution. A candidate only exists when the weld's taint
    /// half passed (or the analysis did not run — fail-open), so this is
    /// the causal receipt that survives into the corpus for the
    /// mutator/telemetry to read. `#[serde(default)]` keeps pre-021 corpora
    /// byte-compatible (deserialize as the all-false / `Generic` default).
    #[serde(default)]
    pub taint_provenance: TaintProvenanceTag,
    /// Feature 023 — the campaign step (phase) this candidate fired at. Value
    /// path: the `best_inflow_step` idx. Structural path:
    /// `FunctionAuthData.material_at_step[contract]`. The shared `(product,
    /// vuln, phase)` coordinate the kind-aware mutator pins on.
    /// `#[serde(default)]` (None) keeps pre-023 corpora byte-compatible.
    #[serde(default)]
    pub phase: Option<usize>,
    pub set: bool,
}

impl_serdeany!(PromotionCandidate);

/// Per-leak-class promotion slots. A singleton `PromotionCandidate` is too
/// small once multiple closed-loop producers can fire in the same run:
/// Permission/Ownership structural pins and Value/Invariant lever pins should
/// not starve each other merely because one oracle reported first. Each class
/// keeps its own high-water candidate, and consumers pick the class family they
/// need (structural vs. lever) at read time.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct PromotionCandidates {
    #[serde(default)]
    pub by_kind: HashMap<LeakClass, PromotionCandidate>,
}

impl_serdeany!(PromotionCandidates);

impl PromotionCandidates {
    pub fn record(&mut self, candidate: PromotionCandidate) -> bool {
        if !candidate.set {
            return false;
        }
        match self.by_kind.get(&candidate.kind) {
            Some(existing) if existing.set && existing.best_inflow >= candidate.best_inflow => false,
            _ => {
                self.by_kind.insert(candidate.kind, candidate);
                true
            }
        }
    }

    pub fn get(&self, kind: LeakClass) -> Option<&PromotionCandidate> {
        self.by_kind.get(&kind).filter(|candidate| candidate.set)
    }

    pub fn first_set(&self, kinds: &[LeakClass]) -> Option<&PromotionCandidate> {
        kinds.iter().find_map(|kind| self.get(*kind))
    }

    pub fn from_singleton(candidate: &PromotionCandidate) -> Self {
        let mut candidates = PromotionCandidates::default();
        candidates.record(candidate.clone());
        candidates
    }
}

/// Feature 021 — the causal-provenance receipt welded onto a
/// `PromotionCandidate`. Deliberately minimal and serializable so it
/// round-trips in the corpus and is assertable in unit tests without a live
/// taint reexecution. Semantics:
/// - `causally_linked`: attacker input was tied to this candidate's execution
///   (value-confirmed storage provenance, or a sink-cleared tainted call) — see
///   `cmp_linearity::injection_causal_link_confirmed`.
/// - `analysis_ran`: whether the taint reexecution actually ran; when `false`,
///   `causally_linked` is the fail-open default (`true`) and carries no
///   evidentiary weight.
/// - `dim`: the located economic dimension of the taint (Price / Balance /
///   Accumulator / Generic), the latent hook for the delivery-archetype
///   taxonomy.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct TaintProvenanceTag {
    pub causally_linked: bool,
    pub analysis_ran: bool,
    pub dim: TaintDim,
}

impl CampaignTargetCache {
    /// Build the cache from the ABI registry by scanning for known selector
    /// patterns. Delegates with no preset selectors → hardcoded
    /// PRIME/EXPLOIT fallback.
    pub fn new(
        abi_map: &ABIAddressToInstanceMap,
        borrowable_tokens: Vec<EVMAddress>,
    ) -> Self {
        Self::new_with_preset(abi_map, borrowable_tokens, &[], None, &[])
    }

    /// Candidate-based target discovery. When `preset_selectors` is non-empty
    /// (a preset matched the target), the prime/exploit chain candidates
    /// are drawn from the matched EXPLOIT'S OWN vocabulary — so the
    /// campaign chains what THIS exploit actually uses, adapting to the
    /// target instead of hunting a hardcoded function menu the exploit's
    /// selectors may fall entirely outside of. Empty → falls back to the
    /// hardcoded PRIME/EXPLOIT_SELECTORS. This removes the hidden
    /// candidate-bias at discovery: the same "candidates, not a fixed
    /// prior" principle the preset system already uses.
    ///
    /// `guidance_prime_extra` — selectors extracted from guidance kill-chain
    /// entries (functions that reach a drain sink). Unioned with PRIME_SELECTORS
    /// in the fallback branch so the guidance dictionary drives discovery instead
    /// of the 6-entry hardcoded list. Ignored when preset_selectors is non-empty
    /// (preset already supplies the full candidate vocabulary).
    pub fn new_with_preset(
        abi_map: &ABIAddressToInstanceMap,
        borrowable_tokens: Vec<EVMAddress>,
        preset_selectors: &[[u8; 4]],
        _contract_families: Option<()>,
        guidance_prime_extra: &[[u8; 4]],
    ) -> Self {
        let (prime_sels, exploit_sels): (&[[u8; 4]], &[[u8; 4]]) = if !preset_selectors.is_empty() {
            // Every matched-exploit selector is a candidate for both ends of the chain;
            // pick_prime_and_exploit (value-flow-aware) then orders them.
            (preset_selectors, preset_selectors)
        } else {
            (PRIME_SELECTORS, EXPLOIT_SELECTORS)
        };

        // In the fallback branch, union guidance-derived primes with the hardcoded list
        // so the kill-chain dictionary drives discovery, not just a 6-entry constant.
        let combined_primes: Vec<[u8; 4]>;
        let prime_sels = if preset_selectors.is_empty() && !guidance_prime_extra.is_empty() {
            let mut seen = std::collections::HashSet::new();
            combined_primes = prime_sels.iter().chain(guidance_prime_extra.iter())
                .filter(|s| seen.insert(**s))
                .copied()
                .collect();
            &combined_primes[..]
        } else {
            prime_sels
        };

        let mut prime_targets = find_targets_by_selector(abi_map, prime_sels);
        let mut exploit_targets = find_targets_by_selector(abi_map, exploit_sels);
        let mut reflexive_targets = find_targets_by_selector(abi_map, REFLEXIVE_LEVER_SELECTORS);



        Self {
            prime_targets,
            exploit_targets,
            borrowable_tokens,
            generic_targets: find_generic_targets(abi_map),
            // Feature 015: independent scan for the data-mined reflexive-lever catalogue
            // (Curve skew levers + lending-fork rate-warpers), regardless of what the
            // prime/exploit allowlists matched.
            reflexive_targets,
        }
    }

    /// Returns true if this cache has enough targets to form at least a 2-step
    /// campaign.
    pub fn is_viable(&self) -> bool {
        (!self.prime_targets.is_empty() && !self.exploit_targets.is_empty()) || !self.generic_targets.is_empty()
    }
}

/// Deterministic state machine that builds a multi-step campaign sequence.
///
/// Uses the pre-filtered `CampaignTargetCache` for O(1) target selection.
/// When `topology_report` is provided, exploit classes ranked by the topology
/// engine are used to prioritize targets (e.g., preferring same-contract
/// prime/exploit pairs for vault-like vulnerability patterns).
///
/// Builds one of:
///   - Borrow → ABI(prime) → ABI(exploit)  (when borrowable tokens available)
///   - ABI(prime) → ABI(exploit)            (no borrow step, still useful for
///     state chaining)
///
/// # Returns
/// `Some(CampaignSequence)` if a viable 2+ step chain was constructed,
/// `None` if insufficient targets were found.
pub fn plan_campaign(
    cache: &CampaignTargetCache,
    guidance_meta: Option<&GuidanceMetadata>,
    temporal_skimming: bool,
) -> Option<CampaignSequence> {
    // Deterministic entry (tests / no live fuzzer RNG): seed a fixed local RNG.
    // With single-candidate pools the sampled draw resolves to the sole element,
    // so structure is stable; production goes through the value-flow entry below
    // with the live `state.rand_mut()`.
    let mut rand = StdRand::with_seed(0xC0FFEE);
    // Deterministic/test entry keeps reflexive promotion OFF; the live fuzzer path
    // (mutator) passes `self.effective_reflexive`.
    plan_campaign_sampled(
        cache,
        guidance_meta,
        temporal_skimming,
        false,
        false,
        None,
        None,
        None,
        None,
        &mut rand,
    )
}

/// Feature 015 selectors of reflexive-skew liquidity primitives.
/// `add_liquidity(uint256[N],uint256)` = 0x4515cef3 (Curve StableSwap 3-pool
/// form); `remove_liquidity_imbalance(uint256[N],uint256)` = 0x9fdaea0c. Kept
/// as named references (tests + the canonical yDAI lever) and as the two
/// highest-priority entries of `REFLEXIVE_LEVER_SELECTORS`.
const SEL_ADD_LIQUIDITY: [u8; 4] = [0x45, 0x15, 0xce, 0xf3];
const SEL_REMOVE_LIQUIDITY_IMBALANCE: [u8; 4] = [0x9f, 0xda, 0xea, 0x0c];

/// Feature 015 — the reflexive-lever catalogue, data-mined from `calls.db`
/// across the 57 cross-step reflexive incidents (see
/// `.speckit/research/reflexive-lever-corpus-mining.md`). These are the
/// attacker state-WARPERS: a mutating call whose write is consumed by a
/// value-gating read (`get_virtual_price` / `exchangeRate` / `pricePerShare`) a
/// few steps later. Ordered by promote priority — the Curve StableSwap skew
/// levers first (the canonical yDAI family), then the lending-fork rate-warpers
/// (Compound/Aave) that dominate the mined set. Pure positioners
/// (`mint`/`enterMarkets`) are deliberately EXCLUDED: they open the position a
/// later warper exploits, so they belong to the prime step, not the promoted
/// lever. The independent scan populates `reflexive_targets` from whichever of
/// these the harvested vocabulary exposes; `maybe_promote_lever` hoists the
/// first match in this order.
const REFLEXIVE_LEVER_SELECTORS: &[[u8; 4]] = &[
    // Curve StableSwap skew levers (canonical yDAI family; highest promote priority)
    SEL_ADD_LIQUIDITY,              // add_liquidity(uint256[3],uint256) = 0x4515cef3
    [0x0b, 0x4c, 0x7e, 0x4d],       // add_liquidity(uint256[2],uint256)
    [0x02, 0x9b, 0x2f, 0x34],       // add_liquidity(uint256[4],uint256)
    SEL_REMOVE_LIQUIDITY_IMBALANCE, // remove_liquidity_imbalance(uint256[3],uint256) = 0x9fdaea0c
    [0xe3, 0x10, 0x32, 0x73],       // remove_liquidity_imbalance(uint256[2],uint256)
    [0x18, 0xa7, 0xbd, 0x76],       // remove_liquidity_imbalance(uint256[4],uint256)
    [0x1a, 0x4d, 0x01, 0xd2],       // remove_liquidity_one_coin(uint256,int128,uint256)
    [0x3d, 0xf0, 0x21, 0x24],       // exchange(int128,int128,uint256,uint256)
    [0xa6, 0x41, 0x7e, 0xd6],       // exchange_underlying(int128,int128,uint256,uint256)
    // Lending-fork rate-warpers (Compound/Aave; the mined generalization)
    [0xc5, 0xeb, 0xea, 0xec], // borrow(uint256)                          Compound
    [0xea, 0xc5, 0xb6, 0xe1], // borrow(uint256,uint256,uint256,uint16,address) Aave
    [0xdb, 0x00, 0x6a, 0x75], // redeem(uint256)
    [0x85, 0x2a, 0x12, 0xe3], // redeemUnderlying(uint256)
    [0xf5, 0xe3, 0xc4, 0x62], // liquidateBorrow(address,uint256,address)
    [0x0e, 0x75, 0x27, 0x02], // repayBorrow(uint256)
    [0x37, 0x1f, 0xd8, 0xe6], // repay(uint256)
    [0x57, 0x3a, 0xde, 0x81], // repay(address,uint256,uint256,address)    Aave
];

/// Feature 015 — a-priori Promote. If the harvested vocabulary (target cache)
/// contains a reflexive-skew liquidity primitive, return a pinned lever step
/// for it. `add_liquidity` is the primary skew lever (it moves the pool balance
/// the vault reads); we fall back to `remove_liquidity_imbalance`. Keyed on
/// selector presence so it fires on both the preset path (selectors seeded into
/// the cache) and the onchain path (harvested ABIs).
fn maybe_promote_lever(cache: &CampaignTargetCache) -> Option<ConciseEVMInput> {
    // Hoist the highest-priority warper the target actually exposes: Curve skew
    // levers first (canonical yDAI), then the lending-fork rate-warpers.
    // Priority order lives in REFLEXIVE_LEVER_SELECTORS so the a-priori promote
    // and the scan stay in lockstep.
    for want in REFLEXIVE_LEVER_SELECTORS.iter().copied() {
        if let Some((addr, _sel, abi)) = cache.reflexive_targets.iter().find(|(_, sel, _)| *sel == want) {
            return Some(build_abi_step(*addr, Some(abi.clone())));
        }
    }
    None
}

/// Feature 024 (post-hoc → planner socket) — build a step for a structural pin
/// `(contract, selector)` by finding its ABI in the harvested cache
/// (prime/exploit/reflexive pools). Mirrors `maybe_promote_lever`. `None` if
/// the selector wasn't harvested for that contract (then the structural move
/// can't be seeded — the sampler must find it, as today).
fn build_structural_step(
    cache: &CampaignTargetCache,
    contract: EVMAddress,
    selector: [u8; 4],
) -> Option<ConciseEVMInput> {
    cache
        .prime_targets
        .iter()
        .chain(cache.exploit_targets.iter())
        .chain(cache.reflexive_targets.iter())
        .find(|(a, s, _)| *a == contract && *s == selector)
        .map(|(a, _, abi)| build_abi_step(*a, Some(abi.clone())))
}

/// Structural-sampling campaign planner. The planner's ONLY job is to propose
/// an atomic frame (borrow → sampled prime → sampled exploit) by drawing
/// uniformly from the harvested contract vocabulary, `get_next_call`-style. It
/// deliberately does NOT consult any per-selector "value-flow" signal: the
/// authoritative economic feedback is the primitive net-realized ledger
/// (`flashloan_data.earned/owed`, `net_realized()` in feedbacks.rs) that
/// already gates the objective/fitness layer. The planner PROPOSES structure;
/// the machine-primitive ledger DISPOSES — monotonic filtering keeps only
/// sampled sequences that yield genuine token/ETH inflows. A prior version
/// anchored the prime on `observed_values` (a syntactic ABI-return pool), which
/// read `approve`'s `bool true` as profit and collapsed chains toward
/// `approve → approve`. That proxy is removed; the ledger is the single source
/// of economic truth.
pub fn plan_campaign_sampled<R: Rand>(
    cache: &CampaignTargetCache,
    guidance_meta: Option<&GuidanceMetadata>,
    temporal_skimming: bool,
    effective_reflexive: bool,
    dimension_warp: bool,
    // Feature 024 — post-hoc → planner socket: the structural (permission-leak / ownership)
    // candidate's (contract, selector) to RE-SEED into the sampled campaign so the exploit is
    // reliably reached each iteration. None on the value/a-priori paths ⇒ byte-identical behavior.
    structural_pin: Option<(EVMAddress, [u8; 4])>,
    // Feature 031 — runtime-discovered Value lever: the PromotionCandidate's (contract, selector)
    // when kind==Value. Injected as the Lever step (before exploit, marked `promoted`) with higher
    // priority than the static REFLEXIVE_LEVER_SELECTORS list. None ⇒ byte-identical behavior.
    value_lever_pin: Option<(EVMAddress, [u8; 4])>,
    // §7d content re-point: a topology-classified capital-source contract for the Borrow slot
    // (from per-contract families), preferred over the blind first borrowable. None ⇒ old behavior.
    borrow_authority: Option<EVMAddress>,
    // Feature 029 Phase 2 — divergence pin: the secant-converged `x` value to pre-load on the
    // first non-borrow step's `txn_value`. When set, the campaign starts at the divergence-peaked
    // txn_value instead of re-learning it from scratch. None ⇒ byte-identical behavior.
    divergence_value: Option<u128>,
    rand: &mut R,
) -> Option<CampaignSequence> {
    let mut steps: Vec<ConciseEVMInput> = Vec::new();
    // Feature 015: indices of promoted reflexive-skew lever steps.
    let mut promoted: Vec<usize> = Vec::new();

    // Step 0 (optional): Borrow step — acquire capital via flashloan
    // §7d content re-point: prefer a topology-classified capital-source token
    // (Borrow authority) over the blind first borrowable; falls back to
    // `.first()` when topology has no opinion.
    let borrow_tok = borrow_authority.or_else(|| cache.borrowable_tokens.first().copied());
    if let Some(token_addr) = borrow_tok {
        steps.push(build_borrow_step(token_addr));
    }

    // Populate prime + exploit steps (with concrete function ABIs), respecting
    // hints
    let (prime_step, exploit_step) = pick_prime_and_exploit(cache, guidance_meta, rand);
    let prime_step_clone = prime_step.clone();
    let exploit_step_clone = exploit_step.clone();
    if let Some((addr, abi)) = prime_step {
        steps.push(build_abi_step(addr, abi));
    }
    // Feature 024 — structural Prime pin (Permission/Ownership). Must land BEFORE
    // the Lever so the assembled sequence respects BPLE order: Borrow → Prime →
    // Lever → Exploit. "Held" by re-planning: the persistent structural
    // candidate re-seeds every iteration, mirroring how `promoted` is
    // re-derived. Skipped if already present, or if the selector isn't in the
    // cache. NOT added to `promoted` ⇒ secant never amplifies it.
    if let Some((sc, ssel)) = structural_pin {
        let present = steps
            .iter()
            .any(|st| st.contract == sc && st.data.as_ref().map(|d| d.function).unwrap_or_default() == ssel);
        if !present {
            if let Some(step) = build_structural_step(cache, sc, ssel) {
                steps.push(step);
            }
        }
    }

    // Feature 031 — dynamic Value lever (unconditional, not gated on
    // effective_reflexive). Runtime ground truth: the oracle found exactly
    // which selector is the lever on THIS target. Covers all Value-kind lever
    // types (reflexive-price, donation/sync, ERC4626 share-price, novel
    // protocols) — not just the 14 known selectors below.
    let dynamic_fired = if let Some((vc, vsel)) = value_lever_pin {
        if let Some(lever) = build_structural_step(cache, vc, vsel) {
            promoted.push(steps.len());
            steps.push(lever);
            true
        } else {
            false
        }
    } else {
        false
    };
    // Feature 015 — cold-start fallback: static reflexive-price list
    // (Curve/Compound/Aave). Fires ONLY when no runtime candidate exists yet
    // AND --reflexive-lever was passed. For known Curve-family targets this
    // gives an immediate head start before runtime discovery produces a
    // candidate. For novel protocols returns None → runs flat until the dynamic
    // path fills in. NOT a general lever system — bootstrap for one pattern.
    if !dynamic_fired && effective_reflexive {
        if let Some(lever) = maybe_promote_lever(cache) {
            promoted.push(steps.len());
            steps.push(lever);
        }
    }
    // Feature 015 Phase 2 (a-posteriori Promote): on the reflexive path, if NO
    // a-priori lever matched (the target exposes no registered reflexive
    // primitive), arm the executor to record per-step attacker-inflow
    // boundaries so the feedback can discover the ledger-moving belly call at
    // runtime. One lever/frame: only arm when `promoted` is still empty. Off
    // the reflexive path this stays `false` ⇒ no executor overhead.
    let aposteriori = effective_reflexive && promoted.is_empty();

    // Feature 029 Phase 2 — divergence pin: apply the secant-converged txn_value to
    // the first non-borrow step so Phase 3 starts at the divergence peak
    // instead of re-learning it from scratch. Skipped when divergence_pin is
    // None (default/old behavior).
    if let Some(dval) = divergence_value {
        if let Some(first_charge) = steps.iter_mut().find(|s| s.input_type != EVMInputTy::Borrow) {
            first_charge.txn_value = Some(EVMU256::from(dval));
        }
    }

    if let Some((addr, abi)) = exploit_step {
        steps.push(build_abi_step(addr, abi));
    }

    // Minimum viable campaign: at least 2 steps
    if steps.len() < 2 {
        return None;
    }

    // Temporal Pre-condition Skimming: Insert a warp (block advance) between
    // the prime step (state priming) and the exploit step. The warp simulates
    // block progression during which state divergence (interest accrual, reward
    // accumulation, oracle price changes) can occur off-screen.
    let mut warps: Vec<(usize, u64)> = Vec::new();
    // Feature 017 Wire B: engage warp when dimension-driven (timestamp taint
    // reached SSTORE) OR flag-driven. dimension_warp gates the static read so
    // the feature is off by default.
    let ts_located = dimension_warp;
    if temporal_skimming || ts_located {
        // The exploit step is always the last step. Insert warp before it.
        // Index is steps.len() - 1 (0-indexed).
        let exploit_idx = steps.len() - 1;
        // Default warp: 10 blocks (~2 minutes). Sufficient to trigger most
        // reward-accrual and timelock-based divergence patterns.
        warps.push((exploit_idx, 10));
    }

    let mut linkages = Vec::new();
    let mut prime_idx = None;
    let mut exploit_idx = None;

    if let Some((prime_addr, Some(ref p_abi))) = prime_step_clone {
        for (i, step) in steps.iter().enumerate() {
            if step.contract == prime_addr && step.data.as_ref().map(|d| d.function) == Some(p_abi.function) {
                prime_idx = Some(i);
                break;
            }
        }
    }

    if let Some((exploit_addr, Some(ref e_abi))) = exploit_step_clone {
        for (i, step) in steps.iter().enumerate() {
            if step.contract == exploit_addr && step.data.as_ref().map(|d| d.function) == Some(e_abi.function) {
                exploit_idx = Some(i);
                break;
            }
        }
    }

    if let Some(p_idx) = prime_idx {
        if let Some(e_idx) = exploit_idx {
            if p_idx < e_idx {
                if let Some((prime_addr, Some(ref p_abi))) = prime_step_clone {
                    let from_registry_key = format!("{:?}_{}_return", prime_addr, hex::encode(p_abi.function));
                    linkages.push(StepLinkage {
                        from_step: p_idx,
                        from_registry_key,
                        to_step: e_idx,
                        to_param_index: 0,
                    });
                }
            }
        }
    }

    Some(CampaignSequence {
        steps,
        linkages,
        warps,
        promoted,
        aposteriori,
    })
}

/// Pick prime and exploit target addresses, using topology intelligence
/// to prefer same-contract pairs when the top-ranked exploit class
/// suggests a single-contract vulnerability (ERC-4626 vaults, staking
/// pools, etc.).
type PickedStep = Option<(EVMAddress, Option<BoxedABI>)>;

fn pick_prime_and_exploit<R: Rand>(
    cache: &CampaignTargetCache,
    guidance_meta: Option<&GuidanceMetadata>,
    rand: &mut R,
) -> (PickedStep, PickedStep) {
    // 1. Uniform selector-level sample for the prime step
    let prime = sample_by_selector(&cache.prime_targets, rand).map(|(a, _, abi)| (*a, Some(abi.clone())));

    // 2. If a prime step was selected and guidance is available, try to pick a matching next candidate as exploit step
    if let Some((prime_addr, Some(prime_abi))) = &prime {
        let prime_name = unsafe {
            crate::evm::abi::FUNCTION_SIG.get(&prime_abi.function)
                .map(|sig| sig.split('(').next().unwrap_or("").trim().to_string())
        };
        if let (Some(p_name), Some(g_meta)) = (prime_name, guidance_meta) {
            let prime_contract_name = g_meta.addr_to_name.get(&format!("{:?}", prime_addr).to_lowercase());
            let lookup_key = match prime_contract_name {
                Some(c_name) => format!("{}:{}", c_name, p_name),
                None => p_name.clone(),
            };

            let next_candidates = g_meta.guidance.scheduler.next_candidates(&lookup_key);
            if !next_candidates.is_empty() {
                // Find all exploit targets matching any next candidate function name
                let mut guided_exploits = Vec::new();
                for (exp_addr, exp_sel, exp_abi) in &cache.exploit_targets {
                    if let Some(exp_name) = unsafe {
                        crate::evm::abi::FUNCTION_SIG.get(exp_sel)
                            .map(|sig| sig.split('(').next().unwrap_or("").trim().to_string())
                    } {
                        let exp_contract_name = g_meta.addr_to_name.get(&format!("{:?}", exp_addr).to_lowercase());
                        let exp_key = match exp_contract_name {
                            Some(c_name) => format!("{}:{}", c_name, exp_name),
                            None => exp_name.clone(),
                        };

                        if next_candidates.contains(&exp_key) || next_candidates.contains(&exp_name) {
                            guided_exploits.push((*exp_addr, Some(exp_abi.clone())));
                        }
                    }
                }
                if !guided_exploits.is_empty() {
                    if let Some(idx) = sample_idx(guided_exploits.len(), rand) {
                        return (Some((*prime_addr, Some(prime_abi.clone()))), Some(guided_exploits[idx].clone()));
                    }
                }
            }
        }
    }

    // 3. Fallback: Uniform selector-level sample for the exploit step
    let exploit = sample_by_selector(&cache.exploit_targets, rand).map(|(a, _, abi)| (*a, Some(abi.clone())));
    if prime.is_some() && exploit.is_some() {
        return (prime, exploit);
    }

    // 4. Second fallback: generic target matching
    if let Some(gi) = sample_idx(cache.generic_targets.len(), rand) {
        let (addr, prime_abi, exploit_abi) = &cache.generic_targets[gi];
        return (Some((*addr, prime_abi.clone())), Some((*addr, exploit_abi.clone())));
    }

    (prime, exploit)
}

/// Selector-level uniform sample: draw a distinct selector (vocabulary word)
/// uniformly from `targets`, then a contract carrying it. Mirrors fuzzland's
/// `get_next_call` (draw from the selector SET `interesting_signatures`, then
/// resolve a contract), so a selector present on many contracts is one word
/// with one vote — not weighted by contract-multiplicity. `None` when empty.
fn sample_by_selector<'a, R: Rand>(
    targets: &'a [(EVMAddress, [u8; 4], BoxedABI)],
    rand: &mut R,
) -> Option<&'a (EVMAddress, [u8; 4], BoxedABI)> {
    // Distinct selectors, insertion-order-stable (determinism under a fixed seed).
    let mut selectors: Vec<[u8; 4]> = Vec::new();
    for (_, sel, _) in targets {
        if !selectors.contains(sel) {
            selectors.push(*sel);
        }
    }
    let sel = *selectors.get(sample_idx(selectors.len(), rand)?)?;
    // Contracts carrying the chosen selector; pick one uniformly (keeps diversity).
    let carriers: Vec<usize> = targets
        .iter()
        .enumerate()
        .filter(|(_, (_, s, _))| *s == sel)
        .map(|(i, _)| i)
        .collect();
    Some(&targets[carriers[sample_idx(carriers.len(), rand)?]])
}

/// Uniform random index into a slice of `len` elements (get_next_call-style
/// draw). `None` when empty. The one primitive behind the campaign's candidate
/// sampling.
fn sample_idx<R: Rand>(len: usize, rand: &mut R) -> Option<usize> {
    if len == 0 {
        None
    } else {
        Some(rand.below(len as u64) as usize)
    }
}

/// Find all contracts whose ABI list includes any of the given selectors.
fn find_targets_by_selector(
    abi_map: &ABIAddressToInstanceMap,
    selectors: &[[u8; 4]],
) -> Vec<(EVMAddress, [u8; 4], BoxedABI)> {
    let mut results = Vec::new();
    for (addr, abis) in &abi_map.map {
        for abi in abis {
            if abi.function == [0u8; 4] {
                continue;
            }
            if selectors.contains(&abi.function) {
                results.push((*addr, abi.function, abi.clone()));
            }
        }
    }
    results
}

/// TESTING-ONLY name-heuristic fallback (gated behind
/// `campaign_generic_fallback`, off by default). When the feature is disabled
/// this returns empty, so the `generic_targets` field, `is_viable`, and the
/// `pick_prime_and_exploit` fallback all become no-ops and production stays
/// exact-selector (machine-truth) only.
///
/// When enabled: a contract with >= 2 functions AND >= 1 trigger-NAMED function
/// is treated as campaignable. Recognizes simple/novel staking/vault/timelock
/// fixtures the selector allowlist doesn't cover. NOT for production — see the
/// const above.
#[cfg(not(feature = "campaign_generic_fallback"))]
fn find_generic_targets(_abi_map: &ABIAddressToInstanceMap) -> Vec<(EVMAddress, Option<BoxedABI>, Option<BoxedABI>)> {
    Vec::new()
}

#[cfg(feature = "campaign_generic_fallback")]
fn find_generic_targets(abi_map: &ABIAddressToInstanceMap) -> Vec<(EVMAddress, Option<BoxedABI>, Option<BoxedABI>)> {
    let mut out = Vec::new();
    for (addr, abis) in &abi_map.map {
        let fns: Vec<&BoxedABI> = abis.iter().filter(|a| a.function != [0u8; 4]).collect();
        if fns.len() < 2 {
            continue;
        }
        // Exploit = first trigger-named function (pinned so the probe calls it).
        let exploit = fns.iter().find(|a| {
            fn_name_lc(a)
                .map(|n| EXPLOIT_NAME_PATTERNS.iter().any(|p| n.contains(p)))
                .unwrap_or(false)
        });
        let Some(exploit) = exploit else { continue };
        let exploit_sel = exploit.function;
        // Prime = first function that is NOT the exploit (benign setup step).
        let prime = fns.iter().find(|a| a.function != exploit_sel);
        out.push((*addr, prime.map(|a| (*a).clone()), Some((*exploit).clone())));
    }
    out
}

/// Build a Borrow step that acquires tokens via flashloan.
fn build_borrow_step(token: EVMAddress) -> ConciseEVMInput {
    ConciseEVMInput {
        input_type: EVMInputTy::Borrow,
        caller: EVMAddress::default(),
        contract: token,
        data: None,
        txn_value: Some(EVMU256::from(1_000_000_000_000_000_000u64)), // 1 ETH worth
        step: false,
        env: Default::default(),
        liquidation_percent: 0,
        randomness: vec![],
        repeat: 1,
        layer: 0,
        call_leak: u32::MAX,
        return_data: None,
        swap_data: HashMap::new(),
        nested_actions: Vec::new(),
        campaign: None,
    }
}

/// Build an ABI step for a target contract.
/// Parameters are resolved by the mutator's existing `mutate_with_vm_slots`
/// path.
#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::*;
    use crate::evm::{
        abi::{ABIAddressToInstanceMap, AEmpty, AUnknown, BoxedABI},
        input::EVMInputTy,
    };

    // Serializes tests that call `set_func_with_signature`, which writes the global
    // `static mut FUNCTION_SIG` HashMap — concurrent writes are a data race (UB).
    static FUNCTION_SIG_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn make_abi(selector: [u8; 4]) -> BoxedABI {
        let mut abi = BoxedABI::new(Box::new(AUnknown {
            concrete: BoxedABI::new(Box::new(AEmpty {})),
            size: 0,
        }));
        abi.set_func(selector);
        abi
    }

    #[test]
    fn test_cache_empty_returns_none() {
        let abi_map = ABIAddressToInstanceMap { map: HashMap::new() };
        let cache = CampaignTargetCache::new(&abi_map, Vec::new());
        assert!(!cache.is_viable());
        assert!(plan_campaign(&cache, None, false).is_none());
    }

    #[test]
    fn test_cache_prime_only_not_viable() {
        let mut map = HashMap::new();
        let addr = EVMAddress::default();
        map.insert(addr, vec![make_abi(PRIME_SELECTORS[0])]);
        let abi_map = ABIAddressToInstanceMap { map };
        let cache = CampaignTargetCache::new(&abi_map, Vec::new());
        assert!(!cache.is_viable());
        assert!(plan_campaign(&cache, None, false).is_none());
    }

    /// Simple/novel fixture (selectors NOT in the allowlist) is recognized via
    /// the name-based generic fallback: a contract with >=2 functions incl.
    /// a trigger-named one (`claimJackpot`) forms a same-contract 2-step
    /// campaign.
    #[cfg(feature = "campaign_generic_fallback")]
    #[test]
    fn test_generic_target_recognized_by_name() {
        let _g = FUNCTION_SIG_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let addr = EVMAddress::repeat_byte(0x11);
        let claim_sel = [0x11u8, 0x11, 0x11, 0x11];
        let dep_sel = [0x22u8, 0x22, 0x22, 0x22];
        let mut claim = make_abi(claim_sel);
        claim.set_func_with_signature(claim_sel, "claimJackpot", "()");
        let mut dep = make_abi(dep_sel);
        dep.set_func_with_signature(dep_sel, "deposit", "(uint256)");

        let mut map = HashMap::new();
        map.insert(addr, vec![claim, dep]);
        let abi_map = ABIAddressToInstanceMap { map };
        let cache = CampaignTargetCache::new(&abi_map, Vec::new());

        assert!(
            cache.prime_targets.is_empty(),
            "these selectors aren't in the allowlist"
        );
        assert!(cache.exploit_targets.is_empty());
        assert!(
            cache.generic_targets.iter().any(|(a, _, _)| *a == addr),
            "recognized via name fallback"
        );
        // The exploit step is pinned to the trigger function (claimJackpot).
        let (_, _, exploit_abi) = cache.generic_targets.iter().find(|(a, _, _)| *a == addr).unwrap();
        assert_eq!(
            exploit_abi.as_ref().map(|a| a.function),
            Some(claim_sel),
            "exploit step pinned to the trigger function's selector"
        );
        assert!(cache.is_viable());

        let campaign = plan_campaign(&cache, None, true).expect("generic fallback must yield a campaign");
        assert_eq!(campaign.steps.len(), 2, "single-contract prime->exploit 2-step chain");
        assert_eq!(campaign.warps.len(), 1, "temporal warp inserted before exploit step");
    }

    /// A contract with a trigger name but only ONE function is not
    /// campaignable.
    #[cfg(feature = "campaign_generic_fallback")]
    #[test]
    fn test_generic_single_function_not_viable() {
        let _g = FUNCTION_SIG_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let addr = EVMAddress::repeat_byte(0x33);
        let sel = [0x33u8, 0x33, 0x33, 0x33];
        let mut claim = make_abi(sel);
        claim.set_func_with_signature(sel, "claim", "()");
        let mut map = HashMap::new();
        map.insert(addr, vec![claim]);
        let abi_map = ABIAddressToInstanceMap { map };
        let cache = CampaignTargetCache::new(&abi_map, Vec::new());
        assert!(cache.generic_targets.is_empty(), "needs >= 2 functions");
    }

    #[test]
    fn test_plan_campaign_prime_and_exploit() {
        let mut map = HashMap::new();
        let prime_addr = EVMAddress::from([0x01; 20]);
        let exploit_addr = EVMAddress::from([0x02; 20]);
        map.insert(prime_addr, vec![make_abi(PRIME_SELECTORS[0])]);
        map.insert(exploit_addr, vec![make_abi(EXPLOIT_SELECTORS[0])]);
        let abi_map = ABIAddressToInstanceMap { map };
        let cache = CampaignTargetCache::new(&abi_map, Vec::new());
        assert!(cache.is_viable());
        let campaign = plan_campaign(&cache, None, false).expect("should produce campaign");
        assert_eq!(campaign.steps.len(), 2);
        assert_eq!(campaign.steps[0].input_type, EVMInputTy::ABI);
        assert_eq!(campaign.steps[0].contract, prime_addr);
        assert_eq!(campaign.steps[1].input_type, EVMInputTy::ABI);
        assert_eq!(campaign.steps[1].contract, exploit_addr);
    }

    #[test]
    fn test_plan_campaign_with_borrow() {
        let mut map = HashMap::new();
        let token = EVMAddress::from([0x03; 20]);
        let prime_addr = EVMAddress::from([0x01; 20]);
        let exploit_addr = EVMAddress::from([0x02; 20]);
        map.insert(prime_addr, vec![make_abi(PRIME_SELECTORS[0])]);
        map.insert(exploit_addr, vec![make_abi(EXPLOIT_SELECTORS[0])]);
        let abi_map = ABIAddressToInstanceMap { map };
        let cache = CampaignTargetCache::new(&abi_map, vec![token]);
        assert!(cache.is_viable());
        let campaign = plan_campaign(&cache, None, false).expect("should produce campaign");
        assert_eq!(campaign.steps.len(), 3);
        assert_eq!(campaign.steps[0].input_type, EVMInputTy::Borrow);
        assert_eq!(campaign.steps[0].contract, token);
        assert_eq!(campaign.steps[1].input_type, EVMInputTy::ABI);
        assert_eq!(campaign.steps[2].input_type, EVMInputTy::ABI);
    }


    #[test]
    fn test_planner_adds_warp_when_temporal_skimming_enabled() {
        let mut map = HashMap::new();
        let prime_addr = EVMAddress::from([0x01; 20]);
        let exploit_addr = EVMAddress::from([0x02; 20]);
        map.insert(prime_addr, vec![make_abi(PRIME_SELECTORS[0])]);
        map.insert(exploit_addr, vec![make_abi(EXPLOIT_SELECTORS[0])]);
        let abi_map = ABIAddressToInstanceMap { map };
        let cache = CampaignTargetCache::new(&abi_map, Vec::new());

        // temporal_skimming = true → should insert warp before exploit step
        let campaign = plan_campaign(&cache, None, true).expect("should produce campaign");
        assert_eq!(campaign.steps.len(), 2);
        assert_eq!(campaign.warps.len(), 1, "should have 1 warp entry");
        assert_eq!(campaign.warps[0].0, 1, "warp should be before exploit step (index 1)");
        assert_eq!(campaign.warps[0].1, 10, "warp should default to 10 blocks");
    }

    #[test]
    fn test_planner_no_warp_when_temporal_skimming_disabled() {
        let mut map = HashMap::new();
        let prime_addr = EVMAddress::from([0x01; 20]);
        let exploit_addr = EVMAddress::from([0x02; 20]);
        map.insert(prime_addr, vec![make_abi(PRIME_SELECTORS[0])]);
        map.insert(exploit_addr, vec![make_abi(EXPLOIT_SELECTORS[0])]);
        let abi_map = ABIAddressToInstanceMap { map };
        let cache = CampaignTargetCache::new(&abi_map, Vec::new());

        // temporal_skimming = false → no warps (backward compatible)
        let campaign = plan_campaign(&cache, None, false).expect("should produce campaign");
        assert!(campaign.warps.is_empty(), "no warps when temporal_skimming is disabled");
    }

    /// Feature 015 — the cheapest proof the Promote path works end-to-end: a
    /// yDAI-like fixture (prime + exploit + a Curve pool exposing
    /// `add_liquidity`) must, with `effective_reflexive=true`, hoist the
    /// lever into the frame and record its index in `promoted`, and the
    /// promoted step must carry the `add_liquidity` selector.
    #[test]
    fn test_effective_reflexive_promoted_into_frame() {
        let mut map = HashMap::new();
        let prime_addr = EVMAddress::from([0x01; 20]);
        let exploit_addr = EVMAddress::from([0x02; 20]);
        let pool_addr = EVMAddress::from([0x0c; 20]); // Curve pool (the skew lever host)
        map.insert(prime_addr, vec![make_abi(PRIME_SELECTORS[0])]);
        map.insert(exploit_addr, vec![make_abi(EXPLOIT_SELECTORS[0])]);
        map.insert(pool_addr, vec![make_abi(SEL_ADD_LIQUIDITY)]);
        let abi_map = ABIAddressToInstanceMap { map };
        let cache = CampaignTargetCache::new(&abi_map, Vec::new());

        // The independent scan finds the lever even though it's not in PRIME_SELECTORS.
        assert!(
            cache
                .reflexive_targets
                .iter()
                .any(|(a, s, _)| *a == pool_addr && *s == SEL_ADD_LIQUIDITY),
            "reflexive scan must discover the Curve pool's add_liquidity"
        );

        let mut rand = StdRand::with_seed(0xC0FFEE);
        let campaign = plan_campaign_sampled(&cache, None, false, true, false, None, None, None, None, &mut rand)
            .expect("viable prime+exploit → campaign");
        assert_eq!(campaign.promoted.len(), 1, "exactly one lever promoted");
        let lever_idx = campaign.promoted[0];
        let lever_sel = campaign.steps[lever_idx]
            .data
            .as_ref()
            .expect("promoted lever has a pinned ABI")
            .function;
        assert_eq!(lever_sel, SEL_ADD_LIQUIDITY, "promoted step is the add_liquidity lever");
        // The lever sits between prime and exploit (never last — the exploit reads
        // after it).
        assert!(lever_idx < campaign.steps.len() - 1, "lever precedes the exploit step");
    }

    /// Feature 015 generalization (corpus-mined): a lending-fork target
    /// exposing only a rate-warper (`borrow`, NO Curve pool present) is
    /// discovered by the expanded REFLEXIVE_LEVER_SELECTORS catalogue and
    /// promoted a-priori — proving the archetype covers the mined lending
    /// majority, not just Curve. a-posteriori stays disarmed.
    #[test]
    fn test_reflexive_lending_lever_promoted() {
        let mut map = HashMap::new();
        map.insert(EVMAddress::from([0x01; 20]), vec![make_abi(PRIME_SELECTORS[0])]);
        map.insert(EVMAddress::from([0x02; 20]), vec![make_abi(EXPLOIT_SELECTORS[0])]);
        let ctoken = EVMAddress::from([0x0d; 20]); // Compound cToken (the rate-warp host)
        let borrow_sel = [0xc5, 0xeb, 0xea, 0xec]; // borrow(uint256), NOT a Curve selector
        map.insert(ctoken, vec![make_abi(borrow_sel)]);
        let abi_map = ABIAddressToInstanceMap { map };
        let cache = CampaignTargetCache::new(&abi_map, Vec::new());

        assert!(
            cache
                .reflexive_targets
                .iter()
                .any(|(a, s, _)| *a == ctoken && *s == borrow_sel),
            "mined catalogue must discover the lending rate-warper (no Curve pool present)"
        );

        let mut rand = StdRand::with_seed(0xC0FFEE);
        let campaign = plan_campaign_sampled(&cache, None, false, true, false, None, None, None, None, &mut rand)
            .expect("viable prime+exploit → campaign");
        assert_eq!(campaign.promoted.len(), 1, "the lending lever is promoted");
        let sel = campaign.steps[campaign.promoted[0]]
            .data
            .as_ref()
            .expect("promoted lever has a pinned ABI")
            .function;
        assert_eq!(sel, borrow_sel, "promoted step is the borrow rate-warper");
        assert!(!campaign.aposteriori, "a-priori fired ⇒ a-posteriori disarmed");
    }

    /// Feature 024 — post-hoc → planner socket: a structural pin `(contract,
    /// selector)` whose ABI was harvested is re-seeded into the sampled
    /// campaign; an un-harvested selector yields no step (the sampler must
    /// find it, as today). Proves the socket the capstone named.
    #[test]
    fn structural_pin_seeds_step_into_plan() {
        let mut map = HashMap::new();
        let a = EVMAddress::from([0x01; 20]);
        let b = EVMAddress::from([0x02; 20]);
        map.insert(a, vec![make_abi(PRIME_SELECTORS[0])]);
        map.insert(b, vec![make_abi(EXPLOIT_SELECTORS[0])]);
        let abi_map = ABIAddressToInstanceMap { map };
        let cache = CampaignTargetCache::new(&abi_map, Vec::new());

        // The helper: harvested (contract, selector) → a step; un-harvested → None.
        let pin = (a, PRIME_SELECTORS[0]);
        assert!(
            build_structural_step(&cache, pin.0, pin.1).is_some(),
            "harvested (contract, selector) yields a structural step"
        );
        assert!(
            build_structural_step(&cache, a, [0xde, 0xad, 0xbe, 0xef]).is_none(),
            "un-harvested selector yields no step"
        );

        // The plan with the pin contains a step for it (byte-identical off the None
        // path).
        let mut rand = StdRand::with_seed(0x5EED);
        let campaign = plan_campaign_sampled(
            &cache,
            None,
            false,
            false,
            false,
            Some(pin),
            None,
            None,
            None,
            &mut rand,
        )
        .expect("viable prime+exploit → campaign");
        assert!(
            campaign
                .steps
                .iter()
                .any(|st| st.contract == pin.0 && st.data.as_ref().map(|d| d.function).unwrap_or_default() == pin.1),
            "structural_pin's (contract, selector) is present in the planned campaign"
        );
    }

    /// Feature 031 — dynamic Value lever: a runtime-discovered (contract,
    /// selector) passed as `value_lever_pin` is injected as the Lever step
    /// and marked `promoted`, WITHOUT requiring `--reflexive-lever`
    /// (effective_reflexive=false). Uses SEL_ADD_LIQUIDITY on a distinct
    /// lever address — the key property tested is flag-independence, not
    /// selector novelty. (Selectors not in any cache pool are skipped
    /// silently per spec; that is correct behaviour.)
    #[test]
    fn value_lever_pin_seeds_lever_step() {
        let mut map = HashMap::new();
        let prime_addr = EVMAddress::from([0x01; 20]);
        let exploit_addr = EVMAddress::from([0x02; 20]);
        let lever_addr = EVMAddress::from([0x0e; 20]); // distinct address from prime/exploit
        map.insert(prime_addr, vec![make_abi(PRIME_SELECTORS[0])]);
        map.insert(exploit_addr, vec![make_abi(EXPLOIT_SELECTORS[0])]);
        // SEL_ADD_LIQUIDITY on lever_addr → lands in reflexive_targets, findable by
        // build_structural_step. The lever_addr is distinct so the pin is unambiguous.
        map.insert(lever_addr, vec![make_abi(SEL_ADD_LIQUIDITY)]);
        let abi_map = ABIAddressToInstanceMap { map };
        let cache = CampaignTargetCache::new(&abi_map, Vec::new());

        let mut rand = StdRand::with_seed(0xBEEF);
        // effective_reflexive=false: static fallback is OFF. Dynamic pin must fire
        // alone.
        let campaign = plan_campaign_sampled(
            &cache,
            None,
            false,
            false,
            false,
            None,
            Some((lever_addr, SEL_ADD_LIQUIDITY)),
            None,
            None,
            &mut rand,
        )
        .expect("viable prime+exploit+lever → campaign");

        assert_eq!(campaign.promoted.len(), 1, "exactly one lever promoted");
        let lever_idx = campaign.promoted[0];
        let lever_sel = campaign.steps[lever_idx]
            .data
            .as_ref()
            .expect("promoted lever has ABI")
            .function;
        assert_eq!(lever_sel, SEL_ADD_LIQUIDITY, "dynamic pin is the promoted lever step");
        assert_eq!(
            campaign.steps[lever_idx].contract, lever_addr,
            "correct contract pinned"
        );
        // Lever must precede the exploit (always the last step).
        assert!(lever_idx < campaign.steps.len() - 1, "lever precedes exploit");
    }

    /// Feature 031 — static list fallback: with no value_lever_pin but
    /// effective_reflexive=true and a known reflexive selector in the
    /// cache, the cold-start fallback still fires. Proves the fallback is
    /// byte-identical to pre-031 behaviour on the reflexive path.
    #[test]
    fn value_lever_pin_none_falls_back_to_static_list() {
        let mut map = HashMap::new();
        let pool_addr = EVMAddress::from([0x0c; 20]);
        map.insert(EVMAddress::from([0x01; 20]), vec![make_abi(PRIME_SELECTORS[0])]);
        map.insert(EVMAddress::from([0x02; 20]), vec![make_abi(EXPLOIT_SELECTORS[0])]);
        map.insert(pool_addr, vec![make_abi(SEL_ADD_LIQUIDITY)]);
        let abi_map = ABIAddressToInstanceMap { map };
        let cache = CampaignTargetCache::new(&abi_map, Vec::new());

        let mut rand = StdRand::with_seed(0xC0FFEE);
        // No value_lever_pin → dynamic_fired=false → fallback fires on
        // effective_reflexive path.
        let campaign = plan_campaign_sampled(&cache, None, false, true, false, None, None, None, None, &mut rand)
            .expect("viable campaign");

        assert_eq!(campaign.promoted.len(), 1, "static fallback promotes add_liquidity");
        let lever_sel = campaign.steps[campaign.promoted[0]]
            .data
            .as_ref()
            .expect("ABI")
            .function;
        assert_eq!(lever_sel, SEL_ADD_LIQUIDITY, "fallback yields the known Curve lever");
    }

    /// §7d content re-point: `borrow_authority` (a topology-classified
    /// capital-source token) overrides the blind
    /// `borrowable_tokens.first()`; `None` is byte-identical to old behavior.
    #[test]
    fn borrow_authority_overrides_first_borrowable() {
        let mut map = HashMap::new();
        map.insert(EVMAddress::from([0x01; 20]), vec![make_abi(PRIME_SELECTORS[0])]);
        map.insert(EVMAddress::from([0x02; 20]), vec![make_abi(EXPLOIT_SELECTORS[0])]);
        let abi_map = ABIAddressToInstanceMap { map };
        let t_first = EVMAddress::from([0xaa; 20]);
        let t_auth = EVMAddress::from([0xbb; 20]);
        let cache = CampaignTargetCache::new(&abi_map, vec![t_first, t_auth]);

        let mut rand = StdRand::with_seed(0xB0);
        let with_auth = plan_campaign_sampled(
            &cache,
            None,
            false,
            false,
            false,
            None,
            None,
            Some(t_auth),
            None,
            &mut rand,
        )
        .expect("viable campaign");
        let borrow = with_auth
            .steps
            .iter()
            .find(|s| matches!(s.input_type, crate::evm::input::EVMInputTy::Borrow));
        assert_eq!(
            borrow.map(|s| s.contract),
            Some(t_auth),
            "borrow_authority overrides borrowable_tokens.first()"
        );

        let no_auth = plan_campaign_sampled(&cache, None, false, false, false, None, None, None, None, &mut rand)
            .expect("viable campaign");
        let borrow0 = no_auth
            .steps
            .iter()
            .find(|s| matches!(s.input_type, crate::evm::input::EVMInputTy::Borrow));
        assert_eq!(
            borrow0.map(|s| s.contract),
            Some(t_first),
            "None keeps the blind first-borrowable (byte-identical)"
        );
    }

    /// Off-path proof: with `effective_reflexive=false` the same fixture yields
    /// NO promotion, so the feature is genuinely inert when disabled
    /// (constitution: zero code path off).
    #[test]
    fn test_effective_reflexive_inert_when_disabled() {
        let mut map = HashMap::new();
        map.insert(EVMAddress::from([0x01; 20]), vec![make_abi(PRIME_SELECTORS[0])]);
        map.insert(EVMAddress::from([0x02; 20]), vec![make_abi(EXPLOIT_SELECTORS[0])]);
        map.insert(EVMAddress::from([0x0c; 20]), vec![make_abi(SEL_ADD_LIQUIDITY)]);
        let abi_map = ABIAddressToInstanceMap { map };
        let cache = CampaignTargetCache::new(&abi_map, Vec::new());

        let mut rand = StdRand::with_seed(0xC0FFEE);
        let campaign = plan_campaign_sampled(&cache, None, false, false, false, None, None, None, None, &mut rand)
            .expect("viable prime+exploit → campaign");
        assert!(
            campaign.promoted.is_empty(),
            "no promotion when effective_reflexive is off"
        );
        assert_eq!(campaign.steps.len(), 2, "plain prime→exploit frame, lever untouched");
    }

    // ── Feature 015 Phase 2 (T10) — a-posteriori arming ──

    /// On a target with NO registered reflexive archetype (no
    /// `add_liquidity`/imbalance in the vocabulary), the reflexive path
    /// arms the executor's per-step inflow snapshot instead of
    /// promoting a-priori: `aposteriori == true`, `promoted` empty. This is the
    /// generalization trigger — "no archetype fired, so go discover the
    /// lever at runtime".
    #[test]
    fn test_aposteriori_armed_when_no_archetype() {
        let mut map = HashMap::new();
        // Prime + exploit only — deliberately NO Curve pool / reflexive selector
        // present.
        map.insert(EVMAddress::from([0x01; 20]), vec![make_abi(PRIME_SELECTORS[0])]);
        map.insert(EVMAddress::from([0x02; 20]), vec![make_abi(EXPLOIT_SELECTORS[0])]);
        let abi_map = ABIAddressToInstanceMap { map };
        let cache = CampaignTargetCache::new(&abi_map, Vec::new());
        assert!(cache.reflexive_targets.is_empty(), "fixture has no reflexive archetype");

        let mut rand = StdRand::with_seed(0xC0FFEE);
        let campaign = plan_campaign_sampled(&cache, None, false, true, false, None, None, None, None, &mut rand)
            .expect("viable prime+exploit → campaign");
        assert!(campaign.promoted.is_empty(), "no a-priori lever to promote");
        assert!(
            campaign.aposteriori,
            "reflexive path with no archetype must arm a-posteriori"
        );
    }

    /// When an a-priori archetype DOES fire, a-posteriori stays disarmed (the
    /// lever is already in the frame; one lever/frame). `promoted`
    /// populated ⇒ `aposteriori == false`.
    #[test]
    fn test_aposteriori_disarmed_when_apriori_fires() {
        let mut map = HashMap::new();
        map.insert(EVMAddress::from([0x01; 20]), vec![make_abi(PRIME_SELECTORS[0])]);
        map.insert(EVMAddress::from([0x02; 20]), vec![make_abi(EXPLOIT_SELECTORS[0])]);
        map.insert(EVMAddress::from([0x0c; 20]), vec![make_abi(SEL_ADD_LIQUIDITY)]);
        let abi_map = ABIAddressToInstanceMap { map };
        let cache = CampaignTargetCache::new(&abi_map, Vec::new());

        let mut rand = StdRand::with_seed(0xC0FFEE);
        let campaign = plan_campaign_sampled(&cache, None, false, true, false, None, None, None, None, &mut rand)
            .expect("viable prime+exploit → campaign");
        assert_eq!(campaign.promoted.len(), 1, "a-priori lever promoted");
        assert!(!campaign.aposteriori, "a-priori match ⇒ a-posteriori disarmed");
    }

    /// Off the reflexive path, a-posteriori is never armed (zero executor
    /// overhead).
    #[test]
    fn test_aposteriori_off_when_flag_off() {
        let mut map = HashMap::new();
        map.insert(EVMAddress::from([0x01; 20]), vec![make_abi(PRIME_SELECTORS[0])]);
        map.insert(EVMAddress::from([0x02; 20]), vec![make_abi(EXPLOIT_SELECTORS[0])]);
        let abi_map = ABIAddressToInstanceMap { map };
        let cache = CampaignTargetCache::new(&abi_map, Vec::new());

        let mut rand = StdRand::with_seed(0xC0FFEE);
        let campaign = plan_campaign_sampled(&cache, None, false, false, false, None, None, None, None, &mut rand)
            .expect("viable prime+exploit → campaign");
        assert!(!campaign.aposteriori, "flag off ⇒ never armed");
    }

    // Feature 021 — the welded taint receipt defaults to "no evidence" and
    // round-trips.
    #[test]
    fn taint_provenance_tag_default_is_empty() {
        let tag = TaintProvenanceTag::default();
        assert!(!tag.causally_linked);
        assert!(!tag.analysis_ran);
        assert_eq!(tag.dim, TaintDim::Generic);
    }

    // Feature 021 — `#[serde(default)]` on `taint_provenance` keeps a pre-021
    // corpus (JSON with no `taint_provenance` key) deserializing cleanly to the
    // empty tag: no corpus-break.
    #[test]
    fn promotion_candidate_deserializes_pre021_corpus() {
        // A pre-021 serialized candidate: has `kind` (020) but NO `taint_provenance`
        // (021).
        let pre021 = r#"{
            "contract": "0x0000000000000000000000000000000000000001",
            "selector": [1,2,3,4],
            "best_inflow": 42,
            "kind": "Value",
            "set": true
        }"#;
        let cand: PromotionCandidate =
            serde_json::from_str(pre021).expect("pre-021 corpus must deserialize via serde(default)");
        assert_eq!(cand.taint_provenance, TaintProvenanceTag::default());
        assert!(cand.set);
        assert_eq!(cand.best_inflow, 42);
    }

    // Feature 021 — a stamped candidate round-trips its taint receipt through the
    // corpus.
    #[test]
    fn promotion_candidate_roundtrips_taint_provenance() {
        let cand = PromotionCandidate {
            contract: EVMAddress::from([0x01; 20]),
            selector: [0xa9, 0x05, 0x9c, 0xbb],
            best_inflow: 1_000_000_000_000_000_000,
            kind: LeakClass::Value,
            taint_provenance: TaintProvenanceTag {
                causally_linked: true,
                analysis_ran: true,
                dim: TaintDim::Price,
            },
            phase: Some(3),
            set: true,
        };
        let json = serde_json::to_string(&cand).expect("serialize");
        let back: PromotionCandidate = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.taint_provenance, cand.taint_provenance);
        assert_eq!(back.taint_provenance.dim, TaintDim::Price);
    }

    fn candidate(kind: LeakClass, contract_byte: u8, selector: [u8; 4], score: u128) -> PromotionCandidate {
        PromotionCandidate {
            contract: EVMAddress::from([contract_byte; 20]),
            selector,
            best_inflow: score,
            kind,
            taint_provenance: TaintProvenanceTag::default(),
            phase: None,
            set: true,
        }
    }

    #[test]
    fn promotion_candidates_keep_independent_kind_slots() {
        let mut candidates = PromotionCandidates::default();
        let permission = candidate(LeakClass::Permission, 0x10, [0xaa; 4], 0);
        let invariant = candidate(LeakClass::Invariant, 0x20, [0xbb; 4], 0);
        let ownership = candidate(LeakClass::Ownership, 0x30, [0xcc; 4], 2);

        assert!(candidates.record(permission.clone()));
        assert!(candidates.record(invariant.clone()));
        assert!(candidates.record(ownership.clone()));

        assert_eq!(
            candidates.get(LeakClass::Permission).unwrap().contract,
            permission.contract
        );
        assert_eq!(
            candidates.get(LeakClass::Invariant).unwrap().contract,
            invariant.contract
        );
        assert_eq!(
            candidates.get(LeakClass::Ownership).unwrap().contract,
            ownership.contract
        );
    }

    #[test]
    fn promotion_candidates_high_water_only_within_same_kind() {
        let mut candidates = PromotionCandidates::default();

        assert!(candidates.record(candidate(LeakClass::Value, 0x10, [0xaa; 4], 100)));
        assert!(!candidates.record(candidate(LeakClass::Value, 0x11, [0xbb; 4], 50)));
        assert_eq!(
            candidates.get(LeakClass::Value).unwrap().contract,
            EVMAddress::from([0x10; 20])
        );

        assert!(candidates.record(candidate(LeakClass::Value, 0x12, [0xcc; 4], 150)));
        assert_eq!(
            candidates.get(LeakClass::Value).unwrap().contract,
            EVMAddress::from([0x12; 20])
        );

        assert!(candidates.record(candidate(LeakClass::Invariant, 0x20, [0xdd; 4], 0)));
        assert_eq!(
            candidates.get(LeakClass::Value).unwrap().contract,
            EVMAddress::from([0x12; 20])
        );
        assert_eq!(
            candidates.get(LeakClass::Invariant).unwrap().contract,
            EVMAddress::from([0x20; 20])
        );
    }

    #[test]
    fn promotion_candidates_first_set_uses_consumer_preference_order() {
        let mut candidates = PromotionCandidates::default();
        candidates.record(candidate(LeakClass::Permission, 0x10, [0xaa; 4], 0));
        candidates.record(candidate(LeakClass::Ownership, 0x20, [0xbb; 4], 1));
        candidates.record(candidate(LeakClass::Invariant, 0x30, [0xcc; 4], 0));
        candidates.record(candidate(LeakClass::Value, 0x40, [0xdd; 4], 100));

        let structural = candidates
            .first_set(&[LeakClass::Ownership, LeakClass::Permission])
            .expect("structural candidate");
        assert_eq!(structural.kind, LeakClass::Ownership);

        let lever = candidates
            .first_set(&[LeakClass::Value, LeakClass::Invariant])
            .expect("lever candidate");
        assert_eq!(lever.kind, LeakClass::Value);
    }

    #[test]
    fn test_planner_adds_warp_when_dimension_warp_enabled() {
        let mut map = HashMap::new();
        let prime_addr = EVMAddress::from([0x01; 20]);
        let exploit_addr = EVMAddress::from([0x02; 20]);
        map.insert(prime_addr, vec![make_abi(PRIME_SELECTORS[0])]);
        map.insert(exploit_addr, vec![make_abi(EXPLOIT_SELECTORS[0])]);
        let abi_map = ABIAddressToInstanceMap { map };
        let cache = CampaignTargetCache::new(&abi_map, Vec::new());

        let mut rand = StdRand::with_seed(0xC0FFEE);
        // dimension_warp = true → should insert warp before exploit step
        let campaign = plan_campaign_sampled(
            &cache,
            None,
            false, // temporal_skimming
            false, // effective_reflexive
            true,  // dimension_warp
            None,  // structural_pin
            None,  // value_lever_pin
            None,  // borrow_authority
            None,  // divergence_value
            &mut rand,
        ).expect("should produce campaign");

        assert_eq!(campaign.steps.len(), 2);
        assert_eq!(campaign.warps.len(), 1, "should have 1 warp entry");
        assert_eq!(campaign.warps[0].0, 1, "warp should be before exploit step (index 1)");
    }


    #[test]
    fn test_campaign_step_linkage_planning() {
        let _guard = FUNCTION_SIG_TEST_LOCK.lock().unwrap();
        let mut map = HashMap::new();
        let target_addr = EVMAddress::from([0x03; 20]);
        let prime_sel = PRIME_SELECTORS[0];
        let mut abi_prime = make_abi(prime_sel);
        abi_prime.set_func_with_signature(prime_sel, "deposit", "(uint256)");

        let exploit_sel = EXPLOIT_SELECTORS[0];
        let mut abi_exploit = make_abi(exploit_sel);
        abi_exploit.set_func_with_signature(exploit_sel, "withdraw", "(uint256)");

        map.insert(target_addr, vec![abi_prime, abi_exploit]);
        let abi_map = ABIAddressToInstanceMap { map };
        let cache = CampaignTargetCache::new(&abi_map, Vec::new());

        let mut rand = StdRand::with_seed(12345);
        let campaign = plan_campaign_sampled(
            &cache,
            None,
            false,
            false,
            false,
            None,
            None,
            None,
            None,
            &mut rand,
        ).expect("should plan campaign");

        assert!(!campaign.linkages.is_empty(), "Linkages list must not be empty");
        let linkage = &campaign.linkages[0];
        assert_eq!(linkage.from_step, 0);
        assert_eq!(linkage.to_step, 1);
        assert_eq!(linkage.to_param_index, 0);
        assert_eq!(
            linkage.from_registry_key,
            format!("{:?}_{}_return", target_addr, hex::encode(prime_sel))
        );
    }

    #[test]
    fn test_campaign_planning_with_guidance() {
        let _g = FUNCTION_SIG_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let target_addr = EVMAddress::from([0x03; 20]);
        let borrow_sel = [0xde, 0xad, 0xbe, 0xef];
        let withdraw_sel = [0xca, 0xfe, 0xba, 0xbe];
        let unrelated_sel = [0xba, 0xad, 0xf0, 0x0d];

        unsafe {
            crate::evm::abi::FUNCTION_SIG.insert(borrow_sel, "borrow(uint256)".to_string());
            crate::evm::abi::FUNCTION_SIG.insert(withdraw_sel, "withdraw(uint256)".to_string());
            crate::evm::abi::FUNCTION_SIG.insert(unrelated_sel, "unrelated(uint256)".to_string());
        }

        let mut map = HashMap::new();
        map.insert(target_addr, vec![
            make_abi(borrow_sel),
            make_abi(withdraw_sel),
            make_abi(unrelated_sel),
        ]);
        let abi_map = ABIAddressToInstanceMap { map };
        let cache = CampaignTargetCache::new_with_preset(
            &abi_map,
            Vec::new(),
            &[borrow_sel, withdraw_sel, unrelated_sel],
            None,
            &[],
        );

        // Construct mock guidance: after "Target:borrow" -> ["Target:withdraw"]
        let mut after = HashMap::new();
        after.insert("Target:borrow".to_string(), vec!["Target:withdraw".to_string()]);

        let guidance = Guidance {
            version: 1,
            generated_at: 0.0,
            meta: crate::evm::guidance::Meta {
                num_functions: 3,
                num_params: 3,
                num_kill_chains: 0,
                num_invariants: 0,
                num_contracts: 1,
            },
            functions: HashMap::new(),
            scheduler: crate::evm::guidance::SchedulerIndex {
                after,
                entry_points: Vec::new(),
            },
            mutator: crate::evm::guidance::MutatorIndex {
                high_value_params: Vec::new(),
            },
            oracle: crate::evm::guidance::OracleIndex {
                invariants: Vec::new(),
                num_invariants: 0,
            },
            contracts: vec![
                crate::evm::guidance::ContractEntry {
                    id: "1".to_string(),
                    name: "Target".to_string(),
                    address: format!("{:?}", target_addr),
                    is_library: None,
                    is_interface: None,
                    is_proxy: None,
                    protocol: None,
                }
            ],
            slot_influence_weights: HashMap::new(),
            storage_layout: HashMap::new(),
        };

        let guidance_meta = GuidanceMetadata::new(guidance);

        let mut rand = StdRand::with_seed(12345);
        let campaign = plan_campaign_sampled(
            &cache,
            Some(&guidance_meta),
            false,
            false,
            false,
            None,
            None,
            None,
            None,
            &mut rand,
        ).expect("should plan campaign");

        // The exploit step must be withdraw_sel because guidance prioritized it
        let exploit_step = &campaign.steps[1];
        let exploit_sig = exploit_step.data.as_ref().map(|d| d.function).unwrap_or_default();
        assert_eq!(exploit_sig, withdraw_sel);
    }
}

fn build_abi_step(target: EVMAddress, abi: Option<BoxedABI>) -> ConciseEVMInput {
    // Pin the concrete function (`abi`) so the step calls it directly — required
    // for the executor's controlled warp probe to exercise the time-gated
    // function instead of hitting the fallback with empty calldata. Args are
    // still mutated via the `mutate_with_vm_slots` path. `None` falls back to
    // the contract.
    ConciseEVMInput {
        input_type: EVMInputTy::ABI,
        caller: EVMAddress::default(),
        contract: target,
        data: abi,
        txn_value: None,
        step: false,
        env: Default::default(),
        liquidation_percent: 0,
        randomness: vec![],
        repeat: 1,
        layer: 0,
        call_leak: u32::MAX,
        return_data: None,
        swap_data: HashMap::new(),
        nested_actions: Vec::new(),
        campaign: None,
    }
}
