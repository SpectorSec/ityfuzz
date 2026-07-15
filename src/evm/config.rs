use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::{self, Debug},
    rc::Rc,
    str::FromStr,
};

/// Configuration for the EVM fuzzer
use crate::evm::contract_utils::ContractLoader;
use crate::{
    evm::{
        onchain::endpoints::OnChainConfig,
        oracles::erc20::IERC20OracleFlashloan,
        types::EVMAddress,
    },
    oracle::{Oracle, Producer},
};

pub enum FuzzerTypes {
    CMP,
    DATAFLOW,
    BASIC,
}

#[derive(Copy, Clone)]
pub enum StorageFetchingMode {
    Dump,
    OneByOne,
}

impl FromStr for StorageFetchingMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "dump" => Ok(StorageFetchingMode::Dump),
            "onebyone" => Ok(StorageFetchingMode::OneByOne),
            _ => Err(format!("Unknown storage fetching mode: {}", s)),
        }
    }
}

impl FromStr for FuzzerTypes {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cmp" => Ok(FuzzerTypes::CMP),
            "dataflow" => Ok(FuzzerTypes::DATAFLOW),
            "basic" => Ok(FuzzerTypes::BASIC),
            _ => Err(format!("Unknown fuzzer type: {}", s)),
        }
    }
}

#[allow(clippy::type_complexity)]
pub struct Config<VS, Addr, Code, By, Loc, SlotTy, Out, I, S, CI, E> {
    pub onchain: Option<OnChainConfig>,
    pub onchain_storage_fetching: Option<StorageFetchingMode>,
    pub etherscan_api_key: String,
    pub flashloan: bool,
    pub concolic: bool,
    pub concolic_caller: bool,
    pub concolic_timeout: u32,
    pub concolic_num_threads: usize,
    /// Feature 011 (Part A): rank the extraction gradient by realized ETH value
    /// instead of raw token units. Off ⇒ original token-unit gradient (unchanged).
    pub impact_eth_gradient: bool,
    pub contract_loader: ContractLoader,
    pub oracle: Vec<Rc<RefCell<dyn Oracle<VS, Addr, Code, By, Loc, SlotTy, Out, I, S, CI, E>>>>,
    pub producers: Vec<Rc<RefCell<dyn Producer<VS, Addr, Code, By, Loc, SlotTy, Out, I, S, CI, E>>>>,
    pub replay_file: Option<String>,
    pub flashloan_oracle: Rc<RefCell<IERC20OracleFlashloan>>,
    pub selfdestruct_oracle: bool,
    /// Feature 020-B — SnapshotDelta oracle (LeakClass::Ownership). Selected by
    /// `-d ownership_leak` / `-d ownership` / `-d all`.
    pub ownership_oracle: bool,
    pub reentrancy_oracle: bool,
    // pub state_comp_oracle: Option<String>,
    // pub state_comp_matching: Option<String>,
    pub work_dir: String,
    pub write_relationship: bool,
    pub run_forever: bool,
    pub sha3_bypass: bool,
    pub base_path: String,
    pub echidna_oracle: bool,
    pub invariant_oracle: bool,
    pub nft_oracle: bool,
    pub fee_on_transfer_oracle: bool,
    pub approval_oracle: bool,
    pub crosschain_oracle: bool,
    pub rebasing_oracle: bool,
    pub panic_on_bug: bool,
    pub spec_id: String,
    pub only_fuzz: HashSet<EVMAddress>,
    pub typed_bug: bool,
    pub arbitrary_external_call: bool,
    pub math_calculate_oracle: bool,
    pub local_files_basedir_pattern: Option<String>,
    pub load_corpus: String,
    pub value_capture: bool,
    pub campaign_orchestrator: bool,
    pub ghost_identities: bool,
    pub temporal_skimming: bool,
    /// Feature 015: enable reflexive-lever promotion + ledger-secant amplification.
    /// Implies `campaign_orchestrator` + `impact_eth_gradient` (auto-enabled with a
    /// warning if unset — the lever is inert without them).
    pub reflexive_lever: bool,
    /// Feature 017: enable Dimension-Driven Warp coupling. When active, the planner
    /// gates the warp lever on TIMESTAMP_DIM_LOCATED (ts_seen reaches SSTORE) as well
    /// as the --temporal-skimming flag. Additive path.
    pub dimension_warp: bool,
    /// Feature 019 Phase A: Causal Identity permission-leak materiality gate. When set,
    /// registers `FunctionAuthTracer` and switches `FunctionOracle` to require a
    /// material sink (SSTORE pre≠post / value-CALL) before firing — killing the
    /// `burn(0,0)` no-op false positive. Off = pre-019 behavior.
    pub causal_identity: bool,
    /// Feature 013 Phase 1: shallow injection detection at CALL boundaries.
    pub injection_detect: bool,
    /// Feature 013 Phase 3: persistent cross-execution taint via FuzzHost.
    pub injection_persist: bool,
    /// Feature 013 Phase 4: value-confirmed provenance (TaintProvenance).
    pub injection_provenance: bool,
    /// Feature 013 Phase 5: scheduler wiring (mutation bias from injection flags).
    pub injection_feedback: bool,
    /// Feature 014 Phase 1: oracle-gated value movement detection.
    pub oracle_detection: bool,
    /// Feature 014 Phase 2: flash loan oracle manipulation detection.
    pub flashloan_detection: bool,
    /// Feature 014 Phase 3: missing updatedAt staleness check detection.
    pub oracle_staleness: bool,
    /// Feature 014 Phase 4: empty state guard (first-deposit inflation) detection.
    pub empty_state_guard: bool,
    /// Feature 014 Phase 5: DoS via state-dependent revert detection.
    pub dos_detection: bool,
    pub guidance_file: String,
    pub state_file: Option<String>,
    #[cfg(feature = "use_presets")]
    pub preset_file_path: String,
    /// Use ONLY --preset-file-path templates, skip the baked-in corpus (isolation).
    #[cfg(feature = "use_presets")]
    pub preset_only: bool,
}

impl<VS, Addr, Code, By, Loc, SlotTy, Out, I, S, CI, E> Debug
    for Config<VS, Addr, Code, By, Loc, SlotTy, Out, I, S, CI, E>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Config")
            .field("onchain", &self.onchain)
            // .field("onchain_storage_fetching", &self.onchain_storage_fetching)
            .field("flashloan", &self.flashloan)
            .field("concolic", &self.concolic)
            .field("concolic_caller", &self.concolic_caller)
            .field("contract_loader", &self.contract_loader)
            // .field("oracle", &self.oracle)
            // .field("producers", &self.producers)
            .field("replay_file", &self.replay_file)
            // .field("flashloan_oracle", &self.flashloan_oracle)
            .field("selfdestruct_oracle", &self.selfdestruct_oracle)
            .field("ownership_oracle", &self.ownership_oracle)
            // .field("state_comp_oracle", &self.state_comp_oracle)
            // .field("state_comp_matching", &self.state_comp_matching)
            .field("work_dir", &self.work_dir)
            .field("write_relationship", &self.write_relationship)
            .field("run_forever", &self.run_forever)
            .field("sha3_bypass", &self.sha3_bypass)
            .field("base_path", &self.base_path)
            .field("echidna_oracle", &self.echidna_oracle)
            .field("panic_on_bug", &self.panic_on_bug)
            .field("spec_id", &self.spec_id)
            .field("only_fuzz", &self.only_fuzz)
            .field("typed_bug", &self.typed_bug)
            // .field("builder", &self.builder)
            .finish()
    }
}
