pub mod abi;
pub mod blaz;
pub mod bytecode_analyzer;
pub mod bytecode_iterator;
pub mod concolic;
pub mod config;
pub mod contract_utils;
pub mod corpus_initializer;
pub mod cov_stage;
pub mod feedbacks;
pub mod host;
pub mod input;
pub mod leak_class;
pub mod middlewares;
pub mod minimizer;
pub mod mutator;
pub mod liquidation;
pub mod liquidation_router;
pub mod onchain;

pub mod oracle;
pub mod oracles;
pub mod planner;
pub mod presets;
pub mod producers;
pub mod scheduler;
pub mod slot_detector;
pub mod solution;
pub mod srcmap;
pub mod tokens;
pub mod types;
pub mod utils;
pub mod vm;
pub mod guidance;
pub mod state_loader;

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt,
    fs::OpenOptions,
    io::Write,
    path::Path,
    rc::Rc,
    str::FromStr,
};

use blaz::{
    offchain_artifacts::OffChainArtifact,
    offchain_config::OffchainConfig,
};
use clap::Parser;
use config::{Config, StorageFetchingMode};
use contract_utils::ContractLoader;
use ethers::types::Transaction;
use input::{ConciseEVMInput, EVMInput};
use itertools::Itertools;
use num_cpus;
use onchain::endpoints::{Chain, OnChainConfig};
use oracles::{erc20::IERC20OracleFlashloan, v2_pair::PairBalanceOracle};
use producers::erc20::ERC20Producer;
use serde::Deserialize;
use serde_json::json;
use tracing::{debug, warn};
use types::{EVMAddress, EVMFuzzState, EVMU256};
use vm::EVMState;

use self::types::EVMQueueExecutor;
use crate::{
    fuzzers::evm_fuzzer::evm_fuzzer,
    oracle::{Oracle, Producer},
    state::FuzzState,
};

pub const PRESET_WETH: &str = "0x4200000000000000000000000000000000000006";

pub fn parse_constructor_args_string(input: String) -> HashMap<String, Vec<String>> {
    let mut map = HashMap::new();

    if input.is_empty() {
        return map;
    }

    let pairs: Vec<&str> = input.split(';').collect();
    for pair in pairs {
        let key_value: Vec<&str> = pair.split(':').collect();
        if key_value.len() == 2 {
            let values: Vec<String> = key_value[1].split(',').map(|s| s.to_string()).collect();
            map.insert(key_value[0].to_string(), values);
        }
    }

    map
}

#[derive(Deserialize)]
struct Data {
    body: RPCCall,
}

#[derive(Deserialize)]
struct RPCCall {
    method: String,
    params: Option<serde_json::Value>,
}

/// CLI for ItyFuzz for EVM smart contracts
#[derive(Parser, Debug, Default)]
#[command(author, version, about, long_about = None, trailing_var_arg = true, allow_hyphen_values = true)]
pub struct EvmArgs {
    /// Glob pattern / address to find contracts
    #[arg(short, long, default_value = "none")]
    target: String,

    #[arg(long, default_value = "false")]
    fetch_tx_data: bool,

    #[arg(long, default_value = "http://localhost:5001/data")]
    proxy_address: String,

    /// Constructor arguments for the contract, separated by semicolon. Example:
    /// https://docs.ityfuzz.rs/docs-evm-contract/constructor-for-offchain-fuzzing
    #[arg(long, default_value = "")]
    constructor_args: String,

    /// Target type (glob, address, anvil_fork, config, setup)
    /// (Default: Automatically infer from target)
    #[arg(long)]
    target_type: Option<String>,

    /// Onchain - Chain type
    /// (eth,goerli,sepolia,bsc,chapel,polygon,mumbai,fantom,avalanche,optimism,
    /// arbitrum,gnosis,base,celo,zkevm,zkevm_testnet,blast,local)
    #[arg(short, long)]
    chain_type: Option<String>,

    /// Onchain - Block number (Default: 0 / latest)
    #[arg(long, short = 'b')]
    onchain_block_number: Option<u64>,

    /// Onchain Customize - RPC endpoint URL (Default: inferred from
    /// chain-type), Example: https://rpc.ankr.com/eth
    #[arg(long, short = 'u')]
    onchain_url: Option<String>,

    /// Onchain Customize - Chain ID (Default: inferred from chain-type)
    #[arg(long, short = 'i')]
    onchain_chain_id: Option<u32>,

    /// Onchain Customize - Block explorer URL (Default: inferred from
    /// chain-type), Example: https://api.etherscan.io/api
    #[arg(long, short = 'e')]
    onchain_explorer_url: Option<String>,

    /// Onchain Customize - Chain name (used as Moralis handle of chain)
    /// (Default: inferred from chain-type)
    #[arg(long, short = 'n')]
    onchain_chain_name: Option<String>,

    /// Onchain Etherscan API Key (Default: None)
    #[arg(long, short = 'k')]
    onchain_etherscan_api_key: Option<String>,

    /// Onchain which fetching method to use (dump, onebyone) (Default:
    /// onebyone)
    #[arg(long, default_value = "onebyone")]
    onchain_storage_fetching: String,

    /// Enable Concolic (Experimental)
    #[arg(long, default_value = "false")]
    concolic: bool,

    /// Support Treating Caller as Symbolically  (Experimental)
    #[arg(long, default_value = "false")]
    concolic_caller: bool,

    /// Time limit for concolic execution (ms) (Default: 1000, 0 for no limit)
    #[arg(long, default_value = "1000")]
    concolic_timeout: u32,

    /// Number of threads for concolic execution (Default: number of cpus)
    #[arg(long, default_value = "0")]
    concolic_num_threads: usize,

    /// Feature 011 (Part A): rank the fund-extraction gradient by realized ETH
    /// value (via the liquidation engine) instead of raw token units. Opt-in;
    /// off by default the gradient behaves exactly as before.
    #[arg(long, default_value = "false")]
    impact_eth_gradient: bool,


    /// Enable the economic outcome detection subsystem.
    ///
    /// Despite the legacy `--flashloan` name, this flag controls the entire
    /// fund-loss detection layer: synthetic-capital injection so the fuzzer's
    /// attacker can attempt drains, owed/earned accounting per call frame, and
    /// portfolio valuation via registered swap routes (Uniswap V2/V3, Curve, ERC-4626,
    /// Compound cTokens, Aave aTokens, Lido wstETH, Sudoswap — not Uniswap-only). The `Fund Loss`
    /// finding in oracles/erc20.rs ONLY fires when this is enabled — without it
    /// the entire balance-based outcome detection is dormant regardless of any
    /// `-d` selection. Default-on in this fork; pass `--flashloan=false` to
    /// disable. Long-form alias `--economic-oracle` is preferred for new scripts.
    #[arg(short, long, alias = "economic-oracle", default_value = "false")]
    flashloan: bool,

    /// Panic when a typed_bug() is called (Default: false)
    #[arg(long, default_value = "false")]
    panic_on_bug: bool,

    /// Detectors enabled (all, high_confidence, ...). Refer to https://docs.ityfuzz.rs/docs-evm-contract/detecting-common-vulns
    /// (Default: high_confidence)
    #[arg(long, short, default_value = "high_confidence")]
    detectors: String, // <- internally this is known as oracles

    // /// Matching style for state comparison oracle (Select from "Exact",
    // /// "DesiredContain", "StateContain")
    // #[arg(long, default_value = "Exact")]
    // state_comp_matching: String,
    /// Replay?
    #[arg(long, short)]
    replay_file: Option<String>,

    /// Path of work dir, saves corpus, logs, and other stuffs
    #[arg(long, short, default_value = "work_dir")]
    work_dir: String,

    /// Write contract relationship to files
    #[arg(long, default_value = "false")]
    write_relationship: bool,

    /// Do not quit when a bug is found, continue find new bugs
    #[arg(long, default_value = "false")]
    run_forever: bool,

    /// random seed
    #[arg(long, default_value = "1667840158231589000")]
    seed: u64,

    /// Whether bypass all SHA3 comparisons, this may break original logic of
    /// contracts  (Experimental)
    #[arg(long, default_value = "false")]
    sha3_bypass: bool,

    /// Enable Value Capture Middleware (Phase 1)
    #[arg(long, default_value = "false")]
    value_capture: bool,

    /// Enable Campaign Orchestrator for multi-step exploit synthesis (Phase 3)
    #[arg(long, default_value = "false")]
    campaign_orchestrator: bool,

    /// Enable Ghost Identities (identity spoofing / confused deputy) for privileged function access
    #[arg(long, default_value = "false")]
    ghost_identities: bool,

    /// Enable Temporal Pre-condition Skimming (multi-block state priming).
    /// When active, campaign steps can include block-advance (warp) operations
    /// between prime and exploit steps to detect cross-round state divergence.
    #[arg(long, default_value = "false")]
    temporal_skimming: bool,

    /// Feature 015: enable the Reflexive Lever Pipeline — promote a reflexive-skew
    /// liquidity lever (`add_liquidity`/`remove_liquidity_imbalance`) into the campaign
    /// frame and amount-tune it with the ledger-secant. Reaches reflexive-body exploits
    /// (Yearn yDAI, Harvest) that the 2-step frame cannot. Auto-enables
    /// `campaign_orchestrator` + `impact_eth_gradient` (warns if you disabled them).
    #[arg(long, default_value = "false")]
    reflexive_lever: bool,

    /// Feature 017: enable Dimension-Driven Warp coupling. When active, the campaign
    /// planner engages the warp lever (block advance between prime and exploit) when
    /// the taint engine's Timestamp-presence bit reaches SSTORE during reexecution —
    /// even without --temporal-skimming. Additive to the existing flag-driven path.
    #[arg(long, default_value = "false")]
    dimension_warp: bool,

    /// Feature 019 Phase A: enable the Causal Identity permission-leak materiality gate.
    /// Registers the inline `FunctionAuthTracer` and switches the permission-leak
    /// oracle to require a material sink (SSTORE pre≠post or a value-CALL) in the
    /// privileged contract before firing — suppressing no-op privileged calls such as
    /// `burn(0x0, 0)`. Additive; off by default (pre-019 behavior).
    #[arg(long, default_value = "false")]
    causal_identity: bool,

    /// Feature 013 Phase 1: shallow injection detection at CALL boundaries.
    /// When enabled, the reexecution taint engine reads shadow-stack and memory taint
    /// at each CALL/DELEGATECALL/STATICCALL boundary and sets static flags for
    /// post-execution oracle consumption.
    #[arg(long, default_value = "false")]
    injection_detect: bool,

    /// Feature 013 Phase 3: persistent cross-execution taint via FuzzHost.
    /// Enables host-level tainted_storage HashMap so SLOAD reads can merge persistent
    /// taint from prior executions in the same campaign. Without this, all taint is
    /// per-execution (resets each iteration).
    #[arg(long, default_value = "false")]
    injection_persist: bool,

    /// Feature 013 Phase 4: value-confirmed provenance (TaintProvenance struct).
    /// Upgrades the host tainted_storage to store the actual written value, enabling
    /// verification that a storage slot still holds attacker-written data (eliminates
    /// false attribution from overwritten slots).
    #[arg(long, default_value = "false")]
    injection_provenance: bool,

    /// Feature 014 Phase 1: oracle-gated value movement detection.
    /// Tracks opcode proximity between oracle CALLs and comparisons that gate value
    /// transfers to financial sinks.
    #[arg(long, default_value = "false")]
    oracle_detection: bool,

    /// Feature 014 Phase 2: flash loan oracle manipulation detection.
    /// Tracks multi-CALL sequences: oracle read → borrow → oracle read → exploit → repay.
    #[arg(long, default_value = "false")]
    flashloan_detection: bool,

    /// Feature 014 Phase 3: missing updatedAt staleness check detection.
    /// Checks whether latestRoundData() CALL is followed by a TIMESTAMP comparison
    /// within 50 opcodes.
    #[arg(long, default_value = "false")]
    oracle_staleness: bool,

    /// Feature 014 Phase 4: empty state guard (first-deposit inflation) detection.
    /// Checks whether deposit/mint functions check totalSupply > 0 before transferring
    /// value (ERC-4626 inflation attack guard).
    #[arg(long, default_value = "false")]
    empty_state_guard: bool,

    /// Feature 014 Phase 5: DoS via state-dependent revert detection.
    /// Checks whether REVERT is gated by tainted storage (from 013 Phase 3).
    #[arg(long, default_value = "false")]
    dos_detection: bool,

    /// Bounty/production profile: enable the full attacker-REACH set as a unit —
    /// flashloan (economic capital: borrow → acquire → liquidate + fund-loss accounting),
    /// value-capture, campaign-orchestrator, ghost-identities, temporal-skimming. Three
    /// clean tiers: `-d` = what we DETECT, `--bounty` = how far the attacker REACHES,
    /// `--concolic` = how hard we SOLVE (kept separate/opt-in — heaviest, a search
    /// strategy, not reach). Individual flags still work and are OR'd with this. NOTE
    /// heavy: the full reach set is memory-intensive and can MEM_ABORT a ~3.5GB box.
    #[arg(long, default_value = "false")]
    bounty: bool,

    /// Only fuzz contracts with the addresses provided, separated by comma
    #[arg(long, default_value = "")]
    only_fuzz: String,

    /// Only needed when using combined.json (source map info).
    /// This is the base path when running solc compile (--base-path passed to
    /// solc). Also, please convert it to absolute path if you are not sure.
    #[arg(long, default_value = "")]
    base_path: String,

    /// Spec ID.
    /// Frontier,Homestead,Tangerine,Spurious,Byzantium,Constantinople,
    /// Petersburg,Istanbul,MuirGlacier,Berlin,London,Merge,Shanghai,Cancun,
    /// Latest
    #[arg(long, default_value = "Latest")]
    spec_id: String,

    /// Builder Artifacts url. If specified, will use this artifact to derive
    /// code coverage.
    #[arg(long, default_value = "")]
    builder_artifacts_url: String,

    /// Builder Artifacts file. If specified, will use this artifact to derive
    /// code coverage.
    #[arg(long, default_value = "")]
    builder_artifacts_file: String,

    /// Offchain Config Url. If specified, will deploy based on offchain config
    /// file.
    #[arg(long, default_value = "")]
    offchain_config_url: String,

    /// Offchain Config File. If specified, will deploy based on offchain config
    /// file.
    #[arg(long, default_value = "")]
    offchain_config_file: String,

    /// Load corpus from directory. If not specified, will use empty corpus.
    #[arg(long, default_value = "")]
    load_corpus: String,

    /// [DEPRECATED] Specify the setup file that deploys all the contract.
    /// Fuzzer invokes setUp() to deploy.
    #[arg(long, default_value = "")]
    setup_file: String,

    /// Specify the deployment script contract that deploys all the contract.
    /// Fuzzer invokes constructor or setUp() of this script to deploy.
    /// For example, if you have contract X in file Y that deploys all the
    /// contracts, you can specify --deployment-script Y:X
    #[arg(long, short = 'm', default_value = "")]
    deployment_script: String,

    /// Forcing a contract to use the given abi. This is useful when the
    /// contract is a complex proxy or decompiler has trouble to detect the abi.
    /// Format: address:abi_file,...
    #[arg(long, default_value = "")]
    force_abi: String,

    /// Preset file. If specified, will load the preset file and match past
    /// exploit template.
    #[cfg(feature = "use_presets")]
    #[arg(long, default_value = "")]
    preset_file_path: String,

    /// Use ONLY the templates from --preset-file-path, skipping the baked-in
    /// DefiHacksPresets corpus entirely. This isolates the preset language: the
    /// mutator's preset budget (20%) speaks only the supplied exploit's shape, with
    /// no dilution from the 1000+ historical templates. Controlled-experiment switch
    /// for measuring exactly how the fuzzer mutates around one known exploit.
    #[cfg(feature = "use_presets")]
    #[arg(long, default_value = "false")]
    preset_only: bool,

    #[arg(long, default_value = "")]
    base_directory: String,

    /// Path to the compiled semantic guidance JSON file.
    #[arg(long, default_value = "")]
    guidance_file: String,

    /// Path to a pre-fetched state snapshot JSON (Architecture A offline fuzzing).
    /// Produced by 11_snapshot_state.py — loads storage, balances, and bytecodes
    /// directly into REVM, eliminating RPC calls during fuzzing.
    #[arg(long)]
    state_file: Option<String>,

    /// Command to build the contract. If specified, will use this command to
    /// build contracts instead of using bins and abis.
    #[arg()]
    build_command: Vec<String>,
}

impl fmt::Display for EvmArgs {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "EvmArgs {{\n")?;
        write!(f, "    target: {},\n", self.target)?;
        write!(f, "    fetch_tx_data: {},\n", self.fetch_tx_data)?;
        write!(f, "    proxy_address: {},\n", self.proxy_address)?;
        write!(f, "    constructor_args: {},\n", self.constructor_args)?;
        write!(f, "    target_type: {:?},\n", self.target_type)?;
        write!(f, "    chain_type: {:?},\n", self.chain_type)?;
        write!(f, "    onchain_block_number: {:?},\n", self.onchain_block_number)?;
        write!(f, "    onchain_url: {:?},\n", self.onchain_url)?;
        write!(f, "    onchain_chain_id: {:?},\n", self.onchain_chain_id)?;
        write!(f, "    onchain_explorer_url: {:?},\n", self.onchain_explorer_url)?;
        write!(f, "    onchain_chain_name: {:?},\n", self.onchain_chain_name)?;
        write!(
            f,
            "    onchain_etherscan_api_key: {:?},\n",
            self.onchain_etherscan_api_key
        )?;
        write!(f, "    onchain_storage_fetching: {},\n", self.onchain_storage_fetching)?;
        write!(f, "    concolic: {},\n", self.concolic)?;
        write!(f, "    concolic_caller: {},\n", self.concolic_caller)?;
        write!(f, "    concolic_timeout: {},\n", self.concolic_timeout)?;
        write!(f, "    concolic_num_threads: {},\n", self.concolic_num_threads)?;
        write!(f, "    flashloan: {},\n", self.flashloan)?;
        write!(f, "    panic_on_bug: {},\n", self.panic_on_bug)?;
        write!(f, "    detectors: {},\n", self.detectors)?;
        write!(f, "    replay_file: {:?},\n", self.replay_file)?;
        write!(f, "    work_dir: {},\n", self.work_dir)?;
        write!(f, "    write_relationship: {},\n", self.write_relationship)?;
        write!(f, "    run_forever: {},\n", self.run_forever)?;
        write!(f, "    seed: {},\n", self.seed)?;
        write!(f, "    sha3_bypass: {},\n", self.sha3_bypass)?;
        write!(f, "    value_capture: {},\n", self.value_capture)?;
        write!(f, "    campaign_orchestrator: {},\n", self.campaign_orchestrator)?;
        write!(f, "    only_fuzz: {},\n", self.only_fuzz)?;
        write!(f, "    base_path: {},\n", self.base_path)?;
        write!(f, "    spec_id: {},\n", self.spec_id)?;
        write!(f, "    builder_artifacts_url: {},\n", self.builder_artifacts_url)?;
        write!(f, "    builder_artifacts_file: {},\n", self.builder_artifacts_file)?;
        write!(f, "    offchain_config_url: {},\n", self.offchain_config_url)?;
        write!(f, "    offchain_config_file: {},\n", self.offchain_config_file)?;
        write!(f, "    load_corpus: {},\n", self.load_corpus)?;
        write!(f, "    setup_file: {},\n", self.setup_file)?;
        write!(f, "    deployment_script: {},\n", self.deployment_script)?;
        write!(f, "    force_abi: {},\n", self.force_abi)?;
        #[cfg(feature = "use_presets")]
        write!(f, "    preset_file_path: {},\n", self.preset_file_path)?;
        write!(f, "    base_directory: {},\n", self.base_directory)?;
        write!(f, "    guidance_file: {},\n", self.guidance_file)?;
        write!(f, "    state_file: {:?},\n", self.state_file)?;
        write!(f, "    build_command: {:?},\n", self.build_command)?;
        write!(f, "}}")
    }
}

enum EVMTargetType {
    Glob,
    Address,
    AnvilFork,
    Config,
    Setup,
}

impl EVMTargetType {
    fn as_str(&self) -> &'static str {
        match self {
            EVMTargetType::Glob => "glob",
            EVMTargetType::Address => "address",
            EVMTargetType::AnvilFork => "anvil_fork",
            EVMTargetType::Config => "config",
            EVMTargetType::Setup => "setup",
        }
    }

    fn from_str(s: &str) -> Self {
        match s {
            "glob" => EVMTargetType::Glob,
            "address" => EVMTargetType::Address,
            "anvil_fork" => EVMTargetType::AnvilFork,
            "config" => EVMTargetType::Config,
            "setup" => EVMTargetType::Setup,
            _ => panic!("Invalid target type"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OracleType {
    ERC20,
    Pair,
    Reentrancy,
    ArbitraryCall,
    MathCalculate,
    Echidna,
    StateComparison,
    TypedBug,
    SelfDestruct,
    Invariant,
    NFT,
    FeeOnTransfer,
    Approval,
    CrossChain,
    Rebasing,
    Function,
    Ownership,
    ERC4626,
    Freshness,    // Feature 036: stale-price / Ghost-#3 detector (auto-activated, ABI fingerprint)
    TemporalSkim, // Feature 036: time-extracted-value skim detector (--temporal-skimming flag)
}

impl OracleType {
    fn as_str(&self) -> &'static str {
        match self {
            OracleType::ERC20 => "erc20",
            OracleType::Pair => "pair",
            OracleType::Reentrancy => "reentrancy",
            OracleType::ArbitraryCall => "arbitrary_call",
            OracleType::MathCalculate => "math_calculate",
            OracleType::Echidna => "echidna",
            OracleType::StateComparison => "state_comparison",
            OracleType::TypedBug => "typed_bug",
            OracleType::SelfDestruct => "selfdestruct",
            OracleType::Invariant => "invariant",
            OracleType::NFT => "nft",
            OracleType::FeeOnTransfer => "fee_on_transfer",
            OracleType::Approval => "approval",
            OracleType::CrossChain => "crosschain",
            OracleType::Rebasing => "rebasing",
            OracleType::Function => "function",
            OracleType::Ownership => "ownership",
            OracleType::ERC4626 => "erc4626",
            OracleType::Freshness => "freshness",
            OracleType::TemporalSkim => "temporal_skim",
        }
    }

    fn from_str(s: &str) -> Self {
        match s {
            "erc20" => OracleType::ERC20,
            "pair" => OracleType::Pair,
            "reentrancy" => OracleType::Reentrancy,
            "arbitrary_call" => OracleType::ArbitraryCall,
            "math_calculate" => OracleType::MathCalculate,
            "echidna" => OracleType::Echidna,
            "state_comparison" => OracleType::StateComparison,
            "typed_bug" => OracleType::TypedBug,
            "selfdestruct" => OracleType::SelfDestruct,
            "invariant" => OracleType::Invariant,
            "nft" => OracleType::NFT,
            "fee_on_transfer" => OracleType::FeeOnTransfer,
            "approval" => OracleType::Approval,
            "crosschain" => OracleType::CrossChain,
            "rebasing" => OracleType::Rebasing,
            "function" => OracleType::Function,
            "ownership" | "ownership_leak" => OracleType::Ownership,
            "erc4626" => OracleType::ERC4626,
            "freshness" => OracleType::Freshness,
            "temporal_skim" => OracleType::TemporalSkim,
            _ => panic!("Invalid detector type: {}", s),
        }
    }

    fn from_strs(s: &str) -> Vec<Self> {
        let mut results = Vec::new();

        for detector in s.split(',') {
            let detector = detector.trim();
            if detector.is_empty() {
                continue;
            }

            if detector == "all" {
                return vec![
                    OracleType::ERC20,
                    OracleType::ERC4626,
                    OracleType::Pair,
                    OracleType::Reentrancy,
                    OracleType::ArbitraryCall,
                    OracleType::MathCalculate,
                    OracleType::Echidna,
                    OracleType::StateComparison,
                    OracleType::TypedBug,
                    OracleType::SelfDestruct,
                    OracleType::Invariant,
                    OracleType::NFT,
                    OracleType::FeeOnTransfer,
                    OracleType::Approval,
                    OracleType::CrossChain,
                    OracleType::Rebasing,
                    OracleType::Function,
                    OracleType::Ownership,
                    OracleType::Freshness,
                    OracleType::TemporalSkim,
                ];
            }
            if detector == "high_confidence" {
                return vec![
                    OracleType::ERC20,
                    OracleType::Pair,
                    OracleType::ArbitraryCall,
                    OracleType::Echidna,
                    OracleType::TypedBug,
                    OracleType::SelfDestruct,
                    OracleType::Invariant,
                    OracleType::Function,
                ];
            }

            // Feature 020-C — LeakClass SSOT drives selection. A CANONICAL class string
            // (e.g. "value_leak", "permission_leak", "ownership_leak") expands to that
            // primitive's full oracle set via `LeakClass::oracles()`. Matched on `as_str()`
            // ONLY — never the legacy per-oracle aliases in `LeakClass::from_str` — so bare
            // oracle names ("fee_on_transfer", "function") keep their exact single-oracle
            // mapping and existing `-d <oracle>` invocations stay byte-identical.
            if let Some(lc) = crate::evm::leak_class::LeakClass::ALL
                .iter()
                .find(|lc| lc.as_str() == detector)
            {
                results.extend_from_slice(lc.oracles());
                continue;
            }

            results.push(OracleType::from_str(detector));
        }
        results
    }
}

#[allow(clippy::type_complexity)]
pub fn evm_main(mut args: EvmArgs) {
    args.setup_file = args.deployment_script;
    let target = args.target.clone();
    if !args.base_directory.is_empty() {
        std::env::set_current_dir(args.base_directory).unwrap();
    }

    let work_dir = args.work_dir.clone();
    let work_path = Path::new(work_dir.as_str());
    let _ = std::fs::create_dir_all(work_path);

    let mut target_type: EVMTargetType = match args.target_type {
        Some(v) => EVMTargetType::from_str(v.as_str()),
        None => {
            // infer target type from args
            if args.target.starts_with("0x") {
                EVMTargetType::Address
            } else {
                EVMTargetType::Glob
            }
        }
    };

    let is_onchain = args.chain_type.is_some() || args.onchain_url.is_some();

    let mut onchain = if is_onchain {
        let block_number = args.onchain_block_number.unwrap_or(0);
        let chain_type = args.chain_type.as_ref().map(|s| s.as_str());
        let custom_url = args.onchain_url.as_ref().map(|s| s.as_str());
        match (chain_type, custom_url) {
            // Custom URL overrides chain default even when -c is specified.
            (_, Some(url)) => Some(OnChainConfig::new_raw(
                url.to_string(),
                args.onchain_chain_id.unwrap_or_else(|| {
                    chain_type
                        .and_then(|c| Chain::from_str(c).ok())
                        .unwrap_or(Chain::ETH)
                        .get_chain_id()
                }),
                block_number,
                args.onchain_explorer_url.clone().unwrap_or_else(|| {
                    chain_type
                        .and_then(|c| Chain::from_str(c).ok())
                        .unwrap_or(Chain::ETH)
                        .get_chain_etherscan_base()
                }),
                args.onchain_chain_name.clone().unwrap_or_else(|| {
                    chain_type.unwrap_or("eth").to_string()
                }),
            )),
            (Some(chain_str), None) => {
                let chain = Chain::from_str(chain_str).expect("Invalid chain type");
                Some(OnChainConfig::new(chain, block_number))
            }
            (None, None) => unreachable!(), // is_onchain guarantees at least one is set
        }
    } else {
        None
    };

    solution::init_cli_args(target, work_dir, &onchain);
    let _onchain_clone = onchain.clone();

    // Deploy the UniversalLiquidationSimulator onto the fork so the pair
    // discovery path can use it for ERC-4626, Curve, and fee-on-transfer tokens.
    if let Some(ref mut oc) = onchain {
        oc.deploy_liquidation_simulator();
    }

    let etherscan_api_key = match args.onchain_etherscan_api_key {
        Some(v) => v,
        None => std::env::var("ETHERSCAN_API_KEY").unwrap_or_default(),
    };

    if onchain.is_some() && !etherscan_api_key.is_empty() {
        onchain.as_mut().unwrap().etherscan_api_key = etherscan_api_key.split(',').map(|s| s.to_string()).collect();
    }
    let erc20_producer = Rc::new(RefCell::new(ERC20Producer::new()));

    let flashloan_oracle = Rc::new(RefCell::new(IERC20OracleFlashloan::new(erc20_producer.clone())));

    // let harness_code = "oracle_harness()";
    // let mut harness_hash: [u8; 4] = [0; 4];
    // set_hash(harness_code, &mut harness_hash);
    // let mut function_oracle =
    //     FunctionHarnessOracle::new_no_condition(EVMAddress::zero(),
    // Vec::from(harness_hash));

    let mut oracles: Vec<
        Rc<
            RefCell<
                dyn Oracle<
                    EVMState,
                    EVMAddress,
                    revm_interpreter::bytecode::Bytecode,
                    bytes::Bytes,
                    EVMAddress,
                    revm_primitives::ruint::Uint<256, 4>,
                    Vec<u8>,
                    EVMInput,
                    FuzzState<
                        EVMInput,
                        EVMState,
                        EVMAddress,
                        EVMAddress,
                        Vec<u8>,
                        ConciseEVMInput,
                    >,
                    ConciseEVMInput,
                    EVMQueueExecutor,
                >,
            >,
        >,
    > = vec![];

    let mut producers: Vec<
        Rc<
            RefCell<
                dyn Producer<
                    EVMState,
                    EVMAddress,
                    _,
                    _,
                    EVMAddress,
                    EVMU256,
                    Vec<u8>,
                    EVMInput,
                    EVMFuzzState,
                    ConciseEVMInput,
                    EVMQueueExecutor,
                >,
            >,
        >,
    > = vec![];

    let oracle_types = OracleType::from_strs(args.detectors.as_str());

    if oracle_types.contains(&OracleType::Pair) {
        oracles.push(Rc::new(RefCell::new(PairBalanceOracle::new())));
    }

    if oracle_types.contains(&OracleType::ERC20) {
        oracles.push(flashloan_oracle.clone());
        producers.push(erc20_producer);
    }

    let is_onchain = onchain.is_some();
    let mut state: EVMFuzzState = FuzzState::new(args.seed);

    let mut proxy_deploy_codes: Vec<String> = vec![];

    if args.fetch_tx_data {
        match reqwest::blocking::get(&args.proxy_address).and_then(|r| r.text()) {
            Ok(response) => {
                match serde_json::from_str::<Vec<Data>>(&response) {
                    Ok(data) => {
                        for d in data {
                            if d.body.method != "eth_sendRawTransaction" {
                                continue;
                            }
                            let tx = match d.body.params {
                                Some(v) => v,
                                None => continue,
                            };
                            let params: Vec<String> = match serde_json::from_value(tx) {
                                Ok(p) => p,
                                Err(e) => { warn!("--fetch-tx-data: failed to parse tx params: {}", e); continue; }
                            };
                            if params.is_empty() { continue; }
                            let data = params[0].trim_start_matches("0x");
                            let bytes_data = match hex::decode(data) {
                                Ok(b) => b,
                                Err(e) => { warn!("--fetch-tx-data: hex decode failed: {}", e); continue; }
                            };
                            let transaction: Transaction = match rlp::decode(&bytes_data) {
                                Ok(t) => t,
                                Err(e) => { warn!("--fetch-tx-data: RLP decode failed: {}", e); continue; }
                            };
                            proxy_deploy_codes.push(hex::encode(transaction.input));
                        }
                    }
                    Err(e) => warn!("--fetch-tx-data: failed to parse proxy response as JSON: {}", e),
                }
            }
            Err(e) => warn!(
                "--fetch-tx-data: could not reach proxy at {} ({}). Continuing without constructor args — start the proxy server or remove --fetch-tx-data.",
                args.proxy_address, e
            ),
        }
    }

    let constructor_args_map = parse_constructor_args_string(args.constructor_args);

    if !args.builder_artifacts_url.is_empty() || !args.builder_artifacts_file.is_empty() || args.build_command.len() > 0
    {
        if onchain.is_some() {
            target_type = EVMTargetType::AnvilFork;
        } else if !args.setup_file.is_empty() {
            target_type = EVMTargetType::Setup;
        } else if !args.offchain_config_url.is_empty() || !args.offchain_config_file.is_empty() {
            target_type = EVMTargetType::Config;
        } else {
            panic!("Please specify --deployment-script (The contract that deploys the project) or --offchain-config-file (JSON for deploying the project)");
        }
    }

    let offchain_artifacts = if !args.builder_artifacts_url.is_empty() {
        Some(OffChainArtifact::from_json_url(args.builder_artifacts_url).expect("failed to parse builder artifacts"))
    } else if !args.builder_artifacts_file.is_empty() {
        Some(OffChainArtifact::from_file(args.builder_artifacts_file).expect("failed to parse builder artifacts"))
    } else if args.build_command.len() > 0 {
        let command = args.build_command.join(" ");
        Some(OffChainArtifact::from_command(command).expect("Failed to build the project"))
    } else {
        None
    };

    let offchain_config = if !args.offchain_config_url.is_empty() {
        Some(OffchainConfig::from_json_url(args.offchain_config_url).expect("failed to parse offchain config"))
    } else if !args.offchain_config_file.is_empty() {
        Some(OffchainConfig::from_file(args.offchain_config_file).expect("failed to parse offchain config"))
    } else {
        None
    };

    let force_abis = args
        .force_abi
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|x| {
            let runes = x.split(':').collect_vec();
            assert_eq!(runes.len(), 2, "Invalid force abi format");
            let abi = std::fs::read_to_string(runes[1]).expect("Failed to read abi file");
            (runes[0].to_string(), abi)
        })
        .collect::<HashMap<_, _>>();

    let mut contract_loader = match target_type {
        EVMTargetType::Glob => ContractLoader::from_glob(
            args.target.as_str(),
            &mut state,
            &proxy_deploy_codes,
            &constructor_args_map,
            args.target.clone(),
            Some(args.base_path.clone()),
        ),
        EVMTargetType::Config => ContractLoader::from_config(
            &offchain_artifacts.expect("offchain artifacts is required for config target type"),
            &offchain_config.expect("offchain config is required for config target type"),
        ),
        EVMTargetType::AnvilFork => {
            let addresses: Vec<EVMAddress> = args
                .target
                .split(',')
                .map(|s| EVMAddress::from_str(s).unwrap())
                .collect();
            ContractLoader::from_fork(
                &offchain_artifacts.expect("offchain artifacts is required for config target type"),
                onchain.as_mut().expect("onchain is required to fork anvil"),
                HashSet::from_iter(addresses),
            )
        }
        EVMTargetType::Setup => ContractLoader::from_setup(
            &offchain_artifacts.expect("offchain artifacts is required for config target type"),
            args.setup_file,
            args.work_dir.clone(),
            &etherscan_api_key,
        ),
        EVMTargetType::Address => {
            if onchain.is_none() {
                panic!("Onchain is required for address target type");
            }
            let mut args_target = args.target.clone();

            let addresses: Vec<EVMAddress> = args_target
                .split(',')
                .map(|s| EVMAddress::from_str(s).unwrap())
                .collect();
            ContractLoader::from_address(
                onchain.as_mut().unwrap(),
                HashSet::from_iter(addresses),
            )
        }
    };

    contract_loader.force_abi(force_abis);

    // Feature 015: --reflexive-lever is inert without its two prerequisites, so it
    // auto-enables them. Warn loudly when it has to, so the run's actual configuration
    // is never silently different from the flags the user typed.
    if args.reflexive_lever {
        if !(args.campaign_orchestrator || args.bounty) {
            warn!("--reflexive-lever auto-enabled --campaign-orchestrator (the lever has no frame to promote into without it)");
        }
        if !args.impact_eth_gradient {
            warn!("--reflexive-lever auto-enabled --impact-eth-gradient (the ledger objective the lever amplifies)");
        }
    }

    let mut config = Config {
        contract_loader,
        only_fuzz: if !args.only_fuzz.is_empty() {
            args.only_fuzz
                .split(',')
                .map(|s| EVMAddress::from_str(s).expect("failed to parse only fuzz"))
                .collect()
        } else {
            HashSet::new()
        },
        onchain,
        concolic: args.concolic,
        concolic_caller: args.concolic_caller,
        concolic_timeout: args.concolic_timeout,
        concolic_num_threads: {
            if args.concolic_num_threads == 0 {
                num_cpus::get()
            } else {
                args.concolic_num_threads
            }
        },
        impact_eth_gradient: args.impact_eth_gradient || args.reflexive_lever,
        oracle: oracles,
        producers,
        flashloan: args.flashloan || args.bounty,
        onchain_storage_fetching: if is_onchain {
            Some(
                StorageFetchingMode::from_str(args.onchain_storage_fetching.as_str())
                    .expect("unknown storage fetching mode"),
            )
        } else {
            None
        },
        replay_file: args.replay_file,
        flashloan_oracle,
        selfdestruct_oracle: oracle_types.contains(&OracleType::SelfDestruct),
        ownership_oracle: oracle_types.contains(&OracleType::Ownership),
        reentrancy_oracle: oracle_types.contains(&OracleType::Reentrancy),
        work_dir: args.work_dir.clone(),
        write_relationship: args.write_relationship,
        run_forever: args.run_forever,
        sha3_bypass: args.sha3_bypass,
        base_path: args.base_path,
        echidna_oracle: oracle_types.contains(&OracleType::Echidna),
        invariant_oracle: oracle_types.contains(&OracleType::Invariant),
        nft_oracle: oracle_types.contains(&OracleType::NFT),
        fee_on_transfer_oracle: oracle_types.contains(&OracleType::FeeOnTransfer),
        approval_oracle: oracle_types.contains(&OracleType::Approval),
        crosschain_oracle: oracle_types.contains(&OracleType::CrossChain),
        rebasing_oracle: oracle_types.contains(&OracleType::Rebasing),
        panic_on_bug: args.panic_on_bug,
        spec_id: args.spec_id,
        typed_bug: oracle_types.contains(&OracleType::TypedBug),
        arbitrary_external_call: oracle_types.contains(&OracleType::ArbitraryCall),
        math_calculate_oracle: oracle_types.contains(&OracleType::MathCalculate),
        local_files_basedir_pattern: match target_type {
            EVMTargetType::Glob => Some(args.target),
            _ => None,
        },
        #[cfg(feature = "use_presets")]
        preset_file_path: args.preset_file_path,
        #[cfg(feature = "use_presets")]
        preset_only: args.preset_only,
        load_corpus: args.load_corpus,
        value_capture: args.value_capture || args.bounty,
        campaign_orchestrator: args.campaign_orchestrator || args.bounty || args.reflexive_lever,
        ghost_identities: args.ghost_identities || args.bounty,
        temporal_skimming: args.temporal_skimming || args.bounty,
        // Feature 015: reflexive lever auto-enables its two hard prerequisites (above:
        // campaign_orchestrator + impact_eth_gradient OR'd with args.reflexive_lever).
        reflexive_lever: args.reflexive_lever,
        dimension_warp: args.dimension_warp,
        causal_identity: args.causal_identity,
        injection_detect: args.injection_detect,
        injection_persist: args.injection_persist,
        injection_provenance: args.injection_provenance,
        injection_feedback: false,
        oracle_detection: args.oracle_detection,
        flashloan_detection: args.flashloan_detection,
        oracle_staleness: args.oracle_staleness,
        empty_state_guard: args.empty_state_guard,
        dos_detection: args.dos_detection,
        guidance_file: args.guidance_file.clone(),
        state_file: args.state_file.clone(),
        etherscan_api_key,
    };

    let mut abis_map: HashMap<String, Vec<Vec<serde_json::Value>>> = HashMap::new();

    for contract_info in config.contract_loader.contracts.clone() {
        let abis: Vec<serde_json::Value> = contract_info
            .abi
            .iter()
            .map(|config| {
                json!({
                    hex::encode(config.function): format!("{}{}", &config.function_name, &config.abi)
                })
            })
            .collect();
        abis_map
            .entry(hex::encode(contract_info.deployed_address))
            .or_default()
            .push(abis);
    }

    let json_str = serde_json::to_string(&abis_map).expect("Failed to serialize ABI map to JSON");

    let abis_json = format!("{}/abis.json", args.work_dir.clone().as_str());

    utils::try_write_file(&abis_json, &json_str, true).unwrap();

    // Pre-detect ERC-20 balance storage slots for all known contracts.
    // This ensures seed_erc20_balances writes to the correct slot even for
    // non-OpenZeppelin tokens (e.g., DAI's DS-Token uses slot 8).
    if let Some(ref mut oc) = config.onchain {
        use onchain::ChainConfig;
        use slot_detector::detect_balance_slot;
        for contract_info in &config.contract_loader.contracts {
            detect_balance_slot(contract_info.deployed_address, oc as &mut dyn ChainConfig);
        }
    }

    evm_fuzzer(config, &mut state)
}

fn test_evm_offchain_setup() {
    let mut args = EvmArgs {
        proxy_address: String::from("http://localhost:5001/data"),
        onchain_storage_fetching: String::from("onebyone"),
        concolic_timeout: 1000,
        detectors: String::from("high_confidence"),
        work_dir: String::from("work_dir"),
        seed: 1667840158231589000,
        spec_id: String::from("Latest"),
        // deployment_script: String::from("test/foundry/invariants/BaseInvariant.t.sol:BaseInvariant"),
        deployment_script: String::from("CounterLibByLibTest"),
        build_command: vec![String::from("forge"), String::from("build")],
        ..Default::default()
    };

    args.setup_file = args.deployment_script;
    if !args.base_directory.is_empty() {
        std::env::set_current_dir(args.base_directory).unwrap();
    }

    let work_dir = args.work_dir.clone();
    let work_path = Path::new(work_dir.as_str());
    let _ = std::fs::create_dir_all(work_path);

    let mut target_type: EVMTargetType = EVMTargetType::Setup;

    let erc20_producer = Rc::new(RefCell::new(ERC20Producer::new()));

    let flashloan_oracle = Rc::new(RefCell::new(IERC20OracleFlashloan::new(erc20_producer.clone())));

    let mut oracles: Vec<
        Rc<
            RefCell<
                dyn Oracle<
                    EVMState,
                    EVMAddress,
                    revm_interpreter::bytecode::Bytecode,
                    bytes::Bytes,
                    EVMAddress,
                    revm_primitives::ruint::Uint<256, 4>,
                    Vec<u8>,
                    EVMInput,
                    FuzzState<EVMInput, EVMState, EVMAddress, EVMAddress, Vec<u8>, ConciseEVMInput>,
                    ConciseEVMInput,
                    EVMQueueExecutor,
                >,
            >,
        >,
    > = vec![];

    let mut producers: Vec<
        Rc<
            RefCell<
                dyn Producer<
                    EVMState,
                    EVMAddress,
                    _,
                    _,
                    EVMAddress,
                    EVMU256,
                    Vec<u8>,
                    EVMInput,
                    EVMFuzzState,
                    ConciseEVMInput,
                    EVMQueueExecutor,
                >,
            >,
        >,
    > = vec![];

    let oracle_types = OracleType::from_strs(args.detectors.as_str());

    if oracle_types.contains(&OracleType::Pair) {
        oracles.push(Rc::new(RefCell::new(PairBalanceOracle::new())));
    }

    if oracle_types.contains(&OracleType::ERC20) {
        oracles.push(flashloan_oracle.clone());
        producers.push(erc20_producer);
    }

    let mut state: EVMFuzzState = FuzzState::new(args.seed);

    let offchain_artifacts = if !args.builder_artifacts_url.is_empty() {
        Some(OffChainArtifact::from_json_url(args.builder_artifacts_url).expect("failed to parse builder artifacts"))
    } else if !args.builder_artifacts_file.is_empty() {
        Some(OffChainArtifact::from_file(args.builder_artifacts_file).expect("failed to parse builder artifacts"))
    } else if args.build_command.len() > 0 {
        let command = args.build_command.join(" ");
        Some(OffChainArtifact::from_command(command).expect("Failed to build the project"))
    } else {
        None
    };

    let mut contract_loader = ContractLoader::from_setup(
        &offchain_artifacts.expect("offchain artifacts is required for config target type"),
        args.setup_file,
        args.work_dir.clone(),
        "",
    );

    // Feature 015: --reflexive-lever is inert without its two prerequisites, so it
    // auto-enables them. Warn loudly when it has to, so the run's actual configuration
    // is never silently different from the flags the user typed.
    if args.reflexive_lever {
        if !(args.campaign_orchestrator || args.bounty) {
            warn!("--reflexive-lever auto-enabled --campaign-orchestrator (the lever has no frame to promote into without it)");
        }
        if !args.impact_eth_gradient {
            warn!("--reflexive-lever auto-enabled --impact-eth-gradient (the ledger objective the lever amplifies)");
        }
    }

    let config = Config {
        contract_loader,
        only_fuzz: HashSet::new(),
        onchain: None,
        concolic: args.concolic,
        concolic_caller: args.concolic_caller,
        concolic_timeout: args.concolic_timeout,
        concolic_num_threads: {
            if args.concolic_num_threads == 0 {
                num_cpus::get()
            } else {
                args.concolic_num_threads
            }
        },
        impact_eth_gradient: args.impact_eth_gradient || args.reflexive_lever,
        oracle: oracles,
        producers,
        flashloan: args.flashloan || args.bounty,
        onchain_storage_fetching: None,
        replay_file: args.replay_file,
        flashloan_oracle,
        selfdestruct_oracle: oracle_types.contains(&OracleType::SelfDestruct),
        ownership_oracle: oracle_types.contains(&OracleType::Ownership),
        reentrancy_oracle: oracle_types.contains(&OracleType::Reentrancy),
        work_dir: args.work_dir.clone(),
        write_relationship: args.write_relationship,
        run_forever: args.run_forever,
        sha3_bypass: args.sha3_bypass,
        base_path: args.base_path,
        echidna_oracle: oracle_types.contains(&OracleType::Echidna),
        invariant_oracle: oracle_types.contains(&OracleType::Invariant),
        nft_oracle: oracle_types.contains(&OracleType::NFT),
        fee_on_transfer_oracle: oracle_types.contains(&OracleType::FeeOnTransfer),
        approval_oracle: oracle_types.contains(&OracleType::Approval),
        crosschain_oracle: oracle_types.contains(&OracleType::CrossChain),
        rebasing_oracle: oracle_types.contains(&OracleType::Rebasing),
        panic_on_bug: args.panic_on_bug,
        spec_id: args.spec_id,
        typed_bug: oracle_types.contains(&OracleType::TypedBug),
        arbitrary_external_call: oracle_types.contains(&OracleType::ArbitraryCall),
        math_calculate_oracle: oracle_types.contains(&OracleType::MathCalculate),
        local_files_basedir_pattern: match target_type {
            EVMTargetType::Glob => Some(args.target),
            _ => None,
        },
        #[cfg(feature = "use_presets")]
        preset_file_path: args.preset_file_path,
        #[cfg(feature = "use_presets")]
        preset_only: args.preset_only,
        load_corpus: args.load_corpus,
        value_capture: args.value_capture || args.bounty,
        campaign_orchestrator: args.campaign_orchestrator || args.bounty || args.reflexive_lever,
        ghost_identities: args.ghost_identities || args.bounty,
        temporal_skimming: args.temporal_skimming || args.bounty,
        // Feature 015: reflexive lever auto-enables campaign_orchestrator (above) +
        // impact_eth_gradient (OR'd with args.reflexive_lever at their fields).
        reflexive_lever: args.reflexive_lever,
        dimension_warp: args.dimension_warp,
        causal_identity: args.causal_identity,
        injection_detect: args.injection_detect,
        injection_persist: args.injection_persist,
        injection_provenance: args.injection_provenance,
        injection_feedback: false,
        oracle_detection: args.oracle_detection,
        flashloan_detection: args.flashloan_detection,
        oracle_staleness: args.oracle_staleness,
        empty_state_guard: args.empty_state_guard,
        dos_detection: args.dos_detection,
        guidance_file: args.guidance_file.clone(),
        state_file: args.state_file.clone(),
        etherscan_api_key: String::from(""),
    };

    let mut abis_map: HashMap<String, Vec<Vec<serde_json::Value>>> = HashMap::new();

    for contract_info in config.contract_loader.contracts.clone() {
        let abis: Vec<serde_json::Value> = contract_info
            .abi
            .iter()
            .map(|config| {
                json!({
                    hex::encode(config.function): format!("{}{}", &config.function_name, &config.abi)
                })
            })
            .collect();
        abis_map
            .entry(hex::encode(contract_info.deployed_address))
            .or_default()
            .push(abis);
    }

    let json_str = serde_json::to_string(&abis_map).expect("Failed to serialize ABI map to JSON");

    debug!("work_dir: {:?}", args.work_dir.clone().as_str());
    let abis_json = format!("{}/abis.json", args.work_dir.clone().as_str());

    utils::try_write_file(&abis_json, &json_str, true).unwrap();
    evm_fuzzer(config, &mut state)
}

#[cfg(test)]
mod test {
    use super::parse_constructor_args_string;
    use super::OracleType;

    #[test]
    fn test_parse_constructor_args_string() {
        let input =
            "Test1:88,0x97C6D26d7E0D316850A967b46845E15a32666d25;Test2:88,0x97C6D26d7E0D316850A967b46845E15a32666d25"
                .to_string();
        let ret = parse_constructor_args_string(input);
        // println!("constructor args: {:?}", ret);
    }

    /// Feature 020-C golden test: legacy bare-oracle `-d` strings resolve BYTE-IDENTICALLY
    /// (single oracle each), while canonical LeakClass strings expand to the primitive's full
    /// oracle set. Guards against the split-brain regression where routing a bare name through
    /// the class would silently widen its oracle set.
    #[test]
    fn detector_routing_legacy_vs_class() {
        // Legacy bare oracle names — unchanged single-oracle mapping.
        assert_eq!(OracleType::from_strs("function"), vec![OracleType::Function]);
        assert_eq!(OracleType::from_strs("reentrancy"), vec![OracleType::Reentrancy]);
        // Critically: "fee_on_transfer" stays a SINGLE oracle, NOT the 3-oracle Value class.
        assert_eq!(OracleType::from_strs("fee_on_transfer"), vec![OracleType::FeeOnTransfer]);

        // Canonical class strings expand via LeakClass::oracles() (the SSOT).
        // §1 audit: orphan bindings added — Permission now includes Approval, Value includes
        // Pair+MathCalculate, Ownership includes SelfDestruct, Invariant includes TypedBug,
        // Message includes CrossChain.
        assert_eq!(
            OracleType::from_strs("permission_leak"),
            vec![OracleType::Function, OracleType::Approval]
        );
        // Feature 036: Freshness + TemporalSkim added to Value class.
        assert_eq!(
            OracleType::from_strs("value_leak"),
            vec![OracleType::FeeOnTransfer, OracleType::ERC20, OracleType::Rebasing, OracleType::ERC4626, OracleType::Pair, OracleType::MathCalculate, OracleType::Freshness, OracleType::TemporalSkim]
        );
        assert_eq!(
            OracleType::from_strs("ownership_leak"),
            vec![OracleType::Ownership, OracleType::NFT, OracleType::SelfDestruct]
        );
        assert_eq!(OracleType::from_strs("control_flow_leak"), vec![OracleType::Reentrancy]);

        // Aggregate keywords are untouched by the class layer and still include Ownership (020-B).
        assert!(OracleType::from_strs("all").contains(&OracleType::Ownership));
        // ERC4626 was previously absent from the "all" list despite being a registered OracleType
        // and listed in LeakClass::Value.oracles(). Fixed: -d all now includes it.
        assert!(OracleType::from_strs("all").contains(&OracleType::ERC4626));
        // Feature 036: Freshness and TemporalSkim now have OracleType identity and appear in "all".
        assert!(OracleType::from_strs("all").contains(&OracleType::Freshness));
        assert!(OracleType::from_strs("all").contains(&OracleType::TemporalSkim));
        assert!(!OracleType::from_strs("high_confidence").contains(&OracleType::Freshness));
        assert!(!OracleType::from_strs("high_confidence").contains(&OracleType::TemporalSkim));
        assert!(!OracleType::from_strs("high_confidence").contains(&OracleType::Ownership));
    }
}
