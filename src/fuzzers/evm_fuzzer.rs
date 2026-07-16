use std::{cell::RefCell, collections::HashMap, fs::File, io::Read, ops::Deref, path::Path, process::exit, rc::Rc};

use bytes::Bytes;
use glob::glob;
use itertools::Itertools;
use libafl::{
    feedbacks::Feedback,
    prelude::{HasMetadata, MaxMapFeedback, SimpleEventManager, SimpleMonitor, StdMapObserver},
    Evaluator,
    Fuzzer,
};
use libafl_bolts::tuples::tuple_list;
use revm_interpreter::bytecode::Bytecode;
use tracing::{debug, error, info};

use crate::{
    evm::{
        abi::{ABIAddressToInstanceMap, BoxedABI},
        blaz::builder::ArtifactInfoMetadata,
        concolic::{
            concolic_host::CONCOLIC_TIMEOUT,
            concolic_stage::{ConcolicFeedbackWrapper, ConcolicStage},
        },
        config::Config,
        contract_utils::FIX_DEPLOYER,
        corpus_initializer::EVMCorpusInitializer,
        cov_stage::CoverageStage,
        feedbacks::{Sha3WrappedFeedback, TokenBalanceFeedback},
        host::{
            FuzzHost,
            ACTIVE_MATCH_EXT_CALL,
            CALL_UNTIL,
            CMP_MAP,
            JMP_MAP,
            PANIC_ON_BUG,
            READ_MAP,
            WRITE_MAP,
            WRITE_RELATIONSHIPS,
        },
        input::{ConciseEVMInput, EVMInput},
        middlewares::{
            call_printer::CallPrinter,
            cheatcode::Cheatcode,
            coverage::{Coverage, EVAL_COVERAGE},
            dos_detector::DoSDetector,
            empty_state_guard::EmptyStateGuard,
            fee_on_transfer_detector::FeeOnTransferDetector,
            flashloan_oracle::FlashloanOracle,
            middleware::Middleware,
            oracle_staleness::OracleStaleness,
            oracle_tracker::OracleTracker,
            function_auth::FunctionAuthTracer,
            reentrancy::ReentrancyTracer,
            sha3_bypass::{Sha3Bypass, Sha3TaintAnalysis},
            value_capture::ValueCaptureMiddleware,
        },
        minimizer::EVMMinimizer,
        mutator::FuzzMutator,
        planner::CampaignTargetCache,
        onchain::{flashloan::Flashloan, offchain::OffChainConfig, ChainConfig, OnChain, WHITELIST_ADDR},
        oracles::{
            approval::SuspiciousApprovalOracle,
            arb_call::ArbitraryCallOracle,
            arb_transfer::ArbitraryERC20TransferOracle,
            echidna::EchidnaOracle,
            fee_on_transfer::FeeOnTransferOracle,
            invariant::InvariantOracle,
            nft::NFTOwnershipOracle,
            reentrancy::ReentrancyOracle,
            selfdestruct::SelfdestructOracle,
            typed_bug::TypedBugOracle,
        },
        presets::ExploitTemplate,
        scheduler::{PowerABIMutationalStage, PowerABIScheduler, UncoveredBranchesMetadata},
        types::{fixed_address, EVMAddress, EVMFuzzMutator, EVMFuzzState, EVMQueueExecutor, EVMU256},
        vm::{EVMExecutor, EVMState},
    },
    executor::FuzzExecutor,
    evm::feedbacks::DivergenceFeedback,
    feedback::{CmpFeedback, DataflowFeedback, OracleFeedback},
    fuzzer::{ItyFuzzer, REPLAY, RUN_FOREVER},
    oracle::BugMetadata,
    scheduler::SortedDroppingScheduler,
    state::{FuzzState, HasCaller, HasExecutionResult, HasPresets},
};

#[allow(clippy::type_complexity)]
pub fn evm_fuzzer(
    config: Config<
        EVMState,
        EVMAddress,
        Bytecode,
        Bytes,
        EVMAddress,
        EVMU256,
        Vec<u8>,
        EVMInput,
        EVMFuzzState,
        ConciseEVMInput,
        EVMQueueExecutor,
    >,
    state: &mut EVMFuzzState,
) {
    info!("\n\n ================ EVM Fuzzer Start ===================\n\n");
    let mut config = config;

    // --- Dynamic Auto-Activation based on Compiled Guidance ---
    let guidance_path = if !config.guidance_file.is_empty() {
        Some(config.guidance_file.clone())
    } else if std::path::Path::new("spectrefuzz.guidance").exists() {
        info!("[guidance] found default spectrefuzz.guidance in current directory");
        Some("spectrefuzz.guidance".to_string())
    } else {
        info!("[guidance] no compiled semantic guidance file provided or found. Running baseline concolic.");
        None
    };

    let mut loaded_guidance = None;
    if let Some(path) = guidance_path {
        info!("[guidance] loading compiled semantic guidance from {}", path);
        if let Ok(guidance) = crate::evm::guidance::Guidance::load(&path) {
            let meta = &guidance.meta;
            info!(
                "[guidance] successfully digested: {} contracts, {} functions, {} kill chains, {} invariants",
                meta.num_contracts, meta.num_functions, meta.num_kill_chains, meta.num_invariants
            );
            
            // Auto-configure config flags based on guidance content:
            // 1. Invariants -> InvariantOracle
            if !guidance.oracle.invariants.is_empty() && !config.invariant_oracle {
                config.invariant_oracle = true;
                info!("[guidance] invariants found: auto-activating InvariantOracle");
            }
            
            // 2. Scan functions for sinks
            let mut has_delegatecall = false;
            let mut has_selfdestruct = false;
            for fn_entry in guidance.functions.values() {
                for chain in &fn_entry.kill_chains.chains {
                    if chain.sink.contains("delegatecall") || chain.sink.contains("call") {
                        has_delegatecall = true;
                    }
                    if chain.sink.contains("selfdestruct") {
                        has_selfdestruct = true;
                    }
                }
            }
            
            if has_delegatecall && !config.arbitrary_external_call {
                config.arbitrary_external_call = true;
                info!("[guidance] delegatecall/call sinks found: auto-activating ArbitraryCallOracle");
            }
            if has_selfdestruct && !config.selfdestruct_oracle {
                config.selfdestruct_oracle = true;
                info!("[guidance] selfdestruct sinks found: auto-activating SelfdestructOracle");
            }
            
            // 3. Scan state variables for oracle-related names
            let mut has_oracle_feeds = false;
            for slots in guidance.slot_influence_weights.values() {
                for state_var in slots.keys() {
                    let sv_lower = state_var.to_lowercase();
                    if sv_lower.contains("oracle") || sv_lower.contains("feed") || sv_lower.contains("aggregator") || sv_lower.contains("price") {
                        has_oracle_feeds = true;
                        break;
                    }
                }
            }
            if has_oracle_feeds {
                if !config.reentrancy_oracle {
                    config.reentrancy_oracle = true;
                    info!("[guidance] oracle-related variables found: auto-activating ReentrancyOracle");
                }
                if !config.oracle_staleness {
                    config.oracle_staleness = true;
                    info!("[guidance] oracle-related variables found: auto-activating OracleStaleness middleware");
                }
            }
            
            // 4. Force enable economic/flashloans if guidance is loaded
            if !config.flashloan {
                config.flashloan = true;
                info!("[guidance] economic exploit path loaded: auto-activating flashloan simulation and economic oracle");
            }
            
            loaded_guidance = Some(guidance);
        } else {
            error!("[guidance] failed to parse guidance file at {}", path);
        }
    }

    // create work dir if not exists
    let _path = Path::new(config.work_dir.as_str());

    let monitor = SimpleMonitor::new(|s| info!("{}", s));
    let mut mgr = SimpleEventManager::new(monitor);
    let infant_scheduler = SortedDroppingScheduler::new();
    let scheduler = PowerABIScheduler::new();

    let jmps = unsafe { &mut JMP_MAP };
    let cmps = unsafe { &mut CMP_MAP };
    let reads = unsafe { &mut READ_MAP };
    let writes = unsafe { &mut WRITE_MAP };
    let jmp_observer = unsafe { StdMapObserver::new("jmp", jmps) };

    let deployer = fixed_address(FIX_DEPLOYER);
    let mut fuzz_host = FuzzHost::new(scheduler.clone(), config.work_dir.clone());
    fuzz_host.set_spec_id(config.spec_id);

    // **Note**: cheatcode should be the first middleware because it consumes the
    // step if it is a call to cheatcode_address, and this step should not be
    // visible to other middlewares.
    fuzz_host.add_middlewares(Rc::new(RefCell::new(Cheatcode::new(&config.etherscan_api_key))));

    macro_rules! create_onchain {
        ($onchain: expr) => {{
            let mid = Rc::new(RefCell::new(OnChain::new(
                // scheduler can be cloned because it never uses &mut self
                $onchain,
                config.onchain_storage_fetching.unwrap(),
            )));

            debug!("onchain middleware enabled");
            fuzz_host.add_middlewares(mid.clone());
            mid
        }};
    }

    let onchain_middleware = match config.onchain.clone() {
        Some(onchain) => Some(create_onchain!(onchain)),
        None => {
            // enable active match for offchain fuzzing (todo: handle this more elegantly)
            match &config.contract_loader.setup_data.clone().map(|s| s.onchain_middleware) {
                Some(Some(mid)) => {
                    let mid = Rc::new(RefCell::new(mid.clone()));
                    fuzz_host.add_middlewares(mid.clone());
                    Some(mid)
                }
                _ => {
                    unsafe {
                        ACTIVE_MATCH_EXT_CALL = false;
                    }
                    None
                }
            }
        }
    };

    if config.write_relationship {
        unsafe {
            WRITE_RELATIONSHIPS = true;
        }
    }

    if config.run_forever {
        unsafe {
            RUN_FOREVER = true;
        }
    }

    unsafe {
        PANIC_ON_BUG = config.panic_on_bug;
    }

    if !config.only_fuzz.is_empty() {
        unsafe {
            WHITELIST_ADDR = Some(config.only_fuzz.clone());
        }
    }

    if config.flashloan {
        // we should use real balance of tokens in the contract instead of providing
        // flashloan to contract as well for on chain env
        {
            let chain_cfg: Option<Box<dyn ChainConfig>> = if let Some(onchain) = config.onchain.clone() {
                Some(Box::new(onchain) as Box<dyn ChainConfig>)
            } else if let Some(ref setup_data) = config.contract_loader.setup_data {
                if setup_data.v2_pairs.is_empty() {
                    None
                } else {
                    Some(Box::new(OffChainConfig::new(setup_data).unwrap()) as Box<dyn ChainConfig>)
                }
            } else {
                None
            };

            fuzz_host.add_flashloan_middleware(Flashloan::new(true, chain_cfg, config.flashloan_oracle));
        }
    }
    let sha3_taint = Rc::new(RefCell::new(Sha3TaintAnalysis::new()));

    debug!("sha3 bypass enabled (unconditional)");
    fuzz_host.add_middlewares(Rc::new(RefCell::new(Sha3Bypass::new(sha3_taint.clone()))));

    if config.reentrancy_oracle {
        debug!("reentrancy oracle enabled");
        fuzz_host.add_middlewares(Rc::new(RefCell::new(ReentrancyTracer::new())));
    }

    if config.value_capture {
        debug!("value capture middleware enabled");
        fuzz_host.add_middlewares(Rc::new(RefCell::new(ValueCaptureMiddleware::new())));
    }

    if config.fee_on_transfer_oracle {
        // Inline per-transfer fee measurement feeds FeeOnTransferOracle's evidence
        // (EVMState::fee_observations). Without it the oracle sees nothing.
        debug!("fee-on-transfer inline detector enabled");
        fuzz_host.add_middlewares(Rc::new(RefCell::new(FeeOnTransferDetector::new())));
    }

    if config.causal_identity {
        // Feature 019 Phase A: inline materiality tracker feeds the FunctionOracle's
        // permission-leak gate (EVMState::permission_leak_metadata). Records SSTORE
        // pre≠post deltas and value-CALLs so the oracle can suppress no-op privileged
        // calls (the burn(0,0) false positive). Order vs cmp_linearity is irrelevant —
        // cmp_linearity runs on a separate reexecution pass (feedbacks.rs:138); this
        // reads the accumulated taint bus best-effort, gating on the same-pass delta.
        debug!("causal-identity permission-leak detector enabled");
        fuzz_host.add_middlewares(Rc::new(RefCell::new(FunctionAuthTracer::new())));
    }

    // Feature 014 Phase 1: oracle-gated value movement detection.
    if config.oracle_detection {
        debug!("oracle-gated transfer detector enabled");
        fuzz_host.add_middlewares(Rc::new(RefCell::new(OracleTracker::new())));
    }

    // Feature 014 Phase 3: oracle staleness (missing updatedAt check) detection.
    if config.oracle_staleness {
        debug!("oracle staleness detector enabled");
        fuzz_host.add_middlewares(Rc::new(RefCell::new(OracleStaleness::new())));
    }

    // Feature 014 Phase 4: empty state guard (first-deposit inflation) detection.
    if config.empty_state_guard {
        debug!("empty state guard detector enabled");
        fuzz_host.add_middlewares(Rc::new(RefCell::new(EmptyStateGuard::new())));
    }

    // Feature 014 Phase 2: flash loan oracle manipulation detection.
    if config.flashloan_detection {
        debug!("flash loan oracle manipulation detector enabled");
        fuzz_host.add_middlewares(Rc::new(RefCell::new(FlashloanOracle::new())));
    }

    // Feature 014 Phase 5: DoS via state-dependent revert detection.
    // Depends on Feature 013 Phase 3 (persistent taint on host.tainted_storage).
    if config.dos_detection {
        debug!("DoS via state-dependent revert detector enabled");
        fuzz_host.add_middlewares(Rc::new(RefCell::new(DoSDetector::new())));
    }

    let mut evm_executor: EVMQueueExecutor = EVMExecutor::new(fuzz_host, deployer);

    if config.replay_file.is_some() {
        // add coverage middleware for replay
        unsafe {
            REPLAY = true;
        }
    }

    // moved here to ensure state has ArtifactInfoMetadata during corpus
    // initialization
    if !state.has_metadata::<ArtifactInfoMetadata>() {
        state.add_metadata(ArtifactInfoMetadata::new());
    }
    let mut corpus_initializer = EVMCorpusInitializer::new(
        &mut evm_executor,
        scheduler.clone(),
        infant_scheduler.clone(),
        state,
        config.work_dir.clone(),
    );

    let mut artifacts = corpus_initializer.initialize(&mut config.contract_loader.clone());

    // Legacy topology hints storage removed.

    let mut instance_map = ABIAddressToInstanceMap::new();
    artifacts.address_to_abi_object.iter().for_each(|(addr, abi)| {
        instance_map.map.insert(*addr, abi.clone());
    });

    // Matched-preset selectors → candidate-based campaign target discovery (below).
    // Empty when no preset feature/match → campaign falls back to hardcoded selectors.
    #[allow(unused_mut)]
    let mut preset_chain_selectors: Vec<[u8; 4]> = Vec::new();

    #[cfg(feature = "use_presets")]
    {
        // Start with the baked-in DefiHacksPresets corpus — always loaded,
        // no flag required. This is the third capability gate flipped to
        // default-on in this fork (after Sha3Bypass and flashloan accounting).
        // The README's "80% of previous hacks" claim depends on these
        // templates being matched against the target's deployed contracts.
        // --preset-only isolates the preset language: use ONLY the --preset-file-path
        // templates, skipping the baked-in corpus entirely, so the mutator's preset
        // budget speaks one exploit's shape with zero dilution (controlled experiment).
        let mut exploit_templates = if config.preset_only && !config.preset_file_path.is_empty() {
            info!("[presets] --preset-only: skipping baked-in corpus, using only {}", config.preset_file_path);
            Vec::new()
        } else {
            let baked = ExploitTemplate::baked_in();
            info!("[presets] loaded {} baked-in exploit templates", baked.len());
            baked
        };

        // If --preset-file-path is set, add those templates. Additive by default (layered
        // on the baked set, fuzzland's original behavior); the sole set under --preset-only.
        if !config.preset_file_path.is_empty() {
            let extra = ExploitTemplate::from_filename(config.preset_file_path.clone());
            info!("[presets] loaded {} template(s) from {}", extra.len(), config.preset_file_path);
            exploit_templates.extend(extra);
        }

        let mut sig_to_addr_abi_map = HashMap::new();
        let mut matched_templates = vec![];
        for template in exploit_templates {
            // to match, all function_sigs in the template
            // must exists in all abi.function
            let mut function_sigs = template.function_sigs.clone();
            for (addr, abis) in &artifacts.address_to_abi_object {
                for abi in abis {
                    for (idx, function_sig) in function_sigs.iter().enumerate() {
                        if abi.function == function_sig.value {
                            debug!("matched: {:?} @ {:?}", abi.function, addr);
                            sig_to_addr_abi_map.insert(function_sig.value, (*addr, abi.clone()));
                            function_sigs.remove(idx);
                            break;
                        }
                    }
                }
                if function_sigs.is_empty() {
                    matched_templates.push(template);
                    break;
                }
            }
        }
        let has_preset_match = !matched_templates.is_empty();
        info!("[presets] has_preset_match: {} ({} template(s) fully matched this target)", has_preset_match, matched_templates.len());

        // Candidate-based campaign discovery: the matched exploit's OWN selectors become
        // the campaign's prime/exploit chain candidates (not a hardcoded menu).
        preset_chain_selectors = sig_to_addr_abi_map.keys().cloned().collect();
        info!("[presets] {} matched selector(s) feed campaign candidate discovery", preset_chain_selectors.len());

        state.init_presets(has_preset_match, matched_templates.clone(), sig_to_addr_abi_map);
    }
    let cov_middleware = Rc::new(RefCell::new(Coverage::new(
        artifacts.address_to_name.clone(),
        config.work_dir.clone(),
    )));

    evm_executor.host.add_middlewares(cov_middleware.clone());

    if let Some(guidance) = loaded_guidance {
        state.add_metadata(crate::evm::guidance::GuidanceMetadata::new(guidance));
    }

    state.add_metadata(instance_map);

    evm_executor.host.initialize(state);
    evm_executor.host.initial_block_timestamp = Some(artifacts.initial_env.block.timestamp);

    if let Some(ref path) = config.state_file {
        crate::evm::state_loader::load_snapshot(&mut evm_executor.host, path);
    }

    // Feature 014 Phase 0: populate oracle_selectors from known ABI interfaces.
    for (addr, abis) in &artifacts.address_to_abi_object {
        let selectors: Vec<[u8; 4]> = abis
            .iter()
            .map(|a| a.function)
            .filter(|sel| crate::evm::oracles::freshness::is_oracle_interface(sel))
            .collect();
        if !selectors.is_empty() {
            evm_executor.host.oracle_selectors.insert(*addr, selectors);
        }
    }

    // now evm executor is ready, we can clone it

    let evm_executor_ref = Rc::new(RefCell::new(evm_executor));

    let meta = state.metadata_map_mut().get_mut::<ArtifactInfoMetadata>().unwrap();
    for (addr, build_artifact) in &artifacts.build_artifacts {
        meta.add(*addr, build_artifact.clone());
    }

    for (addr, bytecode) in &mut artifacts.address_to_bytecode {
        unsafe {
            cov_middleware.deref().borrow_mut().on_insert(
                None,
                &mut evm_executor_ref.deref().borrow_mut().host,
                state,
                bytecode,
                *addr,
            );
        }
    }

    let mut feedback = MaxMapFeedback::new(&jmp_observer);
    feedback.init_state(state).expect("Failed to init state");
    // let calibration = CalibrationStage::new(&feedback);
    if config.concolic {
        unsafe { CONCOLIC_TIMEOUT = config.concolic_timeout };
        // Feature 009: the linearity reexecution / dispatch triage only matters when
        // concolic is on (it manages the concolic budget). Gate it on this flag.
        crate::evm::middlewares::cmp_linearity::lin_set_concolic_enabled(true);
    }

    let concolic_stage = ConcolicStage::new(
        config.concolic,
        config.concolic_caller,
        evm_executor_ref.clone(),
        config.concolic_num_threads,
    );
    if config.campaign_orchestrator {
        // The executor stores CampaignIntermediateStates as metadata after a campaign
        // run; its SerdeAny impl needs explicit registration (the others use
        // impl_serdeany!). Without this, the first insert panics "inserted without
        // registration". Safe here: single-threaded setup, before fuzzing starts.
        unsafe {
            crate::evm::types::CampaignIntermediateStatesEVM::register();
        }
        let instance_map = state.metadata_map().get::<ABIAddressToInstanceMap>().cloned();

        // Extract guidance kill-chain prime selectors: functions that reach a drain sink.
        // These expand the candidate pool beyond the 6-entry hardcoded PRIME_SELECTORS list
        // so the guidance dictionary drives discovery for any target, not just known archetypes.
        let guidance_prime_sels: Vec<[u8; 4]> = {
            let mut sels: Vec<[u8; 4]> = state
                .metadata_map()
                .get::<crate::evm::guidance::GuidanceMetadata>()
                .map(|g| {
                    g.guidance.functions.values()
                        .filter(|f| !f.kill_chains.chains.is_empty())
                        .filter_map(|f| f.selector.as_deref())
                        .filter_map(|s| hex::decode(s.trim_start_matches("0x")).ok())
                        .filter(|b| b.len() == 4)
                        .map(|b| [b[0], b[1], b[2], b[3]])
                        .collect()
                })
                .unwrap_or_default();
            sels.sort();
            sels.dedup();
            sels
        };
        if !guidance_prime_sels.is_empty() {
            info!("[guidance] {} kill-chain prime selectors extracted for campaign discovery", guidance_prime_sels.len());
        }

        let cache = instance_map
            .map(|m| CampaignTargetCache::new_with_preset(&m, Vec::new(), &preset_chain_selectors, None, &guidance_prime_sels))
            .unwrap_or_else(|| CampaignTargetCache::new_with_preset(&ABIAddressToInstanceMap::default(), Vec::new(), &preset_chain_selectors, None, &guidance_prime_sels));
        // [pool-tel] one-shot: how many (contract,selector) entries per selector in the
        // campaign candidate pool. Exposes contract-multiplicity bias (e.g. approve on
        // every ERC20 → over-weighted in uniform (contract,selector) sampling).
        {
            use std::collections::BTreeMap;
            let mut prime_counts: BTreeMap<String, usize> = BTreeMap::new();
            for (_a, sel, _abi) in &cache.prime_targets {
                *prime_counts.entry(format!("0x{}", hex::encode(sel))).or_default() += 1;
            }
            info!("[pool-tel] prime_targets={} entries; per-selector contract-counts: {:?}", cache.prime_targets.len(), prime_counts);
        }
        state.add_metadata(cache);
    }

    let mutator: EVMFuzzMutator = FuzzMutator::new(infant_scheduler.clone(), config.campaign_orchestrator, config.ghost_identities, config.temporal_skimming, config.reflexive_lever, config.dimension_warp);

    state.metadata_map_mut().insert(UncoveredBranchesMetadata::new());
    let std_stage = PowerABIMutationalStage::new(mutator);

    let call_printer_mid = Rc::new(RefCell::new(CallPrinter::new(artifacts.address_to_name.clone())));

    let coverage_obs_stage = CoverageStage::new(
        evm_executor_ref.clone(),
        cov_middleware.clone(),
        call_printer_mid.clone(),
        config.work_dir.clone(),
    );

    let mut stages = tuple_list!(std_stage, concolic_stage, coverage_obs_stage);

    let mut executor = FuzzExecutor::new(evm_executor_ref.clone(), tuple_list!(jmp_observer));

    #[cfg(feature = "deployer_is_attacker")]
    state.add_caller(&deployer);
    let cmp_feedback = CmpFeedback::new(cmps, infant_scheduler.clone(), evm_executor_ref.clone());

    // Build attacker set for TokenBalanceFeedback from callers_pool.
    let attackers: std::collections::HashSet<EVMAddress> =
        state.callers_pool.iter().cloned().collect();
    // Feature 011 (Part A): hand the gradient the liquidation engine only when the
    // ETH-value mode is on; otherwise `None` ⇒ original token-unit gradient.
    let eth_engine_ref = config.impact_eth_gradient.then(|| evm_executor_ref.clone());
    let balance_feedback = TokenBalanceFeedback::new(
        attackers,
        infant_scheduler.clone(),
        config.impact_eth_gradient,
        eth_engine_ref,
        config.reflexive_lever,
    );

    // Combine: any new coverage ceiling OR any new fund-extraction ceiling OR
    // any new same-execution oracle-divergence ceiling makes the state
    // interesting and gets added to infant corpus. DivergenceFeedback is safe
    // here only because ItyFuzzer runs objective/oracle feedback before this
    // infant gate and clears divergence before each target execution.
    let divergence_feedback = DivergenceFeedback::new(infant_scheduler.clone());
    let infant_feedback = libafl::feedbacks::EagerOrFeedback::new(
        libafl::feedbacks::EagerOrFeedback::new(cmp_feedback, balance_feedback),
        divergence_feedback,
    );
    let infant_result_feedback = DataflowFeedback::new(reads, writes);

    let mut oracles = config.oracle;

    if config.echidna_oracle {
        let echidna_oracle = EchidnaOracle::new(
            artifacts
                .address_to_abi
                .iter()
                .flat_map(|(address, abis)| {
                    abis.iter()
                        .filter(|abi| abi.function_name.starts_with("echidna_") && abi.abi == "()")
                        .map(|abi| (*address, abi.function.to_vec()))
                        .collect_vec()
                })
                .collect_vec(),
            artifacts
                .address_to_abi
                .iter()
                .flat_map(|(_address, abis)| {
                    abis.iter()
                        .filter(|abi| abi.function_name.starts_with("echidna_") && abi.abi == "()")
                        .map(|abi| (abi.function.to_vec(), abi.function_name.clone()))
                        .collect_vec()
                })
                .collect::<HashMap<Vec<u8>, String>>(),
        );
        oracles.push(Rc::new(RefCell::new(echidna_oracle)));
    }

    if config.invariant_oracle {
        let invariant_oracle = InvariantOracle::new(
            artifacts
                .address_to_abi
                .iter()
                .flat_map(|(address, abis)| {
                    abis.iter()
                        .filter(|abi| abi.function_name.starts_with("invariant_") && abi.abi == "()")
                        .map(|abi| (*address, abi.function.to_vec()))
                        .collect_vec()
                })
                .collect_vec(),
            artifacts
                .address_to_abi
                .iter()
                .flat_map(|(_address, abis)| {
                    abis.iter()
                        .filter(|abi| abi.function_name.starts_with("invariant_") && abi.abi == "()")
                        .map(|abi| (abi.function.to_vec(), abi.function_name.clone()))
                        .collect_vec()
                })
                .collect::<HashMap<Vec<u8>, String>>(),
        );
        oracles.push(Rc::new(RefCell::new(invariant_oracle)));
    }

    if config.nft_oracle {
        let nft_oracle = NFTOwnershipOracle::new(artifacts.address_to_name.clone());
        oracles.push(Rc::new(RefCell::new(nft_oracle)));
    }

    if config.fee_on_transfer_oracle {
        let fee_oracle = FeeOnTransferOracle::new(artifacts.address_to_name.clone());
        oracles.push(Rc::new(RefCell::new(fee_oracle)));
    }

    if config.approval_oracle {
        use std::collections::HashSet as StdHashSet;
        let known_contracts: StdHashSet<EVMAddress> = artifacts.address_to_name.keys().cloned().collect();
        let approval_oracle = SuspiciousApprovalOracle::new(
            known_contracts,
            artifacts.address_to_name.clone(),
        );
        oracles.push(Rc::new(RefCell::new(approval_oracle)));
    }

    if config.crosschain_oracle {
        use crate::evm::oracles::crosschain::CrossChainOracle;
        use std::collections::HashSet as StdHashSet;
        // Start with an empty trusted_bridges set. In practice, users can add
        // known bridge endpoints via the onchain config; the fuzzer will flag
        // any non-trusted caller that successfully invokes a receiver.
        let trusted_bridges: StdHashSet<EVMAddress> = StdHashSet::new();
        let cc_oracle = CrossChainOracle::new(trusted_bridges, artifacts.address_to_name.clone());
        oracles.push(Rc::new(RefCell::new(cc_oracle)));
    }

    if config.rebasing_oracle {
        use crate::evm::oracles::rebasing::RebasingOracle;
        use std::collections::HashSet as StdHashSet;
        let known_contracts: StdHashSet<EVMAddress> = artifacts.address_to_name.keys().cloned().collect();
        let rebasing_oracle = RebasingOracle::new(known_contracts, artifacts.address_to_name.clone());
        oracles.push(Rc::new(RefCell::new(rebasing_oracle)));
    }

    if config.temporal_skimming {
        use crate::evm::oracles::temporal_skim::TemporalSkimOracle;
        let temporal_oracle = TemporalSkimOracle::new(artifacts.address_to_name.clone());
        oracles.push(Rc::new(RefCell::new(temporal_oracle)));
        info!("Temporal Skim oracle activated (--temporal-skimming)");
    }

    // Auto-detected from ABI fingerprinting — no config flag needed.
    // ERC-4626: activated whenever convertToAssets(uint256) is found in any contract ABI.
    if !artifacts.erc4626_vaults.is_empty() {
        use crate::evm::oracles::erc4626::ERC4626Oracle;
        let erc4626_oracle = ERC4626Oracle::new(
            artifacts.erc4626_vaults.clone(),
            artifacts.address_to_name.clone(),
        );
        oracles.push(Rc::new(RefCell::new(erc4626_oracle)));
        info!("ERC-4626 share-price oracle auto-activated for {} vault(s)", artifacts.erc4626_vaults.len());
    }

    // EIP-712: corpus seeds already injected in corpus_initializer.
    // Log detection for visibility.
    if !artifacts.eip712_contracts.is_empty() {
        info!("EIP-712 domain separator detected in {} contract(s) — zero-sig seeds injected", artifacts.eip712_contracts.len());
    }

    // Freshness oracle: auto-activated when Chainlink-style oracle contracts
    // are detected in the ABI (latestRoundData / latestAnswer / getRoundData).
    // Monitors updatedAt field post-execution; flags stale data accepted without
    // a freshness check — Ghost #3 from the DeFi ghost taxonomy.
    if !artifacts.oracle_contracts.is_empty() {
        use crate::evm::oracles::freshness::FreshnessOracle;
        let freshness_oracle = FreshnessOracle::new(
            artifacts.oracle_contracts.clone(),
            artifacts.address_to_name.clone(),
            3600, // 1-hour default staleness threshold (Chainlink heartbeat)
        );
        oracles.push(Rc::new(RefCell::new(freshness_oracle)));
        info!(
            "Freshness oracle auto-activated: {} oracle contract(s) monitored (max staleness: 3600s)",
            artifacts.oracle_contracts.len()
        );
    }

    // Permission-leak oracle: auto-activated when privileged functions are
    // detected in the ABI. All attacker callers are treated as unauthorized;
    // the deployer (address_to_name key that is not in callers_pool) is implicitly
    // allowed by populating the allowed set with only non-attacker addresses.
    if !artifacts.privileged_functions.is_empty() {
        use crate::evm::oracles::function::FunctionOracle;
        let attackers: std::collections::HashSet<EVMAddress> =
            state.callers_pool.iter().cloned().collect();
        let mut fn_oracle = FunctionOracle::new(artifacts.address_to_name.clone());
        // Feature 019 Phase A: switch on the materiality gate when --causal-identity is set.
        fn_oracle.set_causal_identity(config.causal_identity);
        for (contract, selector, fn_name) in &artifacts.privileged_functions {
            // Allow only non-attacker callers (i.e., the deployer).
            // Any address in callers_pool is a fuzzer-controlled attacker.
            let deployers: std::collections::HashSet<EVMAddress> = artifacts
                .address_to_name
                .keys()
                .filter(|a| !attackers.contains(*a))
                .cloned()
                .collect();
            fn_oracle.add_rule(*contract, *selector, fn_name.clone(), deployers);
        }
        info!(
            "Permission-leak oracle auto-activated: {} privileged function(s) monitored",
            artifacts.privileged_functions.len()
        );
        oracles.push(Rc::new(RefCell::new(fn_oracle)));
    }

    // if let Some(path) = config.state_comp_oracle {
    //     let mut file = File::open(path.clone()).expect("Failed to open state comp
    // oracle file");     let mut buf = String::new();
    //     file.read_to_string(&mut buf)
    //         .expect("Failed to read state comp oracle file");

    //     let evm_state =
    // serde_json::from_str::<EVMState>(buf.as_str()).expect("Failed to parse state
    // comp oracle file");

    //     let oracle = Rc::new(RefCell::new(StateCompOracle::new(
    //         evm_state,
    //         config.state_comp_matching.unwrap(),
    //     )));
    //     oracles.push(oracle);
    // }

    if config.arbitrary_external_call {
        oracles.push(Rc::new(RefCell::new(ArbitraryCallOracle::new(
            artifacts.address_to_name.clone(),
        ))));
        oracles.push(Rc::new(RefCell::new(
            ArbitraryERC20TransferOracle::new(artifacts.address_to_name.clone()),
        )));
    }

    if config.typed_bug {
        oracles.push(Rc::new(RefCell::new(TypedBugOracle::new(
            artifacts.address_to_name.clone(),
        ))));
    }

    state.add_metadata(BugMetadata::new());

    if config.selfdestruct_oracle {
        oracles.push(Rc::new(RefCell::new(SelfdestructOracle::new(
            artifacts.address_to_name.clone(),
        ))));
    }

    // Feature 020-B — SnapshotDelta oracle (LeakClass::Ownership). Post-hoc governance-state gate:
    // fires when an authority-bearing storage slot (EIP-1967 proxy slots + registered owner slots)
    // is relocated across a tx. Distinct bug type from the permission leak.
    if config.ownership_oracle {
        use crate::evm::oracles::snapshot_delta::SnapshotDeltaOracle;
        oracles.push(Rc::new(RefCell::new(SnapshotDeltaOracle::new(
            artifacts.address_to_name.clone(),
        ))));
    }

    if config.reentrancy_oracle {
        oracles.push(Rc::new(RefCell::new(ReentrancyOracle::new(
            artifacts.address_to_name.clone(),
        ))));
    }

    // Legacy topology oracle activation removed.

    if let Some(m) = onchain_middleware.clone() {
        m.borrow_mut().add_abi(artifacts.address_to_abi.clone());
    }

    let mut producers = config.producers;

    let objective: OracleFeedback<
        '_,
        EVMState,
        EVMAddress,
        Bytecode,
        Bytes,
        EVMAddress,
        revm_primitives::ruint::Uint<256, 4>,
        Vec<u8>,
        EVMInput,
        FuzzState<EVMInput, EVMState, EVMAddress, EVMAddress, Vec<u8>, ConciseEVMInput>,
        ConciseEVMInput,
        EVMQueueExecutor,
    > = OracleFeedback::new(&mut oracles, &mut producers, evm_executor_ref.clone());
    let wrapped_feedback = ConcolicFeedbackWrapper::new(Sha3WrappedFeedback::new(
        feedback,
        sha3_taint,
        evm_executor_ref.clone(),
        config.sha3_bypass,
    ));

    let mut fuzzer: ItyFuzzer<_, _, _, _, _, _, _, _, _, _, _, _, _, _, EVMMinimizer> = ItyFuzzer::new(
        scheduler,
        infant_scheduler,
        wrapped_feedback,
        infant_feedback,
        infant_result_feedback,
        objective,
        EVMMinimizer::new(evm_executor_ref.clone()),
        config.work_dir,
    );

    let initial_vm_state = artifacts.initial_state.clone();
    let mut testcases = vec![];
    let to_load_glob: String;

    if let Some(files) = config.replay_file.clone() {
        to_load_glob = files;
    } else {
        to_load_glob = config.load_corpus;
    }

    if !to_load_glob.is_empty() {
        'process_file: for file in glob(to_load_glob.as_str()).expect("Failed to read glob pattern") {
            let mut f = File::open(file.as_ref().expect("glob issue")).expect("Failed to open file");
            let mut transactions = String::new();
            let mut deserialized_transactions = vec![];
            f.read_to_string(&mut transactions).expect("Failed to read file");
            for txn in transactions.split('\n') {
                if txn.len() < 4 {
                    continue;
                }
                let deserialized_tx = serde_json::from_slice::<ConciseEVMInput>(txn.as_bytes());
                if deserialized_tx.is_err() {
                    error!("Failed to deserialize file: {:?}", file);
                    continue 'process_file;
                }
                deserialized_transactions.push(deserialized_tx.unwrap());
            }
            testcases.push(deserialized_transactions);
        }
    }

    macro_rules! load_code {
        ($txn: expr) => {
            if let Some(onchain_mid) = onchain_middleware.clone() {
                onchain_mid.borrow_mut().load_code(
                    $txn.contract,
                    &mut evm_executor_ref.clone().deref().borrow_mut().host,
                    false,
                    true,
                    false,
                    $txn.caller,
                    state,
                );
            }
        };
    }

    match config.replay_file {
        None => {
            // load initial corpus
            for testcase in testcases {
                let mut vm_state = initial_vm_state.clone();
                for txn in testcase {
                    load_code!(txn);
                    let (inp, call_until) = txn.to_input(vm_state.clone());
                    unsafe {
                        CALL_UNTIL = call_until;
                    }
                    fuzzer
                        .evaluate_input_events(state, &mut executor, &mut mgr, inp, false)
                        .unwrap();
                    vm_state = state.get_execution_result().new_state.clone();
                }
            }
            let res = fuzzer.fuzz_loop(&mut stages, &mut executor, state, &mut mgr);

            // it is not possible to reach here unless an exception is thrown
            let rv = res.err().unwrap().to_string();
            if rv == "No items in No entries in corpus" {
                error!("There is nothing to fuzz. Please check the target you provided.");
                return;
            } else {
                error!("{}", rv);
            }
            exit(1);
        }
        Some(_) => {
            unsafe {
                EVAL_COVERAGE = true;
            }

            let printer = Rc::new(RefCell::new(CallPrinter::new(artifacts.address_to_name.clone())));
            evm_executor_ref.borrow_mut().host.add_middlewares(printer.clone());

            for testcase in testcases {
                let mut vm_state = initial_vm_state.clone();
                let mut idx = 0;
                for txn in testcase {
                    load_code!(txn);
                    idx += 1;
                    // let splitter = txn.split(" ").collect::<Vec<&str>>();
                    info!("============ Execution {} ===============", idx);
                    let (inp, call_until) = txn.to_input(vm_state.clone());
                    printer.borrow_mut().cleanup();

                    unsafe {
                        CALL_UNTIL = call_until;
                    }

                    fuzzer
                        .evaluate_input_events(state, &mut executor, &mut mgr, inp, false)
                        .unwrap();

                    info!("============ Execution result {} =============", idx);
                    info!("reverted: {:?}", state.get_execution_result().clone().reverted);
                    info!("call trace:\n{}", printer.deref().borrow().get_trace());
                    info!("output: {:?}", hex::encode(state.get_execution_result().clone().output));

                    // debug!(
                    //     "new_state: {:?}",
                    //     state.get_execution_result().clone().new_state.state
                    // );

                    vm_state = state.get_execution_result().new_state.clone();
                    if config.value_capture {
                        info!("Observed values: {:?}", vm_state.state.observed_values);
                    }
                    info!("================================================");
                }
            }

            // dump coverage:
            cov_middleware.borrow_mut().record_instruction_coverage();
            // unsafe {
            //     EVAL_COVERAGE = false;
            //     CALL_UNTIL = u32::MAX;
            // }

            // fuzzer
            //     .fuzz_loop(&mut stages, &mut executor, state, &mut mgr)
            //     .expect("Fuzzing failed");
        }
    }
}
