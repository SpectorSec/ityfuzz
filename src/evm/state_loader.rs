use std::{collections::HashMap, fs, sync::Arc};

use libafl::prelude::Scheduler;
use revm_interpreter::bytecode::Bytecode;
use revm_primitives::Bytes as PrimBytes;
use serde::Deserialize;
use tracing::info;

use crate::evm::{
    host::FuzzHost,
    types::{EVMAddress, EVMFuzzState, EVMU256},
};

#[derive(Deserialize)]
struct SnapshotFile {
    state:   HashMap<String, HashMap<String, String>>,
    balance: HashMap<String, String>,
    code:    HashMap<String, String>,
}

fn parse_addr(s: &str) -> EVMAddress {
    s.parse().unwrap_or_else(|_| panic!("snapshot: invalid address '{s}'"))
}

fn parse_u256(s: &str) -> EVMU256 {
    let hex = s.strip_prefix("0x").unwrap_or(s);
    EVMU256::from_str_radix(hex, 16).unwrap_or_else(|_| panic!("snapshot: invalid U256 '{s}'"))
}

/// Inject a pre-fetched state snapshot directly into the fuzzer's REVM instance.
///
/// Call this after `evm_executor.host.initialize(state)` so the snapshot
/// values are not overwritten by the corpus initializer's deploy phase.
/// Contracts that appear in both the snapshot and the deployer's output keep
/// the snapshot's storage (snapshot wins via HashMap::insert).
pub fn load_snapshot<SC>(host: &mut FuzzHost<SC>, path: &str)
where
    SC: Scheduler<State = EVMFuzzState> + Clone,
{
    let raw = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("snapshot: cannot read '{path}': {e}"));
    let snap: SnapshotFile = serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("snapshot: invalid JSON in '{path}': {e}"));

    let mut n_state = 0usize;
    let mut n_balance = 0usize;
    let mut n_code = 0usize;

    for (addr_str, slots) in &snap.state {
        let addr = parse_addr(addr_str);
        let storage: HashMap<EVMU256, EVMU256> = slots
            .iter()
            .map(|(k, v)| (parse_u256(k), parse_u256(v)))
            .collect();
        host.evmstate.state.insert(addr, storage);
        n_state += 1;
    }

    for (addr_str, bal_str) in &snap.balance {
        if bal_str == "0x0" {
            continue;
        }
        let addr = parse_addr(addr_str);
        host.evmstate.balance.insert(addr, parse_u256(bal_str));
        n_balance += 1;
    }

    for (addr_str, code_hex) in &snap.code {
        if code_hex.len() <= 2 {
            continue; // "0x" = empty
        }
        let addr = parse_addr(addr_str);
        let stripped = code_hex.strip_prefix("0x").unwrap_or(code_hex);
        let bytes_vec = hex::decode(stripped)
            .unwrap_or_else(|e| panic!("snapshot: invalid bytecode hex for {addr_str}: {e}"));
        let bytecode = Bytecode::new_legacy(PrimBytes::from(bytes_vec));
        host.code.insert(addr, Arc::new(bytecode));
        n_code += 1;
    }

    info!(
        "[state-file] loaded snapshot: {} storage accounts, {} balances, {} bytecodes",
        n_state, n_balance, n_code
    );
}
