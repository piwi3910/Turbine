//! Lab scripts and manifests, checked without contacting a lab host.
//!
//! - `scripts/lab/phase1-novanas.yaml` loads through the real configuration model.
//! - `scripts/lab-test.sh --dry-run` and `scripts/lab-serve.sh --dry-run` print the commands they
//!   would run, including the rendered k3s Job; `ssh`, `rsync`, `scp`, `curl` and `kubectl` are
//!   replaced by stubs that record any call, so a dry run that reaches for a host fails the test.
//! - The rendered Jobs carry the cached workspace slots, the kernel build, the environment and
//!   the read-only model mount that the GPU tests and `turbine-server` rely on.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_norway::Value;

const CI_ROOT: &str = "/home/piwi/turbine-ci";
const CACHE: &str = "/home/piwi/turbine-ci/cache";
const MODEL_DIR: &str = "/models/llama-3.2-3b-instruct";
const MOE_MODEL_DIR: &str = "/models/olmoe-1b-7b-0125-instruct";
/// The vLLM-ROCm image the baseline Job pins (tag and digest, checked on Docker Hub).
const VLLM_IMAGE: &str = "rocm/vllm:rocm7.14.1_rdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0@sha256:19ad8dc5fb3012f2d5810995f73e8bf069302056d2b4aa6fdf2d2ae5fa9a68ab";
const SSH: &str = "ssh -o BatchMode=yes -o ConnectTimeout=10 piwi@192.168.10.203";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repository root")
}

/// Walk a dotted path of mapping keys and sequence indexes.
fn at<'a>(v: &'a Value, path: &str) -> &'a Value {
    path.split('.').fold(v, |cur, key| {
        let next = match key.parse::<usize>() {
            Ok(i) => cur.get(i),
            Err(_) => cur.get(key),
        };
        next.unwrap_or_else(|| panic!("missing {key} in {path}"))
    })
}

fn str_at<'a>(v: &'a Value, path: &str) -> &'a str {
    at(v, path)
        .as_str()
        .unwrap_or_else(|| panic!("{path} is not a string"))
}

fn container(job: &Value) -> &Value {
    let containers = at(job, "spec.template.spec.containers")
        .as_sequence()
        .expect("containers");
    assert_eq!(containers.len(), 1, "one container per lab Job");
    &containers[0]
}

fn env_opt(job: &Value, name: &str) -> Option<String> {
    at(container(job), "env")
        .as_sequence()
        .expect("env list")
        .iter()
        .find(|e| str_at(e, "name") == name)
        .map(|e| str_at(e, "value").to_string())
}

fn env(job: &Value, name: &str) -> String {
    env_opt(job, name).unwrap_or_else(|| panic!("env {name} is not set"))
}

/// The shell script the container runs: `command: [bash, -c]` with the script in `args`.
fn script(job: &Value) -> String {
    let c = container(job);
    let command: Vec<&str> = at(c, "command")
        .as_sequence()
        .expect("command list")
        .iter()
        .map(|v| v.as_str().expect("command entry"))
        .collect();
    assert_eq!(command, ["bash", "-c"], "the Job runs a bash setup script");
    str_at(c, "args.0").to_string()
}

/// `(mountPath, hostPath, readOnly)` for every mount, sorted.
fn mounts(job: &Value) -> Vec<(String, String, bool)> {
    let volumes = at(job, "spec.template.spec.volumes")
        .as_sequence()
        .expect("volumes");
    let mut out: Vec<_> = at(container(job), "volumeMounts")
        .as_sequence()
        .expect("volumeMounts")
        .iter()
        .map(|m| {
            let name = str_at(m, "name");
            let volume = volumes
                .iter()
                .find(|v| str_at(v, "name") == name)
                .unwrap_or_else(|| panic!("volume {name} is mounted but not declared"));
            (
                str_at(m, "mountPath").to_string(),
                str_at(volume, "hostPath.path").to_string(),
                m.get("readOnly").and_then(Value::as_bool).unwrap_or(false),
            )
        })
        .collect();
    out.sort();
    out
}

fn amd_gpus(job: &Value) -> u64 {
    // `amd.com/gpu` contains a dot, so it cannot go through the dotted-path helper.
    at(container(job), "resources.limits")
        .get("amd.com/gpu")
        .and_then(Value::as_u64)
        .expect("amd.com/gpu limit")
}

/// Asserts that `needles` occur in `hay` in this order.
fn assert_in_order(hay: &str, needles: &[&str]) {
    let mut from = 0;
    for n in needles {
        match hay[from..].find(n) {
            Some(i) => from += i + n.len(),
            None => panic!("{n:?} missing (or out of order) in:\n{hay}"),
        }
    }
}

/// Runs `scripts/<script> <args>` with stubs for every host-contacting tool first on PATH.
/// Returns the output and whether any stub was called.
fn lab_script(script: &str, tag: &str, args: &[&str]) -> (Output, Option<String>) {
    let stubs =
        std::env::temp_dir().join(format!("turbine-lab-scripts-{}-{tag}", std::process::id()));
    let _ = fs::remove_dir_all(&stubs);
    fs::create_dir_all(&stubs).expect("stub dir");
    let calls = stubs.join("calls.log");
    for tool in ["ssh", "rsync", "scp", "curl", "kubectl"] {
        let path = stubs.join(tool);
        fs::write(
            &path,
            format!(
                "#!/bin/sh\necho \"{tool} $*\" >> '{}'\nexit 97\n",
                calls.display()
            ),
        )
        .expect("write stub");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).expect("chmod stub");
        }
    }
    let path = format!(
        "{}:{}",
        stubs.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let out = Command::new("bash")
        .arg(repo_root().join("scripts").join(script))
        .args(args)
        .current_dir(repo_root())
        .env("PATH", path)
        .output()
        .expect("run bash");
    let called = fs::read_to_string(&calls).ok();
    let _ = fs::remove_dir_all(&stubs);
    (out, called)
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// A successful dry run that contacted no host; returns its stdout.
fn dry_run(script: &str, tag: &str, args: &[&str]) -> String {
    let (out, called) = lab_script(script, tag, args);
    let text = stdout(&out);
    println!("{text}");
    assert_eq!(called, None, "{tag}: a dry run contacted a host");
    assert!(
        out.status.success(),
        "{tag}: exit {:?}: {}",
        out.status,
        stderr(&out)
    );
    text
}

/// The body of the heredoc a dry run prints for the remote command `cmd`.
fn heredoc<'a>(text: &'a str, cmd: &str) -> &'a str {
    let open = format!("{SSH} '{cmd}' <<'EOF'\n");
    let start = text
        .find(&open)
        .unwrap_or_else(|| panic!("no {open:?} in:\n{text}"))
        + open.len();
    let len = text[start..].find("\nEOF\n").expect("heredoc end");
    &text[start..start + len]
}

/// The Job a dry run applies (the rendered template), parsed; the first one when it applies
/// several.
fn applied_job(text: &str) -> Value {
    applied_jobs(text).remove(0)
}

/// Every Job a dry run applies, in order, parsed.
fn applied_jobs(text: &str) -> Vec<Value> {
    let open = format!("{SSH} 'export KUBECTL_KUBERC=false; kubectl apply -f -' <<'EOF'\n");
    let mut jobs = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find(&open) {
        let manifest = heredoc(rest, "export KUBECTL_KUBERC=false; kubectl apply -f -");
        assert!(
            !manifest.contains("__"),
            "unrendered placeholder:\n{manifest}"
        );
        jobs.push(serde_norway::from_str(manifest).unwrap_or_else(|e| panic!("{e}:\n{manifest}")));
        rest = &rest[i + open.len() + manifest.len()..];
    }
    assert!(!jobs.is_empty(), "no kubectl apply in:\n{text}");
    jobs
}

/// The run id a dry run announces (`run <id>: job ...`).
fn run_id(text: &str, tool: &str) -> String {
    let prefix = format!("{tool}: novanas: run ");
    let line = text
        .lines()
        .find(|l| l.starts_with(&prefix))
        .unwrap_or_else(|| panic!("no run line in:\n{text}"));
    let id = line[prefix.len()..].split(':').next().unwrap().to_string();
    assert!(
        !id.is_empty()
            && id
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
        "run id {id:?} is not DNS-1123 safe"
    );
    id
}

/// The NUL-separated test command a `lab-test.sh` dry run uploads, one argument per line.
fn test_command(text: &str, id: &str) -> Vec<String> {
    let cmd = format!("cat > {CI_ROOT}/runs/{id}/test-command");
    let hf = format!("{cmd} && touch {CI_ROOT}/runs/{id}/hf-reference");
    let body = if text.contains(&format!("'{hf}'")) {
        heredoc(text, &hf)
    } else {
        heredoc(text, &cmd)
    };
    body.lines().map(str::to_string).collect()
}

#[test]
fn phase1_novanas_config_loads() {
    let path = repo_root().join("scripts/lab/phase1-novanas.yaml");
    let c = turbine_core::config::load(&path, &[]).expect("phase1-novanas.yaml loads");
    assert_eq!(c.server.listen.to_string(), "0.0.0.0:18000");
    assert_eq!(c.model.path, Path::new(MODEL_DIR));
    assert_eq!(
        c.model.served_name.as_deref(),
        Some("meta-llama/Llama-3.2-3B-Instruct")
    );
    assert_eq!(c.execution.backend.as_str(), "hip");
}

#[test]
fn phase2_novanas_configs_load_with_the_scheduler_defaults() {
    for (file, dir, name) in [
        (
            "phase2-novanas-llama.yaml",
            MODEL_DIR,
            "meta-llama/Llama-3.2-3B-Instruct",
        ),
        (
            "phase2-novanas-olmoe.yaml",
            MOE_MODEL_DIR,
            "allenai/OLMoE-1B-7B-0125-Instruct",
        ),
    ] {
        let path = repo_root().join("scripts/lab").join(file);
        let c = turbine_core::config::load(&path, &[])
            .unwrap_or_else(|e| panic!("{file} does not load: {e}"));
        assert_eq!(c.server.listen.to_string(), "0.0.0.0:18000", "{file}");
        assert_eq!(c.model.path, Path::new(dir), "{file}");
        assert_eq!(c.model.served_name.as_deref(), Some(name), "{file}");
        assert_eq!(c.execution.backend.as_str(), "hip", "{file}");
        // The spec's Phase 2 defaults, spelled out in the file.
        let s = &c.scheduler;
        assert!(s.continuous_batching && s.chunked_prefill, "{file}");
        assert_eq!(
            (
                s.max_running_requests,
                s.max_batch_tokens,
                s.prefill_chunk_tokens,
                s.max_queued_requests
            ),
            (64, 8192, 2048, 256),
            "{file}"
        );
        assert_eq!(c.kv.gpu.max_bytes.map(|b| b.0), Some(8 << 30), "{file}");
        assert_eq!(c.kv.block_tokens, 128, "{file}");
    }
}

/// Every tree sync into a cached build workspace compares by checksum and never carries source
/// mtimes over (no `-a`, no `-t`): a changed file must get a fresh mtime, or cargo's mtime
/// fingerprints reuse artifacts that a newer-dated tree of another checkout left in the slot.
#[test]
fn lab_tree_syncs_never_preserve_source_mtimes() {
    let root = repo_root();
    for file in [
        "scripts/lab-test.sh",
        "scripts/lab-serve.sh",
        "scripts/lab/novanas-test-job.yaml",
        "scripts/lab/novanas-serve-job.yaml",
        "scripts/remote-cargo.sh",
    ] {
        let text = fs::read_to_string(root.join(file)).expect(file);
        let syncs: Vec<&str> = text
            .lines()
            .map(str::trim)
            .filter(|l| l.contains("rsync -") && l.contains("--delete"))
            .collect();
        assert!(!syncs.is_empty(), "{file} has no tree sync");
        for line in syncs {
            let flags = line
                .split_whitespace()
                .skip_while(|w| *w != "rsync")
                .nth(1)
                .expect("rsync flags");
            assert!(
                flags.starts_with('-') && !flags.starts_with("--"),
                "{file}: {line}"
            );
            assert!(flags.contains('c'), "{file}: no checksum compare: {line}");
            assert!(
                !flags.contains('a') && !flags.contains('t'),
                "{file}: preserves mtimes: {line}"
            );
        }
    }
}

#[test]
fn lab_test_dry_run_applies_a_one_gpu_job_with_cached_slots() {
    let text = dry_run("lab-test.sh", "test-default", &["--dry-run", "novanas"]);
    let id = run_id(&text, "lab-test");
    let job_name = format!("turbine-lab-test-{id}");
    let build_name = format!("turbine-lab-build-{id}");
    let run_dir = format!("{CI_ROOT}/runs/{id}");
    // The GPU-less build Job runs to completion before the GPU Job is applied.
    assert_in_order(
        &text,
        &[
            &format!("{SSH} 'mkdir -p {run_dir}/src {CACHE}/slots"),
            "rsync -rlpcz --delete --exclude target/ --exclude .git/ --exclude .claude/",
            &format!("piwi@192.168.10.203:{run_dir}/src/"),
            &format!("cat > {run_dir}/test-command"),
            "kubectl apply -f -",
            &format!("kubectl -n turbine-ci logs -f job/{build_name}"),
            &format!("wait until job {build_name} succeeds or fails"),
            "kubectl apply -f -",
            &format!("kubectl -n turbine-ci logs -f job/{job_name}"),
            &format!("wait until job {job_name} succeeds or fails"),
        ],
    );
    // A run never deletes anything; cleanup (interrupt, unschedulable) names only its own Jobs.
    assert!(!text.contains(" delete "), "{text}");
    assert!(
        text.contains("-exec rm -rf {} +"),
        "stale uploads are pruned: {text}"
    );

    let jobs = applied_jobs(&text);
    assert_eq!(jobs.len(), 2, "a build Job and a GPU Job");
    let (build, job) = (&jobs[0], &jobs[1]);
    assert_eq!(str_at(build, "metadata.name"), build_name);
    assert_eq!(str_at(build, "metadata.labels.turbine-lab-role"), "build");
    assert_eq!(str_at(build, "metadata.labels.turbine-lab-run"), id);
    assert_eq!(amd_gpus(build), 0, "the build claims no R9700");
    assert_eq!(env(build, "TURBINE_LAB_PHASE"), "build");
    assert_eq!(env(job, "TURBINE_LAB_PHASE"), "test");
    // Same template otherwise: same mounts, and the slot handling is shared.
    assert_eq!(mounts(build), mounts(job));
    assert_eq!(script(build), script(job));

    let job = job.clone();
    assert_eq!(str_at(&job, "metadata.name"), job_name);
    assert_eq!(str_at(&job, "metadata.namespace"), "turbine-ci");
    assert_eq!(str_at(&job, "metadata.labels.turbine-lab-role"), "test");
    assert_eq!(str_at(&job, "metadata.labels.turbine-lab-run"), id);
    assert_eq!(amd_gpus(&job), 1, "one R9700 by default");
    assert_eq!(env(&job, "TURBINE_EXPECT_AMD"), "1");
    assert_eq!(env(&job, "TURBINE_LAB_RUN_ID"), id);
    assert_eq!(env(&job, "TURBINE_LAB_ROOT"), CI_ROOT);
    assert_eq!(env(&job, "TURBINE_LAB_CACHE"), CACHE);
    assert_eq!(env(&job, "TURBINE_LAB_SLOTS"), "2");
    assert_eq!(env(&job, "TURBINE_TEST_BACKEND"), "hip");
    assert_eq!(env(&job, "TURBINE_TEST_MODEL_DIR"), MODEL_DIR);
    assert_eq!(env(&job, "TURBINE_TEST_MOE_MODEL_DIR"), MOE_MODEL_DIR);
    assert_eq!(env(&job, "TURBINE_ROCM_PATH"), "/opt/rocm/rocm/core-7.14");
    assert_eq!(env(&job, "UV_CACHE_DIR"), format!("{CACHE}/uv"));
    // The slot, not the manifest, picks the target and kernel build dirs.
    for var in [
        "CARGO_TARGET_DIR",
        "KERNEL_BUILD_DIR",
        "TURBINE_KERNEL_LIBRARY",
    ] {
        assert_eq!(env_opt(&job, var), None, "{var} must come from the slot");
    }

    let m = mounts(&job);
    for want in [
        ("/models", "/home/piwi/turbine-models", true),
        (CI_ROOT, CI_ROOT, false),
        (
            "/usr/local/cargo/registry",
            "/home/piwi/turbine-ci/cache/cargo/registry",
            false,
        ),
        (
            "/usr/local/cargo/git",
            "/home/piwi/turbine-ci/cache/cargo/git",
            false,
        ),
        (
            "/var/cache/apt/archives",
            "/home/piwi/turbine-ci/cache/apt/archives",
            false,
        ),
        (
            "/var/lib/apt/lists",
            "/home/piwi/turbine-ci/cache/apt/lists",
            false,
        ),
    ] {
        let want = (want.0.to_string(), want.1.to_string(), want.2);
        assert!(m.contains(&want), "{want:?} missing from {m:?}");
    }

    let s = script(&job);
    assert!(s.contains("set -euo pipefail"), "{s}");
    assert_in_order(
        &s,
        &[
            "apt-get install -y -qq cmake ninja-build python3 rsync",
            "if [[ $PHASE == test && -f \"$RUN_DIR/slot\" ]]; then PREFERRED=\"$(cat \"$RUN_DIR/slot\")\"; fi",
            "for n in $PREFERRED $(seq 0 $((TURBINE_LAB_SLOTS - 1)))",
            "exec {LOCK_FD}>\"$TURBINE_LAB_CACHE/slots/test-$n.lock\"",
            "flock -n \"$LOCK_FD\"",
            "export CARGO_TARGET_DIR=\"$SLOT_DIR/target\"",
            "export KERNEL_BUILD_DIR=\"$SLOT_DIR/kernels\"",
            "export TURBINE_KERNEL_LIBRARY=\"$KERNEL_BUILD_DIR/libturbine_hip.so\"",
            "rsync -rlpc --delete \"$RUN_DIR/src/\" \"$SLOT_DIR/src/\"",
            "mapfile -d '' TEST_CMD < \"$RUN_DIR/test-command\"",
            "if [[ $PHASE == test ]]; then rm -rf \"$RUN_DIR\"; fi",
            "cd \"$SLOT_DIR/src\"",
            "CMAKE_HOME_DIRECTORY:INTERNAL=$PWD/kernels/rocm",
            "if [[ ! -f \"$KERNEL_BUILD_DIR/CMakeCache.txt\" ]]",
            "cmake -S kernels/rocm -B \"$KERNEL_BUILD_DIR\" -G Ninja",
            "-DGPU_TARGETS=gfx1201",
            "cmake --build \"$KERNEL_BUILD_DIR\"",
            "test -f \"$TURBINE_KERNEL_LIBRARY\"",
            "if [[ $PHASE == build ]]; then",
            "BUILD_CMD+=(--no-run)",
            "\"${BUILD_CMD[@]}\"",
            "echo \"${SLOT#test-}\" > \"$RUN_DIR/slot\"",
            "exit 0",
            "if [[ $HF_REFERENCE -eq 1 ]]",
            "https://astral.sh/uv/",
            "exec \"${TEST_CMD[@]}\"",
        ],
    );
    // The kernel build dir is never wiped wholesale: only a foreign CMake cache is dropped.
    assert!(!s.contains("rm -rf \"$KERNEL_BUILD_DIR\"\n"), "{s}");

    assert_eq!(
        test_command(&text, &id),
        [
            "cargo",
            "test",
            "--no-fail-fast",
            "--workspace",
            "--",
            "--include-ignored",
            "--show-output",
            "--skip",
            "hf_reference_matches_cpu",
            "--skip",
            "hostmem_",
            "--skip",
            "rccl_init_with_a_missing_peer",
            "--skip",
            "cold_tp2",
            "--skip",
            "tp_forward_profile",
        ]
    );
}

#[test]
fn lab_test_gpus_2_is_the_inventory_path() {
    let text = dry_run(
        "lab-test.sh",
        "test-gpus2",
        &["--dry-run", "novanas", "--gpus", "2"],
    );
    let jobs = applied_jobs(&text);
    assert_eq!(amd_gpus(&jobs[0]), 0, "the build claims no R9700");
    assert_eq!(amd_gpus(&jobs[1]), 2);
    assert_eq!(env(&jobs[1], "TURBINE_EXPECT_AMD"), "2");
}

/// P3 Task 17: `--features <list>` reaches the in-container `cargo test` (the fault-injection
/// build of `tests/fault.rs`), before the harness arguments; `--stop` takes no features.
#[test]
fn lab_test_forwards_cargo_features() {
    let text = dry_run(
        "lab-test.sh",
        "test-features",
        &["--dry-run", "novanas", "--features", "fault-injection"],
    );
    let id = run_id(&text, "lab-test");
    assert_eq!(
        test_command(&text, &id),
        [
            "cargo",
            "test",
            "--no-fail-fast",
            "--workspace",
            "--features",
            "fault-injection",
            "--",
            "--include-ignored",
            "--show-output",
            "--skip",
            "hf_reference_matches_cpu",
            "--skip",
            "hostmem_",
            "--skip",
            "rccl_init_with_a_missing_peer",
            "--skip",
            "cold_tp2",
            "--skip",
            "tp_forward_profile",
        ]
    );
    for args in [
        &[
            "novanas",
            "--stop",
            "0926-abc",
            "--features",
            "fault-injection",
        ][..],
        &["novanas", "--features"][..],
        &["novanas", "--features", "bad feature"][..],
    ] {
        let (out, called) = lab_script("lab-test.sh", "features-usage", args);
        assert_eq!(out.status.code(), Some(2), "{args:?}: {}", stderr(&out));
        assert_eq!(called, None, "{args:?} contacted a host");
    }
}

/// Shorter test cycles (user decision 2026-09-27): `--tier quick` skips the slow perf/timing
/// tests via `--skip`, `--tier perf` runs only those (as bare filters), `--tier full` (the
/// default) changes nothing; `--stop` takes no tier.
#[test]
fn lab_test_tier_selects_or_skips_the_slow_tests() {
    let slow = [
        "serving_mix",
        "forward_profile",
        "moe_prefill_timings",
        "moe_ep_local_timings",
        "host_step_costs",
        "decode_forward_timing",
        "decode_op_timings",
        "fused_projection_timings",
        "prefill_op_timings",
        "every_implementation_matches_cpu",
        "implementations_enumerated",
        "gemm_matches_cpu",
        "gemm_table_matches_cpu",
        "norm_rope_silu_embedding_add_match_cpu",
        "paged_prefill_ck_128_matches_cpu",
        "paged_and_moe_ops",
        "moe_experts_small_m_matches_cpu",
        "moe_experts_grouped_matches_cpu",
        "logits_reduce_matches_cpu",
        "host_staging_does_not_wait_for_the_stream",
        "prefill_shapes_match_cpu",
        "moe_decode_tier_timings",
        "decode_attention_timings",
    ];

    let text = dry_run(
        "lab-test.sh",
        "tier-quick",
        &["--dry-run", "novanas", "--tier", "quick"],
    );
    let id = run_id(&text, "lab-test");
    let cmd = test_command(&text, &id);
    let mut want = vec![
        "cargo".to_string(),
        "test".to_string(),
        "--no-fail-fast".to_string(),
        "--workspace".to_string(),
        "--".to_string(),
        "--include-ignored".to_string(),
        "--show-output".to_string(),
        "--skip".to_string(),
        "hf_reference_matches_cpu".to_string(),
        "--skip".to_string(),
        "hostmem_".to_string(),
        "--skip".to_string(),
        "rccl_init_with_a_missing_peer".to_string(),
        "--skip".to_string(),
        "cold_tp2".to_string(),
        "--skip".to_string(),
        "tp_forward_profile".to_string(),
    ];
    for t in slow {
        want.push("--skip".to_string());
        want.push(t.to_string());
    }
    assert_eq!(cmd, want);

    let text = dry_run(
        "lab-test.sh",
        "tier-perf",
        &["--dry-run", "novanas", "--tier", "perf"],
    );
    let id = run_id(&text, "lab-test");
    let cmd = test_command(&text, &id);
    let mut want = vec![
        "cargo".to_string(),
        "test".to_string(),
        "--no-fail-fast".to_string(),
        "--workspace".to_string(),
        "--".to_string(),
        "--include-ignored".to_string(),
        "--show-output".to_string(),
        "--skip".to_string(),
        "hf_reference_matches_cpu".to_string(),
        "--skip".to_string(),
        "hostmem_".to_string(),
        "--skip".to_string(),
        "rccl_init_with_a_missing_peer".to_string(),
        "--skip".to_string(),
        "cold_tp2".to_string(),
        "--skip".to_string(),
        "tp_forward_profile".to_string(),
    ];
    for t in slow {
        want.push(t.to_string());
    }
    assert_eq!(cmd, want);

    // full (the default) is unchanged.
    let text = dry_run(
        "lab-test.sh",
        "tier-full",
        &["--dry-run", "novanas", "--tier", "full"],
    );
    let id = run_id(&text, "lab-test");
    assert_eq!(
        test_command(&text, &id),
        [
            "cargo",
            "test",
            "--no-fail-fast",
            "--workspace",
            "--",
            "--include-ignored",
            "--show-output",
            "--skip",
            "hf_reference_matches_cpu",
            "--skip",
            "hostmem_",
            "--skip",
            "rccl_init_with_a_missing_peer",
            "--skip",
            "cold_tp2",
            "--skip",
            "tp_forward_profile",
        ]
    );

    for (tag, args) in [
        ("tier-bad", &["novanas", "--tier", "bogus"][..]),
        ("tier-missing", &["novanas", "--tier"][..]),
        (
            "tier-stop",
            &["novanas", "--stop", "0926-abc", "--tier", "quick"][..],
        ),
    ] {
        let (out, called) = lab_script("lab-test.sh", tag, args);
        assert_eq!(out.status.code(), Some(2), "{args:?}: {}", stderr(&out));
        assert_eq!(called, None, "{args:?} contacted a host");
    }
}

#[test]
fn lab_test_passes_a_subset_and_opts_into_the_hf_reference() {
    let text = dry_run(
        "lab-test.sh",
        "test-subset",
        &[
            "--dry-run",
            "novanas",
            "--",
            "-p",
            "turbine-model",
            "--test",
            "tiny_model",
        ],
    );
    let id = run_id(&text, "lab-test");
    assert_eq!(
        test_command(&text, &id),
        [
            "cargo",
            "test",
            "--no-fail-fast",
            "-p",
            "turbine-model",
            "--test",
            "tiny_model",
            "--",
            "--include-ignored",
            "--show-output",
            "--skip",
            "hf_reference_matches_cpu",
            "--skip",
            "hostmem_",
            "--skip",
            "rccl_init_with_a_missing_peer",
            "--skip",
            "cold_tp2",
            "--skip",
            "tp_forward_profile",
        ]
    );
    assert!(
        !text.contains("&& touch "),
        "no hf-reference marker: {text}"
    );

    let text = dry_run(
        "lab-test.sh",
        "test-hf",
        &[
            "--dry-run",
            "novanas",
            "--with-hf-reference",
            "--",
            "-p",
            "turbine-model",
            "--test",
            "golden",
            "--",
            "hf_reference",
        ],
    );
    let id = run_id(&text, "lab-test");
    assert!(
        text.contains(&format!("touch {CI_ROOT}/runs/{id}/hf-reference")),
        "{text}"
    );
    assert_eq!(
        test_command(&text, &id),
        [
            "cargo",
            "test",
            "--no-fail-fast",
            "-p",
            "turbine-model",
            "--test",
            "golden",
            "--",
            "--include-ignored",
            "--show-output",
            "--skip",
            "hostmem_",
            "--skip",
            "rccl_init_with_a_missing_peer",
            "--skip",
            "cold_tp2",
            "--skip",
            "tp_forward_profile",
            "hf_reference",
        ]
    );
}

#[test]
fn lab_test_runs_get_unique_jobs_and_stop_touches_only_its_own() {
    let a = dry_run("lab-test.sh", "test-a", &["--dry-run", "novanas"]);
    let b = dry_run("lab-test.sh", "test-b", &["--dry-run", "novanas"]);
    let (ida, idb) = (run_id(&a, "lab-test"), run_id(&b, "lab-test"));
    assert_ne!(ida, idb, "two runs share a Job name");
    assert!(!a.contains(&idb) && !b.contains(&ida));

    let text = dry_run(
        "lab-test.sh",
        "test-stop",
        &["--dry-run", "novanas", "--stop", &ida],
    );
    assert_eq!(text.matches(" delete ").count(), 1, "one delete: {text}");
    assert!(
        text.contains(&format!(
            "kubectl -n turbine-ci delete job turbine-lab-build-{ida} turbine-lab-test-{ida} --ignore-not-found"
        )),
        "{text}"
    );
    assert!(
        text.contains(&format!("rm -rf {CI_ROOT}/runs/{ida}'")),
        "{text}"
    );
    for absent in ["rsync", "apply", " -l ", "turbine-lab-serve"] {
        assert!(
            !text.contains(absent),
            "--stop must not run {absent}: {text}"
        );
    }
}

#[test]
fn lab_test_usage_errors_exit_2_without_contacting_a_host() {
    for (tag, args) in [
        ("t-noargs", &[][..]),
        ("t-host", &["localhost"][..]),
        ("t-gpus3", &["novanas", "--gpus", "3"][..]),
        ("t-gpus-missing", &["novanas", "--gpus"][..]),
        ("t-gpus-spark", &["dgx-spark", "--gpus", "1"][..]),
        ("t-stop-spark", &["dgx-spark", "--stop", "0926-ab"][..]),
        ("t-stop-noid", &["novanas", "--stop"][..]),
        ("t-stop-badid", &["novanas", "--stop", "../x"][..]),
        (
            "t-stop-gpus",
            &["novanas", "--gpus", "2", "--stop", "0926-ab"][..],
        ),
        ("t-unknown", &["novanas", "-p", "turbine-core"][..]),
    ] {
        let (out, called) = lab_script("lab-test.sh", tag, args);
        assert_eq!(called, None, "{tag}: contacted a host");
        assert_eq!(out.status.code(), Some(2), "{tag}: {}", stderr(&out));
        assert!(stderr(&out).contains("usage:"), "{tag}: {}", stderr(&out));
    }
}

#[test]
fn novanas_serve_job_matches_the_test_job() {
    let test = applied_job(&dry_run(
        "lab-test.sh",
        "serve-cmp-test",
        &["--dry-run", "novanas"],
    ));
    let text = dry_run(
        "lab-serve.sh",
        "serve-cmp",
        &["--dry-run", "novanas", "scripts/lab/phase1-novanas.yaml"],
    );
    let id = run_id(&text, "lab-serve");
    let serve = applied_job(&text);
    assert_eq!(
        str_at(&serve, "metadata.name"),
        format!("turbine-lab-serve-{id}")
    );
    assert_eq!(str_at(&serve, "metadata.namespace"), "turbine-ci");
    assert_eq!(str_at(&serve, "metadata.labels.turbine-lab"), "true");
    assert_eq!(str_at(&serve, "metadata.labels.turbine-lab-role"), "serve");
    assert_eq!(
        at(&serve, "spec.template.spec.hostNetwork").as_bool(),
        Some(true)
    );
    assert_eq!(amd_gpus(&serve), 1);

    // Same mounts and caches; same ROCm root and runtime libraries.
    assert_eq!(mounts(&serve), mounts(&test));
    for var in [
        "TURBINE_LAB_ROOT",
        "TURBINE_LAB_CACHE",
        "TURBINE_ROCM_PATH",
        "LD_LIBRARY_PATH",
        "TURBINE_AMD_SMI_LIBRARY",
    ] {
        assert_eq!(env(&serve, var), env(&test, var), "{var}");
    }
    assert_eq!(env(&serve, "TURBINE_LAB_RUN_ID"), id);

    let s = script(&serve);
    assert_in_order(
        &s,
        &[
            "apt-get install -y -qq cmake ninja-build python3 rsync",
            "SLOT=serve-0",
            "flock -w 1800 \"$LOCK_FD\"",
            "export CARGO_TARGET_DIR=\"$SLOT_DIR/target\"",
            "rsync -rlpc --delete \"$RUN_DIR/src/\" \"$SLOT_DIR/src/\"",
            "cp \"$RUN_DIR/config.yaml\" \"$SLOT_DIR/config.yaml\"",
            "cmake -S kernels/rocm -B \"$KERNEL_BUILD_DIR\" -G Ninja",
            "cmake --build \"$KERNEL_BUILD_DIR\"",
            "cargo build --release -p turbine-server",
            "exec \"$CARGO_TARGET_DIR/release/turbine-server\" --config \"$SLOT_DIR/config.yaml\"",
        ],
    );
}

#[test]
fn lab_serve_dry_run_prints_the_start_sequence() {
    let text = dry_run(
        "lab-serve.sh",
        "start",
        &["--dry-run", "novanas", "scripts/lab/phase1-novanas.yaml"],
    );
    let id = run_id(&text, "lab-serve");
    let run_dir = format!("{CI_ROOT}/runs/{id}");
    let ready =
        "curl -s -o /dev/null -w %{http_code} --max-time 5 http://192.168.10.203:18000/ready";
    assert_in_order(
        &text,
        &[
            ready,
            "refuse to start unless that fails to connect",
            &format!("{SSH} 'mkdir -p {run_dir}/src {CACHE}/slots"),
            "rsync -rlpcz --delete --exclude target/ --exclude .git/ --exclude .claude/",
            &format!("piwi@192.168.10.203:{run_dir}/src/"),
            &format!("phase1-novanas.yaml piwi@192.168.10.203:{run_dir}/config.yaml"),
            "kubectl apply -f -",
            &format!("kubectl -n turbine-ci logs -f job/turbine-lab-serve-{id}"),
            ready,
        ],
    );
    assert!(
        !text.contains(" delete "),
        "a start deletes nothing: {text}"
    );
}

#[test]
fn lab_serve_dry_run_stop_deletes_only_the_serve_jobs() {
    let text = dry_run("lab-serve.sh", "stop", &["--dry-run", "novanas", "--stop"]);
    assert!(
        text.contains(
            "kubectl -n turbine-ci delete job -l turbine-lab-role=serve --ignore-not-found"
        ),
        "{text}"
    );
    let deletes = text.matches(" delete ").count();
    assert_eq!(deletes, 1, "exactly one delete: {text}");
    for absent in ["rsync", "apply", "turbine-lab-test", "role=test"] {
        assert!(
            !text.contains(absent),
            "--stop must not run {absent}: {text}"
        );
    }
}

#[test]
fn lab_serve_usage_errors_exit_2_without_contacting_a_host() {
    let missing = "scripts/lab/does-not-exist.yaml";
    for (tag, args, expect) in [
        ("noargs", &[][..], "usage:"),
        (
            "host",
            &["dgx-spark", "scripts/lab/phase1-novanas.yaml"][..],
            "usage:",
        ),
        ("extra", &["novanas", "--stop", "x"][..], "usage:"),
        ("missing", &["novanas", missing][..], missing),
    ] {
        let (out, called) = lab_script("lab-serve.sh", tag, args);
        assert_eq!(called, None, "{tag}: contacted a host");
        assert_eq!(out.status.code(), Some(2), "{tag}: {}", stderr(&out));
        assert!(stderr(&out).contains(expect), "{tag}: {}", stderr(&out));
    }
}

#[test]
fn lab_serve_vllm_dry_run_applies_the_pinned_baseline_job() {
    for (slug, name, max_len) in [
        (
            "llama-3.2-3b-instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "32768",
        ),
        (
            "olmoe-1b-7b-0125-instruct",
            "allenai/OLMoE-1B-7B-0125-Instruct",
            "4096",
        ),
    ] {
        let text = dry_run(
            "lab-serve.sh",
            &format!("vllm-{slug}"),
            &["--dry-run", "novanas", "--vllm", slug],
        );
        let id = run_id(&text, "lab-serve");
        let job_name = format!("turbine-lab-vllm-{id}");
        let models = "curl -s -o /dev/null -w %{http_code} --max-time 5 \
                      http://192.168.10.203:18100/v1/models";
        assert_in_order(
            &text,
            &[
                models,
                "refuse to start unless that fails to connect",
                &format!("{SSH} 'test -d /home/piwi/turbine-models/{slug}'"),
                "kubectl apply -f -",
                &format!("kubectl -n turbine-ci logs -f job/{job_name}"),
                models,
            ],
        );
        // Nothing is uploaded, nothing deleted, and Turbine's port is not probed.
        for absent in ["rsync", " delete ", ":18000"] {
            assert!(!text.contains(absent), "{slug}: {absent} in:\n{text}");
        }

        let job = applied_job(&text);
        assert_eq!(str_at(&job, "metadata.name"), job_name);
        assert_eq!(str_at(&job, "metadata.namespace"), "turbine-ci");
        // Role serve: `lab-serve.sh novanas --stop` removes it with the Turbine serve Job.
        assert_eq!(str_at(&job, "metadata.labels.turbine-lab-role"), "serve");
        assert_eq!(str_at(&job, "metadata.labels.turbine-lab-kind"), "vllm");
        assert_eq!(str_at(&job, "metadata.labels.turbine-lab-run"), id);
        assert_eq!(
            at(&job, "spec.template.spec.hostNetwork").as_bool(),
            Some(true)
        );
        assert_eq!(amd_gpus(&job), 1);
        let c = container(&job);
        assert_eq!(str_at(c, "image"), VLLM_IMAGE);
        let words = |key: &str| -> Vec<String> {
            at(c, key)
                .as_sequence()
                .expect("list")
                .iter()
                .map(|v| v.as_str().expect("string entry").to_string())
                .collect()
        };
        assert_eq!(words("command"), ["vllm", "serve"]);
        assert_eq!(
            words("args").join(" "),
            format!(
                "/models/{slug} --served-model-name {name} --host 0.0.0.0 --port 18100 \
                 --dtype bfloat16 --kv-cache-dtype auto --max-model-len {max_len}"
            )
        );
        assert_eq!(env(&job, "HF_HUB_OFFLINE"), "1", "vLLM never downloads");

        // The weights read-only from the same host directory as the Turbine Jobs.
        let model_mount = at(c, "volumeMounts")
            .as_sequence()
            .expect("volumeMounts")
            .iter()
            .find(|m| str_at(m, "mountPath") == "/models")
            .expect("/models mount");
        assert_eq!(
            model_mount.get("readOnly").and_then(Value::as_bool),
            Some(true)
        );
        let volume = at(&job, "spec.template.spec.volumes")
            .as_sequence()
            .expect("volumes")
            .iter()
            .find(|v| str_at(v, "name") == str_at(model_mount, "name"))
            .expect("models volume");
        assert_eq!(str_at(volume, "hostPath.path"), "/home/piwi/turbine-models");
    }
}

#[test]
fn lab_serve_vllm_usage_errors_exit_2_without_contacting_a_host() {
    for (tag, args, expect) in [
        ("vllm-noslug", &["novanas", "--vllm"][..], "usage:"),
        (
            "vllm-badslug",
            &["novanas", "--vllm", "qwen3-moe"][..],
            "unknown model slug for --vllm: qwen3-moe",
        ),
        (
            "vllm-path",
            &["novanas", "--vllm", "../llama-3.2-3b-instruct"][..],
            "unknown model slug",
        ),
        (
            "vllm-extra",
            &["novanas", "--vllm", "llama-3.2-3b-instruct", "x"][..],
            "usage:",
        ),
        (
            "vllm-spark",
            &["dgx-spark", "--vllm", "llama-3.2-3b-instruct"][..],
            "usage:",
        ),
    ] {
        let (out, called) = lab_script("lab-serve.sh", tag, args);
        assert_eq!(called, None, "{tag}: contacted a host");
        assert_eq!(out.status.code(), Some(2), "{tag}: {}", stderr(&out));
        assert!(stderr(&out).contains(expect), "{tag}: {}", stderr(&out));
    }
}

/// The `turbine-server` command line of a rendered serve Job.
fn server_command(job: &Value) -> String {
    let s = script(job);
    s.lines()
        .map(str::trim)
        .find(|l| l.starts_with("exec \"$CARGO_TARGET_DIR/release/turbine-server\""))
        .unwrap_or_else(|| panic!("no turbine-server exec line in:\n{s}"))
        .to_string()
}

#[test]
fn lab_serve_set_passthrough_reaches_the_server_command() {
    let exec =
        "exec \"$CARGO_TARGET_DIR/release/turbine-server\" --config \"$SLOT_DIR/config.yaml\"";
    let plain = applied_job(&dry_run(
        "lab-serve.sh",
        "set-none",
        &["--dry-run", "novanas", "scripts/lab/phase1-novanas.yaml"],
    ));
    assert_eq!(server_command(&plain), exec);

    let text = dry_run(
        "lab-serve.sh",
        "set-two",
        &[
            "--dry-run",
            "novanas",
            "scripts/lab/phase1-novanas.yaml",
            "--set",
            "scheduler.max_batch_tokens=4096",
            "--set",
            "execution.kernel_library=/opt/lib/libturbine_hip.so",
        ],
    );
    assert_eq!(
        server_command(&applied_job(&text)),
        format!(
            "{exec} --set scheduler.max_batch_tokens=4096 \
             --set execution.kernel_library=/opt/lib/libturbine_hip.so"
        )
    );
}

#[test]
fn lab_serve_set_usage_errors_exit_2_without_contacting_a_host() {
    let cfg = "scripts/lab/phase1-novanas.yaml";
    for (tag, args, expect) in [
        ("set-novalue", &["novanas", cfg, "--set"][..], "usage:"),
        (
            "set-noeq",
            &["novanas", cfg, "--set", "scheduler"][..],
            "--set",
        ),
        (
            "set-nokey",
            &["novanas", cfg, "--set", "=4096"][..],
            "--set",
        ),
        (
            "set-unsafe",
            &["novanas", cfg, "--set", "model.path=/x; rm -rf /"][..],
            "--set",
        ),
        ("set-other", &["novanas", cfg, "--runs", "3"][..], "usage:"),
        (
            "set-vllm",
            &[
                "novanas",
                "--vllm",
                "llama-3.2-3b-instruct",
                "--set",
                "a.b=1",
            ][..],
            "usage:",
        ),
        (
            "set-stop",
            &["novanas", "--stop", "--set", "a.b=1"][..],
            "usage:",
        ),
    ] {
        let (out, called) = lab_script("lab-serve.sh", tag, args);
        assert_eq!(called, None, "{tag}: contacted a host");
        assert_eq!(out.status.code(), Some(2), "{tag}: {}", stderr(&out));
        assert!(stderr(&out).contains(expect), "{tag}: {}", stderr(&out));
    }
}

#[test]
fn lab_serve_stop_with_a_run_id_deletes_only_that_run() {
    let id = "0926120000-0123abcd";
    let text = dry_run(
        "lab-serve.sh",
        "stop-id",
        &["--dry-run", "novanas", "--stop", id],
    );
    assert!(
        text.contains(&format!(
            "kubectl -n turbine-ci delete job -l turbine-lab-role=serve,turbine-lab-run={id} \
             --ignore-not-found"
        )),
        "{text}"
    );
    assert_eq!(text.matches(" delete ").count(), 1, "one delete: {text}");
}

#[test]
fn lab_perf_dry_run_serves_benches_and_stops_both_engines() {
    let text = dry_run(
        "lab-perf.sh",
        "perf-llama",
        &[
            "--dry-run",
            "novanas",
            "llama",
            "--set",
            "scheduler.max_batch_tokens=4096",
        ],
    );
    let bench = |port: u16, requests: u32| {
        format!(
            "turbine-bench --url http://192.168.10.203:{port} --concurrency 16 \
             --requests {requests} --prompt-words 512 --max-tokens 256 --ignore-eos --output json"
        )
    };
    let (warm_t, run_t) = (bench(18000, 16), bench(18000, 200));
    let (warm_v, run_v) = (bench(18100, 16), bench(18100, 200));
    assert_in_order(
        &text,
        &[
            "cargo build --release -p turbine-bench --bin turbine-bench",
            "+ scripts/lab-serve.sh novanas scripts/lab/phase2c-novanas-llama.yaml \
             --set scheduler.max_batch_tokens=4096",
            "--config \"$SLOT_DIR/config.yaml\" --set scheduler.max_batch_tokens=4096",
            &warm_t,
            &run_t,
            &run_t,
            &run_t,
            "+ scripts/lab-serve.sh novanas --stop ",
            "delete job -l turbine-lab-role=serve,turbine-lab-run=",
            "+ scripts/lab-serve.sh novanas --vllm llama-3.2-3b-instruct",
            VLLM_IMAGE,
            &warm_v,
            &run_v,
            &run_v,
            &run_v,
            "+ scripts/lab-serve.sh novanas --stop ",
            "delete job -l turbine-lab-role=serve,turbine-lab-run=",
            "lab-perf: dry run: nothing contacted",
        ],
    );
    assert_eq!(text.matches(&run_t).count(), 3, "{text}");
    assert_eq!(text.matches(&run_v).count(), 3, "{text}");
    // Each engine is stopped by its own run id, never by the bare role label.
    assert!(
        !text.contains("delete job -l turbine-lab-role=serve --ignore-not-found"),
        "{text}"
    );

    // --skip-vllm: Turbine only; the verdict compares with the recorded vLLM number.
    let text = dry_run(
        "lab-perf.sh",
        "perf-olmoe-skip",
        &[
            "--dry-run",
            "novanas",
            "olmoe",
            "--skip-vllm",
            "--runs",
            "1",
        ],
    );
    assert_in_order(
        &text,
        &[
            "+ scripts/lab-serve.sh novanas scripts/lab/phase2c-novanas-olmoe.yaml",
            &warm_t,
            &run_t,
            "+ scripts/lab-serve.sh novanas --stop ",
        ],
    );
    assert_eq!(text.matches(&run_t).count(), 1, "{text}");
    assert!(
        !text.contains("--vllm") && !text.contains(":18100"),
        "{text}"
    );
    assert!(text.contains("vllm=535(recorded)"), "{text}");

    // --config replaces the Phase 2c lab config.
    let text = dry_run(
        "lab-perf.sh",
        "perf-config",
        &[
            "--dry-run",
            "novanas",
            "llama",
            "--skip-vllm",
            "--config",
            "scripts/lab/phase2-novanas-llama.yaml",
        ],
    );
    assert!(
        text.contains("+ scripts/lab-serve.sh novanas scripts/lab/phase2-novanas-llama.yaml\n"),
        "{text}"
    );
}

#[test]
fn lab_perf_usage_errors_exit_2_without_contacting_a_host() {
    for (tag, args, expect) in [
        ("perf-noargs", &[][..], "usage:"),
        ("perf-host", &["dgx-spark", "llama"][..], "usage:"),
        ("perf-model", &["novanas", "qwen"][..], "usage:"),
        (
            "perf-runs0",
            &["novanas", "llama", "--runs", "0"][..],
            "usage:",
        ),
        (
            "perf-runsx",
            &["novanas", "llama", "--runs", "x"][..],
            "usage:",
        ),
        ("perf-runs", &["novanas", "llama", "--runs"][..], "usage:"),
        ("perf-set", &["novanas", "llama", "--set"][..], "usage:"),
        (
            "perf-setbad",
            &["novanas", "llama", "--set", "nokey"][..],
            "--set",
        ),
        (
            "perf-config",
            &["novanas", "llama", "--config", "scripts/lab/missing.yaml"][..],
            "scripts/lab/missing.yaml",
        ),
        (
            "perf-unknown",
            &["novanas", "llama", "--fast"][..],
            "usage:",
        ),
    ] {
        let (out, called) = lab_script("lab-perf.sh", tag, args);
        assert_eq!(called, None, "{tag}: contacted a host");
        assert_eq!(out.status.code(), Some(2), "{tag}: {}", stderr(&out));
        assert!(stderr(&out).contains(expect), "{tag}: {}", stderr(&out));
    }
}

/// Shorter test cycles (user decision 2026-09-27): lab-bench.sh usage errors, including the new
/// --with-tests / --skip-tests / --golden16 / --quick flags and the GPU-0-only guard (GPU 1's
/// PCIe link is not throughput-comparable). lab-bench.sh has no --dry-run: a well-formed
/// invocation immediately reaches for the host (rsync, then ssh, via scripts/remote-cargo.sh),
/// so these usage cases only cover what must be rejected before any host contact.
#[test]
fn lab_bench_usage_errors_exit_2_without_contacting_a_host() {
    for (tag, args) in [
        ("bench-gpu1", &["--gpu", "1"][..]),
        ("bench-gpu2", &["--gpu", "2"][..]),
        ("bench-model", &["--model", "qwen"][..]),
        ("bench-unknown", &["--fast"][..]),
    ] {
        let (out, called) = lab_script("lab-bench.sh", tag, args);
        assert_eq!(called, None, "{tag}: contacted a host");
        assert_eq!(out.status.code(), Some(2), "{tag}: {}", stderr(&out));
    }
    let (out, _) = lab_script("lab-bench.sh", "bench-gpu1-message", &["--gpu", "1"]);
    assert!(
        stderr(&out).contains("GPU 0"),
        "refusing GPU 1 should explain why: {}",
        stderr(&out)
    );
}

/// The new flags parse (they are not rejected the way an unknown flag or a bad --gpu/--model is
/// in the test above) for --gpu 0 (the default, spelled out) and every combination of them.
/// lab-bench.sh has no --dry-run and reaches for the host immediately after parsing, so this
/// only checks that parsing succeeded (no "usage:" on stderr), not what happens next: what it
/// reaches for after that (scripts/remote-cargo.sh, which resolves this checkout through `git`)
/// is exercised for real by a human running it, not safe to assert on here since this same test
/// binary also runs from a host-synced tree that excludes .git (scripts/remote-cargo.sh, every
/// lab-*.sh upload).
#[test]
fn lab_bench_new_flags_are_not_usage_errors() {
    for (tag, args) in [
        ("flags-default", &["--gpu", "0"][..]),
        ("flags-with-tests", &["--with-tests"][..]),
        ("flags-skip-tests", &["--skip-tests"][..]),
        ("flags-golden16", &["--golden16"][..]),
        ("flags-quick", &["--quick"][..]),
        (
            "flags-all",
            &["--gpu", "0", "--with-tests", "--golden16", "--quick"][..],
        ),
    ] {
        let (out, _) = lab_script("lab-bench.sh", tag, args);
        assert!(
            !stderr(&out).contains("usage:"),
            "{tag}: rejected as a usage error: {}",
            stderr(&out)
        );
    }
}

/// Runs `bash <args>` with the host-contacting tools stubbed as in [`lab_script`]; returns the
/// output and whether any stub was called.
fn bash_with_stubs(args: &[&str], tag: &str) -> (Output, Option<String>) {
    let stubs =
        std::env::temp_dir().join(format!("turbine-lab-scripts-{}-{tag}", std::process::id()));
    let _ = fs::remove_dir_all(&stubs);
    fs::create_dir_all(&stubs).expect("stub dir");
    let calls = stubs.join("calls.log");
    for tool in ["ssh", "rsync", "scp", "curl", "kubectl", "docker"] {
        let path = stubs.join(tool);
        fs::write(
            &path,
            format!(
                "#!/bin/sh\necho \"{tool} $*\" >> '{}'\nexit 97\n",
                calls.display()
            ),
        )
        .expect("write stub");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).expect("chmod stub");
        }
    }
    let path = format!(
        "{}:{}",
        stubs.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let out = Command::new("bash")
        .args(args)
        .current_dir(repo_root())
        .env("PATH", path)
        .output()
        .expect("run bash");
    let called = fs::read_to_string(&calls).ok();
    let _ = fs::remove_dir_all(&stubs);
    (out, called)
}

/// `soak_precondition <host> <bytes>` from `scripts/overload-soak.sh` sourced with
/// `SOAK_SOURCE_ONLY=1`.
fn soak_precondition(host: &str, bytes: &str) -> (Output, Option<String>) {
    let script = repo_root().join("scripts/overload-soak.sh");
    let snippet = format!(
        "SOAK_SOURCE_ONLY=1 source '{}' && soak_precondition {host} {bytes}",
        script.display()
    );
    bash_with_stubs(&["-c", &snippet], "soak-precondition")
}

/// P3 S-19: the soak refuses a busy host (`precondition`, exit 1) and starts nothing: an R9700
/// with 4 GiB VRAM in use is busy, 512 MiB is free; a Spark needs MemAvailable above the 24 GiB
/// container cap plus the 8 GiB host reserve (25 GiB refused, 42 GiB accepted).
#[test]
fn soak_precondition_refuses_busy_gpu() {
    for (host, bytes, ok) in [
        ("novanas", "4294967296", false),
        ("novanas", "536870912", true),
        ("dgx-spark", "26843545600", false),
        ("dgx-spark2", "45097156608", true),
        ("novanas", "lots", false),
    ] {
        let (out, called) = soak_precondition(host, bytes);
        assert_eq!(called, None, "{host} {bytes}: contacted a host");
        assert_eq!(
            out.status.success(),
            ok,
            "{host} {bytes}: {}{}",
            stdout(&out),
            stderr(&out)
        );
        if !ok {
            assert_eq!(out.status.code(), Some(1), "{host} {bytes}");
            assert!(stderr(&out).contains("precondition"), "{}", stderr(&out));
        }
    }
    // Usage errors exit 2 before anything is contacted.
    for args in [
        &[][..],
        &["localhost"][..],
        &["novanas", "--duration", "ten"][..],
        &["novanas", "--model", "/etc"][..],
    ] {
        let (out, called) = lab_script("overload-soak.sh", "soak-usage", args);
        assert_eq!(out.status.code(), Some(2), "{args:?}: {}", stderr(&out));
        assert_eq!(called, None, "{args:?}: contacted a host");
    }
}

/// The soak's server configuration loads through the real configuration model, on the hip
/// backend with the Phase 3 reliability keys it states.
#[test]
fn phase3_soak_config_loads() {
    let path = repo_root().join("scripts/lab/phase3-novanas-soak.yaml");
    let c = turbine_core::config::load(&path, &[]).expect("phase3-novanas-soak.yaml loads");
    assert_eq!(c.execution.backend.as_str(), "hip");
    assert_eq!(c.server.listen.port(), 18000);
    assert!(c.reliability.enabled && c.reliability.adaptive_admission);
    assert_eq!(c.reliability.admission.max_queue, 256);
    assert_eq!(
        c.kv.gpu.max_bytes, None,
        "the kv pool is the budget remainder (C-8)"
    );
}
/// P5 Task 9: `lab-cluster.sh` passes `bash -n`, and its dry run renders a GPU-less build Job and
/// a two-R9700 scenario Job from `scripts/lab/novanas-cluster-job.yaml`, with the weights
/// read-only, no Service or host port, and a cleanup that selects only this run's Jobs.
#[test]
fn cluster_dry_run_manifest() {
    let syntax = Command::new("bash")
        .arg("-n")
        .arg(repo_root().join("scripts/lab-cluster.sh"))
        .output()
        .expect("run bash -n");
    assert!(syntax.status.success(), "bash -n: {}", stderr(&syntax));

    let text = dry_run(
        "lab-cluster.sh",
        "cluster-tp2",
        &["--dry-run", "tp2-novanas"],
    );
    let id = run_id(&text, "lab-cluster");
    let job_name = format!("turbine-lab-cluster-{id}");
    let build_name = format!("turbine-lab-cluster-build-{id}");
    let run_dir = format!("{CI_ROOT}/runs/{id}");
    let selector = format!("-l turbine-lab=true,turbine-lab-run={id}");
    assert_in_order(
        &text,
        &[
            &format!("{SSH} 'mkdir -p {run_dir}/src {CACHE}/slots"),
            "rsync -rlpcz --delete --exclude target/ --exclude .git/ --exclude .claude/",
            &format!("piwi@192.168.10.203:{run_dir}/src/"),
            "kubectl apply -f -",
            &format!("kubectl -n turbine-ci logs -f job/{build_name}"),
            "kubectl apply -f -",
            "Insufficient amd.com/gpu limit 60 s",
            &format!("kubectl -n turbine-ci logs -f job/{job_name}"),
            &format!("kubectl -n turbine-ci delete job {selector} --ignore-not-found"),
            &format!("rm -rf {run_dir}'"),
        ],
    );
    // The only delete is the run's own label selector; never a bare `turbine-lab=true`.
    assert_eq!(text.matches(" delete ").count(), 1, "{text}");
    assert_eq!(
        text.lines().last(),
        Some("lab-cluster: tp2-novanas PASS"),
        "{text}"
    );

    let jobs = applied_jobs(&text);
    assert_eq!(jobs.len(), 2, "a build Job and the scenario Job");
    let (build, job) = (&jobs[0], &jobs[1]);
    assert_eq!(str_at(build, "metadata.name"), build_name);
    assert_eq!(str_at(job, "metadata.name"), job_name);
    for (j, gpus, phase) in [(build, 0, "build"), (job, 2, "run")] {
        assert_eq!(str_at(j, "metadata.namespace"), "turbine-ci");
        assert_eq!(str_at(j, "metadata.labels.turbine-lab"), "true");
        assert_eq!(str_at(j, "metadata.labels.turbine-lab-run"), id);
        assert_eq!(str_at(j, "metadata.labels.turbine-scenario"), "tp2-novanas");
        assert_eq!(amd_gpus(j), gpus);
        assert_eq!(env(j, "TURBINE_LAB_PHASE"), phase);
        assert_eq!(env(j, "TURBINE_LAB_SCENARIO"), "tp2-novanas");
        assert_eq!(str_at(container(j), "image"), "rust:1.97-trixie");
        let m = mounts(j);
        for want in [
            ("/models", "/home/piwi/turbine-models", true),
            ("/opt/rocm/rocm", "/opt/rocm/rocm", true),
        ] {
            let want = (want.0.to_string(), want.1.to_string(), want.2);
            assert!(m.contains(&want), "{want:?} missing from {m:?}");
        }
        assert!(
            at(j, "spec.template.spec").get("hostNetwork").is_none(),
            "loopback only"
        );
    }
    let s = script(job);
    assert_in_order(
        &s,
        &[
            "flock -w 1800 \"$LOCK_FD\"",
            "rsync -rlpc --delete \"$RUN_DIR/src/\" \"$SLOT_DIR/src/\"",
            "cmake --build \"$KERNEL_BUILD_DIR\"",
            "cargo build --release -p turbine-server -p turbine-bench -p turbine-distributed",
            "exec bash scripts/lab-cluster.sh --in-job \"$TURBINE_LAB_SCENARIO\"",
        ],
    );
    let template =
        fs::read_to_string(repo_root().join("scripts/lab/novanas-cluster-job.yaml")).unwrap();
    for manifest in [
        heredoc(&text, "export KUBECTL_KUBERC=false; kubectl apply -f -"),
        template.as_str(),
    ] {
        assert!(!manifest.contains("kind: Service"), "{manifest}");
        assert!(!manifest.contains("hostPort"), "{manifest}");
    }
}

#[test]
fn lab_cluster_scenarios_and_stop_touch_only_their_own_run() {
    for scenario in ["collbench-novanas", "dp2-novanas"] {
        let text = dry_run("lab-cluster.sh", scenario, &["--dry-run", scenario]);
        let job = applied_jobs(&text).remove(1);
        assert_eq!(env(&job, "TURBINE_LAB_SCENARIO"), scenario);
        assert_eq!(amd_gpus(&job), 2);
        assert_eq!(
            text.lines().last(),
            Some(format!("lab-cluster: {scenario} PASS").as_str())
        );
    }
    let text = dry_run(
        "lab-cluster.sh",
        "cluster-stop",
        &["--dry-run", "--stop", "0926-ab"],
    );
    assert!(
        text.contains(
            "kubectl -n turbine-ci delete job -l turbine-lab=true,turbine-lab-run=0926-ab --ignore-not-found --wait=true"
        ),
        "{text}"
    );
    for absent in ["rsync", "apply"] {
        assert!(
            !text.contains(absent),
            "--stop must not run {absent}: {text}"
        );
    }
    for (tag, args) in [
        ("cl-noargs", &[][..]),
        ("cl-scenario", &["tp4-novanas"][..]),
        ("cl-two", &["tp2-novanas", "dp2-novanas"][..]),
        ("cl-stop-noid", &["--stop"][..]),
        ("cl-stop-badid", &["--stop", "../x"][..]),
        ("cl-injob", &["--in-job"][..]),
    ] {
        let (out, called) = lab_script("lab-cluster.sh", tag, args);
        assert_eq!(called, None, "{tag}: contacted a host");
        assert_eq!(out.status.code(), Some(2), "{tag}: {}", stderr(&out));
        assert!(stderr(&out).contains("usage:"), "{tag}: {}", stderr(&out));
    }
}

#[test]
fn phase5_novanas_configs_load() {
    use turbine_core::config::{DeviceSelection, SizeOrAuto};
    use turbine_core::types::DeviceId;
    for (file, dir, tp, dp) in [
        ("phase5-novanas-llama.yaml", MODEL_DIR, 2, 1),
        ("phase5-novanas-olmoe.yaml", MOE_MODEL_DIR, 2, 1),
        ("phase5-novanas-dp2.yaml", MODEL_DIR, 1, 2),
    ] {
        let path = repo_root().join("scripts/lab").join(file);
        let c = turbine_core::config::load(&path, &[])
            .unwrap_or_else(|e| panic!("{file} does not load: {e}"));
        assert_eq!(c.server.listen.to_string(), "127.0.0.1:18000", "{file}");
        assert_eq!(c.model.path, Path::new(dir), "{file}");
        assert_eq!(c.execution.backend.as_str(), "hip", "{file}");
        let p = &c.parallel;
        assert_eq!(p.tensor_parallel_size, SizeOrAuto::Size(tp), "{file}");
        assert_eq!(p.data_parallel_size, SizeOrAuto::Size(dp), "{file}");
        assert_eq!(
            p.devices,
            DeviceSelection::List(vec![DeviceId(0), DeviceId(1)]),
            "{file}"
        );
        assert_eq!(c.kv.block_tokens, 128, "{file}");
    }
}
