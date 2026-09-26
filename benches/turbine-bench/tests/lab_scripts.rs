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
use turbine_core::types::ExecutionBackend;

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

/// The Job a dry run applies (the rendered template), parsed.
fn applied_job(text: &str) -> Value {
    let manifest = heredoc(text, "export KUBECTL_KUBERC=false; kubectl apply -f -");
    assert!(
        !manifest.contains("__"),
        "unrendered placeholder:\n{manifest}"
    );
    serde_norway::from_str(manifest).unwrap_or_else(|e| panic!("{e}:\n{manifest}"))
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
    assert_eq!(c.execution.backend, ExecutionBackend::Hip);
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
        assert_eq!(c.execution.backend, ExecutionBackend::Hip, "{file}");
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
        assert_eq!(c.kv.block_tokens, 16, "{file}");
    }
}

#[test]
fn lab_test_dry_run_applies_a_one_gpu_job_with_cached_slots() {
    let text = dry_run("lab-test.sh", "test-default", &["--dry-run", "novanas"]);
    let id = run_id(&text, "lab-test");
    let job_name = format!("turbine-lab-test-{id}");
    let run_dir = format!("{CI_ROOT}/runs/{id}");
    assert_in_order(
        &text,
        &[
            &format!("{SSH} 'mkdir -p {run_dir}/src {CACHE}/slots"),
            "rsync -az --delete --exclude target/ --exclude .git/ --exclude .claude/",
            &format!("piwi@192.168.10.203:{run_dir}/src/"),
            &format!("cat > {run_dir}/test-command"),
            "kubectl apply -f -",
            &format!("kubectl -n turbine-ci logs -f job/{job_name}"),
        ],
    );
    // A run never deletes anything; cleanup (interrupt, unschedulable) names only its own Job.
    assert!(!text.contains(" delete "), "{text}");
    assert!(
        text.contains("-exec rm -rf {} +"),
        "stale uploads are pruned: {text}"
    );

    let job = applied_job(&text);
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
            "exec {LOCK_FD}>\"$TURBINE_LAB_CACHE/slots/test-$n.lock\"",
            "flock -n \"$LOCK_FD\"",
            "export CARGO_TARGET_DIR=\"$SLOT_DIR/target\"",
            "export KERNEL_BUILD_DIR=\"$SLOT_DIR/kernels\"",
            "export TURBINE_KERNEL_LIBRARY=\"$KERNEL_BUILD_DIR/libturbine_hip.so\"",
            "rsync -a --delete \"$RUN_DIR/src/\" \"$SLOT_DIR/src/\"",
            "mapfile -d '' TEST_CMD < \"$RUN_DIR/test-command\"",
            "rm -rf \"$RUN_DIR\"",
            "cd \"$SLOT_DIR/src\"",
            "CMAKE_HOME_DIRECTORY:INTERNAL=$PWD/kernels/rocm",
            "if [[ ! -f \"$KERNEL_BUILD_DIR/CMakeCache.txt\" ]]",
            "cmake -S kernels/rocm -B \"$KERNEL_BUILD_DIR\" -G Ninja",
            "-DGPU_TARGETS=gfx1201",
            "cmake --build \"$KERNEL_BUILD_DIR\"",
            "test -f \"$TURBINE_KERNEL_LIBRARY\"",
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
    let job = applied_job(&text);
    assert_eq!(amd_gpus(&job), 2);
    assert_eq!(env(&job, "TURBINE_EXPECT_AMD"), "2");
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
            "kubectl -n turbine-ci delete job turbine-lab-test-{ida} --ignore-not-found"
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
            "rsync -a --delete \"$RUN_DIR/src/\" \"$SLOT_DIR/src/\"",
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
            "rsync -az --delete --exclude target/ --exclude .git/ --exclude .claude/",
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
