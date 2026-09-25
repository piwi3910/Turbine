//! Lab scripts and manifests, checked without contacting a lab host.
//!
//! - `scripts/lab/phase1-novanas.yaml` loads through the real configuration model.
//! - `scripts/lab/novanas-{test,serve}-job.yaml` carry the kernel build, the environment and the
//!   read-only model mount that the GPU tests and `turbine-server` rely on.
//! - `scripts/lab-serve.sh --dry-run` prints the commands it would run; `ssh`, `rsync`, `curl`
//!   and `kubectl` are replaced by stubs that record any call, so a dry run that reaches for a
//!   host fails the test.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_norway::Value;
use turbine_core::types::ExecutionBackend;

const CI_ROOT: &str = "/home/piwi/turbine-ci";
const KERNEL_LIBRARY: &str = "/home/piwi/turbine-ci/target/kernels/libturbine_hip.so";
const MODEL_DIR: &str = "/models/llama-3.2-3b-instruct";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repository root")
}

fn manifest(name: &str) -> Value {
    let path = repo_root().join("scripts/lab").join(name);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_norway::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
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

fn env(job: &Value, name: &str) -> String {
    at(container(job), "env")
        .as_sequence()
        .expect("env list")
        .iter()
        .find(|e| str_at(e, "name") == name)
        .map(|e| str_at(e, "value").to_string())
        .unwrap_or_else(|| panic!("env {name} is not set"))
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
fn novanas_test_job_builds_kernels_and_mounts_models() {
    let job = manifest("novanas-test-job.yaml");
    assert_eq!(str_at(&job, "metadata.name"), "turbine-lab-test");
    assert_eq!(str_at(&job, "metadata.namespace"), "turbine-ci");
    assert_eq!(env(&job, "TURBINE_KERNEL_LIBRARY"), KERNEL_LIBRARY);
    assert_eq!(env(&job, "TURBINE_TEST_BACKEND"), "hip");
    assert_eq!(env(&job, "TURBINE_TEST_MODEL_DIR"), MODEL_DIR);
    assert_eq!(env(&job, "TURBINE_ROCM_PATH"), "/opt/rocm/rocm/core-7.14");
    assert_eq!(env(&job, "UV_CACHE_DIR"), format!("{CI_ROOT}/uv-cache"));
    assert!(
        mounts(&job).contains(&(
            "/models".to_string(),
            "/home/piwi/turbine-models".to_string(),
            true
        )),
        "read-only /home/piwi/turbine-models at /models: {:?}",
        mounts(&job)
    );

    let s = script(&job);
    assert!(s.contains("set -euo pipefail"), "{s}");
    assert_in_order(
        &s,
        &[
            "apt-get install -y -qq cmake python3",
            "cmake -S kernels/rocm -B \"$KERNEL_BUILD_DIR\"",
            "-DGPU_TARGETS=gfx1201",
            "cmake --build \"$KERNEL_BUILD_DIR\"",
            "test -f \"$TURBINE_KERNEL_LIBRARY\"",
            "https://astral.sh/uv/",
            "exec cargo test --workspace -- --include-ignored --show-output",
        ],
    );
    assert_eq!(
        env(&job, "KERNEL_BUILD_DIR"),
        format!("{CI_ROOT}/target/kernels")
    );
    assert!(
        KERNEL_LIBRARY.starts_with(&env(&job, "KERNEL_BUILD_DIR")),
        "the library is built into KERNEL_BUILD_DIR"
    );
}

#[test]
fn novanas_serve_job_matches_the_test_job() {
    let test = manifest("novanas-test-job.yaml");
    let serve = manifest("novanas-serve-job.yaml");
    assert_eq!(str_at(&serve, "metadata.name"), "turbine-lab-serve");
    assert_eq!(str_at(&serve, "metadata.namespace"), "turbine-ci");
    assert_eq!(str_at(&serve, "metadata.labels.turbine-lab"), "true");
    assert_eq!(
        at(&serve, "spec.template.spec.hostNetwork").as_bool(),
        Some(true)
    );
    assert_eq!(amd_gpus(&serve), 1);
    assert_eq!(amd_gpus(&test), 2);

    // Same mounts (the serve Job needs no uv); same kernel library and ROCm root.
    assert_eq!(mounts(&serve), mounts(&test));
    for var in [
        "TURBINE_KERNEL_LIBRARY",
        "TURBINE_ROCM_PATH",
        "KERNEL_BUILD_DIR",
        "CARGO_TARGET_DIR",
        "LD_LIBRARY_PATH",
        "TURBINE_AMD_SMI_LIBRARY",
    ] {
        assert_eq!(env(&serve, var), env(&test, var), "{var}");
    }

    let s = script(&serve);
    assert_in_order(
        &s,
        &[
            "apt-get install -y -qq cmake python3",
            "cmake -S kernels/rocm -B \"$KERNEL_BUILD_DIR\"",
            "cmake --build \"$KERNEL_BUILD_DIR\"",
            "cargo build --release -p turbine-server",
            "exec \"$CARGO_TARGET_DIR/release/turbine-server\" --config /home/piwi/turbine-ci/serve/config.yaml",
        ],
    );
}

/// Runs `scripts/lab-serve.sh <args>` with stubs for every host-contacting tool first on PATH.
/// Returns the output and whether any stub was called.
fn lab_serve(tag: &str, args: &[&str]) -> (Output, Option<String>) {
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
        .arg(repo_root().join("scripts/lab-serve.sh"))
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

#[test]
fn lab_serve_dry_run_prints_the_start_sequence() {
    let (out, called) = lab_serve(
        "start",
        &["--dry-run", "novanas", "scripts/lab/phase1-novanas.yaml"],
    );
    let text = stdout(&out);
    println!("{text}");
    assert_eq!(called, None, "a dry run contacted a host");
    assert!(
        out.status.success(),
        "exit {:?}: {}",
        out.status,
        stderr(&out)
    );
    assert_in_order(
        &text,
        &[
            "ssh -o BatchMode=yes -o ConnectTimeout=10 piwi@192.168.10.203",
            "mkdir -p /home/piwi/turbine-ci/src /home/piwi/turbine-ci/target /home/piwi/turbine-ci/cargo-registry /home/piwi/turbine-ci/serve",
            "rsync -az --delete --exclude target/ --exclude .git/",
            "piwi@192.168.10.203:/home/piwi/turbine-ci/src/",
            "phase1-novanas.yaml piwi@192.168.10.203:/home/piwi/turbine-ci/serve/config.yaml",
            "kubectl -n turbine-ci delete job turbine-lab-serve --ignore-not-found --wait=true",
            "kubectl apply -f /home/piwi/turbine-ci/src/scripts/lab/novanas-serve-job.yaml",
            "kubectl -n turbine-ci logs -f job/turbine-lab-serve",
            "curl -s -o /dev/null -w %{http_code} --max-time 5 http://192.168.10.203:18000/ready",
        ],
    );
}

#[test]
fn lab_serve_dry_run_stop_deletes_only_the_serve_job() {
    let (out, called) = lab_serve("stop", &["--dry-run", "novanas", "--stop"]);
    let text = stdout(&out);
    println!("{text}");
    assert_eq!(called, None, "a dry run contacted a host");
    assert!(
        out.status.success(),
        "exit {:?}: {}",
        out.status,
        stderr(&out)
    );
    assert!(
        text.contains("kubectl -n turbine-ci delete job turbine-lab-serve --ignore-not-found"),
        "{text}"
    );
    let deletes = text.matches(" delete ").count();
    assert_eq!(deletes, 1, "exactly one delete: {text}");
    for absent in ["rsync", "apply", "turbine-lab-test"] {
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
        let (out, called) = lab_serve(tag, args);
        assert_eq!(called, None, "{tag}: contacted a host");
        assert_eq!(out.status.code(), Some(2), "{tag}: {}", stderr(&out));
        assert!(stderr(&out).contains(expect), "{tag}: {}", stderr(&out));
    }
}
