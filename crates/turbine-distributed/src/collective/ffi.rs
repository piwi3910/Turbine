//! One runtime-loaded binding to the NCCL C API (P5 S-2). RCCL implements the same API, so the
//! same table loads `librccl.so.1` (backend `rccl`) or `libnccl.so.2` (backend `nccl`); nothing
//! links either library at build time.
//!
//! This is the only module of `turbine-distributed` allowed to contain `unsafe` (contract §1.3).
//! Ownership rules: [`NcclApi`] owns the loaded library and never unloads it (RCCL and NCCL
//! start helper threads that outlive their communicators), so every function pointer copied
//! out of it stays valid for the life of the process. Communicators, device buffers and streams
//! are never created here; callers pass the phase-1 handles they own.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use libloading::Library;

use super::nccl_api::{FLAVORS, NcclFlavor};
use super::{Collective, CollectiveError, CollectiveInit, CollectiveLibrary, UNIQUE_ID_BYTES};

/// The 13 symbols the binding resolves (contract §15.1); a library missing any is refused.
pub(crate) const SYMBOLS: [&str; 13] = [
    "ncclGetVersion",
    "ncclGetUniqueId",
    "ncclCommInitRankConfig",
    "ncclCommGetAsyncError",
    "ncclCommAbort",
    "ncclCommDestroy",
    "ncclAllReduce",
    "ncclAllGather",
    "ncclReduceScatter",
    "ncclBroadcast",
    "ncclGroupStart",
    "ncclGroupEnd",
    "ncclGetErrorString",
];

/// `ncclResult_t`.
type NcclResult = c_int;
/// `ncclComm_t` (opaque).
type NcclComm = *mut c_void;
/// `cudaStream_t` / `hipStream_t` (opaque), from `StreamRef::native_handle`.
type NativeStream = *mut c_void;

/// `ncclUniqueId`: 128 opaque bytes, passed by value to `ncclCommInitRankConfig`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct NcclUniqueId {
    pub(crate) internal: [c_char; UNIQUE_ID_BYTES],
}

type GetVersionFn = unsafe extern "C" fn(version: *mut c_int) -> NcclResult;
type GetUniqueIdFn = unsafe extern "C" fn(id: *mut NcclUniqueId) -> NcclResult;
/// `config` is an `ncclConfig_t*` built by the communicator (P5 Task 7).
type CommInitRankConfigFn = unsafe extern "C" fn(
    comm: *mut NcclComm,
    nranks: c_int,
    id: NcclUniqueId,
    rank: c_int,
    config: *mut c_void,
) -> NcclResult;
type CommGetAsyncErrorFn =
    unsafe extern "C" fn(comm: NcclComm, async_error: *mut NcclResult) -> NcclResult;
type CommFn = unsafe extern "C" fn(comm: NcclComm) -> NcclResult;
type AllReduceFn = unsafe extern "C" fn(
    send: *const c_void,
    recv: *mut c_void,
    count: usize,
    datatype: c_int,
    op: c_int,
    comm: NcclComm,
    stream: NativeStream,
) -> NcclResult;
type AllGatherFn = unsafe extern "C" fn(
    send: *const c_void,
    recv: *mut c_void,
    sendcount: usize,
    datatype: c_int,
    comm: NcclComm,
    stream: NativeStream,
) -> NcclResult;
type BroadcastFn = unsafe extern "C" fn(
    send: *const c_void,
    recv: *mut c_void,
    count: usize,
    datatype: c_int,
    root: c_int,
    comm: NcclComm,
    stream: NativeStream,
) -> NcclResult;
type GroupFn = unsafe extern "C" fn() -> NcclResult;
type GetErrorStringFn = unsafe extern "C" fn(result: NcclResult) -> *const c_char;

/// The resolved entry points. Communicator calls are made by `NcclCollective` (P5 Task 7).
#[expect(
    dead_code,
    reason = "communicator entry points are called by NcclCollective (P5 Task 7)"
)]
pub(crate) struct NcclFns {
    pub(crate) get_version: GetVersionFn,
    pub(crate) get_unique_id: GetUniqueIdFn,
    pub(crate) comm_init_rank_config: CommInitRankConfigFn,
    pub(crate) comm_get_async_error: CommGetAsyncErrorFn,
    pub(crate) comm_abort: CommFn,
    pub(crate) comm_destroy: CommFn,
    pub(crate) all_reduce: AllReduceFn,
    pub(crate) all_gather: AllGatherFn,
    /// Same signature as all-reduce, with `count` = elements received per rank.
    pub(crate) reduce_scatter: AllReduceFn,
    pub(crate) broadcast: BroadcastFn,
    pub(crate) group_start: GroupFn,
    pub(crate) group_end: GroupFn,
    pub(crate) get_error_string: GetErrorStringFn,
}

/// One loaded NCCL-API library and its binding table.
pub struct NcclApi {
    path: PathBuf,
    flavor: &'static NcclFlavor,
    version: i32,
    pub(crate) fns: NcclFns,
    /// Never unloaded; see the module ownership rules.
    _lib: ManuallyDrop<Library>,
}

impl std::fmt::Debug for NcclApi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NcclApi")
            .field("path", &self.path)
            .field("backend", &self.flavor.name)
            .field("version", &self.version)
            .finish_non_exhaustive()
    }
}

impl NcclApi {
    /// Loads the library of `flavor`: `explicit` when given (a failure is fatal and names the
    /// path), else the flavor's default locations in order (RCCL under the ROCm tree, then the
    /// loader path; NCCL from the loader path).
    pub fn load(
        flavor: &'static NcclFlavor,
        explicit: Option<&Path>,
    ) -> Result<Arc<NcclApi>, CollectiveError> {
        if let Some(path) = explicit {
            return NcclApi::load_from(path, flavor);
        }
        let mut tried = Vec::new();
        for candidate in flavor.defaults {
            match open(Path::new(candidate)) {
                // A library that loads but fails the binding checks is reported as such,
                // not skipped in favour of the next candidate.
                Ok(lib) => return bind(lib, Path::new(candidate), flavor),
                Err(e) => tried.push(e),
            }
        }
        let library = flavor
            .defaults
            .last()
            .map_or_else(|| flavor.name.to_string(), |d| d.to_string());
        Err(CollectiveError::Unavailable {
            library,
            detail: tried.join("; "),
        })
    }

    /// Loads exactly `path`. A file named for another flavor (`libnccl*` asked as `rccl`) is
    /// refused; other names are taken as `flavor`.
    pub fn load_from(
        path: &Path,
        flavor: &'static NcclFlavor,
    ) -> Result<Arc<NcclApi>, CollectiveError> {
        let lib = open(path).map_err(|detail| CollectiveError::Unavailable {
            library: path.display().to_string(),
            detail,
        })?;
        bind(lib, path, flavor)
    }

    /// The registered backend name (`rccl`, `nccl`).
    pub fn backend_name(&self) -> &'static str {
        self.flavor.name
    }

    /// `ncclGetVersion` of the loaded library (`NCCL_VERSION_CODE` encoding).
    pub fn version_code(&self) -> i32 {
        self.version
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// `ncclGetErrorString(code)`.
    pub fn error_string(&self, code: i32) -> String {
        // SAFETY: `get_error_string` comes from the library `self` keeps loaded; it takes a
        // plain integer and returns a pointer to a static NUL-terminated string (or NULL).
        let s = unsafe { (self.fns.get_error_string)(code) };
        if s.is_null() {
            return format!("unknown NCCL result {code}");
        }
        // SAFETY: non-NULL results of ncclGetErrorString point at static NUL-terminated text
        // owned by the library, which is never unloaded.
        unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
    }
}

impl CollectiveLibrary for NcclApi {
    fn backend(&self) -> &'static str {
        self.flavor.name
    }
    fn version(&self) -> Option<String> {
        Some(version_text(self.version))
    }
    /// `ncclGetUniqueId`.
    fn unique_id(&self) -> Result<[u8; UNIQUE_ID_BYTES], CollectiveError> {
        let mut id = NcclUniqueId {
            internal: [0; UNIQUE_ID_BYTES],
        };
        // SAFETY: `get_unique_id` comes from the library `self` keeps loaded; `id` is an
        // exclusively borrowed local of the exact C type (128 bytes) the call writes.
        let rc = unsafe { (self.fns.get_unique_id)(&mut id) };
        if rc != 0 {
            return Err(CollectiveError::Backend {
                code: rc,
                message: format!("ncclGetUniqueId: {}", self.error_string(rc)),
            });
        }
        Ok(id.internal.map(|c| c as u8))
    }
    fn open(&self, _init: CollectiveInit) -> Result<Arc<dyn Collective>, CollectiveError> {
        Err(CollectiveError::Unavailable {
            library: self.path.display().to_string(),
            detail: "communicators arrive with P5 Task 7".into(),
        })
    }
}

/// The flavor whose file-name prefix `path` carries (`librccl*` → rccl, `libnccl*` → nccl).
fn flavor_from_name(path: &Path) -> Option<&'static NcclFlavor> {
    let name = path.file_name()?.to_str()?;
    FLAVORS
        .iter()
        .copied()
        .find(|f| name.starts_with(f.file_prefix))
}

/// `2.27.3 (22703)` for an `NCCL_VERSION_CODE`.
fn version_text(code: i32) -> String {
    format!(
        "{}.{}.{} ({code})",
        code / 10_000,
        code / 100 % 100,
        code % 100
    )
}

fn open(path: &Path) -> Result<Library, String> {
    // SAFETY: loading runs the library's initialisers; RCCL and NCCL have none with
    // preconditions. The returned Library is either owned by an NcclApi (never unloaded) or
    // dropped before any symbol is resolved from it.
    unsafe { Library::new(path) }.map_err(|e| format!("{}: {}", path.display(), loader_error(&e)))
}

fn bind(
    lib: Library,
    path: &Path,
    flavor: &'static NcclFlavor,
) -> Result<Arc<NcclApi>, CollectiveError> {
    let unavailable = |detail: String| CollectiveError::Unavailable {
        library: path.display().to_string(),
        detail,
    };
    if let Some(named) = flavor_from_name(path)
        && !std::ptr::eq(named, flavor)
    {
        return Err(unavailable(format!(
            "the file name names backend {} but {} was requested",
            named.name, flavor.name
        )));
    }
    // Report every missing symbol at once rather than the first one only.
    let missing: Vec<&str> = SYMBOLS
        .iter()
        .copied()
        .filter(|name| {
            // SAFETY: only the symbol's presence is checked; the returned address is never
            // dereferenced or called.
            unsafe { lib.get::<*const c_void>(name.as_bytes()) }.is_err()
        })
        .collect();
    if !missing.is_empty() {
        return Err(unavailable(format!(
            "missing symbol(s) {} of the {} the binding needs",
            missing.join(", "),
            SYMBOLS.len()
        )));
    }
    macro_rules! sym {
        ($name:literal, $ty:ty) => {{
            // SAFETY: `$ty` is the C signature nccl.h / rccl.h declare for `$name`; the pointer
            // is copied out and only used while `lib`, moved into the NcclApi below and never
            // unloaded, is loaded.
            let s = unsafe { lib.get::<$ty>($name.as_bytes()) }.map_err(|e| {
                unavailable(format!("missing symbol {}: {}", $name, loader_error(&e)))
            })?;
            *s
        }};
    }
    let fns = NcclFns {
        get_version: sym!("ncclGetVersion", GetVersionFn),
        get_unique_id: sym!("ncclGetUniqueId", GetUniqueIdFn),
        comm_init_rank_config: sym!("ncclCommInitRankConfig", CommInitRankConfigFn),
        comm_get_async_error: sym!("ncclCommGetAsyncError", CommGetAsyncErrorFn),
        comm_abort: sym!("ncclCommAbort", CommFn),
        comm_destroy: sym!("ncclCommDestroy", CommFn),
        all_reduce: sym!("ncclAllReduce", AllReduceFn),
        all_gather: sym!("ncclAllGather", AllGatherFn),
        reduce_scatter: sym!("ncclReduceScatter", AllReduceFn),
        broadcast: sym!("ncclBroadcast", BroadcastFn),
        group_start: sym!("ncclGroupStart", GroupFn),
        group_end: sym!("ncclGroupEnd", GroupFn),
        get_error_string: sym!("ncclGetErrorString", GetErrorStringFn),
    };
    let mut version: c_int = 0;
    // SAFETY: `get_version` was resolved from `lib`, still loaded here; `version` is an
    // exclusively borrowed local int, the out-parameter ncclGetVersion writes.
    let rc = unsafe { (fns.get_version)(&mut version) };
    if rc != 0 {
        return Err(unavailable(format!("ncclGetVersion returned {rc}")));
    }
    let minimum = flavor.min_version;
    if version < minimum {
        return Err(unavailable(format!(
            "{} version {} is below the minimum {}",
            flavor.name,
            version_text(version),
            version_text(minimum)
        )));
    }
    tracing::info!(
        event = "collective_library_loaded",
        backend = flavor.name,
        path = %path.display(),
        version = %version_text(version),
        "NCCL-API library loaded"
    );
    Ok(Arc::new(NcclApi {
        path: path.to_path_buf(),
        flavor,
        version,
        fns,
        _lib: ManuallyDrop::new(lib),
    }))
}

/// libloading's message plus the platform loader's (`dlerror`) text.
fn loader_error(e: &libloading::Error) -> String {
    match std::error::Error::source(e) {
        Some(source) => format!("{e}: {source}"),
        None => e.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::collective::CollectiveError;
    use crate::collective::nccl_api::{NCCL_FLAVOR, RCCL_FLAVOR, RCCL_MIN_VERSION};

    fn stub(variant: &str, file: &str) -> PathBuf {
        Path::new(env!("TURBINE_NCCL_STUB_DIR"))
            .join(variant)
            .join(file)
    }

    fn unavailable(e: CollectiveError) -> (String, String) {
        match e {
            CollectiveError::Unavailable { library, detail } => (library, detail),
            other => panic!("expected Unavailable, got {other:?}"),
        }
    }

    #[test]
    fn missing_library() {
        #[cfg(target_os = "macos")]
        for flavor in [&RCCL_FLAVOR, &NCCL_FLAVOR] {
            let (library, detail) =
                unavailable(NcclApi::load(flavor, None).expect_err("no NCCL-API library on macOS"));
            assert!(
                library.contains(flavor.name),
                "{}: library {library}",
                flavor.name
            );
            // The platform loader's message (dlopen) is carried through.
            assert!(detail.contains("dlopen"), "{}: {detail}", flavor.name);
        }
        let explicit = Path::new("/nonexistent/librccl.so.1");
        let err =
            NcclApi::load(&RCCL_FLAVOR, Some(explicit)).expect_err("explicit path does not exist");
        assert!(
            err.to_string().contains("/nonexistent/librccl.so.1"),
            "{err}"
        );
    }

    #[test]
    fn one_binding_both_libraries() {
        let expected: i32 = env!("TURBINE_NCCL_STUB_VERSION").parse().expect("version");
        let rccl = NcclApi::load_from(&stub("rccl", "librccl.so.1"), &RCCL_FLAVOR)
            .expect("rccl stub loads");
        assert_eq!(rccl.backend_name(), "rccl");
        assert_eq!(rccl.version_code(), expected);
        assert_eq!(rccl.path(), stub("rccl", "librccl.so.1"));

        // The same binding type serves NCCL; the file name decides the backend.
        let nccl: Arc<NcclApi> = NcclApi::load_from(&stub("nccl", "libnccl.so.2"), &NCCL_FLAVOR)
            .expect("nccl stub loads");
        assert_eq!(nccl.backend_name(), "nccl");
        assert_eq!(nccl.version_code(), expected);
        assert_eq!(SYMBOLS.len(), 13);
        assert_eq!(nccl.error_string(4), "invalid argument");
        // Through the registry's trait: the name, the version text and a group id.
        let lib: &dyn CollectiveLibrary = &*nccl;
        assert_eq!(lib.backend(), "nccl");
        assert_eq!(lib.version().as_deref(), Some("2.30.4 (23004)"));
        assert!(lib.unique_id().is_ok());

        // An explicit path goes through the same loader.
        let explicit = NcclApi::load(&NCCL_FLAVOR, Some(&stub("nccl", "libnccl.so.2")))
            .expect("explicit stub path loads");
        assert_eq!(explicit.backend_name(), "nccl");

        // A file named for the other vendor is refused.
        let (_, detail) = unavailable(
            NcclApi::load_from(&stub("rccl", "librccl.so.1"), &NCCL_FLAVOR)
                .expect_err("rccl file for an nccl request"),
        );
        assert!(detail.contains("rccl"), "{detail}");

        let (library, detail) = unavailable(
            NcclApi::load_from(&stub("low", "librccl.so.1"), &RCCL_FLAVOR)
                .expect_err("version below the minimum"),
        );
        assert!(library.ends_with("low/librccl.so.1"), "{library}");
        assert!(detail.contains("21800"), "{detail}");
        assert!(detail.contains(&RCCL_MIN_VERSION.to_string()), "{detail}");

        let (_, detail) = unavailable(
            NcclApi::load_from(&stub("nogroupend", "libnccl.so.2"), &NCCL_FLAVOR)
                .expect_err("ncclGroupEnd missing"),
        );
        assert!(detail.contains("ncclGroupEnd"), "{detail}");
    }
}
