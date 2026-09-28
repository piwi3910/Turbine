//! Read-only parsers over a sysfs root (and the `proc/` directory beside it): NUMA nodes, the
//! PCIe parent chain of a device, network interfaces, InfiniBand/RoCE devices, NVMe controllers
//! and IPv4 addresses. Every reader returns `Err` with the path it could not read, so the caller
//! can degrade that one source to `unknown`.

use std::collections::BTreeMap;
use std::net::Ipv4Addr;
use std::path::{Path, PathBuf};

/// One NUMA node from `devices/system/node/node<N>`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NumaNode {
    pub id: u32,
    pub cpus: Option<String>,
    pub memory_bytes: Option<u64>,
    pub distances: Option<Vec<u32>>,
}

/// A PCI function resolved through `bus/pci/devices/<bdf>`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PciDevice {
    pub bdf: String,
    /// The host bridge directory (`pci0000:00`).
    pub host_bridge: String,
    /// Bridges from the root port down to this device's parent.
    pub chain: Vec<PciFunction>,
    pub function: PciFunction,
}

/// The attributes of one PCI function directory.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PciFunction {
    pub bdf: String,
    pub class: Option<u32>,
    /// Raw `numa_node` (`-1` = the kernel does not know).
    pub numa_node: Option<i32>,
    pub link_gts: Option<f64>,
    pub width: Option<u32>,
}

/// A network interface backed by a PCI function (virtual interfaces are skipped).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NetIf {
    pub name: String,
    pub bdf: String,
    pub speed_mbps: Option<u64>,
    pub mac: Option<String>,
    /// `type` (`ARPHRD_*`): 1 Ethernet, 32 InfiniBand.
    pub arp_type: Option<u32>,
}

/// An RDMA device from `class/infiniband` (port 1).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IbDev {
    pub name: String,
    pub bdf: String,
    pub link_layer: Option<String>,
    pub active: Option<bool>,
    pub rate_gbps: Option<f64>,
}

/// An NVMe controller from `class/nvme`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NvmeCtrl {
    pub name: String,
    pub bdf: String,
    pub model: Option<String>,
}

pub(crate) struct Sysfs {
    root: PathBuf,
    proc_root: PathBuf,
}

impl Sysfs {
    /// `root` is the sysfs mount (`/sys`); `proc/` is read from its parent directory.
    pub(crate) fn new(root: &Path) -> Sysfs {
        let proc_root = root
            .parent()
            .map_or_else(|| PathBuf::from("/proc"), |p| p.join("proc"));
        Sysfs {
            root: root.to_path_buf(),
            proc_root,
        }
    }

    pub(crate) fn hostname(&self) -> Option<String> {
        read_trimmed(&self.proc_root.join("sys/kernel/hostname")).filter(|h| !h.is_empty())
    }

    pub(crate) fn numa_nodes(&self) -> Result<Vec<NumaNode>, String> {
        let dir = self.root.join("devices/system/node");
        let mut nodes = Vec::new();
        for name in list_dir(&dir)? {
            let Some(id) = name
                .strip_prefix("node")
                .and_then(|n| n.parse::<u32>().ok())
            else {
                continue;
            };
            let node = dir.join(&name);
            nodes.push(NumaNode {
                id,
                cpus: read_trimmed(&node.join("cpulist")),
                memory_bytes: read_trimmed(&node.join("meminfo"))
                    .and_then(|m| parse_node_memtotal(&m)),
                distances: read_trimmed(&node.join("distance")).and_then(|d| {
                    d.split_whitespace()
                        .map(|x| x.parse().ok())
                        .collect::<Option<Vec<u32>>>()
                }),
            });
        }
        nodes.sort_by_key(|n| n.id);
        Ok(nodes)
    }

    /// Whether `bus/pci/devices` can be listed at all.
    pub(crate) fn pci_available(&self) -> Result<(), String> {
        list_dir(&self.root.join("bus/pci/devices")).map(drop)
    }

    /// Resolves `bdf` to its canonical device directory and parent chain.
    pub(crate) fn pci_device(&self, bdf: &str) -> Option<PciDevice> {
        let bdf = bdf.to_ascii_lowercase();
        let dir = self.root.join("bus/pci/devices").join(&bdf);
        self.device_at(&dir)
    }

    /// The PCI function a `device` symlink (or directory) points at.
    fn device_at(&self, link: &Path) -> Option<PciDevice> {
        let real = std::fs::canonicalize(link).ok()?;
        let components: Vec<String> = real
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        // The last `pciDDDD:BB` component is the host bridge; the BDFs after it are the chain.
        let host_at = components
            .iter()
            .rposition(|c| c.starts_with("pci") && !is_bdf(c))?;
        let bdfs: Vec<&String> = components[host_at + 1..].iter().collect();
        if bdfs.is_empty() || !bdfs.iter().all(|c| is_bdf(c)) {
            return None;
        }
        let mut dir: PathBuf = components[..=host_at].iter().collect();
        let mut functions = Vec::with_capacity(bdfs.len());
        for b in &bdfs {
            dir.push(b.as_str());
            functions.push(read_function(&dir, b));
        }
        let function = functions.pop()?;
        Some(PciDevice {
            bdf: function.bdf.clone(),
            host_bridge: components[host_at].clone(),
            chain: functions,
            function,
        })
    }

    pub(crate) fn net_interfaces(&self) -> Result<Vec<NetIf>, String> {
        let dir = self.root.join("class/net");
        let mut out = Vec::new();
        for name in list_dir(&dir)? {
            let ifdir = dir.join(&name);
            let Some(dev) = self.device_at(&ifdir.join("device")) else {
                continue; // virtual interface (lo, bridges, veth, …)
            };
            out.push(NetIf {
                name,
                bdf: dev.bdf,
                speed_mbps: read_trimmed(&ifdir.join("speed"))
                    .and_then(|s| s.parse::<i64>().ok())
                    .and_then(|s| u64::try_from(s).ok())
                    .filter(|&s| s > 0),
                mac: read_trimmed(&ifdir.join("address")).filter(|a| !a.is_empty()),
                arp_type: read_trimmed(&ifdir.join("type")).and_then(|t| t.parse().ok()),
            });
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }

    pub(crate) fn infiniband(&self) -> Result<Vec<IbDev>, String> {
        let dir = self.root.join("class/infiniband");
        let mut out = Vec::new();
        for name in list_dir(&dir)? {
            let devdir = dir.join(&name);
            let Some(dev) = self.device_at(&devdir.join("device")) else {
                continue;
            };
            let port = devdir.join("ports/1");
            out.push(IbDev {
                name,
                bdf: dev.bdf,
                link_layer: read_trimmed(&port.join("link_layer")).map(|l| l.to_ascii_lowercase()),
                active: read_trimmed(&port.join("state")).map(|s| s.contains("ACTIVE")),
                rate_gbps: read_trimmed(&port.join("rate")).and_then(|r| parse_ib_rate(&r)),
            });
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }

    pub(crate) fn nvme(&self) -> Result<Vec<NvmeCtrl>, String> {
        let dir = self.root.join("class/nvme");
        let mut out = Vec::new();
        for name in list_dir(&dir)? {
            let cdir = dir.join(&name);
            let Some(dev) = self.device_at(&cdir.join("device")) else {
                continue;
            };
            out.push(NvmeCtrl {
                name,
                bdf: dev.bdf,
                model: read_trimmed(&cdir.join("model")).filter(|m| !m.is_empty()),
            });
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }

    /// IPv4 addresses per interface: the local addresses of `proc/net/fib_trie`, each assigned
    /// to the interface of the most specific `proc/net/route` entry containing it.
    pub(crate) fn ipv4_by_interface(&self) -> Result<BTreeMap<String, Vec<String>>, String> {
        let trie_path = self.proc_root.join("net/fib_trie");
        let route_path = self.proc_root.join("net/route");
        let trie = std::fs::read_to_string(&trie_path)
            .map_err(|e| format!("{}: {e}", trie_path.display()))?;
        let routes = std::fs::read_to_string(&route_path)
            .map_err(|e| format!("{}: {e}", route_path.display()))?;
        let routes = parse_routes(&routes);
        let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for addr in parse_fib_trie_locals(&trie) {
            let best = routes
                .iter()
                .filter(|r| r.mask != 0 && u32::from(addr) & r.mask == r.dest)
                .max_by_key(|r| r.mask.count_ones());
            if let Some(r) = best {
                let list = out.entry(r.iface.clone()).or_default();
                let s = addr.to_string();
                if !list.contains(&s) {
                    list.push(s);
                }
            }
        }
        Ok(out)
    }
}

fn list_dir(dir: &Path) -> Result<Vec<String>, String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut names: Vec<String> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    Ok(names)
}

fn read_trimmed(path: &Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
}

fn read_function(dir: &Path, bdf: &str) -> PciFunction {
    PciFunction {
        bdf: bdf.to_string(),
        class: read_trimmed(&dir.join("class"))
            .and_then(|c| u32::from_str_radix(c.trim_start_matches("0x"), 16).ok()),
        numa_node: read_trimmed(&dir.join("numa_node")).and_then(|n| n.parse().ok()),
        link_gts: read_trimmed(&dir.join("current_link_speed")).and_then(|s| parse_link_speed(&s)),
        width: read_trimmed(&dir.join("current_link_width"))
            .and_then(|w| w.parse().ok())
            .filter(|&w: &u32| w > 0),
    }
}

/// `DDDD:BB:DD.F` in lower-case hex.
pub(crate) fn is_bdf(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 12
        && b[4] == b':'
        && b[7] == b':'
        && b[10] == b'.'
        && b.iter().enumerate().all(|(i, c)| {
            matches!(i, 4 | 7 | 10) || c.is_ascii_digit() || (b'a'..=b'f').contains(c)
        })
}

/// `"32.0 GT/s PCIe"` → 32.0; `"Unknown"` → None.
pub(crate) fn parse_link_speed(s: &str) -> Option<f64> {
    s.split_whitespace()
        .next()?
        .parse::<f64>()
        .ok()
        .filter(|v| *v > 0.0)
}

/// `"200 Gb/sec (4X HDR)"` → 200.0.
pub(crate) fn parse_ib_rate(s: &str) -> Option<f64> {
    let mut words = s.split_whitespace();
    let value = words.next()?.parse::<f64>().ok()?;
    (words.next()? == "Gb/sec" && value > 0.0).then_some(value)
}

/// `Node 0 MemTotal:  131596048 kB` → bytes.
fn parse_node_memtotal(meminfo: &str) -> Option<u64> {
    meminfo.lines().find_map(|l| {
        let mut w = l.split_whitespace();
        (w.next()? == "Node").then_some(())?;
        w.next()?;
        (w.next()? == "MemTotal:").then_some(())?;
        let kb: u64 = w.next()?.parse().ok()?;
        (w.next()? == "kB").then(|| kb.saturating_mul(1024))
    })
}

struct Route {
    iface: String,
    /// Destination and mask in host order.
    dest: u32,
    mask: u32,
}

/// `proc/net/route`: little-endian hex destination and mask per interface.
fn parse_routes(text: &str) -> Vec<Route> {
    text.lines()
        .skip(1)
        .filter_map(|l| {
            let cols: Vec<&str> = l.split_whitespace().collect();
            let hex = |s: &str| {
                u32::from_str_radix(s, 16)
                    .ok()
                    .map(|v| u32::from_be_bytes(v.to_le_bytes()))
            };
            Some(Route {
                iface: (*cols.first()?).to_string(),
                dest: hex(cols.get(1)?)?,
                mask: hex(cols.get(7)?)?,
            })
        })
        .collect()
}

/// Addresses listed as `/32 host LOCAL` in `proc/net/fib_trie`, loopback excluded.
fn parse_fib_trie_locals(text: &str) -> Vec<Ipv4Addr> {
    let mut out = Vec::new();
    let mut last: Option<Ipv4Addr> = None;
    for line in text.lines() {
        let t = line.trim();
        if let Some(addr) = t.strip_prefix("|-- ") {
            last = addr.parse().ok();
        } else if t.starts_with("/32 host LOCAL")
            && let Some(a) = last.take().filter(|a| !a.is_loopback() && !out.contains(a))
        {
            out.push(a);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_parsers() {
        assert!(is_bdf("0000:03:00.0"));
        assert!(!is_bdf("pci0000:00"));
        assert!(!is_bdf("0000:03:00.A"));
        assert_eq!(parse_link_speed("32.0 GT/s PCIe"), Some(32.0));
        assert_eq!(parse_link_speed("Unknown"), None);
        assert_eq!(parse_ib_rate("200 Gb/sec (4X HDR)"), Some(200.0));
        assert_eq!(parse_ib_rate("invalid"), None);
        assert_eq!(
            parse_node_memtotal("Node 0 MemTotal:       1024 kB\nNode 0 MemFree: 1 kB"),
            Some(1_048_576)
        );
        let routes = parse_routes(
            "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT\n\
             eth0\t000AA8C0\t00000000\t0001\t0\t0\t0\t00FFFFFF\t0\t0\t0\n",
        );
        assert_eq!(routes[0].dest, u32::from(Ipv4Addr::new(192, 168, 10, 0)));
        assert_eq!(routes[0].mask, u32::from(Ipv4Addr::new(255, 255, 255, 0)));
    }
}
