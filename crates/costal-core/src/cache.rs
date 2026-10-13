//! The share of L2 each thread has to itself, used to size tiles (docs/06 §2.3).
//!
//! We target only the per-core private L2, never L3: L3 is shared by all cores and other
//! processes, so the share we get is unpredictable.
//! Without cache information we use a fallback; these numbers affect speed only, never output
//! (docs/06 §1).

/// Fallback tile budget when cache info is unavailable: half of the smallest common private L2 (256 KB).
pub const FALLBACK_TILE_BYTES: usize = 128 * 1024;

/// L2 bytes per thread; `None` if it can't be read.
pub fn l2_bytes_per_thread() -> Option<usize> {
    imp::l2_per_thread()
}

/// Tile budget: half of the per-thread L2 (the other half is left for row/column blocks and
/// heaps), or the fallback when it can't be read.
pub fn tile_budget_bytes() -> usize {
    l2_bytes_per_thread()
        .map(|b| (b / 2).clamp(64 * 1024, 4 << 20))
        .unwrap_or(FALLBACK_TILE_BYTES)
}

#[cfg(target_os = "macos")]
mod imp {
    use std::ffi::CString;
    use std::os::raw::{c_char, c_int, c_void};

    extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }

    fn read_u64(name: &str) -> Option<u64> {
        let cname = CString::new(name).ok()?;
        let mut value: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        // SAFETY: the buffer is one u64 and len states its size truthfully; the kernel writes at
        // most len bytes.
        let rc = unsafe {
            sysctlbyname(
                cname.as_ptr(),
                &mut value as *mut u64 as *mut c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        match (rc, len) {
            // Some entries are 32-bit; on little-endian machines the low 4 bytes hold the value
            (0, 4) => Some(value & 0xffff_ffff),
            (0, 8) => Some(value),
            _ => None,
        }
    }

    pub fn l2_per_thread() -> Option<usize> {
        sysctl().or_else(super::cpuid::l2_per_thread)
    }

    fn sysctl() -> Option<usize> {
        // L2 of the performance cores (perflevel0) and how many cores share it; older systems
        // without perflevel fall back to hw.l2cachesize
        let (size, sharing) = match (
            read_u64("hw.perflevel0.l2cachesize"),
            read_u64("hw.perflevel0.cpusperl2"),
        ) {
            (Some(s), Some(c)) if s > 0 && c > 0 => (s, c),
            _ => (read_u64("hw.l2cachesize").filter(|&s| s > 0)?, 1),
        };
        Some((size / sharing) as usize)
    }
}

#[cfg(target_os = "linux")]
mod imp {
    use std::fs;

    /// sysfs first; some VMs (e.g. GitHub's Azure runners) expose no cache directory, so fall
    /// back to CPUID on x86 (docs/03 §8.5).
    pub fn l2_per_thread() -> Option<usize> {
        sysfs().or_else(super::cpuid::l2_per_thread)
    }

    fn sysfs() -> Option<usize> {
        for entry in fs::read_dir("/sys/devices/system/cpu/cpu0/cache")
            .ok()?
            .flatten()
        {
            let dir = entry.path();
            let read = |name: &str| fs::read_to_string(dir.join(name)).ok();
            let (Some(level), Some(kind), Some(size)) = (read("level"), read("type"), read("size"))
            else {
                continue;
            };
            if level.trim() != "2" || kind.trim() == "Instruction" {
                continue;
            }
            let Some(size) = super::parse_size(size.trim()) else {
                continue;
            };
            let sharing = read("shared_cpu_list")
                .map(|s| super::count_cpus(s.trim()))
                .unwrap_or(1)
                .max(1);
            return Some(size / sharing);
        }
        None
    }
}

// Windows and everything else: CPUID on x86, the fallback elsewhere (docs/06 §2.3).
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod imp {
    pub fn l2_per_thread() -> Option<usize> {
        super::cpuid::l2_per_thread()
    }
}

/// One cache level as described by a CPUID "deterministic cache parameters" leaf
/// (Intel leaf 4 and AMD leaf 0x8000001D share the layout).
#[cfg(any(target_arch = "x86", target_arch = "x86_64", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CacheLeaf {
    /// 0 = no more caches, 1 = data, 2 = instruction, 3 = unified.
    kind: u32,
    level: u32,
    size: usize,
    /// Logical processors sharing this cache.
    sharing: usize,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64", test))]
fn decode_cache_leaf(eax: u32, ebx: u32, ecx: u32) -> CacheLeaf {
    let ways = ((ebx >> 22) & 0x3ff) as usize + 1;
    let partitions = ((ebx >> 12) & 0x3ff) as usize + 1;
    let line = (ebx & 0xfff) as usize + 1;
    let sets = ecx as usize + 1;
    CacheLeaf {
        kind: eax & 0x1f,
        level: (eax >> 5) & 0x7,
        size: ways * partitions * line * sets,
        sharing: (((eax >> 14) & 0xfff) + 1) as usize,
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod cpuid {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{__cpuid, __cpuid_count, CpuidResult};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{__cpuid, __cpuid_count, CpuidResult};

    // Vendor strings as (ebx, edx, ecx)
    const AMD: [u32; 3] = [0x6874_7541, 0x6974_6e65, 0x444d_4163]; // "AuthenticAMD"
    const HYGON: [u32; 3] = [0x6f67_7948, 0x6e65_476e, 0x656e_6975]; // "HygonGenuine"

    #[allow(unused_unsafe)] // the intrinsics are safe fns on newer toolchains
    fn cpuid(leaf: u32, sub: u32) -> CpuidResult {
        // SAFETY: CPUID exists on every x86-64 CPU and on every x86 CPU Rust targets.
        unsafe {
            if sub == 0 {
                __cpuid(leaf)
            } else {
                __cpuid_count(leaf, sub)
            }
        }
    }

    /// L2 bytes per logical processor from CPUID; `None` if the CPU doesn't report it.
    pub fn l2_per_thread() -> Option<usize> {
        let id = cpuid(0, 0);
        let vendor = [id.ebx, id.edx, id.ecx];
        let leaf = if vendor == AMD || vendor == HYGON {
            let max_ext = cpuid(0x8000_0000, 0).eax;
            if max_ext < 0x8000_001D {
                // Older AMD: leaf 0x80000006 gives the per-core L2 size in KB, no sharing info
                return (max_ext >= 0x8000_0006)
                    .then(|| ((cpuid(0x8000_0006, 0).ecx >> 16) as usize) << 10)
                    .filter(|&b| b > 0);
            }
            0x8000_001D
        } else if id.eax >= 4 {
            4
        } else {
            return None;
        };
        for sub in 0..16 {
            let r = cpuid(leaf, sub);
            let c = super::decode_cache_leaf(r.eax, r.ebx, r.ecx);
            if c.kind == 0 {
                break;
            }
            if c.level == 2 && c.kind != 2 {
                return Some(c.size / c.sharing.max(1));
            }
        }
        None
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
mod cpuid {
    pub fn l2_per_thread() -> Option<usize> {
        None
    }
}

/// Parse a sysfs cache size such as "1024K" or "2M".
#[cfg(any(target_os = "linux", test))]
fn parse_size(s: &str) -> Option<usize> {
    let (num, mul) = match s.as_bytes().last()? {
        b'K' => (&s[..s.len() - 1], 1 << 10),
        b'M' => (&s[..s.len() - 1], 1 << 20),
        b'G' => (&s[..s.len() - 1], 1 << 30),
        _ => (s, 1),
    };
    num.parse::<usize>().ok().map(|n| n * mul)
}

/// Count CPUs in a sysfs CPU list, e.g. "0-3,8-11" → 8.
#[cfg(any(target_os = "linux", test))]
fn count_cpus(list: &str) -> usize {
    list.split(',')
        .filter(|s| !s.is_empty())
        .map(|r| match r.split_once('-') {
            Some((a, b)) => {
                let (a, b) = (
                    a.parse::<usize>().unwrap_or(0),
                    b.parse::<usize>().unwrap_or(0),
                );
                b.saturating_sub(a) + 1
            }
            None => 1,
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sysfs_sizes() {
        assert_eq!(parse_size("1024K"), Some(1 << 20));
        assert_eq!(parse_size("2M"), Some(2 << 20));
        assert_eq!(parse_size("512"), Some(512));
        assert_eq!(parse_size("x"), None);
    }

    #[test]
    fn counts_cpu_lists() {
        assert_eq!(count_cpus("0"), 1);
        assert_eq!(count_cpus("0-3,8-11"), 8);
        assert_eq!(count_cpus("0,2"), 2);
    }

    #[test]
    fn decodes_cpuid_cache_leaf() {
        // A unified L2: 16 ways, 1 partition, 64-byte lines, 1024 sets = 1 MiB, shared by 2
        let eax = 3 | (2 << 5) | (1 << 14);
        let ebx = (15 << 22) | 63;
        let c = decode_cache_leaf(eax, ebx, 1023);
        assert_eq!(
            c,
            CacheLeaf {
                kind: 3,
                level: 2,
                size: 1 << 20,
                sharing: 2
            }
        );
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn cpuid_l2_is_plausible() {
        // Runs the real CPUID path; any x86 L2 per thread falls in this range
        if let Some(b) = cpuid::l2_per_thread() {
            assert!((16 << 10..=256 << 20).contains(&b), "{b}");
        }
    }

    #[test]
    fn budget_is_within_bounds() {
        let b = tile_budget_bytes();
        assert!((64 * 1024..=4 << 20).contains(&b), "{b}");
    }
}
