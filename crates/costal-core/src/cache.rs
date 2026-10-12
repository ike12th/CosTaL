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

    pub fn l2_per_thread() -> Option<usize> {
        for entry in fs::read_dir("/sys/devices/system/cpu/cpu0/cache").ok()? {
            let dir = entry.ok()?.path();
            let read = |name: &str| fs::read_to_string(dir.join(name)).ok();
            if read("level")?.trim() != "2" || read("type")?.trim() == "Instruction" {
                continue;
            }
            let size = super::parse_size(read("size")?.trim())?;
            let sharing = read("shared_cpu_list")
                .map(|s| super::count_cpus(s.trim()))
                .unwrap_or(1)
                .max(1);
            return Some(size / sharing);
        }
        None
    }
}

// Windows and others: use the fallback for now. Detection via GetLogicalProcessorInformationEx
// is still to do (docs/06 §2.3).
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod imp {
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
    fn budget_is_within_bounds() {
        let b = tile_budget_bytes();
        assert!((64 * 1024..=4 << 20).contains(&b), "{b}");
    }
}
