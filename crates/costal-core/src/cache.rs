//! 每个线程能独占的 L2 份额，用来定瓦片尺寸（docs/06 §2.3）。
//!
//! 只瞄准每核独占的那份 L2，不瞄准 L3：L3 被所有核和其它进程共享，能分到多少不可预测。
//! 读不到缓存信息就用保底值；这些数只影响速度，不影响输出（docs/06 §1）。

/// 读不到缓存信息时的保底瓦片预算：常见最小独占 L2（256 KB）的一半。
pub const FALLBACK_TILE_BYTES: usize = 128 * 1024;

/// 每个线程分到的 L2 字节数；读不到返回 `None`。
pub fn l2_bytes_per_thread() -> Option<usize> {
    imp::l2_per_thread()
}

/// 瓦片预算：每线程 L2 的一半（另一半留给行块、列块和堆），读不到时用保底值。
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
        // SAFETY: 缓冲区是一个 u64，len 如实给出其大小；内核按 len 截断写入。
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
            // 有的项是 32 位；小端机上低 4 字节就是值
            (0, 4) => Some(value & 0xffff_ffff),
            (0, 8) => Some(value),
            _ => None,
        }
    }

    pub fn l2_per_thread() -> Option<usize> {
        // 性能核（perflevel0）的 L2 与共享它的核数；老系统没有 perflevel 时退回 hw.l2cachesize
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

// Windows 等：先用保底值。GetLogicalProcessorInformationEx 的探测待补（docs/06 §2.3）。
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod imp {
    pub fn l2_per_thread() -> Option<usize> {
        None
    }
}

/// 解析 sysfs 的缓存大小，如 "1024K"、"2M"。
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

/// 数 sysfs 的 CPU 列表里有几个 CPU，如 "0-3,8-11" → 8。
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
