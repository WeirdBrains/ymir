use std::sync::Mutex;

use super::*;
use crate::bench::report::Namer;
use crate::bench::score::{last_number, normalize, percentile, run_python, words};
use crate::bench::suites::Suite;
use ymir_core::client::TimedChat;

#[test]
fn percentiles_use_nearest_rank() {
    let v = [5.0, 1.0, 3.0, 2.0, 4.0];
    assert_eq!(median(&v), Some(3.0));
    assert_eq!(percentile(&v, 90.0), Some(5.0));
    assert_eq!(percentile(&[7.0], 90.0), Some(7.0));
    assert_eq!(median(&[]), None);
}

#[test]
fn answers_are_judged_strictly_but_ignore_reasoning_and_decoration() {
    assert_eq!(normalize("<think>hmm, Sydney?</think>\n**Canberra.**"), "canberra");
    assert_ne!(normalize("The capital is Canberra"), "canberra");
    assert_eq!(last_number("17*23 = 391, plus 5 is 396."), Some(396));
    assert_eq!(last_number("19,134"), Some(19134));
    assert_eq!(words("<think>x</think>Tom, Lee; SAM and Ria"), "tom lee sam and ria");
    assert!(score::json_ok("```json\n{\"name\":\"Ada\",\"age\":36,\"tags\":[\"x\",\"y\"]}\n```"));
    assert!(!score::json_ok("Sure! {\"name\":\"Ada\"}"));
}

fn hard(name: &str) -> suites::Check {
    suites::quality_suite(Suite::Hard, Some(600)).into_iter().find(|c| c.name == name).unwrap()
}

#[test]
fn hard_checks_judge_shapes_exactly() {
    let j = |n: &str, a: &str| judge(&hard(n), a, true);
    assert_eq!(j("logic_seats", "B"), Some(true));
    assert_eq!(j("logic_seats", "C."), Some(false));
    assert_eq!(j("math_change", "1965"), Some(true));
    assert_eq!(j("follow_lines", "ALPHA\n6\nammag\n"), Some(true));
    assert_eq!(j("follow_lines", "ALPHA\n6\ngamma"), Some(false));
    let ok = "Waves roll over the sand and whisper gently to every sleeping shell.";
    assert_eq!(j("follow_ocean", ok), Some(true));
    assert_eq!(j("follow_ocean", "Waves roll over the sand and whisper gently to every sleeping shell"), Some(false));
    assert_eq!(j("follow_ocean", "Waves roll over the blue sand and whisper to every sleeping shell."), Some(false));
    assert_eq!(j("follow_coffee", "- Dark roast wakes sleepy minds\n- BREW IT HOT EVERY MORNING\n- Sugar softens bitter edges slowly"), Some(true));
    assert_eq!(j("follow_coffee", "- Dark roast wakes sleepy minds\n- brew it hot every morning\n- Sugar softens bitter edges slowly"), Some(false));
    assert_eq!(j("follow_bwords", "babble,baboon,bagels,bakery,ballad"), Some(true));
    assert_eq!(j("follow_bwords", "bakery,babble,baboon,bagels,ballad"), Some(false));
    assert_eq!(j("follow_acrostic", "Calm tides arrive\nOcean breathes slowly\nDrifting salt air\nEvening settles in"), Some(true));
    assert_eq!(j("follow_acrostic", "Calm tides arrive\nWaves breathe slowly\nDrifting salt air\nEvening settles in"), Some(false));
    let two = "Bread rises because yeast eats sugar and releases small bubbles of gas inside the dough. Those bubbles stretch the gluten and make it light!";
    assert_eq!(j("follow_two", two), Some(true));
    assert_eq!(j("follow_two", &two.replace("light!", "light.")), Some(false));
    let call = |n: &str, a: &str| judge(&hard(n), a, true);
    assert_eq!(call("tool_weather", "```json\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Oslo\",\"units\":\"f\"}}\n```"), Some(true));
    assert_eq!(call("tool_weather", "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Oslo\",\"units\":\"F\"}}"), Some(false));
    assert_eq!(
        call("tool_table", "{\"name\":\"book_table\",\"arguments\":{\"restaurant\":\"Luigi's\",\"party_size\":4,\"time\":\"19:00\"}}"),
        Some(true)
    );
    assert_eq!(
        call("tool_table", "{\"name\":\"book_table\",\"arguments\":{\"restaurant\":\"Luigi's\",\"party_size\":\"4\",\"time\":\"19:00\"}}"),
        Some(false)
    );
    assert_eq!(call("tool_alarm", "{\"name\":\"set_alarm\",\"arguments\":{\"hour\":18,\"minute\":45,\"label\":\"gym\"}}"), Some(true));
    assert_eq!(call("tool_none", "{\"name\": null, \"arguments\": {}}"), Some(true));
    assert_eq!(call("tool_none", "Paris"), Some(false));
    assert_eq!(
        call("schema_movie", "{\"title\":\"Arrival\",\"year\":2016,\"genres\":[\"science fiction\",\"drama\",\"mystery\"],\"rating\":null}"),
        Some(true)
    );
}

#[test]
fn needles_scale_with_context_and_sit_at_three_depths() {
    let small = suites::quality_suite(Suite::Hard, Some(800));
    let big = suites::quality_suite(Suite::Hard, Some(8000));
    let find = |v: &[suites::Check], n: &str| v.iter().find(|c| c.name == n).unwrap().prompt.len();
    assert!(find(&big, "recall_middle") > 5 * find(&small, "recall_middle"));
    for n in ["recall_start", "recall_middle", "recall_end"] {
        let c = hard(n);
        assert_eq!(c.category, "retrieval");
        let secret = if n == "recall_start" {
            "violet-anchor-4821"
        } else if n == "recall_middle" {
            "crimson-willow-7302"
        } else {
            "indigo-compass-1957"
        };
        assert_eq!(judge(&c, &format!("The passcode is {secret}."), true), Some(true));
        assert_eq!(judge(&c, "amber-falcon-1234", true), Some(false));
        assert_eq!(c.prompt.matches(secret).count(), 1, "the needle appears once, in the list");
    }
}

#[test]
fn the_hard_suite_expectations_hold_up() {
    if !score::python_available() {
        return;
    }
    // Reference solutions pass their own tests, so a failing model is the model's fault.
    let solutions = [
        (
            "code_intervals",
            "def merge_intervals(iv):\n    out = []\n    for a, b in sorted(iv):\n        if out and a <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], b)\n        else:\n            out.append([a, b])\n    return out",
        ),
        (
            "code_brackets",
            "def is_balanced(s):\n    p = {')':'(',']':'[','}':'{'}\n    st = []\n    for c in s:\n        if c in '([{':\n            st.append(c)\n        elif c in p:\n            if not st or st.pop() != p[c]:\n                return False\n    return not st",
        ),
        (
            "code_lru",
            "from collections import OrderedDict\nclass LRUCache:\n    def __init__(self, capacity):\n        self.c = capacity\n        self.d = OrderedDict()\n    def get(self, k):\n        if k not in self.d:\n            return -1\n        self.d.move_to_end(k)\n        return self.d[k]\n    def put(self, k, v):\n        self.d[k] = v\n        self.d.move_to_end(k)\n        if len(self.d) > self.c:\n            self.d.popitem(last=False)",
        ),
        (
            "code_substring",
            "def longest_unique(s):\n    seen = {}\n    best = start = 0\n    for i, ch in enumerate(s):\n        if ch in seen and seen[ch] >= start:\n            start = seen[ch] + 1\n        seen[ch] = i\n        best = max(best, i - start + 1)\n    return best",
        ),
        (
            "code_coins",
            "def min_coins(coins, amount):\n    inf = amount + 1\n    dp = [0] + [inf] * amount\n    for a in range(1, amount + 1):\n        for c in coins:\n            if c <= a and dp[a - c] + 1 < dp[a]:\n                dp[a] = dp[a - c] + 1\n    return dp[amount] if dp[amount] != inf else -1",
        ),
        (
            "code_islands",
            "def count_islands(grid):\n    seen = set()\n    n = 0\n    for r in range(len(grid)):\n        for c in range(len(grid[r])):\n            if grid[r][c] == '1' and (r, c) not in seen:\n                n += 1\n                st = [(r, c)]\n                seen.add((r, c))\n                while st:\n                    y, x = st.pop()\n                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):\n                        a, b = y + dy, x + dx\n                        if 0 <= a < len(grid) and 0 <= b < len(grid[a]) and grid[a][b] == '1' and (a, b) not in seen:\n                            seen.add((a, b))\n                            st.append((a, b))\n    return n",
        ),
        (
            "code_decode",
            "def num_decodings(s):\n    if not s or s[0] == '0':\n        return 0\n    a, b = 1, 1\n    for i in range(1, len(s)):\n        c = 0\n        if s[i] != '0':\n            c += b\n        if 10 <= int(s[i-1:i+1]) <= 26:\n            c += a\n        a, b = b, c\n    return b",
        ),
        (
            "code_rain",
            "def trap(h):\n    if not h:\n        return 0\n    l, r = 0, len(h) - 1\n    lm = rm = t = 0\n    while l < r:\n        if h[l] < h[r]:\n            lm = max(lm, h[l])\n            t += lm - h[l]\n            l += 1\n        else:\n            rm = max(rm, h[r])\n            t += rm - h[r]\n            r -= 1\n    return t",
        ),
    ];
    for (name, code) in solutions {
        let check = hard(name);
        assert_eq!(judge(&check, &format!("```python\n{code}\n```"), true), Some(true), "{name}");
        // And a stub fails, so the tests do test something.
        assert_eq!(judge(&check, "```python\ndef nothing(*a):\n    return None\n```", true), Some(false), "{name} stub");
    }
    // The arithmetic behind the number answers.
    assert_eq!(pow_last_digit(7, 222), 9);
    assert_eq!(((12 / 3) * 240 + 5 * 415), 3035);
    assert_eq!(5000 - 3035, 1965);
    assert_eq!((480 - 90) / (12 - 6), 65);
    assert_eq!((1..=200).filter(|n| n % 3 == 0 || n % 5 == 0).sum::<i32>(), 9368);
    assert_eq!((1..).find(|n| n % 5 == 2 && n % 7 == 3 && n % 9 == 4), Some(157));
    assert_eq!(3600 / 180, 20);
    assert_eq!(150 * 60 / 30, 300);
    assert_eq!(80 * 125 / 100 * 80 / 100 - 6, 74);
}

fn pow_last_digit(b: u64, e: u32) -> u64 {
    (0..e).fold(1, |a, _| a * b % 10)
}

#[test]
fn model_code_is_run_with_a_time_limit() {
    if !score::python_available() {
        return;
    }
    assert!(run_python("def f():\n    return 2", "assert f() == 2\nprint('OK')").unwrap());
    assert!(!run_python("def f():\n    return 3", "assert f() == 2\nprint('OK')").unwrap());
    assert!(!run_python("while True:\n    pass", "print('OK')").unwrap());
}

#[test]
fn shared_output_hides_peer_ids_and_addresses() {
    let m = "dial /ip4/100.64.0.9/tcp/4848/p2p/12D3KooWabc failed: refused by 10.0.0.1:4848 (peer 12D3KooWxyz)";
    let s = report::scrub(m, false);
    assert!(!s.contains("100.64") && !s.contains("12D3Koo") && !s.contains("10.0.0.1"), "{s}");
    assert!(s.contains("failed: refused"));
    assert_eq!(report::scrub(m, true), m);
}

#[test]
fn stamps_are_utc() {
    assert_eq!(stamp(0), "19700101T000000Z");
    assert_eq!(stamp(1_795_267_771), "20261121T132931Z");
}

#[test]
fn samples_compute_speeds_from_the_stream() {
    let r = TimedChat {
        text: "x".into(),
        ttft: Some(Duration::from_millis(500)),
        total: Duration::from_millis(2500),
        prompt_tokens: Some(1100),
        cached_tokens: Some(100),
        completion_tokens: Some(41),
        served_by: Some("m".into()),
        ..Default::default()
    };
    let s = sample_of("p", &r, None);
    assert_eq!(s.decode_tps, Some(20.0));
    assert_eq!(s.pp_tps, Some(2000.0));
    assert_eq!(s.node, "m");
    assert_eq!(sample_of("p", &r, Some("pinned")).node, "pinned");
}

#[test]
fn nodes_are_picked_by_name_word_or_peer_id() {
    let nodes = vec![
        ("12D3KooWAAAAAAAAAA1".to_string(), "m1".to_string()),
        ("12D3KooWBBBBBBBBBB2".to_string(), "m2".to_string()),
        ("12D3KooWCCCCCCCCCC3".to_string(), "star base".to_string()),
    ];
    assert_eq!(resolve_node("M2", &nodes).unwrap(), vec!["12D3KooWBBBBBBBBBB2".to_string()]);
    assert_eq!(resolve_node("base", &nodes).unwrap(), vec!["12D3KooWCCCCCCCCCC3".to_string()]);
    assert_eq!(resolve_node("12D3KooWAAAA", &nodes).unwrap(), vec!["12D3KooWAAAAAAAAAA1".to_string()]);
    assert!(resolve_node("nope", &nodes).is_err());
    assert!(resolve_node("12D3Koo", &nodes).is_err(), "a prefix shared by every node is ambiguous");
    assert!(model_matches("qwen*", "qwen3-14b") && !model_matches("qwen", "qwen3-14b"));
}

#[test]
fn anonymous_labels_follow_the_roster_not_the_run() {
    let roster = vec![("peer-z".to_string(), "zed".to_string()), ("peer-a".to_string(), "alpha".to_string())];
    let mut n = Namer::new(false, &roster);
    // peer-a sorts first, so it is node-a however the names come up.
    assert_eq!(n.name("zed"), "node-b");
    assert_eq!(n.name("alpha"), "node-a");
    assert_eq!(n.name("peer-a"), "node-a");
    assert_eq!(n.name("Ymir on zed"), "node-b");
    assert_eq!(n.name("someone else"), "node-c");
    assert_eq!(Namer::new(true, &roster).name("zed"), "zed");
}

/// A pinned backend with two nodes serving overlapping models; the second node is slower.
struct Mock {
    asked: Mutex<Vec<(String, String)>>,
}

fn target(node: &str, model: &str, state: &str, purpose: &str) -> Target {
    Target {
        node: node.into(),
        peer: Some(format!("peer-{node}")),
        role: "model".into(),
        model: model.into(),
        purpose: purpose.into(),
        state: state.into(),
        replicas: 1,
        context: Some(8192),
    }
}

#[async_trait]
impl Backend for Mock {
    fn via(&self) -> &'static str {
        "weirdnet"
    }

    fn pinned(&self) -> bool {
        true
    }

    async fn listing(&self) -> Result<Listing> {
        Ok(Listing {
            nodes: vec![("peer-real-a".into(), "real-a".into()), ("peer-real-b".into(), "real-b".into()), ("peer-real-c".into(), "real-c".into())],
            targets: vec![
                target("real-a", "chatty", "warm", "chat"),
                target("real-b", "chatty", "warm", "chat"),
                target("real-b", "cold", "loadable", "chat"),
                target("real-a", "vec", "warm", "embedding"),
            ],
            unreachable: vec![("real-c".into(), "no route".into())],
        })
    }

    async fn chat(&self, t: &Target, prompt: &str, o: &AskOptions) -> Result<TimedChat> {
        self.asked.lock().unwrap().push((t.node.clone(), t.model.clone()));
        let text = if prompt.contains("17 * 23") {
            "396"
        } else if prompt.contains("capital of Australia") {
            "Canberra"
        } else if prompt.contains("JSON") {
            "{\"name\":\"Ada\",\"age\":36,\"tags\":[\"x\",\"y\"]}"
        } else {
            "a few words of mock answer"
        };
        let slow = if t.node == "real-b" { 2 } else { 1 };
        Ok(TimedChat {
            text: text.into(),
            ttft: Some(Duration::from_millis(10 * slow)),
            total: Duration::from_millis(10 * slow + 100 * slow),
            prompt_tokens: Some(50),
            cached_tokens: Some(0),
            completion_tokens: Some(u64::from(o.max_tokens.min(11))),
            served_by: Some(if t.node == "real-b" && t.model == "chatty" { "real-c".into() } else { t.node.clone() }),
            ..Default::default()
        })
    }

    async fn embed(&self, _: &Target, _: &[String]) -> Result<(usize, Duration)> {
        Ok((4, Duration::from_millis(5)))
    }
}

fn args(dir: &std::path::Path) -> BenchArgs {
    BenchArgs {
        repeats: 2,
        warmup: 1,
        timeout: 20,
        busy_retries: 0,
        busy_wait: 0,
        out_dir: dir.to_path_buf(),
        suite: Suite::Quick,
        ..Default::default()
    }
}

#[tokio::test]
async fn runs_a_node_by_model_matrix_one_request_at_a_time() {
    let dir = tempfile::tempdir().unwrap();
    let mock = Mock { asked: Mutex::new(Vec::new()) };
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let (jp, mp, cp) = run(&mock, &args(dir.path()), &mut out, &mut err).await.unwrap();
    let report: Value = serde_json::from_str(&std::fs::read_to_string(jp).unwrap()).unwrap();
    let md = std::fs::read_to_string(mp).unwrap();
    let csv = std::fs::read_to_string(cp).unwrap();
    for text in [report.to_string(), md.clone(), csv.clone()] {
        assert!(!text.contains("real-"), "node names are anonymised by default");
    }
    assert!(md.contains("## Side by side") && md.contains("node-a") && md.contains("node-b"));
    let rows = report["rows"].as_array().unwrap();
    let find = |m: &str, n: &str| rows.iter().find(|r| r["model"] == m && r["node"] == n).unwrap();
    // The same model on two nodes gives two rows, and the slower node measures slower.
    let (a, b) = (find("chatty", "node-a"), find("chatty", "node-b"));
    assert!(a["decode_tps"]["median"].as_f64().unwrap() > b["decode_tps"]["median"].as_f64().unwrap());
    // node-b placed the model on node-c, and says so.
    assert_eq!(b["served_by"], json!(["node-c"]));
    assert!(md.contains("Forwarded"));
    // The quick suite has one code question, scored only where python3 and the sandbox are.
    assert_eq!(a["quality"]["ran"], if score::python_available() { 4 } else { 3 });
    assert!(find("cold", "node-b")["skipped"].as_str().unwrap().contains("not loaded"));
    assert_eq!(find("vec", "node-a")["embedding"]["dims"], 4);
    assert!(md.contains("node-c: skipped, node unreachable: no route"));
    assert!(csv.lines().next().unwrap().starts_with("started,suite,via,model,node"));
    assert_eq!(csv.lines().count(), 1 + 4);
    // Strictly one request after another, grouped by node, nothing asked of an unreachable node.
    let asked = mock.asked.lock().unwrap();
    assert!(asked.iter().all(|(n, _)| n != "real-c"));
    let nodes: Vec<&String> = asked.iter().map(|(n, _)| n).collect();
    let switches = nodes.windows(2).filter(|w| w[0] != w[1]).count();
    assert_eq!(switches, 1, "all of one node's requests before the next node's: {nodes:?}");
}

#[tokio::test]
async fn picks_nodes_and_models_and_refuses_cold_loading_unasked() {
    let dir = tempfile::tempdir().unwrap();
    let mock = Mock { asked: Mutex::new(Vec::new()) };
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let mut a = args(dir.path());
    a.node = vec!["real-b".into()];
    a.model = vec!["chatty".into(), "ghost".into()];
    let (jp, ..) = run(&mock, &a, &mut out, &mut err).await.unwrap();
    let report: Value = serde_json::from_str(&std::fs::read_to_string(jp).unwrap()).unwrap();
    assert_eq!(report["rows"].as_array().unwrap().len(), 1);
    assert!(report["skipped"].to_string().contains("model ghost: not served"));
    assert!(mock.asked.lock().unwrap().iter().all(|(n, m)| n == "real-b" && m == "chatty"));

    // --cold warns, and without --yes and without a terminal it will not load anything.
    let mut c = args(dir.path());
    c.node = vec!["real-b".into()];
    c.model = vec!["cold".into()];
    c.cold = true;
    let mut err = Vec::new();
    let e = run(&mock, &c, &mut Vec::new(), &mut err).await.unwrap_err();
    assert!(format!("{e:#}").contains("--yes"), "{e:#}");
    let warned = String::from_utf8(err).unwrap();
    assert!(warned.contains("WARNING") && warned.contains("evict") && warned.contains("cold on real-b"));
    c.yes = true;
    let (jp, ..) = run(&mock, &c, &mut Vec::new(), &mut Vec::new()).await.unwrap();
    let report: Value = serde_json::from_str(&std::fs::read_to_string(jp).unwrap()).unwrap();
    assert_eq!(report["rows"][0]["state_at_start"], "loadable");

    // Unknown nodes are an error, not an empty run.
    let mut bad = args(dir.path());
    bad.node = vec!["nowhere".into()];
    assert!(run(&mock, &bad, &mut Vec::new(), &mut Vec::new()).await.is_err());
}

#[tokio::test]
async fn compares_two_results() {
    let dir = tempfile::tempdir().unwrap();
    let mock = Mock { asked: Mutex::new(Vec::new()) };
    let a = args(dir.path());
    let (ja, ..) = run(&mock, &a, &mut Vec::new(), &mut Vec::new()).await.unwrap();
    let mut b = args(dir.path());
    b.suite = Suite::Quick;
    b.node = vec!["real-a".into()];
    tokio::time::sleep(Duration::from_millis(1100)).await;
    let (jb, ..) = run(&mock, &b, &mut Vec::new(), &mut Vec::new()).await.unwrap();
    let mut out = Vec::new();
    compare(&CompareArgs { a: ja, b: jb, out: Some(dir.path().join("cmp.md")) }, &mut out).unwrap();
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains("| chatty | node-a |") && text.contains(" → "), "{text}");
    assert!(text.contains("Only in A: chatty on node-b"), "{text}");
    assert!(dir.path().join("cmp.md").exists());
}
