//! The prompt suites: what is timed and what is scored.
//!
//! `quick` is a smoke test, `standard` the original suite (every current model passes it, so it
//! shows speed more than ability) and `hard` is built so models separate: multi-step reasoning,
//! instruction following with exact output shapes, longer code tasks run against tests, and recall
//! over a long context.

use clap::ValueEnum;
use serde_json::Value;

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Suite {
    Quick,
    Standard,
    Hard,
}

impl Suite {
    pub fn name(self) -> &'static str {
        match self {
            Suite::Quick => "quick",
            Suite::Standard => "standard",
            Suite::Hard => "hard",
        }
    }

    /// Tokens a quality answer may use, thinking included, when `--quality-tokens` is not given.
    pub fn quality_tokens(self) -> u64 {
        match self {
            Suite::Quick => 1024,
            Suite::Standard => 2048,
            Suite::Hard => 3072,
        }
    }
}

/// A performance prompt: a name, the text and the most tokens to generate.
pub struct PerfPrompt {
    pub name: &'static str,
    pub text: String,
    pub max_tokens: u64,
}

/// Roughly how many tokens one generated fact takes.
const TOKENS_PER_FACT: u32 = 22;

fn facts(n: u32) -> String {
    (1..=n).map(|i| format!("Fact {i}: the item numbered {i} weighs {} grams and is kept in box {}.\n", i * 7 + 3, i % 9 + 1)).collect()
}

/// `context` is the long prompt's size in tokens (about); `decode_max` the story's token cap.
pub fn perf_suite(suite: Suite, context: Option<u32>, decode_max: Option<u64>) -> Vec<PerfPrompt> {
    let n = context.map_or(60, |c| (c / TOKENS_PER_FACT).max(2));
    let mut v = vec![
        PerfPrompt { name: "short", text: "Explain in two sentences why the sky is blue.".into(), max_tokens: 128 },
        PerfPrompt {
            name: "decode",
            text: "Write a story of about 200 words about a lighthouse keeper and a stray cat.".into(),
            max_tokens: decode_max.unwrap_or(if suite == Suite::Quick { 128 } else { 256 }),
        },
    ];
    if suite != Suite::Quick || context.is_some() {
        let asked = (n / 3).max(1);
        v.push(PerfPrompt {
            name: "long_prompt",
            text: format!("Read these facts.\n{}\nIn one short sentence, say how heavy item {asked} is.", facts(n)),
            max_tokens: 48,
        });
    }
    v
}

/// A quality check: a prompt and how its answer is judged.
pub struct Check {
    pub name: &'static str,
    pub category: &'static str,
    pub prompt: String,
    pub kind: Kind,
}

pub enum Kind {
    /// The text, normalized, equals one of these.
    Exact(&'static [&'static str]),
    /// The words of the text (letters only, lower case) are exactly these.
    Words(&'static str),
    /// The last whole number in the text is this one.
    Number(i64),
    /// A JSON object with these fields (name "Ada", age 36, tags ["x", "y"]).
    Json,
    /// A JSON value equal to this one.
    JsonEq(&'static str),
    /// The non-empty lines are exactly these.
    Lines(&'static [&'static str]),
    /// A predicate over the answer.
    Func(fn(&str) -> bool),
    /// Python code that, followed by these lines, prints OK.
    Code(&'static str),
}

fn c(name: &'static str, category: &'static str, prompt: &str, kind: Kind) -> Check {
    Check { name, category, prompt: prompt.to_string(), kind }
}

fn standard() -> Vec<Check> {
    vec![
        c("arithmetic_1", "arithmetic", "Compute 17 * 23 + 5. Reply with only the final number.", Kind::Number(396)),
        c("arithmetic_2", "arithmetic", "What is 12345 + 6789? Reply with only the number.", Kind::Number(19134)),
        c(
            "json_format",
            "instruction",
            "Reply with only a JSON object, no prose and no code fences, with exactly these keys: \"name\" set to the string \"Ada\", \"age\" set to the number 36, and \"tags\" set to an array of the two strings \"x\" and \"y\".",
            Kind::Json,
        ),
        c(
            "code_fib",
            "code",
            "Write a Python function fib(n) that returns the n-th Fibonacci number, with fib(0) = 0 and fib(1) = 1. Reply with only the code.",
            Kind::Code("assert [fib(i) for i in range(10)] == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]\nassert fib(30) == 832040\nprint('OK')\n"),
        ),
        c(
            "code_palindrome",
            "code",
            "Write a Python function is_palindrome(s) that returns True when s reads the same backwards, ignoring case and anything that is not a letter or digit. Reply with only the code.",
            Kind::Code(
                "assert is_palindrome('A man, a plan, a canal: Panama') is True\nassert is_palindrome('hello') is False\nassert is_palindrome('') is True\nprint('OK')\n",
            ),
        ),
        c("fact_capital", "recall", "What is the capital of Australia? Reply with one word.", Kind::Exact(&["canberra"])),
        c("fact_symbol", "recall", "What is the chemical symbol for gold? Reply with only the symbol.", Kind::Exact(&["au"])),
        c("fact_leap", "recall", "How many days are in a leap year? Reply with only the number.", Kind::Number(366)),
    ]
}

fn four_s_words(s: &str) -> bool {
    let t = super::score::strip_think(s);
    let t = t.trim();
    let w: Vec<&str> = t.split(' ').collect();
    !t.contains('\n') && w.len() == 4 && w.iter().all(|x| x.len() > 1 && x.starts_with('s') && x.chars().all(|c| c.is_ascii_lowercase()))
}

fn hard() -> Vec<Check> {
    let retrieval = format!("Read these facts.\n{}\nWhich box holds the item that weighs 346 grams? Reply with only the box number.", facts(80));
    vec![
        c(
            "reason_change",
            "reasoning",
            "A shop sells pens at 3 for $2.40 and notebooks at $4.15 each. Maya buys 12 pens and 5 notebooks and pays with a $50 bill. How many cents of change does she get? Reply with only the whole number of cents.",
            Kind::Number(1965),
        ),
        c(
            "reason_order",
            "reasoning",
            "Tom is taller than Sam. Sam is taller than Ria. Ria is taller than Joe. Lee is shorter than Tom but taller than Sam. List all five from tallest to shortest, names only, separated by commas.",
            Kind::Words("tom lee sam ria joe"),
        ),
        c(
            "reason_clock",
            "reasoning",
            "A meeting starts at 14:35 and lasts 1 hour 50 minutes, then there is a 25-minute break. At what time does the break end? Reply with only the time in HH:MM 24-hour format.",
            Kind::Exact(&["16:50"]),
        ),
        c(
            "reason_letters",
            "reasoning",
            "How many times does the letter 'e' appear in the word 'excellence'? Reply with only the number.",
            Kind::Number(4),
        ),
        c("reason_primes", "reasoning", "How many prime numbers are there between 50 and 100? Reply with only the number.", Kind::Number(10)),
        c("reason_modpow", "reasoning", "What is the last digit of 7 to the power 222? Reply with only the digit.", Kind::Number(9)),
        c(
            "follow_lines",
            "instruction",
            "Write exactly three lines and nothing else. Line 1: the word alpha in upper case. Line 2: the number of letters in the word banana. Line 3: the word gamma spelled backwards, in lower case.",
            Kind::Lines(&["ALPHA", "6", "ammag"]),
        ),
        c(
            "follow_json",
            "instruction",
            "Reply with only JSON, no prose and no code fence: an object with the key \"items\" holding an array of the squares of 2, 3, 4 and 5 in ascending order, and the key \"total\" holding their sum.",
            Kind::JsonEq("{\"items\":[4,9,16,25],\"total\":54}"),
        ),
        c(
            "follow_words",
            "instruction",
            "Reply with exactly four words, all lower case, each starting with the letter s, separated by single spaces, with no punctuation and nothing else.",
            Kind::Func(four_s_words),
        ),
        c(
            "code_lru",
            "code",
            "Write a Python class LRUCache with __init__(self, capacity), get(self, key) that returns -1 when the key is missing, and put(self, key, value) that evicts the least recently used item when the capacity is exceeded. Both get and put count as a use. Reply with only the code.",
            Kind::Code(
                "c = LRUCache(2)\nc.put(1, 1)\nc.put(2, 2)\nassert c.get(1) == 1\nc.put(3, 3)\nassert c.get(2) == -1\nc.put(4, 4)\nassert c.get(1) == -1\nassert c.get(3) == 3 and c.get(4) == 4\nc.put(3, 30)\nassert c.get(3) == 30\nprint('OK')\n",
            ),
        ),
        c(
            "code_roman",
            "code",
            "Write a Python function roman_to_int(s) that converts a Roman numeral from 1 to 3999 to an integer, handling subtractive forms like IV and CM. Reply with only the code.",
            Kind::Code(
                "assert roman_to_int('IV') == 4\nassert roman_to_int('LVIII') == 58\nassert roman_to_int('MCMXCIV') == 1994\nassert roman_to_int('MMMCMXCIX') == 3999\nprint('OK')\n",
            ),
        ),
        c(
            "code_intervals",
            "code",
            "Write a Python function merge_intervals(iv) that takes a list of [start, end] lists in any order, merges the ones that overlap or touch (so [1, 2] and [2, 3] merge), and returns the merged list of lists sorted by start. Reply with only the code.",
            Kind::Code(
                "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]]) == [[1,5]]\nassert merge_intervals([]) == []\nassert merge_intervals([[5,6],[1,2]]) == [[1,2],[5,6]]\nprint('OK')\n",
            ),
        ),
        c(
            "code_brackets",
            "code",
            "Write a Python function is_balanced(s) that returns True when every (, [ and { in s is closed by the matching bracket in the right order, ignoring other characters, and False otherwise. Reply with only the code.",
            Kind::Code(
                "assert is_balanced('([]{})') is True\nassert is_balanced('(]') is False\nassert is_balanced('((') is False\nassert is_balanced('a(b[c]d)e') is True\nassert is_balanced('') is True\nassert is_balanced(']') is False\nprint('OK')\n",
            ),
        ),
        c("recall_tungsten", "recall", "What is the atomic number of tungsten? Reply with only the number.", Kind::Number(74)),
        Check { name: "recall_long_context", category: "recall", prompt: retrieval, kind: Kind::Number(5) },
    ]
}

pub fn quality_suite(suite: Suite) -> Vec<Check> {
    match suite {
        Suite::Standard => standard(),
        Suite::Hard => hard(),
        Suite::Quick => standard().into_iter().filter(|c| ["arithmetic_1", "json_format", "code_fib", "fact_capital"].contains(&c.name)).collect(),
    }
}

/// The expected value of a [`Kind::JsonEq`].
pub fn expected_json(s: &str) -> Value {
    serde_json::from_str(s).unwrap_or(Value::Null)
}
