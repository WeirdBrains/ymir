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
    /// The text, lower-cased, contains this (case-insensitive).
    Has(String),
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

fn body(s: &str) -> String {
    super::score::strip_think(s)
}

fn word_list(s: &str) -> Vec<&str> {
    s.split_whitespace().collect()
}

fn lines_of(s: &str) -> Vec<String> {
    s.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string).collect()
}

fn has_word(s: &str, w: &str) -> bool {
    s.split(|c: char| !c.is_alphanumeric()).any(|x| x.eq_ignore_ascii_case(w))
}

/// One sentence of exactly twelve words that starts with `Waves`, ends with a full stop and
/// avoids the letter z and the words blue and water.
fn ocean_sentence(s: &str) -> bool {
    let t = body(s);
    let t = t.trim();
    let w = word_list(t);
    !t.contains('\n')
        && w.len() == 12
        && w[0] == "Waves"
        && t.ends_with('.')
        && t.matches('.').count() == 1
        && !t.to_lowercase().contains('z')
        && !has_word(t, "blue")
        && !has_word(t, "water")
}

/// Three bullet lines of five words, the second in capitals, none with the word "the".
fn coffee_bullets(s: &str) -> bool {
    let l = lines_of(&body(s));
    l.len() == 3
        && l.iter().all(|x| x.starts_with("- ") && word_list(&x[2..]).len() == 5 && !has_word(x, "the"))
        && l[1].chars().filter(|c| c.is_alphabetic()).all(char::is_uppercase)
        && l[1].chars().any(char::is_alphabetic)
}

/// Five distinct six-letter lower-case words starting with b, in alphabetical order, joined by
/// commas with nothing else.
fn b_words(s: &str) -> bool {
    let t = body(s);
    let t = t.trim();
    let w: Vec<&str> = t.split(',').collect();
    w.len() == 5
        && w.iter().all(|x| x.len() == 6 && x.starts_with('b') && x.chars().all(|c| c.is_ascii_lowercase()))
        && w.windows(2).all(|p| p[0] < p[1])
}

/// Four lines spelling CODE down their first letters, three to six words each, no commas.
fn acrostic(s: &str) -> bool {
    let l = lines_of(&body(s));
    l.len() == 4
        && l.iter().zip("CODE".chars()).all(|(x, c)| x.chars().next().is_some_and(|f| f.eq_ignore_ascii_case(&c)))
        && l.iter().all(|x| (3..=6).contains(&word_list(x).len()) && !x.contains(','))
}

/// Exactly two sentences, `because` in the first, an exclamation mark ending the second, 20 to 30
/// words in all and no letter x.
fn two_sentences(s: &str) -> bool {
    let t = body(s);
    let t = t.trim().to_string();
    let Some(split) = t.find(|c| ".!?".contains(c)) else { return false };
    let (first, second) = t.split_at(split + 1);
    let second = second.trim();
    let n = word_list(&t).len();
    has_word(first, "because")
        && second.ends_with('!')
        && second.matches(['.', '!', '?']).count() == 1
        && !second.is_empty()
        && (20..=30).contains(&n)
        && !t.to_lowercase().contains('x')
        && !t.contains('\n')
}

fn call_of(s: &str) -> Option<serde_json::Value> {
    serde_json::from_str(&super::score::extract_block(s)).ok()
}

fn str_is(v: &serde_json::Value, want: &str) -> bool {
    v.as_str().is_some_and(|x| x.trim().eq_ignore_ascii_case(want))
}

fn weather_call(s: &str) -> bool {
    call_of(s).is_some_and(|v| {
        v["name"] == "get_weather"
            && str_is(&v["arguments"]["city"], "Oslo")
            && v["arguments"]["units"] == "f"
            && v["arguments"].as_object().is_some_and(|o| o.len() == 2)
    })
}

fn table_call(s: &str) -> bool {
    call_of(s).is_some_and(|v| {
        let a = &v["arguments"];
        v["name"] == "book_table"
            && a["party_size"].as_i64() == Some(4)
            && a["time"] == "19:00"
            && a["restaurant"].as_str().is_some_and(|r| r.to_lowercase().contains("luigi"))
            && a.as_object().is_some_and(|o| o.len() == 3)
    })
}

fn alarm_call(s: &str) -> bool {
    call_of(s).is_some_and(|v| {
        let a = &v["arguments"];
        v["name"] == "set_alarm" && a["hour"].as_i64() == Some(18) && a["minute"].as_i64() == Some(45) && str_is(&a["label"], "gym")
    })
}

fn no_tool(s: &str) -> bool {
    call_of(s).is_some_and(|v| v["name"].is_null() && v["arguments"].as_object().is_some_and(|o| o.is_empty()))
}

const TOOLS: &str = "You can call these tools. Each takes a JSON object of arguments.\n\
- get_weather(city: string, units: \"c\" or \"f\")\n\
- book_table(restaurant: string, party_size: integer, time: string as 24-hour HH:MM)\n\
- set_alarm(hour: integer 0-23, minute: integer 0-59, label: string)\n\
Reply with only one JSON object {\"name\": <tool>, \"arguments\": {...}} and no prose and no code fence. \
If none of the tools applies, reply with only {\"name\": null, \"arguments\": {}}.\nRequest: ";

/// Needles hidden in a long list of vault passcodes; `context` is the prompt size in tokens, about.
fn needles(context: u32) -> Vec<Check> {
    const A: [&str; 8] = ["amber", "birch", "cobalt", "dune", "ember", "fjord", "garnet", "harbor"];
    const B: [&str; 8] = ["falcon", "lantern", "meadow", "otter", "quartz", "raven", "summit", "thistle"];
    let n = (context / 16).max(30);
    let code = |i: u32| {
        let h = i.wrapping_mul(2654435761).rotate_left(7);
        format!("{}-{}-{}", A[(h % 8) as usize], B[((h >> 4) % 8) as usize], 1000 + h % 9000)
    };
    let picks = [n / 6 + 1, n / 2 + 1, n * 5 / 6 + 1];
    let secrets = ["violet-anchor-4821", "crimson-willow-7302", "indigo-compass-1957"];
    let list: String = (1..=n)
        .map(|i| match picks.iter().position(|p| *p == i) {
            Some(k) => format!("Vault {i}: passcode {}.\n", secrets[k]),
            None => format!("Vault {i}: passcode {}.\n", code(i)),
        })
        .collect();
    let names = ["recall_start", "recall_middle", "recall_end"];
    (0..3)
        .map(|k| Check {
            name: names[k],
            category: "retrieval",
            prompt: format!("Below is a list of vault passcodes.\n{list}\nWhat is the passcode of vault {}? Reply with only the passcode.", picks[k]),
            kind: Kind::Has(secrets[k].to_string()),
        })
        .collect()
}

fn hard(context: Option<u32>) -> Vec<Check> {
    let mut v = vec![
        // multi-step arithmetic with exact answers
        c(
            "math_change",
            "math",
            "A shop sells pens at 3 for $2.40 and notebooks at $4.15 each. Maya buys 12 pens and 5 notebooks and pays with a $50 bill. How many cents of change does she get? Reply with only the whole number of cents.",
            Kind::Number(1965),
        ),
        c(
            "math_tank",
            "math",
            "A 480-litre tank holds 90 litres. A pipe adds 12 litres a minute and a drain removes 6 litres a minute, both open together. After how many minutes is the tank exactly full? Reply with only the number.",
            Kind::Number(65),
        ),
        c(
            "math_marbles",
            "math",
            "Sara has 3 times as many marbles as Ben. Sara gives Ben 14 marbles and now they have the same number. How many marbles do they have together? Reply with only the number.",
            Kind::Number(56),
        ),
        c(
            "math_percent",
            "math",
            "A jacket costs $80. The price goes up by 25%, then drops by 20%, then a $6 coupon is taken off. What is the final price in dollars? Reply with only the number.",
            Kind::Number(74),
        ),
        c(
            "math_multiples",
            "math",
            "What is the sum of all whole numbers from 1 to 200 that are divisible by 3 or by 5 (or both)? Reply with only the number.",
            Kind::Number(9368),
        ),
        c(
            "math_meet",
            "math",
            "Two cyclists start 150 km apart and ride toward each other at 18 km/h and 12 km/h. After how many minutes do they meet? Reply with only the number.",
            Kind::Number(300),
        ),
        c(
            "math_lights",
            "math",
            "Three lights blink every 12, 18 and 30 seconds and all blink together at time 0. How many times do all three blink together in the next hour, counting the blink at exactly 3600 seconds but not the one at time 0? Reply with only the number.",
            Kind::Number(20),
        ),
        c(
            "math_remainder",
            "math",
            "What is the smallest positive integer that leaves remainder 2 when divided by 5, remainder 3 when divided by 7 and remainder 4 when divided by 9? Reply with only the number.",
            Kind::Number(157),
        ),
        c("math_modpow", "math", "What is the last digit of 7 to the power 222? Reply with only the digit.", Kind::Number(9)),
        // logic puzzles
        c(
            "logic_pets",
            "logic",
            "Alice, Bob, Carol and Dave each own exactly one different pet: a cat, a dog, a fish or a bird. Alice owns neither the dog nor the fish. Bob owns neither the cat nor the fish. Carol owns neither the cat nor the dog nor the fish. Dave does not own the bird. Who owns the fish? Reply with only the name.",
            Kind::Exact(&["dave"]),
        ),
        c(
            "logic_liars",
            "logic",
            "A, B and C each either always tell the truth or always lie. A says: \"B is lying.\" B says: \"C is lying.\" C says: \"A and B are both lying.\" Who is telling the truth? Reply with only the letter.",
            Kind::Exact(&["b"]),
        ),
        c(
            "logic_seats",
            "logic",
            "Five people A, B, C, D and E sit in seats 1 to 5 in a row. C sits immediately to the right of B. A is not in seat 1 or seat 5. D is in seat 1 or seat 5. E is not next to D. A sits somewhere to the right of E. A is not next to D. Who sits in seat 3? Reply with only the letter.",
            Kind::Exact(&["b"]),
        ),
        c(
            "logic_day",
            "logic",
            "If the day after tomorrow is Sunday, what day of the week was it 100 days ago? Reply with only the day name.",
            Kind::Exact(&["wednesday"]),
        ),
        c(
            "logic_sheep",
            "logic",
            "A farmer has 17 sheep. All but 9 run away, and then the farmer buys 4 more. How many sheep does the farmer have? Reply with only the number.",
            Kind::Number(13),
        ),
        // longer code tasks run against hidden tests
        c(
            "code_lru",
            "code",
            "Write a Python class LRUCache with __init__(self, capacity), get(self, key) that returns -1 when the key is missing, and put(self, key, value) that evicts the least recently used item when the capacity is exceeded. Both get and put count as a use. Reply with only the code.",
            Kind::Code(
                "c = LRUCache(2)\nc.put(1, 1)\nc.put(2, 2)\nassert c.get(1) == 1\nc.put(3, 3)\nassert c.get(2) == -1\nc.put(4, 4)\nassert c.get(1) == -1\nassert c.get(3) == 3 and c.get(4) == 4\nc.put(3, 30)\nassert c.get(3) == 30\nc.put(5, 5)\nassert c.get(4) == -1\nprint('OK')\n",
            ),
        ),
        c(
            "code_intervals",
            "code",
            "Write a Python function merge_intervals(iv) that takes a list of [start, end] lists in any order, merges the ones that overlap or touch (so [1, 2] and [2, 3] merge), and returns the merged list of lists sorted by start. Reply with only the code.",
            Kind::Code(
                "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]]) == [[1,5]]\nassert merge_intervals([]) == []\nassert merge_intervals([[5,6],[1,2]]) == [[1,2],[5,6]]\nassert merge_intervals([[1,10],[2,3],[4,5]]) == [[1,10]]\nprint('OK')\n",
            ),
        ),
        c(
            "code_substring",
            "code",
            "Write a Python function longest_unique(s) that returns the length of the longest substring of s with no repeated character. Reply with only the code.",
            Kind::Code(
                "assert longest_unique('abcabcbb') == 3\nassert longest_unique('bbbbb') == 1\nassert longest_unique('pwwkew') == 3\nassert longest_unique('') == 0\nassert longest_unique('abba') == 2\nassert longest_unique('tmmzuxt') == 5\nprint('OK')\n",
            ),
        ),
        c(
            "code_coins",
            "code",
            "Write a Python function min_coins(coins, amount) that returns the fewest coins from the list (each usable any number of times) that sum to amount, or -1 if that is impossible. min_coins(anything, 0) is 0. Reply with only the code.",
            Kind::Code(
                "assert min_coins([1,2,5], 11) == 3\nassert min_coins([2], 3) == -1\nassert min_coins([1], 0) == 0\nassert min_coins([186,419,83,408], 6249) == 20\nassert min_coins([5,10], 7) == -1\nprint('OK')\n",
            ),
        ),
        c(
            "code_islands",
            "code",
            "Write a Python function count_islands(grid) where grid is a list of strings of '0' and '1'; return the number of groups of '1' cells connected up, down, left or right. Reply with only the code.",
            Kind::Code(
                "assert count_islands(['11000','11000','00100','00011']) == 3\nassert count_islands([]) == 0\nassert count_islands(['111','010','111']) == 1\nassert count_islands(['101','010','101']) == 5\nassert count_islands(['0']) == 0\nprint('OK')\n",
            ),
        ),
        c(
            "code_decode",
            "code",
            "Write a Python function num_decodings(s) that returns the number of ways to decode a string of digits where 'A' is 1 up to 'Z' is 26 (so '12' is 2: AB or L). A leading zero or a zero not part of 10 or 20 gives 0. Reply with only the code.",
            Kind::Code(
                "assert num_decodings('12') == 2\nassert num_decodings('226') == 3\nassert num_decodings('06') == 0\nassert num_decodings('10') == 1\nassert num_decodings('100') == 0\nassert num_decodings('2101') == 1\nassert num_decodings('1111111111') == 89\nprint('OK')\n",
            ),
        ),
        c(
            "code_rain",
            "code",
            "Write a Python function trap(h) that takes a list of non-negative integers that are bar heights, each bar one unit wide, and returns how many units of rain water are trapped between the bars. Reply with only the code.",
            Kind::Code(
                "assert trap([0,1,0,2,1,0,1,3,2,1,2,1]) == 6\nassert trap([4,2,0,3,2,5]) == 9\nassert trap([]) == 0\nassert trap([3,3,3]) == 0\nassert trap([5,0,5]) == 5\nprint('OK')\n",
            ),
        ),
        c(
            "code_brackets",
            "code",
            "Write a Python function is_balanced(s) that returns True when every (, [ and { in s is closed by the matching bracket in the right order, ignoring other characters, and False otherwise. Reply with only the code.",
            Kind::Code(
                "assert is_balanced('([]{})') is True\nassert is_balanced('(]') is False\nassert is_balanced('((') is False\nassert is_balanced('a(b[c]d)e') is True\nassert is_balanced('') is True\nassert is_balanced(']') is False\nassert is_balanced('([)]') is False\nprint('OK')\n",
            ),
        ),
        // strict multi-constraint instructions
        c(
            "follow_ocean",
            "instruction",
            "Write one sentence about the ocean that has exactly 12 words, starts with the word Waves, ends with a single full stop, never uses the letter z and never uses the words blue or water. Reply with only the sentence.",
            Kind::Func(ocean_sentence),
        ),
        c(
            "follow_coffee",
            "instruction",
            "Write exactly three lines about coffee and nothing else. Each line starts with \"- \" and has exactly five words after it. The second line must be in capital letters. No line may contain the word the.",
            Kind::Func(coffee_bullets),
        ),
        c(
            "follow_bwords",
            "instruction",
            "Give five different English words, each exactly six letters long, all lower case and each starting with the letter b, in alphabetical order, separated by commas with no spaces and nothing else.",
            Kind::Func(b_words),
        ),
        c(
            "follow_acrostic",
            "instruction",
            "Write four lines about the sea. The first letters of the four lines must spell CODE. Each line has between three and six words and no commas. Reply with only the four lines.",
            Kind::Func(acrostic),
        ),
        c(
            "follow_two",
            "instruction",
            "Answer in exactly two sentences about why bread rises. The first sentence must contain the word because. The second must end with an exclamation mark. Together they have between 20 and 30 words, and the letter x must not appear anywhere. Reply with only the two sentences.",
            Kind::Func(two_sentences),
        ),
        c(
            "follow_lines",
            "instruction",
            "Write exactly three lines and nothing else. Line 1: the word alpha in upper case. Line 2: the number of letters in the word banana. Line 3: the word gamma spelled backwards, in lower case.",
            Kind::Lines(&["ALPHA", "6", "ammag"]),
        ),
        // tools and schemas
        Check {
            name: "tool_weather",
            category: "tools",
            prompt: format!("{TOOLS}What is the weather in Oslo? I think in Fahrenheit."),
            kind: Kind::Func(weather_call),
        },
        Check {
            name: "tool_table",
            category: "tools",
            prompt: format!("{TOOLS}Please book a table for four at Luigi's for 7 pm."),
            kind: Kind::Func(table_call),
        },
        Check {
            name: "tool_alarm",
            category: "tools",
            prompt: format!("{TOOLS}Wake me at a quarter to seven in the evening and label it gym."),
            kind: Kind::Func(alarm_call),
        },
        Check { name: "tool_none", category: "tools", prompt: format!("{TOOLS}What is the capital of France?"), kind: Kind::Func(no_tool) },
        c(
            "schema_movie",
            "tools",
            "Fill this JSON schema from the text and reply with only the JSON, no code fence: {\"title\": string, \"year\": integer, \"genres\": array of strings in the order given, \"rating\": number or null}. Use null when the text gives no rating, and no other keys. Text: \"Arrival (2016) is a science fiction drama, and also a mystery. It has not been rated yet.\"",
            Kind::JsonEq("{\"title\":\"Arrival\",\"year\":2016,\"genres\":[\"science fiction\",\"drama\",\"mystery\"],\"rating\":null}"),
        ),
    ];
    v.extend(needles(context.unwrap_or(4000)));
    v
}

pub fn quality_suite(suite: Suite, context: Option<u32>) -> Vec<Check> {
    match suite {
        Suite::Standard => standard(),
        Suite::Hard => hard(context),
        Suite::Quick => standard().into_iter().filter(|c| ["arithmetic_1", "json_format", "code_fib", "fact_capital"].contains(&c.name)).collect(),
    }
}

/// The expected value of a [`Kind::JsonEq`].
pub fn expected_json(s: &str) -> Value {
    serde_json::from_str(s).unwrap_or(Value::Null)
}
