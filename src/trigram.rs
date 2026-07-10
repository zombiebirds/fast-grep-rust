/// Returns true if `pattern` enables case-insensitive matching anywhere
/// via an inline `(?i…)` flag group. The trigram index is built from the
/// raw bytes of source files (case-sensitive), so a `(?i)abc` pattern
/// can match `ABC` in a file even though the trigram `abc` was never
/// recorded — the index would falsely report no candidates. Callers use
/// this as a short-circuit signal to fall back to a full-file scan.
///
/// Walks `(?…)` flag groups looking for `i` not preceded by `-` (the
/// `-i` form disables the flag rather than enabling it). Catches
/// `(?i)`, `(?im)`, `(?mi)`, `(?Ri)`, `(?i:…)`, etc. False positives
/// (treating `(?-i)abc` as case-insensitive) are harmless — we just
/// pay the full-scan cost unnecessarily.
pub fn has_case_insensitive_flag(pattern: &str) -> bool {
    let bytes = pattern.as_bytes();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'(' && bytes[i + 1] == b'?' {
            // Walk the flag-group prefix until `:` (scoped flags) or `)`
            // (top-level flags), checking each char.
            let mut j = i + 2;
            let mut negate = false;
            while j < bytes.len() {
                match bytes[j] {
                    b'-' => negate = true,
                    b'i' if !negate => return true,
                    b':' | b')' => break,
                    _ => {}
                }
                j += 1;
            }
            i = j;
        } else {
            i += 1;
        }
    }
    false
}

/// Decompose a regex pattern into literal trigrams that must appear in any match.
/// Returns a Vec of Vec<[u8;3]> where the outer vec is OR alternatives,
/// and each inner vec is AND-required trigrams for that alternative.
pub fn decompose_pattern(pattern: &str) -> Vec<Vec<[u8; 3]>> {
    decompose_inner(pattern, false)
}

/// Like [`decompose_pattern`], but each literal run is Unicode case-folded
/// (the same fold the CI index applies to content) before trigrams are taken.
/// Used to resolve `(?i)` queries against the case-insensitive companion index.
pub fn decompose_pattern_folded(pattern: &str) -> Vec<Vec<[u8; 3]>> {
    decompose_inner(pattern, true)
}

fn decompose_inner(pattern: &str, fold: bool) -> Vec<Vec<[u8; 3]>> {
    // Prefer AST-based decomposition. It correctly handles nested groups,
    // named captures (`(?P<n>…)`), lookarounds, and complex character
    // classes — cases the string-level scanner got subtly wrong. Fall
    // back to the legacy heuristic only if the AST parser rejects the
    // pattern (very rare for valid regex; the legacy code keeps us safe
    // on any malformed input that somehow reaches this layer).
    if let Some(result) = decompose_via_ast(pattern, fold) {
        return result;
    }
    decompose_via_string(pattern, fold)
}

/// AST-based decomposition. Returns `None` if `pattern` fails to parse —
/// the caller falls back to the string-level heuristic.
fn decompose_via_ast(pattern: &str, fold: bool) -> Option<Vec<Vec<[u8; 3]>>> {
    use regex_syntax::ast::{self, Ast};

    let mut parser = ast::parse::Parser::new();
    let ast = parser.parse(pattern).ok()?;

    // Top-level alternation defines the OR alternatives; a single AST
    // node counts as one alternative (one AND-set of trigrams).
    let branches: Vec<Ast> = match &ast {
        Ast::Alternation(alt) => alt.asts.clone(),
        _ => vec![ast.clone()],
    };

    let mut result = Vec::new();
    let mut folded_buf = Vec::new();
    for branch in &branches {
        let mut current = String::new();
        let mut runs: Vec<String> = Vec::new();
        collect_literal_runs(branch, &mut current, &mut runs);
        if !current.is_empty() {
            runs.push(current);
        }

        let mut trigrams: Vec<[u8; 3]> = Vec::new();
        for run in &runs {
            let bytes: &[u8] = if fold {
                crate::casefold::fold_into(run.as_bytes(), &mut folded_buf);
                &folded_buf
            } else {
                run.as_bytes()
            };
            if bytes.len() >= 3 {
                for w in bytes.windows(3) {
                    trigrams.push([w[0], w[1], w[2]]);
                }
            }
        }
        trigrams.sort();
        trigrams.dedup();
        result.push(trigrams);
    }

    // An empty alternative matches everything — short-circuit to the
    // "no constraint" sentinel that the searcher interprets as a full
    // scan.
    if result.iter().any(|v| v.is_empty()) {
        return Some(vec![vec![]]);
    }
    Some(result)
}

/// Walk `ast` accumulating a current literal run. Whenever we hit a
/// non-literal node we flush the current run into `runs`. Groups and
/// repetitions are transparent (we recurse into their inner AST); a
/// nested alternation breaks the run, matching the legacy "skip groups
/// with alternation" behaviour — extracting literals from inside an OR
/// would manufacture AND constraints that don't hold across branches.
fn collect_literal_runs(
    ast: &regex_syntax::ast::Ast,
    current: &mut String,
    runs: &mut Vec<String>,
) {
    use regex_syntax::ast::{Ast, RepetitionKind, RepetitionRange};

    match ast {
        Ast::Literal(lit) => {
            current.push(lit.c);
        }
        Ast::Concat(concat) => {
            for child in &concat.asts {
                collect_literal_runs(child, current, runs);
            }
        }
        Ast::Group(group) => {
            // Capture / non-capture / named groups all behave the same
            // for literal extraction.
            collect_literal_runs(&group.ast, current, runs);
        }
        Ast::Repetition(rep) => {
            // Repetition is only safe to fuse when the inner literal is
            // guaranteed to contribute at least one character. `*` / `?`
            // / `{0,n}` allow zero occurrences, so they must break the
            // run — fusing them would require trigrams that vanish
            // when the repetition matches the empty string (e.g.
            // `b*c` over `c`).
            //
            // `+` / `{m,}` / `{m,n}` with m >= 1 contribute at least
            // `m` copies; we fuse that minimum so the resulting trigrams
            // are present in every valid match.
            match &rep.op.kind {
                RepetitionKind::ZeroOrOne | RepetitionKind::ZeroOrMore => {
                    if !current.is_empty() {
                        runs.push(std::mem::take(current));
                    }
                }
                RepetitionKind::OneOrMore => {
                    collect_literal_runs(&rep.ast, current, runs);
                }
                RepetitionKind::Range(range) => {
                    let min = match range {
                        RepetitionRange::Exactly(n) => *n,
                        RepetitionRange::AtLeast(n) => *n,
                        RepetitionRange::Bounded(n, _) => *n,
                    };
                    if min == 0 {
                        if !current.is_empty() {
                            runs.push(std::mem::take(current));
                        }
                    } else {
                        for _ in 0..min {
                            collect_literal_runs(&rep.ast, current, runs);
                        }
                    }
                }
            }
        }
        Ast::Flags(_) => {
            // Inline flag group like `(?i)`: zero-width, no literal
            // content. Break the run so we don't accidentally fuse
            // surrounding literals across a flag boundary.
            if !current.is_empty() {
                runs.push(std::mem::take(current));
            }
        }
        Ast::Empty(_) => {}
        Ast::Dot(_)
        | Ast::Assertion(_)
        | Ast::ClassUnicode(_)
        | Ast::ClassPerl(_)
        | Ast::ClassBracketed(_)
        | Ast::Alternation(_) => {
            if !current.is_empty() {
                runs.push(std::mem::take(current));
            }
        }
    }
}

/// Legacy string-level decomposer. Used as a fallback when the AST
/// parser rejects a pattern. Kept verbatim from the previous
/// implementation so behaviour on any input the AST chokes on stays
/// identical.
fn decompose_via_string(pattern: &str, fold: bool) -> Vec<Vec<[u8; 3]>> {
    // Split on top-level '|' (not inside parens/brackets)
    let alternatives = split_alternatives(pattern);
    let mut result = Vec::new();
    let mut folded = Vec::new();
    for alt in &alternatives {
        let literals = extract_literal_runs(alt);
        let mut trigrams = Vec::new();
        for lit in &literals {
            let bytes: &[u8] = if fold {
                crate::casefold::fold_into(lit.as_bytes(), &mut folded);
                &folded
            } else {
                lit.as_bytes()
            };
            if bytes.len() >= 3 {
                for w in bytes.windows(3) {
                    trigrams.push([w[0], w[1], w[2]]);
                }
            }
        }
        trigrams.sort();
        trigrams.dedup();
        result.push(trigrams);
    }
    // Filter out empty alternatives (they match everything)
    if result.iter().any(|v| v.is_empty()) {
        return vec![vec![]];
    }
    result
}

fn split_alternatives(pattern: &str) -> Vec<String> {
    let mut alts = Vec::new();
    let mut current = String::new();
    let mut depth = 0i32;
    let mut bracket = false;
    let mut escape = false;

    for ch in pattern.chars() {
        if escape {
            current.push(ch);
            escape = false;
            continue;
        }
        if ch == '\\' {
            escape = true;
            current.push(ch);
            continue;
        }
        if bracket {
            current.push(ch);
            if ch == ']' {
                bracket = false;
            }
            continue;
        }
        match ch {
            '[' => {
                bracket = true;
                current.push(ch);
            }
            '(' => {
                depth += 1;
                current.push(ch);
            }
            ')' => {
                depth -= 1;
                current.push(ch);
            }
            '|' if depth == 0 => {
                alts.push(std::mem::take(&mut current));
            }
            _ => {
                current.push(ch);
            }
        }
    }
    alts.push(current);
    alts
}

/// Extract contiguous literal runs from a regex pattern.
/// Stops at metacharacters (., *, +, ?, [, (, {, |, ^, $).
fn extract_literal_runs(pattern: &str) -> Vec<String> {
    let mut runs = Vec::new();
    let mut current = String::new();
    let mut chars = pattern.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '\\' {
            // Escaped character — check if it's a literal
            if let Some(&next) = chars.peek() {
                match next {
                    'd' | 'D' | 'w' | 'W' | 's' | 'S' | 'b' | 'B' | 'A' | 'z' | 'Z' | 'p' | 'P' => {
                        // Not a literal — regex escape class
                        if !current.is_empty() {
                            runs.push(std::mem::take(&mut current));
                        }
                        chars.next();
                        // \p{...} and \P{...} — skip the brace-delimited property name
                        if (next == 'p' || next == 'P') && chars.peek() == Some(&'{') {
                            chars.next(); // skip '{'
                            while let Some(c) = chars.next() {
                                if c == '}' {
                                    break;
                                }
                            }
                        }
                    }
                    _ => {
                        // Escaped literal (e.g., \. \* etc)
                        current.push(chars.next().unwrap());
                    }
                }
            }
        } else if ch == '[' {
            // Skip entire character class — contents are not literals
            if !current.is_empty() {
                runs.push(std::mem::take(&mut current));
            }
            // Handle '^' and ']' as first char in class (e.g., [^]b] or []b])
            let mut first = true;
            if chars.peek() == Some(&'^') {
                chars.next();
            }
            while let Some(c) = chars.next() {
                if c == '\\' {
                    chars.next();
                    first = false;
                } else if c == '[' && chars.peek() == Some(&':') {
                    // POSIX class like [:alnum:] inside a bracket expression.
                    // The full pattern is [[:alnum:]] — inner closes at :] and
                    // outer bracket expression continues. Skip to :].
                    chars.next(); // consume ':'
                    while let Some(p) = chars.next() {
                        if p == ':' {
                            if chars.peek() == Some(&']') {
                                chars.next(); // consume closing ']'
                            }
                            break;
                        }
                    }
                    first = false;
                } else if c == ']' && !first {
                    break;
                } else {
                    first = false;
                }
            }
        } else if ch == '{' {
            // Skip repetition quantifier {n}, {n,}, {n,m}
            if !current.is_empty() {
                runs.push(std::mem::take(&mut current));
            }
            while let Some(c) = chars.next() {
                if c == '}' {
                    break;
                }
            }
        } else if ch == '(' {
            // Skip entire group if it contains alternation — extracting literals
            // from inside would produce AND constraints from OR alternatives.
            // For groups without alternation, just skip the group syntax.
            if !current.is_empty() {
                runs.push(std::mem::take(&mut current));
            }
            // Scan ahead to find matching ')' and check for '|'
            let mut depth = 1i32;
            let mut has_alt = false;
            let saved = chars.clone();
            while let Some(c) = chars.next() {
                match c {
                    '\\' => {
                        chars.next();
                    }
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    '|' if depth == 1 => has_alt = true,
                    _ => {}
                }
            }
            if !has_alt {
                // No alternation — re-parse group contents for literals
                chars = saved;
                // Skip non-capturing group syntax (?:, (?P<, etc.)
                if chars.peek() == Some(&'?') {
                    let mut lookahead = chars.clone();
                    lookahead.next();
                    if let Some(&after) = lookahead.peek() {
                        // Flag-group prefix chars the rust `regex` crate accepts:
                        // `i` `m` `s` `x` `u` `U` (standard) plus `R` (CRLF
                        // line-terminator mode); plus `:` (end of flag prefix
                        // before the group body), `P` (named-group `(?P<…>`),
                        // `-` (negate flag), `<`/`!`/`=` (lookaround prefixes).
                        // Missing one of these means the parser falls into
                        // the literal extractor and pulls trigrams from the
                        // regex syntax — which won't exist in source files
                        // and produces a false-empty candidate set.
                        if ":PimRsxuU-<!=".contains(after) {
                            chars.next();
                            while let Some(&c) = chars.peek() {
                                if c == ':' || c == ')' {
                                    chars.next();
                                    break;
                                }
                                chars.next();
                            }
                        }
                    }
                }
            }
        } else if is_meta(ch) {
            if !current.is_empty() {
                runs.push(std::mem::take(&mut current));
            }
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        runs.push(current);
    }
    runs
}

fn is_meta(ch: char) -> bool {
    matches!(
        ch,
        '.' | '*' | '+' | '?' | '[' | ']' | '(' | ')' | '{' | '}' | '|' | '^' | '$'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- decompose_pattern tests (ported from decomposeRegex in trigram.test.ts) ---

    #[test]
    fn extracts_required_trigrams_from_plain_literal() {
        let result = decompose_pattern("hello");
        assert_eq!(result.len(), 1);
        assert!(result[0].contains(&[b'h', b'e', b'l']));
        assert!(result[0].contains(&[b'e', b'l', b'l']));
        assert!(result[0].contains(&[b'l', b'l', b'o']));
    }

    #[test]
    fn returns_empty_trigrams_for_short_patterns() {
        let result = decompose_pattern("ab");
        // Short pattern → single alternative with no trigrams → falls back to vec![vec![]]
        assert!(result.iter().all(|v| v.is_empty()));
    }

    #[test]
    fn handles_alternation_producing_separate_branches() {
        let result = decompose_pattern("hello|world");
        assert_eq!(result.len(), 2);
        assert!(result[0].contains(&[b'h', b'e', b'l']));
        assert!(result[1].contains(&[b'w', b'o', b'r']));
    }

    #[test]
    fn extracts_trigrams_from_literal_parts_with_wildcards() {
        let result = decompose_pattern("function.*async");
        assert_eq!(result.len(), 1);
        assert!(result[0].contains(&[b'f', b'u', b'n']));
        assert!(result[0].contains(&[b'a', b's', b'y']));
    }

    #[test]
    fn handles_escaped_metacharacters_as_literals() {
        let result = decompose_pattern("a\\.b\\.c");
        assert_eq!(result.len(), 1);
        assert!(result[0].contains(&[b'a', b'.', b'b']));
    }

    #[test]
    fn handles_character_classes_by_breaking_literal_run() {
        let result = decompose_pattern("foo[abc]bar");
        assert_eq!(result.len(), 1);
        assert!(result[0].contains(&[b'f', b'o', b'o']));
        assert!(result[0].contains(&[b'b', b'a', b'r']));
    }

    #[test]
    fn handles_shorthand_classes_like_d_w() {
        let result = decompose_pattern("hello\\dworld");
        assert_eq!(result.len(), 1);
        assert!(result[0].contains(&[b'h', b'e', b'l']));
        assert!(result[0].contains(&[b'w', b'o', b'r']));
    }

    #[test]
    fn returns_empty_for_pure_wildcard_patterns() {
        let result = decompose_pattern(".*");
        assert!(result.iter().all(|v| v.is_empty()));
    }

    #[test]
    fn handles_nested_groups_in_alternation() {
        // (foo|bar)baz — parens are not top-level alternation, treated as single alternative
        let result = decompose_pattern("(foo|bar)baz");
        // Should not panic; result depends on implementation details
        assert!(result.len() >= 1);
    }

    #[test]
    fn char_class_not_treated_as_literal() {
        // [A-Z]olland should only produce trigrams from "olland", not from "A-Z"
        let result = decompose_pattern("[A-Z]olland");
        assert_eq!(result.len(), 1);
        assert!(result[0].contains(&[b'o', b'l', b'l']));
        assert!(result[0].contains(&[b'l', b'l', b'a']));
        assert!(!result[0].contains(&[b'A', b'-', b'Z']));
    }

    // --- AST-based improvements over the legacy string-level scanner ---

    #[test]
    fn named_capture_extracts_inner_literal() {
        // The legacy scanner consumed the literal as part of the flag-group
        // prefix and produced no trigrams at all. AST walking sees through
        // `(?P<name>...)` and extracts the inner literal.
        let result = decompose_pattern("(?P<name>hello)");
        assert_eq!(result.len(), 1);
        assert!(result[0].contains(&[b'h', b'e', b'l']));
        assert!(result[0].contains(&[b'e', b'l', b'l']));
        assert!(result[0].contains(&[b'l', b'l', b'o']));
    }

    #[test]
    fn repetition_with_min_two_fuses_minimum() {
        // foo{2,3}bar: the repetition is on `o` with a minimum of 2.
        // The AST walker fuses 2 copies of `o`, producing "foobar" as
        // the required literal run. The trigrams of "foobar" are
        // present in every valid match (`fooobar`, `foooobar`),
        // so the filter is sound.
        let result = decompose_pattern("foo{2,3}bar");
        assert_eq!(result.len(), 1);
        assert!(result[0].contains(&[b'f', b'o', b'o']));
        assert!(result[0].contains(&[b'o', b'o', b'b']));
        assert!(result[0].contains(&[b'o', b'b', b'a']));
        assert!(result[0].contains(&[b'b', b'a', b'r']));
    }

    #[test]
    fn star_repetition_breaks_run() {
        // b*c: zero or more b's, then c. Matches "c", "bc", "bbc", etc.
        // The trigram filter must NOT require "bc" because "c" alone
        // is a valid match. We break the run on `*` to fall back to
        // a full scan (empty trigrams) instead of over-constraining.
        let result = decompose_pattern("b*c");
        assert!(
            result.iter().all(|v| v.is_empty()),
            "expected empty trigrams (full-scan fallback), got {:?}",
            result
        );
    }

    #[test]
    fn question_repetition_breaks_run() {
        // ab?c: matches "ac" or "abc". Break on `?` keeps the
        // required trigram set conservative (falls back to full scan
        // rather than over-requiring "abc").
        let result = decompose_pattern("ab?c");
        assert!(result.iter().all(|v| v.is_empty()));
    }

    #[test]
    fn lookahead_breaks_literal_run() {
        // foo(?=bar)baz: lookahead is a zero-width assertion in the AST
        // and must split the surrounding literals.
        let result = decompose_pattern("foo(?=bar)baz");
        assert_eq!(result.len(), 1);
        assert!(result[0].contains(&[b'f', b'o', b'o']));
        assert!(result[0].contains(&[b'b', b'a', b'z']));
    }

    #[test]
    fn non_capturing_group_is_transparent() {
        // (?:foo)bar — non-capturing group should not appear in trigrams.
        let result = decompose_pattern("(?:foo)bar");
        assert_eq!(result.len(), 1);
        assert!(result[0].contains(&[b'f', b'o', b'o']));
        assert!(result[0].contains(&[b'o', b'o', b'b']));
        assert!(result[0].contains(&[b'o', b'b', b'a']));
        assert!(result[0].contains(&[b'b', b'a', b'r']));
    }

    #[test]
    fn unicode_property_class_breaks_run() {
        // \p{Greek}olland — Unicode property class is not a literal.
        let result = decompose_pattern(r"\p{Greek}olland");
        assert_eq!(result.len(), 1);
        assert!(result[0].contains(&[b'o', b'l', b'l']));
        assert!(result[0].contains(&[b'l', b'l', b'a']));
        assert!(result[0].contains(&[b'l', b'a', b'n']));
        assert!(result[0].contains(&[b'a', b'n', b'd']));
    }
}
